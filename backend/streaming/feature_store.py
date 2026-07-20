"""
Feature Store real-time berbasis Redis untuk MuleRadar.

State per akun di-update setiap transaksi masuk, lalu dipakai untuk:
  1. Feature vektor 13-dim (kompatibel dengan model XGBoost+TGN)
  2. Streaming features (velocity, rapid in-out, device sharing, burst)

Skema Redis per akun:
  acct:{id}            HASH  agregat kumulatif (degree, amount, night, dll)
  acct:{id}:dev        SET   device_id unik
  acct:{id}:in_cp      SET   pengirim unik (fan-in: siapa kirim ke akun ini)
  acct:{id}:out_cp     SET   penerima unik (fan-out: akun ini kirim ke siapa)
  acct:{id}:txwin      ZSET  timestamp transaksi keluar (velocity 1 jam)
  acct:{id}:inwin      ZSET  (timestamp -> amount) dana masuk (rapid in-out)

Semua operasi pakai pipeline supaya 1 transaksi = 1 round-trip.
"""

import os
import sys
import json
import time

import redis

# Backend dir on path agar bisa import feature_defs (definisi kanonik HHI).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_defs import counterparty_hhi as _canon_hhi   # fix HHI 3-Jul

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Window detik untuk fitur streaming
VELOCITY_WINDOW_S = 3600      # 1 jam
RAPID_INOUT_WINDOW_S = 600    # 10 menit
TX_RETENTION_S = 7200         # simpan 2 jam riwayat window (auto-trim)

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

# CATATAN (17-Jul, keputusan arsitektur "Prioritas 1" — fitur real-time):
# 4 fitur LINTAS-AKUN (device_sharing_count, n_institutions, pagerank,
# kcore_number) TIDAK bisa dihitung <5ms per-transaksi (butuh scan graph
# global) — jadi TIDAK di-update() di sini seperti 20 fitur lain. Sebagai
# gantinya, job periodik TERPISAH (streaming/refresh_graph_cache.py) menghitung
# ke-4nya secara batch (Neo4j GDS pageRank/kcore utk 2 fitur graph structural,
# Postgres utk 2 fitur network) lalu HSET langsung ke hash acct:{id} yg SAMA
# dipakai di sini (field baru, tak bentrok field counter existing) — TIDAK
# perlu round-trip Redis tambahan di get_model_features(), tinggal baca dari
# `h` yg sudah di-fetch. Kalau job belum pernah jalan utk akun ini (cold-start
# atau baru muncul), default 0.0 (self-heal siklus refresh berikutnya).
FEATURE_COLS = [
    # 13 baseline
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
    # 7 behavioral
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
    # 4 network/graph — diisi BATCH oleh refresh_graph_cache.py, dibaca di sini
    "device_sharing_count", "n_institutions", "pagerank", "kcore_number",
]

# Ambang structuring (threshold avoidance pattern, currency-agnostic)
_STRUCTURING_BANDS = [
    (9_500, 10_000), (95_000, 100_000), (950_000, 1_000_000),
    (9_500_000, 10_000_000), (95_000_000, 100_000_000), (450_000_000, 500_000_000),
]


class FeatureStore:
    """Redis-backed rolling feature store, update per transaksi."""

    # Lua: set hash field ke max(current, value) secara atomik
    _LUA_HMAX = """
    local cur = redis.call('HGET', KEYS[1], ARGV[1])
    local v = tonumber(ARGV[2])
    if (not cur) or (v > tonumber(cur)) then
        redis.call('HSET', KEYS[1], ARGV[1], v)
    end
    return 1
    """

    # Lua: update max_gap_s dari selisih ts sekarang vs last_tx_ts, lalu set last_tx_ts
    _LUA_UPDATE_GAP = """
    local last = redis.call('HGET', KEYS[1], 'last_tx_ts')
    if last then
        local gap = tonumber(ARGV[1]) - tonumber(last)
        if gap > 0 then
            local cur = redis.call('HGET', KEYS[1], 'max_gap_s')
            if (not cur) or (gap > tonumber(cur)) then
                redis.call('HSET', KEYS[1], 'max_gap_s', gap)
            end
        end
    end
    redis.call('HSET', KEYS[1], 'last_tx_ts', ARGV[1])
    return 1
    """

    def __init__(self, redis_url: str = None):
        self.r = redis.from_url(redis_url or REDIS_URL, decode_responses=True)
        self._hmax_script       = self.r.register_script(self._LUA_HMAX)
        self._update_gap_script = self.r.register_script(self._LUA_UPDATE_GAP)

    def ping(self) -> bool:
        try:
            return self.r.ping()
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────
    # UPDATE — dipanggil per transaksi masuk
    # ──────────────────────────────────────────────────────────────
    def update(self, tx: dict) -> None:
        """
        Update state untuk from_account (pengirim) dan to_account (penerima).
        tx: {from_account, to_account, amount, tx_timestamp, channel, device_id, hour}
        TIDAK menyentuh is_laundering (label) — murni dari sinyal transaksi.
        """
        frm = str(tx["from_account"])
        to = str(tx["to_account"])
        amount = float(tx.get("amount", 0) or 0)
        device = str(tx.get("device_id", "") or "")
        ts = self._epoch(tx.get("tx_timestamp"))
        # Jam-of-day HARUS konsisten dgn training (features.py EXTRACT(HOUR) =
        # jam APA ADANYA dari timestamp, bukan UTC). Fix timezone-skew 3-Jul:
        # jangan pakai gmtime(ts) yang menggeser ke UTC → night_tx_ratio meleset
        # 7 jam kalau data WIB. Pakai jam naive dari timestamp asli.
        hour = tx.get("hour")
        if hour is None:
            hour = self._naive_hour(tx.get("tx_timestamp"))
        is_night = 1 if int(hour) in NIGHT_HOURS else 0

        # Hitung flag behavioral sebelum pipeline
        is_round = 1 if (amount > 0 and abs(amount - round(amount / 10000) * 10000) < 1.0) else 0
        is_struct = 1 if any(lo <= amount < hi for lo, hi in _STRUCTURING_BANDS) else 0
        channel = str(tx.get("channel", "unknown") or "unknown")

        p = self.r.pipeline(transaction=False)

        # Daftar akun aktif (17-Jul, utk refresh_graph_cache.py tahu akun mana
        # yg perlu di-refresh fitur network-nya — bounded ke akun yg BENERAN
        # aktif di jalur real-time, bukan seluruh 2,3 juta akun historis).
        p.sadd("known_accounts", frm)
        p.sadd("known_accounts", to)

        # --- Pengirim (out) ---
        k = f"acct:{frm}"
        p.hincrby(k, "out_degree", 1)
        p.hincrbyfloat(k, "out_amount_sum", amount)
        p.hincrby(k, "total_tx", 1)
        p.hincrby(k, "night_count", is_night)
        self._hmax_script(keys=[k], args=["max_single_tx", amount], client=p)
        p.hset(k, "last_out_ts", ts)
        if device:
            p.sadd(f"{k}:dev", device)
        p.sadd(f"{k}:out_cp", to)
        p.zadd(f"{k}:txwin", {f"{ts}:{amount}": ts})
        # Behavioral counters
        if is_round:  p.hincrby(k, "round_tx_count", 1)
        if is_struct: p.hincrby(k, "structuring_count", 1)
        p.hincrby(f"{k}:chan", channel, 1)

        # --- Penerima (in) ---
        k2 = f"acct:{to}"
        p.hincrby(k2, "in_degree", 1)
        p.hincrbyfloat(k2, "in_amount_sum", amount)
        p.hincrby(k2, "total_tx", 1)
        p.hincrby(k2, "night_count", is_night)
        self._hmax_script(keys=[k2], args=["max_single_tx", amount], client=p)
        p.hset(k2, "last_in_ts", ts)
        if device:
            p.sadd(f"{k2}:dev", device)
        p.sadd(f"{k2}:in_cp", frm)
        p.zadd(f"{k2}:inwin", {f"{ts}:{amount}": ts})
        # Behavioral counters
        p.hincrby(f"{k2}:chan", channel, 1)

        p.execute()

        # Dormancy gap tracking (Lua, luar pipeline karena butuh read-then-write)
        self._update_gap_script(keys=[k],  args=[ts])
        self._update_gap_script(keys=[k2], args=[ts])

        # Trim window lama (probabilistik supaya tidak tiap transaksi)
        if ts % 17 == 0:
            self._trim(frm, ts)
            self._trim(to, ts)

    # ──────────────────────────────────────────────────────────────
    # READ — feature vektor + streaming features
    # ──────────────────────────────────────────────────────────────
    def get_model_features(self, account_id: str) -> list:
        """Return list 20 fitur (urutan FEATURE_COLS) untuk model.predict."""
        import math
        acc = str(account_id)
        k = f"acct:{acc}"
        now = time.time()

        # Fix (17-Jul, audit produksi skalabilitas): SEBELUMNYA 6 round-trip
        # Redis TERPISAH (hgetall, scard×2, zcount, zrange, hgetall) tiap
        # panggil -> di consumer, get_model_features + get_streaming_signals
        # = ~13 round-trip PER TRANSAKSI. Di skala "jutaan/hari" itu
        # bottleneck latency. Sekarang SATU pipeline -> 1 round-trip utk
        # semua read di fungsi ini. Hasil identik (cuma di-batch).
        p = self.r.pipeline(transaction=False)
        p.hgetall(k)                                          # [0] h
        p.scard(f"{k}:in_cp")                                 # [1] u_send
        p.scard(f"{k}:out_cp")                                # [2] u_recv
        p.zcount(f"{k}:txwin", now - VELOCITY_WINDOW_S, now)  # [3] tx_1h
        p.zrange(f"{k}:txwin", -20, -1, withscores=True)      # [4] pairs
        p.hgetall(f"{k}:chan")                                # [5] chan_raw
        res = p.execute()

        h = res[0]
        if not h:
            return [0.0] * len(FEATURE_COLS)

        in_deg  = float(h.get("in_degree", 0))
        out_deg = float(h.get("out_degree", 0))
        in_amt  = float(h.get("in_amount_sum", 0))
        out_amt = float(h.get("out_amount_sum", 0))
        total   = float(h.get("total_tx", 0))
        night   = float(h.get("night_count", 0))
        max_tx  = float(h.get("max_single_tx", 0))
        u_send  = float(res[1] or 0)
        u_recv  = float(res[2] or 0)

        # ── 7 behavioral features (real-time approximation) ──────────────
        # burst_ratio: transaksi dalam 1 jam terakhir / total
        tx_1h   = float(res[3] or 0)
        burst_ratio = tx_1h / max(total, 1)

        # inter_tx_std: std dev gap antar transaksi dari 20 entry txwin terbaru
        pairs = res[4]
        if len(pairs) >= 2:
            ts_vals = sorted(score for _, score in pairs)
            gaps = [ts_vals[i+1] - ts_vals[i] for i in range(len(ts_vals)-1)]
            inter_tx_std = float(sum((g - sum(gaps)/len(gaps))**2 for g in gaps) / len(gaps)) ** 0.5
        else:
            inter_tx_std = 0.0

        # dormancy_days: max gap antar transaksi (dari Lua update_gap)
        max_gap_s    = float(h.get("max_gap_s", 0))
        dormancy_days = max_gap_s / 86400.0

        # counterparty_hhi: definisi KANONIK (feature_defs, fix 3-Jul finding #4).
        # Konsisten dgn detection/features & ml/tgn_dataset → tak ada train/serve skew.
        counterparty_hhi = _canon_hhi(u_recv)

        # channel_entropy: hitung dari Redis Hash channel counts (res[5])
        chan_raw = res[5]
        if chan_raw:
            cnts  = [float(v) for v in chan_raw.values()]
            total_c = sum(cnts)
            if total_c > 0:
                channel_entropy = -sum(
                    (c / total_c) * math.log2(c / total_c + 1e-12)
                    for c in cnts
                )
            else:
                channel_entropy = 0.0
        else:
            channel_entropy = 0.0

        # structuring_score & round_amount_ratio: dari counter incremental
        structuring_score  = float(h.get("structuring_count", 0)) / max(out_deg, 1)
        round_amount_ratio = float(h.get("round_tx_count", 0)) / max(out_deg, 1)

        feats = {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "degree_ratio": out_deg / (in_deg + 1),
            "in_amount_sum": in_amt,
            "out_amount_sum": out_amt,
            "amount_ratio": out_amt / (in_amt + 1),
            "unique_senders": u_send,
            "unique_recipients": u_recv,
            "max_single_tx": max_tx,
            "night_tx_ratio": night / max(total, 1),
            "avg_amount_in": in_amt / (in_deg + 1),
            "avg_amount_out": out_amt / (out_deg + 1),
            "total_tx": total,
            "burst_ratio": burst_ratio,
            "inter_tx_std": inter_tx_std,
            "dormancy_days": dormancy_days,
            "counterparty_hhi": counterparty_hhi,
            "channel_entropy": channel_entropy,
            "structuring_score": structuring_score,
            "round_amount_ratio": round_amount_ratio,
            # 4 network/graph — diisi batch oleh refresh_graph_cache.py.
            # Default 0.0 kalau belum pernah di-refresh utk akun ini.
            "device_sharing_count": float(h.get("device_sharing_count", 0.0)),
            "n_institutions": float(h.get("n_institutions", 0.0)),
            "pagerank": float(h.get("pagerank", 0.0)),
            "kcore_number": float(h.get("kcore_number", 0.0)),
        }
        return [feats[c] for c in FEATURE_COLS]

    def get_streaming_signals(self, account_id: str, now_ts: float = None) -> dict:
        """
        Sinyal real-time yang TIDAK ada di fitur statis:
          - velocity_1h    : jumlah transaksi keluar dalam 1 jam
          - rapid_inout    : True jika dana masuk lalu keluar < 10 menit
          - device_count   : jumlah device unik (sharing detection)
          - burst_cp       : jumlah counterparty unik (fan-out signal)
        """
        acc = str(account_id)
        k = f"acct:{acc}"
        now = now_ts or time.time()

        # Fix (17-Jul, audit produksi skalabilitas): 7 round-trip -> 1 pipeline.
        p = self.r.pipeline(transaction=False)
        p.zcount(f"{k}:txwin", now - VELOCITY_WINDOW_S, now)      # [0] velocity_1h
        p.scard(f"{k}:dev")                                       # [1] device_count
        p.scard(f"{k}:out_cp")                                    # [2] burst_cp (fan-out)
        p.scard(f"{k}:in_cp")                                     # [3] fan_in (collector)
        p.zcount(f"{k}:inwin", now - RAPID_INOUT_WINDOW_S, now)   # [4] recent_in
        p.hget(k, "last_out_ts")                                  # [5] last_out
        p.hget(k, "last_in_ts")                                   # [6] last_in
        res = p.execute()

        velocity_1h = res[0] or 0
        device_count = res[1] or 0
        burst_cp = res[2] or 0
        fan_in = res[3] or 0
        recent_in = res[4] or 0
        last_out = res[5]
        last_in = res[6]
        rapid_inout = False
        if recent_in > 0 and last_out and last_in:
            gap = float(last_out) - float(last_in)
            rapid_inout = 0 <= gap <= RAPID_INOUT_WINDOW_S

        return {
            "velocity_1h": int(velocity_1h),
            "rapid_inout": bool(rapid_inout),
            "device_count": int(device_count),
            "burst_cp": int(burst_cp),
            "fan_in": int(fan_in),
        }

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────
    def _trim(self, account_id: str, now_ts: float):
        """Hapus entry window lebih tua dari TX_RETENTION_S."""
        k = f"acct:{account_id}"
        cutoff = now_ts - TX_RETENTION_S
        self.r.zremrangebyscore(f"{k}:txwin", 0, cutoff)
        self.r.zremrangebyscore(f"{k}:inwin", 0, cutoff)

    @staticmethod
    def _epoch(ts_val) -> float:
        """Konversi berbagai format timestamp ke epoch detik."""
        if ts_val is None:
            return time.time()
        if isinstance(ts_val, (int, float)):
            return float(ts_val)
        s = str(ts_val)
        from datetime import datetime
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s).timestamp()
        except Exception:
            return time.time()

    @staticmethod
    def _naive_hour(ts_val) -> int:
        """
        Jam-of-day dari timestamp APA ADANYA (naive), meniru DB EXTRACT(HOUR).
        TIDAK lewat epoch→gmtime (yang menggeser ke UTC) supaya konsisten dengan
        training features.py. Untuk timestamp string (jalur utama dari DB/Kafka),
        ambil .hour langsung dari datetime naive. Untuk epoch numerik (fallback),
        pakai localtime (asumsi server on-premise = zona data, mis. WIB) —
        BUKAN gmtime.
        """
        if ts_val is None:
            return time.localtime().tm_hour
        if isinstance(ts_val, (int, float)):
            return time.localtime(float(ts_val)).tm_hour
        s = str(ts_val)
        from datetime import datetime
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s[:19], fmt).hour
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s).hour
        except Exception:
            return time.localtime().tm_hour

    def stats(self) -> dict:
        """Statistik store (jumlah akun ter-track)."""
        return {"tracked_accounts": self.r.dbsize()}
