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
import json
import time

import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Window detik untuk fitur streaming
VELOCITY_WINDOW_S = 3600      # 1 jam
RAPID_INOUT_WINDOW_S = 600    # 10 menit
TX_RETENTION_S = 7200         # simpan 2 jam riwayat window (auto-trim)

NIGHT_HOURS = {22, 23, 0, 1, 2, 3}

# Urutan harus sama dengan FEATURE_COLS di detection/model.py
FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
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

    def __init__(self, redis_url: str = None):
        self.r = redis.from_url(redis_url or REDIS_URL, decode_responses=True)
        self._hmax_script = self.r.register_script(self._LUA_HMAX)

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
        hour = tx.get("hour")
        if hour is None:
            hour = time.gmtime(ts).tm_hour
        is_night = 1 if int(hour) in NIGHT_HOURS else 0

        p = self.r.pipeline(transaction=False)

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
        p.sadd(f"{k}:out_cp", to)            # pengirim → penerima (fan-out)
        p.zadd(f"{k}:txwin", {f"{ts}:{amount}": ts})

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
        p.sadd(f"{k2}:in_cp", frm)           # penerima ← pengirim (fan-in)
        p.zadd(f"{k2}:inwin", {f"{ts}:{amount}": ts})

        p.execute()

        # Trim window lama (probabilistik supaya tidak tiap transaksi)
        if ts % 17 == 0:
            self._trim(frm, ts)
            self._trim(to, ts)

    # ──────────────────────────────────────────────────────────────
    # READ — feature vektor + streaming features
    # ──────────────────────────────────────────────────────────────
    def get_model_features(self, account_id: str) -> list:
        """Return list 13 fitur (urutan FEATURE_COLS) untuk model.predict."""
        acc = str(account_id)
        k = f"acct:{acc}"
        h = self.r.hgetall(k)
        if not h:
            return [0.0] * len(FEATURE_COLS)

        in_deg = float(h.get("in_degree", 0))
        out_deg = float(h.get("out_degree", 0))
        in_amt = float(h.get("in_amount_sum", 0))
        out_amt = float(h.get("out_amount_sum", 0))
        total = float(h.get("total_tx", 0))
        night = float(h.get("night_count", 0))
        max_tx = float(h.get("max_single_tx", 0))
        u_send = float(self.r.scard(f"{k}:in_cp") or 0)   # pengirim unik (fan-in)
        u_recv = float(self.r.scard(f"{k}:out_cp") or 0)  # penerima unik (fan-out)

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

        velocity_1h = self.r.zcount(f"{k}:txwin", now - VELOCITY_WINDOW_S, now)
        device_count = self.r.scard(f"{k}:dev") or 0
        burst_cp = self.r.scard(f"{k}:out_cp") or 0    # fan-out
        fan_in = self.r.scard(f"{k}:in_cp") or 0        # fan-in (collector signal)

        # Rapid in-out: ada dana masuk dalam 10 menit terakhir + ada keluar setelahnya
        recent_in = self.r.zcount(f"{k}:inwin", now - RAPID_INOUT_WINDOW_S, now)
        last_out = self.r.hget(k, "last_out_ts")
        last_in = self.r.hget(k, "last_in_ts")
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

    def stats(self) -> dict:
        """Statistik store (jumlah akun ter-track)."""
        return {"tracked_accounts": self.r.dbsize()}
