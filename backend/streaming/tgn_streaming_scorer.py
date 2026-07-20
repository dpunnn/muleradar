"""
TGN-STREAMING scorer (Phase 4.8, 20-Jul) — real-time fast-path pakai TGN,
menggantikan XGBoost, sesuai rencana "lebih pintar walau lebih berat".

INSIGHT (paper Rossi et al. 2020): TGN didesain utk streaming — memory per-akun
di-update INCREMENTAL per-edge (O(1)). Yang batch-only selama ini cuma CARA
pakai-nya (ml/ensemble.py replay semua edge tiap panggil). Di sini: parameter
TERLATIH (msg_mlp, GRU, node_classifier dari tgn_v1.pt) dipakai apa adanya,
sedangkan MEMORY = STATE yg dibangun ulang dari stream, disimpan per-akun di
Redis (bukan buffer 1,98jt beku dari checkpoint).

Flow per transaksi (src->dst, t):
  1. nf_s = scaler(node_features_24(src))            (dari feature_store)
  2. mem_s, mem_d = Redis[acct:src:tgnmem], Redis[acct:dst:tgnmem]  (0 kalau baru)
  3. ea = [amount_scaled, channel_enc, hour]         (amount di-log1p+scale)
  4. dt = log1p(t - last_update[src]) / 20
  5. msg = msg_mlp(cat(mem_s, mem_d, nf_s, ea, dt))   (540-dim)
  6. new_mem_s = GRU(msg, mem_s); new_mem_d = GRU(msg, mem_d)
  7. tulis new_mem + last_update ke Redis
  8. logit = node_classifier(cat(new_mem_s, nf_s)); risk = sigmoid(logit)

STATUS (20-Jul): TERVERIFIKASI & JADI DEFAULT produksi (SCORER_MODE=tgn).
  - Math FAITHFUL thd jalur batch proven (ml/verify_tgn_streaming.py:
    korelasi 1,000000, max-diff 8e-07 vs reference yg hasilkan PR-AUC test
    0,9930). Node scaler EXACT (extract objek scaler dari npz training).
  - Tervalidasi flag kolektor illicit NYATA di jalur produksi (fitur
    feature_store live): base 0,84 > XGBoost 0,71 (ml/validate_tgn_
    production_path.py). Latency ~6,6ms/tx (~900 tx/s di 6 partisi).
RESIDUAL (bukan blocker, terdokumentasi):
  - Amount scaler approx (log1p sampel; exact tak tersimpan di mana pun —
    sekunder, cuma feed message, dampak kecil).
  - 4 fitur graph (pagerank/kcore/device_sharing/n_institutions) butuh
    refresh_graph_cache.py jalan berkala; tanpa itu = 0 (skew) — TGN tetap
    robust flag (tervalidasi), tapi akurasi penuh perlu cache fresh.
  - Memory mulai 0 (cold) tiap akun -> menghangat seiring transaksi. Inheren
    streaming, wajar.
  - Fallback OTOMATIS ke XGBoost kalau checkpoint/scaler TGN hilang.
"""

import os
import pickle
import logging

import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models"))
# Prefer checkpoint STRIPPED (3MB, tanpa buffer memory 2GB) utk load produksi
# ringan+cepat; fallback ke full tgn_v1.pt kalau stripped belum dibuat.
TGN_CKPT_STRIPPED = os.path.join(_MODELS_DIR, "tgn_serving_v1.pt")
TGN_CKPT_FULL = os.path.join(_MODELS_DIR, "tgn_v1.pt")
TGN_CKPT = TGN_CKPT_STRIPPED if os.path.isfile(TGN_CKPT_STRIPPED) else TGN_CKPT_FULL
TGN_SCALERS = os.path.join(_MODELS_DIR, "tgn_serving_scalers.pkl")

MEMORY_DIM = 256
NODE_FEAT_DIM = 24
EDGE_FEAT_DIM = 3
CHANNEL_MAP = {"mobile": 0, "atm": 1, "internet": 2, "teller": 3, "qris": 4}

# TTL memory TGN (fix QC 20-Jul). MASALAH: memory 256 float32 = ~1KB/akun TANPA
# expiry, sedangkan Redis dikonfig `--maxmemory 1gb --maxmemory-policy
# allkeys-lru` (docker-compose). Di skala jutaan akun, LRU akan meng-EVICT
# key SECARA ACAK — termasuk memory akun yg SEDANG AKTIF -> state hilang
# diam-diam -> skor mendadak turun tanpa jejak. Dgn TTL, akun yg TIDAK aktif
# N hari kedaluwarsa secara PREDIKTABEL (mereka toh cold-start lagi kalau
# muncul), sementara akun aktif ter-refresh TTL-nya tiap transaksi.
# Set 0 utk nonaktifkan TTL (butuh Redis maxmemory memadai).
TGN_MEM_TTL_S = int(os.getenv("TGN_MEM_TTL_S", str(30 * 24 * 3600)))  # 30 hari


class TGNStreamingScorer:
    """Skor risiko real-time via TGN, memory-state per-akun di Redis."""

    def __init__(self, store, ckpt_path: str = None, scalers_path: str = None):
        import torch  # import lokal — torch berat, jangan bebani modul yg tak pakai
        self._torch = torch
        self.store = store           # FeatureStore (Redis + get_model_features)
        self.r = store.r

        ckpt_path = ckpt_path or TGN_CKPT
        scalers_path = scalers_path or TGN_SCALERS

        # ── Scaler ───────────────────────────────────────────────────
        with open(scalers_path, "rb") as f:
            sc = pickle.load(f)
        self.node_mean = np.asarray(sc["node_mean"], dtype=np.float32)
        self.node_std = np.asarray(sc["node_std"], dtype=np.float32)
        self.amount_mean = float(sc["amount_mean"])
        self.amount_std = float(sc["amount_std"])
        self.channel_map = sc.get("channel_map", CHANNEL_MAP)
        # Urutan node feature: scaler.feature_cols HARUS == urutan yg dikembalikan
        # feature_store.get_model_features (== feature_defs.FEATURE_COLS, sudah
        # diverifikasi identik). Guard eksplisit di sini biar tak silent-skew.
        from feature_store import FEATURE_COLS as _FS_COLS
        if list(sc["feature_cols"]) != list(_FS_COLS):
            raise RuntimeError(
                "Urutan feature_cols scaler != feature_store.FEATURE_COLS — "
                "skew fitur, TGN streaming tak valid. Re-derive scaler.")

        # ── Model params (msg_mlp, GRU, node_classifier) ─────────────
        from ml.tgn_model import ManualTGN
        model = ManualTGN(
            num_nodes=2, node_feat_dim=NODE_FEAT_DIM, edge_feat_dim=EDGE_FEAT_DIM,
            memory_dim=MEMORY_DIM, hidden_dim=MEMORY_DIM, dropout=0.0)
        # weights_only=True (fix QC 20-Jul, KEAMANAN): default lama (False)
        # meng-unpickle penuh -> checkpoint yg dimanipulasi bisa MENGEKSEKUSI
        # KODE ARBITRER saat di-load. Utk sistem AML produksi itu tak bisa
        # diterima. Checkpoint serving kita cuma berisi tensor + primitif,
        # jadi weights_only=True cukup. Fallback ke False HANYA kalau load
        # gagal (mis. checkpoint lama berisi objek non-tensor) + warning jelas.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception as e:
            logger.warning("torch.load(weights_only=True) gagal utk %s (%s) — "
                           "fallback weights_only=False. PASTIKAN file checkpoint "
                           "tepercaya (bisa eksekusi kode saat unpickle).",
                           ckpt_path, e)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        # buang buffer memory 1,98jt (kita kelola state sendiri di Redis)
        sd = {k: v for k, v in sd.items() if k not in ("memory", "last_update")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # 'memory'/'last_update' sengaja hilang -> abaikan; sisanya harus cocok
        real_missing = [m for m in missing if m not in ("memory", "last_update")]
        if real_missing:
            raise RuntimeError(f"TGN load_state_dict param hilang: {real_missing}")
        model.eval()  # WAJIB: BatchNorm1d di node_classifier pakai running-stats
        self.model = model
        logger.info("TGNStreamingScorer siap (ckpt=%s)", ckpt_path)

    # ──────────────────────────────────────────────────────────────
    def _mem_key(self, acc):  return f"acct:{acc}:tgnmem"
    def _last_key(self, acc): return f"acct:{acc}:tgnlast"

    def _read_memory(self, acc) -> np.ndarray:
        raw = self.r.get(self._mem_key(acc))
        if raw is None:
            return np.zeros(MEMORY_DIM, dtype=np.float32)
        try:
            # decode_responses=True -> string; disimpan sbg latin-1 bytes
            # (latin-1 = pemetaan byte 0-255 <-> char 1:1, jadi lossless).
            b = raw.encode("latin-1") if isinstance(raw, str) else raw
            arr = np.frombuffer(b, dtype=np.float32)
        except Exception:
            logger.warning("TGN memory %s korup/tak ter-decode — reset ke nol", acc)
            return np.zeros(MEMORY_DIM, dtype=np.float32)
        if arr.shape[0] != MEMORY_DIM:
            return np.zeros(MEMORY_DIM, dtype=np.float32)
        # Guard NaN/Inf (fix QC 20-Jul): kalau state pernah tercemar nilai tak
        # hingga (instabilitas numerik / tulisan parsial), SEMUA skor akun itu
        # jadi NaN selamanya tanpa error jelas. Deteksi & reset, jangan diam.
        if not np.isfinite(arr).all():
            logger.warning("TGN memory %s mengandung NaN/Inf — reset ke nol", acc)
            return np.zeros(MEMORY_DIM, dtype=np.float32)
        return arr.copy()

    def _read_last(self, acc) -> float:
        v = self.r.get(self._last_key(acc))
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _write_state(self, acc, mem: np.ndarray, ts: float, pipe=None):
        client = pipe or self.r
        mem = np.asarray(mem, dtype=np.float32)
        # Jangan pernah PERSIST NaN/Inf (kalau lolos, akun itu rusak permanen).
        if not np.isfinite(mem).all():
            logger.warning("TGN memory baru utk %s NaN/Inf — tidak ditulis "
                           "(pertahankan state lama)", acc)
            return
        b = mem.tobytes().decode("latin-1")
        if TGN_MEM_TTL_S > 0:
            client.set(self._mem_key(acc), b, ex=TGN_MEM_TTL_S)
            client.set(self._last_key(acc), ts, ex=TGN_MEM_TTL_S)
        else:
            client.set(self._mem_key(acc), b)
            client.set(self._last_key(acc), ts)

    # ──────────────────────────────────────────────────────────────
    def score_tx(self, tx: dict, apply_update: bool = True,
                 feat_vec=None) -> float:
        """Skor risiko [0,1] utk from_account transaksi ini via TGN.

        apply_update=False (replay/redelivery, idempotensi) -> BACA state &
        hitung skor, TAPI JANGAN tulis memory (state sudah mencerminkan tx ini).

        feat_vec (opt, fix latency 20-Jul): kalau caller SUDAH panggil
        store.get_model_features(src) (mis. realtime_scorer.score()), teruskan
        di sini biar TIDAK fetch Redis DUA KALI (get_model_features = pipeline
        Redis, bagian termahal ~1,5-3ms). None -> fetch sendiri (backward-compat).
        """
        torch = self._torch
        src = str(tx["from_account"])
        dst = str(tx["to_account"])

        # 1. node features (24) -> scale
        if feat_vec is not None:
            feat = np.asarray(feat_vec, dtype=np.float32)
        else:
            feat = np.asarray(self.store.get_model_features(src), dtype=np.float32)
        if feat.shape[0] != NODE_FEAT_DIM:
            # feature_store harus balik 24; kalau tidak, TGN tak bisa skor
            logger.warning("TGN: node feature %d != %d, skip (skor 0)", feat.shape[0], NODE_FEAT_DIM)
            return 0.0
        # Guard input (QC 20-Jul): fitur NaN/Inf dari Redis korup akan merambat
        # jadi skor NaN -> tersimpan ke kolom alerts.risk_score. Bersihkan dulu.
        if not np.isfinite(feat).all():
            logger.warning("TGN: node feature %s ada NaN/Inf — di-nol-kan", src)
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        nf_s = (feat - self.node_mean) / self.node_std

        # 2. memory src+dst
        mem_s = self._read_memory(src)
        mem_d = self._read_memory(dst)

        # 3. edge attr [amount_scaled, channel, hour]
        amount = float(tx.get("amount", 0) or 0)
        amt_scaled = (np.log1p(max(amount, 0.0)) - self.amount_mean) / self.amount_std
        channel = self.channel_map.get(str(tx.get("channel", "")).lower(), 2)
        hour = tx.get("hour")
        if hour is None:
            hour = self.store._naive_hour(tx.get("tx_timestamp"))
        ea = np.array([amt_scaled, float(channel), float(hour)], dtype=np.float32)

        # 4. delta_t (log1p detik sejak update terakhir src) / 20
        ts = self.store._epoch(tx.get("tx_timestamp"))
        last = self._read_last(src)
        dt = np.float32(np.log1p(max(ts - last, 0.0)) / 20.0)

        with torch.no_grad():
            t_mem_s = torch.from_numpy(mem_s).unsqueeze(0)
            t_mem_d = torch.from_numpy(mem_d).unsqueeze(0)
            t_nf_s = torch.from_numpy(nf_s).unsqueeze(0)
            t_ea = torch.from_numpy(ea).unsqueeze(0)
            t_dt = torch.tensor([[dt]], dtype=torch.float32)

            # 5. message -> 6. GRU update (src & dst pakai msg sama, spt update_memory_only)
            msg_raw = torch.cat([t_mem_s, t_mem_d, t_nf_s, t_ea, t_dt], dim=-1)  # (1,540)
            msg = self.model.msg_mlp(msg_raw)
            new_mem_s = self.model.memory_updater(msg, t_mem_s)
            new_mem_d = self.model.memory_updater(msg, t_mem_d)

            # 8. classify src dari memory BARU + node features
            logit = self.model.node_classifier(torch.cat([new_mem_s, t_nf_s], dim=-1))
            risk = float(torch.sigmoid(logit).item())

        # Guard output (QC 20-Jul): jangan pernah kembalikan NaN — hulu
        # (realtime_scorer) akan memfusikannya jadi risk_score NaN lalu
        # ter-INSERT ke alerts.risk_score (kolom numeric) -> data rusak &
        # sulit dilacak. Lebih baik 0 + warning yg kelihatan.
        if not np.isfinite(risk):
            logger.warning("TGN: skor NaN/Inf utk %s — kembalikan 0.0", src)
            return 0.0

        # 7. tulis state (skip kalau replay — idempotensi, sama pola apply_update
        # di realtime_scorer/consumer)
        if apply_update:
            p = self.r.pipeline(transaction=False)
            self._write_state(src, new_mem_s.squeeze(0).numpy(), ts, pipe=p)
            self._write_state(dst, new_mem_d.squeeze(0).numpy(), ts, pipe=p)
            p.execute()

        return risk
