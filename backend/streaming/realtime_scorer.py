"""
Real-Time Risk Scorer — fast path AI untuk streaming.

Arsitektur lambda:
  FAST PATH (di sini): per transaksi → XGBoost pada rolling features
                       + streaming signal boost → risk score instan (<5ms)
  SLOW PATH (batch)  : TGN ensemble re-scoring mendalam (run_detection terjadwal)

TIDAK memakai label is_laundering — murni inference dari perilaku transaksi.
"""

import os
import pickle
import logging

import numpy as np

from feature_store import FeatureStore, FEATURE_COLS

logger = logging.getLogger(__name__)

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models"))
# Fix (17-Jul, keputusan arsitektur "Prioritas 1"): SEBELUMNYA pakai
# xgboost_realtime_v1.pkl (model 20-fitur, PR-AUC cuma 0,2595) krn 4 fitur
# graph/network (device_sharing_count, n_institutions, pagerank,
# kcore_number) belum ada jalur real-time-nya. Sekarang jalur itu ADA:
# streaming/refresh_graph_cache.py — job periodik (Neo4j GDS pageRank/kcore
# + Postgres device/institution) yg nge-cache ke-4 fitur itu ke hash Redis
# acct:{id}, dibaca feature_store.get_model_features() (24 kolom lengkap,
# SAMA urutan dgn feature_defs.FEATURE_COLS kanonik). Jadi sekarang balik
# pakai xgboost_v1.pkl — model UTAMA yg sama dipakai ml/ensemble.py, PR-AUC
# 0,9631 yg sudah dikutip di pitch/proposal. Trade-off jujur: ke-4 fitur ini
# BISA stale sampai ~30 menit (cadence refresh job, lihat catatan cadence di
# refresh_graph_cache.py) — akun yg BARU MUNCUL sejak siklus refresh terakhir
# dapat 0.0 utk ke-4nya (cold-start, sama pola dgn fitur live lain), self-heal
# siklus berikutnya. xgboost_realtime_v1.pkl TIDAK dihapus (biarkan ada utk
# fallback/A-B kalau perlu), tapi tak lagi dipakai default di sini.
XGB_PATH = os.path.join(_MODELS_DIR, "xgboost_v1.pkl")

# Threshold keputusan
# CATATAN (fix 4-Jul, redesign #4): angka 0.5/0.85 ini masih PLACEHOLDER —
# dipilih sebagai angka bulat, BUKAN hasil analisis biaya. FREEZE_THRESHOLD
# khususnya menentukan keputusan HUKUM (bekukan rekening nasabah), harus
# defensible ke regulator, bukan tebakan. Setelah model final (XGBoost+TGN+
# DyGFormer) selesai retrain & ada data validasi nyata, jalankan:
#     python -m ml.threshold_tuning --csv <val_predictions.csv> \
#         --cost-fn <biaya_1_fraud_lolos> --cost-fp <biaya_1_alert_palsu>
# lalu update dua angka ini dari hasilnya (dgn review compliance/risk team,
# bukan otomatis). Lihat ml/threshold_tuning.py untuk detail metodologi.
ALERT_THRESHOLD = 0.5
FREEZE_THRESHOLD = 0.85

# Typology yang secara hukum jelas ilegal → boleh auto-freeze (configurable)
AUTO_FREEZE_TYPOLOGIES = {"judol", "JUDOL_RING"}

def _graduated(value, tiers):
    """Return skor tertinggi yang threshold-nya terlampaui. tiers: [(thr, score), ...] desc."""
    for thr, score in tiers:
        if value >= thr:
            return score
    return 0.0


# Threshold berjenjang per sinyal (triage fast-path, explainable)
FANIN_TIERS    = [(100, 0.55), (50, 0.40), (30, 0.25)]   # collector / judol ring
FANOUT_TIERS   = [(100, 0.45), (50, 0.30), (25, 0.18)]   # smurfing / distribusi
VELOCITY_TIERS = [(50, 0.25), (20, 0.15)]                 # burst transaksi
RAPID_INOUT    = 0.20                                      # masuk-keluar < 10 menit
DEVICE_SHARE   = 0.15                                      # > 5 akun share device


class RealtimeScorer:
    """Fast-path scorer: rolling features + model (XGBoost/TGN) + signal fusion.

    Model base-path bisa dipilih via env SCORER_MODE (Phase 4.8):
      - "tgn" (DEFAULT sejak 20-Jul): TGN-streaming, memory-state per-akun di
        Redis. Math TERVERIFIKASI faithful thd jalur batch (PR-AUC test 0,9930;
        equivalence test corr 1,0, lihat ml/verify_tgn_streaming.py), node
        scaler EXACT, tervalidasi flag kolektor illicit nyata di jalur produksi
        (base 0,84 > XGBoost 0,71). Latency ~6,6ms/tx. Kalau checkpoint/scaler
        TGN hilang -> fallback OTOMATIS ke "xgb".
      - "xgb": XGBoost xgboost_v1.pkl (fallback/pembanding).
    Signal-score (fan_in/velocity/dst) IDENTIK di kedua mode — cuma base
    model yg beda.
    CATATAN OPERASIONAL: utk 4 fitur graph (pagerank/kcore/device_sharing/
    n_institutions) terisi, refresh_graph_cache.py HARUS jalan berkala
    (lihat Prioritas 1). Tanpa itu ke-4 fitur = 0 (skew) — TGN tetap flag
    (robust, tervalidasi) tapi akurasi penuh butuh cache fresh.
    """

    def __init__(self, xgb_path: str = None, store: FeatureStore = None,
                 mode: str = None):
        self.store = store or FeatureStore()
        _requested = (mode or os.getenv("SCORER_MODE", "tgn")).lower()
        # Validasi mode (fix QC 20-Jul): SCORER_MODE typo/ngawur sebelumnya
        # lolos apa adanya -> scorer TETAP jalan (jatuh ke cabang XGBoost di
        # score()) TAPI self.mode melaporkan nilai ngaco -> log consumer
        # menyesatkan ("Base model aktif: NGACO"). Sekarang: normalisasi ke
        # "tgn" + warning eksplisit, biar tak ada mode hantu.
        if _requested not in ("tgn", "xgb"):
            logger.warning("SCORER_MODE='%s' tidak dikenal (valid: tgn|xgb) — "
                           "pakai default 'tgn'", _requested)
            _requested = "tgn"
        self.mode = _requested

        # Base-path XGBoost (default) — selalu di-load sbg fallback.
        path = xgb_path or XGB_PATH
        if os.path.isfile(path):
            with open(path, "rb") as f:
                self.model = pickle.load(f)
            self._model_ok = True
            logger.info("XGBoost loaded for real-time scoring: %s", path)
        else:
            self.model = None
            self._model_ok = False
            logger.warning("XGBoost tidak ada di %s — fallback ke rule-only score", path)

        # Base-path TGN (opt-in) — kalau gagal load, fallback diam ke XGBoost.
        self.tgn = None
        if self.mode == "tgn":
            try:
                from tgn_streaming_scorer import TGNStreamingScorer
                self.tgn = TGNStreamingScorer(store=self.store)
                logger.info("SCORER_MODE=tgn — TGN-streaming aktif (eksperimental)")
            except Exception as e:
                logger.warning("TGN scorer gagal load (%s) — fallback ke XGBoost", e)
                self.mode = "xgb"

    def score(self, tx: dict, apply_update: bool = True) -> dict:
        """
        Skor satu transaksi. Flow:
          1. update feature store (state akun) — HANYA kalau apply_update=True
          2. ambil fitur model + streaming signals
          3. XGBoost predict → base risk
          4. fusi dengan streaming signals → final risk
          5. keputusan: NONE / ALERT / FREEZE

        Return dict: {account_id, risk_score, base_score, risk_level,
                      decision, signals, reasons}

        apply_update (fix 17-Jul, audit produksi at-least-once): FeatureStore
        pakai HINCRBY (out_degree/total_tx/amount_sum) yg TIDAK idempoten —
        kalau consumer beralih ke at-least-once (proses ulang pesan saat
        crash utk cegah data loss), memanggil update() 2x utk tx yg SAMA
        akan DOUBLE-COUNT state -> korupsi sinyal fan_in/degree yg jadi
        andalan scorer. Consumer sekarang punya dedup guard (Redis SET NX
        per tx_id): tx yg SUDAH pernah di-apply -> panggil score(apply_update
        =False) -> LEWATI update state (state sudah mencerminkan tx ini),
        cuma BACA state terkini + hitung keputusan (utk alert idempoten via
        deterministic alert_id di consumer). Lihat consumer.py main loop.
        """
        # 1. Update state (pengirim = subjek utama yang di-skor)
        if apply_update:
            self.store.update(tx)
        account_id = str(tx["from_account"])

        # 2. Features
        feat_vec = self.store.get_model_features(account_id)
        signals = self.store.get_streaming_signals(account_id)

        total_tx = feat_vec[FEATURE_COLS.index("total_tx")]

        # 3. SIGNAL SCORE — bukti perilaku teramati langsung (reliable, explainable)
        reasons = []
        signal_score = 0.0

        s_fanin = _graduated(signals.get("fan_in", 0), FANIN_TIERS)
        if s_fanin:
            signal_score += s_fanin
            reasons.append(f"fanin={signals['fan_in']}")

        s_fanout = _graduated(signals["burst_cp"], FANOUT_TIERS)
        if s_fanout:
            signal_score += s_fanout
            reasons.append(f"fanout={signals['burst_cp']}")

        s_vel = _graduated(signals["velocity_1h"], VELOCITY_TIERS)
        if s_vel:
            signal_score += s_vel
            reasons.append(f"velocity={signals['velocity_1h']}/h")

        if signals["rapid_inout"]:
            signal_score += RAPID_INOUT
            reasons.append("rapid_inout(<10m)")

        if signals["device_count"] > 5:
            signal_score += DEVICE_SHARE
            reasons.append(f"devices={signals['device_count']}")

        # 4. MODEL SCORE — sekunder, dengan cold-start guard.
        # Fitur dari riwayat parsial tidak reliable kalau history sedikit,
        # jadi kontribusi model di-gate oleh jumlah observasi.
        base = 0.0
        if self.mode == "tgn" and self.tgn is not None:
            # Base-path TGN-streaming (Phase 4.8). score_tx kelola memory-state
            # sendiri di Redis + hormati apply_update (idempotensi). Kalau gagal
            # (mis. Redis korup), fallback diam ke 0 — signal_score tetap jalan.
            try:
                # teruskan feat_vec yg SUDAH di-fetch (hindari get_model_features
                # dobel — Redis pipeline mahal, lihat catatan di score_tx)
                base = self.tgn.score_tx(tx, apply_update=apply_update, feat_vec=feat_vec)
            except Exception as e:
                logger.warning("TGN score_tx gagal (%s) — base=0, signal tetap jalan", e)
                base = 0.0
        elif self._model_ok:
            try:
                X = np.array([feat_vec], dtype=np.float32)
                base = float(self.model.predict_proba(X)[0, 1])
            except ValueError as e:
                # Awalnya (16-Jul) ini nangkep bug shape-mismatch nyata: 20 vs
                # 24 kolom. Sejak 17-Jul (feature_store.py FEATURE_COLS 24
                # kolom lengkap, lihat refresh_graph_cache.py), mismatch itu
                # sudah tak seharusnya terjadi lagi — try/except ini SEKARANG
                # murni safety net (mis. kalau ada isi Redis korup/tipe aneh)
                # supaya 1 transaksi anomali tak mematikan seluruh consumer;
                # signal_score (independen) tetap jalan walau model_contrib=0.
                logger.warning("XGBoost predict_proba gagal (data korup?): %s", e)
        # Trust factor: 0 saat history minim, naik ke 1 setelah ~20 transaksi
        trust = min(1.0, total_tx / 20.0)
        model_contrib = base * trust * 0.5   # model maksimal sumbang 0.5

        # 5. Fusi: sinyal teramati + kontribusi model (gated)
        risk = float(min(1.0, signal_score + model_contrib))
        if model_contrib > 0.15:
            reasons.append(f"model={base:.2f}(trust={trust:.1f})")

        # 5. Keputusan
        typology = tx.get("typology")
        if risk >= FREEZE_THRESHOLD and typology in AUTO_FREEZE_TYPOLOGIES:
            decision = "FREEZE"          # auto-freeze hanya untuk typology ilegal jelas
        elif risk >= FREEZE_THRESHOLD:
            decision = "ESCALATE"        # risiko tinggi → eskalasi ke analis (manusia)
        elif risk >= ALERT_THRESHOLD:
            decision = "ALERT"
        else:
            decision = "NONE"

        level = ("HIGH" if risk >= FREEZE_THRESHOLD else
                 "MEDIUM" if risk >= ALERT_THRESHOLD else "LOW")

        return {
            "account_id": account_id,
            "base_score": round(base, 4),
            "risk_score": round(risk, 4),
            "risk_level": level,
            "decision": decision,
            "signals": signals,
            "reasons": reasons,
        }
