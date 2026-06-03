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
XGB_PATH = os.path.join(_MODELS_DIR, "xgboost_v1.pkl")

# Threshold keputusan
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
    """Fast-path scorer: rolling features + XGBoost + streaming signal fusion."""

    def __init__(self, xgb_path: str = None, store: FeatureStore = None):
        self.store = store or FeatureStore()
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

    def score(self, tx: dict) -> dict:
        """
        Skor satu transaksi. Flow:
          1. update feature store (state akun)
          2. ambil fitur model + streaming signals
          3. XGBoost predict → base risk
          4. fusi dengan streaming signals → final risk
          5. keputusan: NONE / ALERT / FREEZE

        Return dict: {account_id, risk_score, base_score, risk_level,
                      decision, signals, reasons}
        """
        # 1. Update state (pengirim = subjek utama yang di-skor)
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
        if self._model_ok:
            X = np.array([feat_vec], dtype=np.float32)
            base = float(self.model.predict_proba(X)[0, 1])
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
