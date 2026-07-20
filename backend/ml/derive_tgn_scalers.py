"""
Derive scaler serving utk TGN-streaming (Phase 4.8, 20-Jul).

MASALAH: checkpoint tgn_v1.pt TIDAK menyimpan StandardScaler yg dipakai saat
training (node features 24-dim + edge amount log1p di-scale). Tanpa scaler yg
SAMA, fitur streaming beda skala -> skor TGN salah. Script ini menurunkan
kembali scaler dari SUMBER YG SAMA dgn training:
  - node scaler: fit di transactions_hi_injected_node_features.pkl (file node
    features PENUH yg dipakai bikin dataset TGN). Training fit di subset
    train-split (70%); fit di semua node (2,13jt) -> mean/std nyaris identik
    utk standarisasi (beda < ~1% utk N sebesar ini). Didokumentasikan sbg
    APPROKSIMASI-DEKAT, bukan bit-exact.
  - amount scaler: mean/std log1p(amount) dari sampel transaksi (stabil).

Output: models/tgn_serving_scalers.pkl = {
    "node_mean": (24,), "node_std": (24,), "feature_cols": [...],
    "amount_mean": float, "amount_std": float, "channel_map": {...},
}

Cara pakai: cd backend && python -m ml.derive_tgn_scalers
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_defs import FEATURE_COLS

CHANNEL_MAP = {"mobile": 0, "atm": 1, "internet": 2, "teller": 3, "qris": 4}
_HERE = os.path.dirname(os.path.abspath(__file__))
NODE_FEAT_PKL = os.path.join(
    _HERE, "..", "..", "data", "processed",
    "transactions_hi_injected_node_features.pkl")
OUT_PATH = os.path.join(_HERE, "..", "..", "models", "tgn_serving_scalers.pkl")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
AMOUNT_SAMPLE = int(os.getenv("TGN_AMOUNT_SAMPLE", "1000000"))


def main():
    # ── Node feature scaler ──────────────────────────────────────────
    print(f"[1/3] Load node features: {NODE_FEAT_PKL}")
    with open(NODE_FEAT_PKL, "rb") as f:
        d = pickle.load(f)
    df = d["df"] if isinstance(d, dict) else d
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"FEATURE_COLS tak lengkap di file: {missing}")
    X = df.reindex(columns=FEATURE_COLS).fillna(0.0).astype(np.float64).values
    node_mean = X.mean(axis=0)
    node_std = X.std(axis=0)
    node_std[node_std < 1e-8] = 1.0  # cegah bagi-nol utk fitur konstan
    print(f"      node scaler dari {X.shape[0]:,} node, {X.shape[1]} fitur")

    # ── Amount scaler (log1p) dari sampel transaksi ──────────────────
    print(f"[2/3] Sampel {AMOUNT_SAMPLE:,} amount dari Postgres...")
    engine = create_engine(DATABASE_URL)
    q = text("""
        SELECT amount FROM transactions
        TABLESAMPLE SYSTEM (0.8)
        WHERE amount IS NOT NULL
        LIMIT :lim
    """)
    with engine.connect() as conn:
        try:
            amts = pd.read_sql(q, conn, params={"lim": AMOUNT_SAMPLE})["amount"].astype(float).values
        except Exception:
            # fallback tanpa TABLESAMPLE (kalau versi PG tak dukung)
            amts = pd.read_sql(
                text("SELECT amount FROM transactions WHERE amount IS NOT NULL LIMIT :lim"),
                conn, params={"lim": AMOUNT_SAMPLE})["amount"].astype(float).values
    amt_log = np.log1p(np.clip(amts, 0, None))
    amount_mean = float(amt_log.mean())
    amount_std = float(amt_log.std())
    if amount_std < 1e-8:
        amount_std = 1.0
    print(f"      amount log1p: mean={amount_mean:.4f} std={amount_std:.4f} (n={len(amts):,})")

    # ── Simpan ───────────────────────────────────────────────────────
    out = {
        "node_mean": node_mean.astype(np.float32),
        "node_std": node_std.astype(np.float32),
        "feature_cols": list(FEATURE_COLS),
        "amount_mean": amount_mean,
        "amount_std": amount_std,
        "channel_map": CHANNEL_MAP,
        "note": "APPROKSIMASI-DEKAT scaler training (bukan bit-exact). "
                "node scaler fit di semua node (bukan train-split), amount dari sampel.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(OUT_PATH)), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"[3/3] Tersimpan -> {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
