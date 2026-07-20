"""
Retrain XGBoost — versi BULK single-pass (4-Jul, rewrite ke-2).

Rewrite dari versi per-chunk (100rb akun x 24 chunk): kelihatan jalan (tak
crash) tapi proyeksi total ~24-30 jam. Root cause: query _SQL_OUT_FILTERED &
_SQL_CHANNEL_FILTERED, walau tabel sudah di-ANALYZE, tetap dieksekusi Postgres
sbg Parallel Seq Scan (scan PENUH 177 juta baris transactions) krn selektivitas
ANY(10rb id) sudah >2% dari tabel — itu scr matematis pilihan planner yang
BENAR per-query, tapi krn diulang 235x (sekali per batch), biaya scan penuh
itu DIBAYAR ULANG 235 kali. Fix: panggil detection.features.extract_features_bulk()
yang scan tabel HANYA SEKALI untuk semua akun sekaligus (lihat docstring
lengkap di detection/features.py).

Checkpoint/resume tetap ada, sekarang di level TS (temporal) chunk saja —
itu satu-satunya bagian yang genuinely butuh diproses per-akun (bukan
full-scan yang bisa di-bulk-kan jadi 1x).

Jalankan:
    cd backend
    python retrain_xgboost.py
"""

import os
import time

from sqlalchemy import create_engine

from detection.features import extract_features_bulk
from detection.model import train_xgboost

TS_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "retrain_ts_checkpoint.pkl")


def main():
    t0 = time.time()
    url = os.getenv(
        "DATABASE_URL",
        "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
    )
    engine = create_engine(url)

    feats = extract_features_bulk(engine, checkpoint_path=TS_CHECKPOINT_PATH)
    n_illicit = int(feats["is_laundering_label"].sum())
    print(
        f"[retrain] SEMUA features siap: shape={feats.shape} illicit={n_illicit:,} "
        f"({n_illicit/len(feats)*100:.2f}%) in {time.time()-t0:.0f}s",
        flush=True,
    )

    print("[retrain] training XGBoost (data inject, HHI kanonik) ...", flush=True)
    train_xgboost(feats)
    print(f"[retrain] RETRAIN_DONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
