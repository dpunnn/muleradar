"""
Latih model XGBoost KHUSUS real-time fast-path (streaming/realtime_scorer.py).

Kenapa terpisah dari xgboost_v1.pkl (model utama, dipakai ml/ensemble.py):
pagerank/kcore_number/device_sharing_count/n_institutions itu fitur graph
GLOBAL (butuh scan seluruh riwayat transaksi) — tidak bisa dihitung cepat
(<5ms) per-transaksi di jalur real-time. feature_store.py (Redis rolling
window) cuma pernah menghasilkan 20 fitur (13 baseline + 7 behavioral),
sedangkan xgboost_v1.pkl dilatih dgn 24 fitur kanonik (feature_defs.py) ->
predict_proba() di real-time crash ValueError shape mismatch (ditemukan
16-Jul saat tes live mode demo-replay).

Fix: model TERPISAH, dilatih HANYA dari 20 fitur yg feature_store.py bisa
hasilkan, disimpan ke models/xgboost_realtime_v1.pkl — xgboost_v1.pkl (dan
angka PR-AUC ensemble 0,9631 yg sudah dikutip di pitch/proposal) TIDAK
disentuh sama sekali.

Data: reuse retrain_features_checkpoint.pkl (300.000 akun, checkpoint
"pre-network-features" dari retrain_xgboost_stacked.py sebelumnya) — sudah
punya seluruh 20 fitur + label, TIDAK perlu re-extract dari Postgres
(hindari re-run query berat 144M baris).

CATATAN JUJUR: checkpoint ini tidak punya kolom first_seen_ts, jadi split
train/test FALLBACK ke random stratified (bukan temporal-inductive spt
model utama) — model ini utk triase cepat real-time, BUKAN pengganti
angka PR-AUC ensemble utama. Jangan kutip PR-AUC model ini sbg angka utama
di pitch/proposal.

Cara pakai:
  python train_realtime_xgboost.py
"""

import os

import pandas as pd

from detection.model import train_xgboost

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "retrain_features_checkpoint.pkl")
REALTIME_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "xgboost_realtime_v1.pkl"
)

# Harus SAMA PERSIS dgn streaming/feature_store.py::FEATURE_COLS (20 fitur
# yg genuinely bisa dihitung real-time dari rolling window Redis).
REALTIME_FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
]


def main():
    print(f"[realtime-train] Load checkpoint: {CHECKPOINT_PATH}")
    feats = pd.read_pickle(CHECKPOINT_PATH)
    print(f"[realtime-train] Shape: {feats.shape}, illicit rate: "
          f"{feats['is_laundering_label'].mean():.4f}")

    missing = [c for c in REALTIME_FEATURE_COLS if c not in feats.columns]
    if missing:
        raise ValueError(f"Checkpoint tidak punya kolom yg dibutuhkan: {missing}")

    train_xgboost(
        feats,
        feature_cols=REALTIME_FEATURE_COLS,
        model_path=REALTIME_MODEL_PATH,
    )
    print(f"[realtime-train] Selesai -> {os.path.abspath(REALTIME_MODEL_PATH)}")


if __name__ == "__main__":
    main()
