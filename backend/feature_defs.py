"""
Feature contract mini — definisi KANONIK fitur yang rawan divergen antar modul.
Fix 3-Jul (audit finding #4): counterparty_hhi sebelumnya punya 3 definisi beda
(features.py TRUE HHI, tgn_dataset & feature_store 1/unique_recipients) → XGBoost
dilatih pada satu distribusi, di-skor pada distribusi lain (train/serve skew).

Semua modul (detection/features, ml/tgn_dataset, streaming/feature_store) HARUS
memakai definisi di sini agar tidak divergen lagi.
"""


FEATURE_COLS = [
    # 13 baseline
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
    # 7 behavioral
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
    # 2 network (fix 6-Jul, audit roadmap #2) — lihat _compute_device_sharing
    # & _compute_institution_diversity di detection/features.py utk justifikasi
    # kenapa masing2 bersih dari noise data-generation (bukan fake feature ke-2).
    "device_sharing_count", "n_institutions",
    # 2 graph structural (7-Jul, dorongan kejar AUC lebih tinggi) — fitur
    # MULTI-HOP pertama (semua di atas 1-hop/langsung dari riwayat akun
    # sendiri). Lihat _compute_graph_structural di detection/features.py.
    "pagerank", "kcore_number",
]


def counterparty_hhi(unique_recipients) -> float:
    """
    Proxy konsentrasi counterparty = 1 / jumlah penerima unik (clip ke >=1).

    Kenapa PROXY, bukan Herfindahl asli (SUM((cnt_i/total)^2)):
    Herfindahl asli butuh hitung transaksi per-penerima. Di skala penuh 181M edge
    itu tidak feasible secara memori (fast-path tgn_dataset sengaja approximate).
    Supaya SATU definisi konsisten di train DAN serve (tidak ada skew), kita pakai
    proxy yang bisa dihitung di SEMUA skala:
      - banyak penerima unik  → hhi kecil (dana tersebar)
      - sedikit penerima unik → hhi besar (dana terkonsentrasi, mule-like)

    Menerima skalar (int/float) atau pandas Series.
    """
    try:
        # pandas Series → vectorized
        import pandas as pd
        if isinstance(unique_recipients, pd.Series):
            return 1.0 / unique_recipients.clip(lower=1)
    except ImportError:
        pass
    return 1.0 / max(float(unique_recipients), 1.0)
