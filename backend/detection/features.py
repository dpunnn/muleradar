"""
MuleRadar Phase 3 — Feature Extraction.
Extract account-level features from PostgreSQL for ML model training & inference.

13 fitur baseline + 7 fitur behavioral (total 20):
  Behavioral baru: burst_ratio, inter_tx_std, dormancy_days, counterparty_hhi,
                   channel_entropy, structuring_score, round_amount_ratio
"""

import logging
import os
import threading
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from feature_defs import counterparty_hhi, FEATURE_COLS   # definisi kanonik (fix HHI 3-Jul)

logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

# ── SQL: 13 fitur baseline ─────────────────────────────────────────────────────
_SQL_IN = text("""
    SELECT to_account AS account_id,
           COUNT(*) AS in_degree,
           SUM(amount) AS in_amount_sum,
           COUNT(DISTINCT from_account) AS unique_senders,
           MAX(amount) AS max_single_tx,
           SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                IN (22,23,0,1,2,3) THEN 1 ELSE 0 END) AS night_tx_count,
           COUNT(*) AS total_tx,
           SUM(is_laundering) AS illicit_count
    FROM transactions
    GROUP BY to_account
""")

_SQL_OUT = text("""
    SELECT from_account AS account_id,
           COUNT(*) AS out_degree,
           SUM(amount) AS out_amount_sum,
           COUNT(DISTINCT to_account) AS unique_recipients,
           MAX(amount) AS max_single_tx_out,
           SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                IN (22,23,0,1,2,3) THEN 1 ELSE 0 END) AS night_tx_count_out,
           COUNT(*) AS total_tx_out,
           SUM(is_laundering) AS illicit_count_out,
           -- round_amount_ratio: fraction of txs with amount divisible by 10000
           SUM(CASE WHEN amount > 0
                    AND ABS(amount - ROUND(amount / 10000.0) * 10000.0) < 1.0
                    THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS round_amount_ratio,
           -- structuring_score: fraction of txs in "just-below-threshold" band
           -- detects amounts 90-99.5% of common round thresholds (threshold avoidance)
           SUM(CASE WHEN
               (amount >= 9500    AND amount < 10000)    OR
               (amount >= 95000   AND amount < 100000)   OR
               (amount >= 950000  AND amount < 1000000)  OR
               (amount >= 9500000 AND amount < 10000000) OR
               (amount >= 95000000 AND amount < 100000000) OR
               (amount >= 450000000 AND amount < 500000000)
               THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS structuring_score
    FROM transactions
    GROUP BY from_account
""")

_SQL_IN_FILTERED = text("""
    SELECT to_account AS account_id,
           COUNT(*) AS in_degree,
           SUM(amount) AS in_amount_sum,
           COUNT(DISTINCT from_account) AS unique_senders,
           MAX(amount) AS max_single_tx,
           SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                IN (22,23,0,1,2,3) THEN 1 ELSE 0 END) AS night_tx_count,
           COUNT(*) AS total_tx,
           SUM(is_laundering) AS illicit_count
    FROM transactions
    WHERE to_account = ANY(:ids)
    GROUP BY to_account
""")

_SQL_OUT_FILTERED = text("""
    SELECT from_account AS account_id,
           COUNT(*) AS out_degree,
           SUM(amount) AS out_amount_sum,
           COUNT(DISTINCT to_account) AS unique_recipients,
           MAX(amount) AS max_single_tx_out,
           SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                IN (22,23,0,1,2,3) THEN 1 ELSE 0 END) AS night_tx_count_out,
           COUNT(*) AS total_tx_out,
           SUM(is_laundering) AS illicit_count_out,
           SUM(CASE WHEN amount > 0
                    AND ABS(amount - ROUND(amount / 10000.0) * 10000.0) < 1.0
                    THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS round_amount_ratio,
           SUM(CASE WHEN
               (amount >= 9500    AND amount < 10000)    OR
               (amount >= 95000   AND amount < 100000)   OR
               (amount >= 950000  AND amount < 1000000)  OR
               (amount >= 9500000 AND amount < 10000000) OR
               (amount >= 95000000 AND amount < 100000000) OR
               (amount >= 450000000 AND amount < 500000000)
               THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS structuring_score
    FROM transactions
    WHERE from_account = ANY(:ids)
    GROUP BY from_account
""")

# counterparty_hhi: DIHAPUS (fix 3-Jul finding #4). Dulu TRUE Herfindahl via
# join mahal (SUM((cnt/total)^2)) — beda definisi dgn tgn_dataset & feature_store
# (1/unique_recipients) → train/serve skew. Sekarang dihitung dari unique_recipients
# pakai feature_defs.counterparty_hhi (kanonik), tanpa query terpisah.

# ── SQL: channel distribution utk SEMUA akun sekaligus (unfiltered) ───────────
# Dipakai extract_features_bulk() — 1x full-table scan utk semua akun, bukan
# per-chunk (lihat catatan arsitektur di extract_features_bulk).
_SQL_CHANNEL = text("""
    SELECT account_id, channel, SUM(cnt) AS cnt
    FROM (
        SELECT from_account AS account_id, channel, COUNT(*) AS cnt
        FROM transactions GROUP BY from_account, channel
        UNION ALL
        SELECT to_account AS account_id, channel, COUNT(*) AS cnt
        FROM transactions GROUP BY to_account, channel
    ) t
    GROUP BY account_id, channel
""")

# ── SQL: channel distribution per account (untuk entropy) ─────────────────────
_SQL_CHANNEL_FILTERED = text("""
    SELECT account_id, channel, SUM(cnt) AS cnt
    FROM (
        SELECT from_account AS account_id, channel, COUNT(*) AS cnt
        FROM transactions WHERE from_account = ANY(:ids)
        GROUP BY from_account, channel
        UNION ALL
        SELECT to_account AS account_id, channel, COUNT(*) AS cnt
        FROM transactions WHERE to_account = ANY(:ids)
        GROUP BY to_account, channel
    ) t
    GROUP BY account_id, channel
""")

# ── SQL: timestamps per account (untuk burst_ratio, inter_tx_std, dormancy_days)
# Cap 500 tx terbaru per akun untuk efisiensi; cukup untuk estimasi pola
#
# Fix insiden OOM 4-Jul: versi lama pakai ROW_NUMBER() OVER (PARTITION BY ...)
# yang HARUS sort SEMUA baris tiap akun dulu sebelum bisa ambil top 500 —
# untuk akun "hub" ekstrem (ditemukan ada yang >6,3 JUTA transaksi — pola
# fan-out/collector, persis yang typology injection cari) itu berarti sort
# jutaan baris per akun, bisa OOM walau work_mem dinaikkan. Rewrite pakai
# LATERAL + ORDER BY ... LIMIT 500: Postgres bisa pakai index composite
# idx_tx_from_ts/idx_tx_to_ts (from_account/to_account, tx_timestamp DESC)
# sebagai bounded top-N scan — BERHENTI begitu dapat 500 baris, tak peduli
# akun itu punya 500 atau 6 juta transaksi. Tak ada sort besar sama sekali.
_SQL_TS_FILTERED = text("""
    SELECT s.account_id, t.tx_timestamp
    FROM unnest(:ids) AS s(account_id)
    CROSS JOIN LATERAL (
        SELECT tx_timestamp FROM transactions
        WHERE from_account = s.account_id
        ORDER BY tx_timestamp DESC
        LIMIT 500
    ) t
    UNION ALL
    SELECT s.account_id, t.tx_timestamp
    FROM unnest(:ids) AS s(account_id)
    CROSS JOIN LATERAL (
        SELECT tx_timestamp FROM transactions
        WHERE to_account = s.account_id
        ORDER BY tx_timestamp DESC
        LIMIT 500
    ) t
""")

# ── Fitur NETWORK causal (7-Jul, sync Postgres pasca-insiden leak) ─────────────
# Fix KRITIS: device_sharing_count/n_institutions/pagerank/kcore_number itu
# fitur LINTAS-AKUN (gantung akun LAIN) — kalau dihitung dari SELURUH tabel
# tanpa batas waktu, insiden leak yg SAMA (kcore causal AUC 0.439 vs 0.772
# "bocor", lihat ml/tgn_dataset.py) akan terulang di jalur Postgres/XGBoost.
# Ketiganya WAJIB pakai WHERE tx_timestamp <= :cutoff — TIDAK seperti
# _SQL_IN/_SQL_OUT/_SQL_CHANNEL di atas (self-aggregate, riwayat akun itu
# sendiri, kategori leak beda & lebih bisa diterima — TETAP unfiltered).

_SQL_FIRST_SEEN = text("""
    SELECT account_id, MIN(ts) AS first_seen
    FROM (
        SELECT from_account AS account_id, tx_timestamp AS ts FROM transactions
        UNION ALL
        SELECT to_account AS account_id, tx_timestamp AS ts FROM transactions
    ) u
    GROUP BY account_id
""")

# Fix (12-Jul): tx_timestamp kolom Postgres BERTIPE `timestamp` asli, sedangkan
# :cutoff dikirim sbg epoch-seconds float (Python) -> psycopg2 error
# "operator does not exist: timestamp without time zone <= numeric". Cast
# eksplisit to_timestamp(:cutoff) (epoch seconds -> timestamptz) lalu ::timestamp
# (buang tz, samakan dgn kolom naive) supaya perbandingan valid.
_SQL_DEVICE_CAUSAL = text("""
    SELECT DISTINCT from_account AS account_id, device_id
    FROM transactions
    WHERE tx_timestamp <= to_timestamp(:cutoff)::timestamp AND device_id IS NOT NULL
""")

_SQL_INST_CAUSAL = text("""
    SELECT DISTINCT account_id, institution_id FROM (
        SELECT from_account AS account_id, institution_id, tx_timestamp
        FROM transactions WHERE tx_timestamp <= to_timestamp(:cutoff)::timestamp
        UNION ALL
        SELECT to_account AS account_id, institution_id, tx_timestamp
        FROM transactions WHERE tx_timestamp <= to_timestamp(:cutoff)::timestamp
    ) u
    WHERE institution_id IS NOT NULL
""")

_SQL_PAIRS_CAUSAL = text("""
    SELECT DISTINCT from_account, to_account
    FROM transactions
    WHERE tx_timestamp <= to_timestamp(:cutoff)::timestamp
""")


def extract_network_features_causal(engine, cutoff_ts) -> pd.DataFrame:
    """
    Ekstrak device_sharing_count/n_institutions/pagerank/kcore_number DARI
    Postgres, causal (cuma edge tx_timestamp <= cutoff_ts) — reuse fungsi yg
    SAMA & SUDAH TERVERIFIKASI dari jalur CSV/TGN (_compute_device_sharing,
    _compute_institution_diversity, _compute_graph_structural), supaya
    definisi tetap SATU (tak divergen lagi antara jalur Postgres vs CSV).
    """
    with engine.connect() as conn:
        conn.execute(text("SET work_mem = '256MB'"))
        device_rows = conn.execute(_SQL_DEVICE_CAUSAL, {"cutoff": cutoff_ts}).mappings().all()
        inst_rows = conn.execute(_SQL_INST_CAUSAL, {"cutoff": cutoff_ts}).mappings().all()
        pair_rows = conn.execute(_SQL_PAIRS_CAUSAL, {"cutoff": cutoff_ts}).mappings().all()

    df_device = pd.DataFrame(device_rows) if device_rows else pd.DataFrame(columns=["account_id", "device_id"])
    df_inst = pd.DataFrame(inst_rows) if inst_rows else pd.DataFrame(columns=["account_id", "institution_id"])
    df_pairs = pd.DataFrame(pair_rows) if pair_rows else pd.DataFrame(columns=["from_account", "to_account"])

    df_dev_feat = _compute_device_sharing(df_device)
    df_inst_feat = _compute_institution_diversity(df_inst)
    df_graph_feat = _compute_graph_structural(df_pairs)

    out = df_dev_feat.merge(df_inst_feat, on="account_id", how="outer")
    out = out.merge(df_graph_feat, on="account_id", how="outer")
    for col in ["device_sharing_count", "n_institutions", "pagerank", "kcore_number"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    return out


# ── Feature column definitions ─────────────────────────────────────────────────
# 13 baseline + 7 behavioral = 20 total
# HARUS identik dengan ml/eval_ablation.py, ml/tgn_dataset.py, detection/model.py
_SCHEMA_COLS = ["account_id"] + FEATURE_COLS + ["is_laundering_label"]


# ── Behavioral feature computation helpers ─────────────────────────────────────

def _compute_channel_entropy(df_channel: pd.DataFrame) -> pd.DataFrame:
    """
    Shannon entropy dari distribusi channel per account.
    entropy = 0: semua transaksi via satu channel (mono-channel, suspek layering)
    entropy tinggi: transaksi tersebar ke banyak channel
    """
    if df_channel.empty:
        return pd.DataFrame(columns=["account_id", "channel_entropy"])

    # Fix (4-Jul): SUM(cnt) di _SQL_CHANNEL_FILTERED bertipe numeric di Postgres
    # (SUM(bigint) -> numeric), psycopg2 balikin sbg decimal.Decimal — Decimal
    # tak bisa dioperasikan langsung dgn Python float (p + 1e-12 di bawah).
    df_channel = df_channel.copy()
    df_channel["cnt"] = df_channel["cnt"].astype(float)

    def entropy(cnts):
        total = cnts.sum()
        if total == 0:
            return 0.0
        p = cnts / total
        return float(-(p * np.log2(p + 1e-12)).sum())

    result = df_channel.groupby("account_id")["cnt"].apply(entropy).reset_index()
    result.columns = ["account_id", "channel_entropy"]
    return result


def _compute_temporal_features(df_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung 3 fitur temporal dari tabel timestamp per account:
      - burst_ratio   : max transaksi dalam window 1 jam / total_tx
      - inter_tx_std  : std dev detik antar transaksi (rendah = pola robotik)
      - dormancy_days : gap terpanjang antar transaksi (dalam hari)
    """
    if df_ts.empty:
        return pd.DataFrame(columns=["account_id", "burst_ratio", "inter_tx_std", "dormancy_days"])

    df_ts = df_ts.copy()
    df_ts["tx_timestamp"] = pd.to_datetime(df_ts["tx_timestamp"])
    # NaT (baris dgn timestamp kosong/gagal parse) jadi int64 sentinel saat
    # di-astype nanti -> subtraction overflow senyap (ketahuan dari
    # RuntimeWarning "overflow encountered in scalar subtract" pas full-run).
    # Buang di sini, bukan biarkan window calc kebawa sentinel value.
    df_ts = df_ts.dropna(subset=["tx_timestamp"])
    if df_ts.empty:
        return pd.DataFrame(columns=["account_id", "burst_ratio", "inter_tx_std", "dormancy_days"])
    df_ts = df_ts.sort_values(["account_id", "tx_timestamp"])

    records = []
    for acc_id, grp in df_ts.groupby("account_id"):
        ts = grp["tx_timestamp"].values  # numpy datetime64 array, sudah sorted
        n = len(ts)

        if n <= 1:
            records.append({
                "account_id": acc_id,
                "burst_ratio": 0.0,
                "inter_tx_std": 0.0,
                "dormancy_days": 0.0,
            })
            continue

        # Convert ke seconds (Unix epoch)
        ts_sec = ts.astype("datetime64[s]").astype(np.int64)

        # inter_tx_std: std dev detik antar transaksi berurutan
        gaps_sec = np.diff(ts_sec).astype(float)
        inter_tx_std = float(np.std(gaps_sec)) if len(gaps_sec) > 0 else 0.0

        # dormancy_days: gap terpanjang dalam hari
        dormancy_days = float(gaps_sec.max() / 86400.0) if len(gaps_sec) > 0 else 0.0

        # burst_ratio: max count dalam window 1 jam / total
        # Sliding window dengan two-pointer O(n)
        window_sec = 3600
        max_in_window = 1
        left = 0
        for right in range(n):
            while ts_sec[right] - ts_sec[left] > window_sec:
                left += 1
            max_in_window = max(max_in_window, right - left + 1)
        burst_ratio = float(max_in_window) / n

        records.append({
            "account_id": acc_id,
            "burst_ratio": burst_ratio,
            "inter_tx_std": inter_tx_std,
            "dormancy_days": dormancy_days,
        })

    return pd.DataFrame(records)


def _compute_device_sharing(df_device: pd.DataFrame, max_cluster_size: int = 20) -> pd.DataFrame:
    """
    device_sharing_count: jumlah akun LAIN yang memakai device_id sama,
    dibatasi ke cluster device KECIL (<= max_cluster_size akun).

    Kenapa dibatasi, bukan raw count (fix 6-Jul, audit roadmap #2): device_id
    akun organik di-assign via hash(account) % n_devices per-chunk 500K baris
    dgn n_devices cuma 500 slot (lihat data/scripts/postprocess.py assign_device)
    -> false-sharing masif murni artefak hashing, BUKAN sinyal. Verifikasi
    empiris di data asli (awk scan 6-Jul, sample 5M baris organik): rata-rata
    3,305 akun/device, SEMUA device organik yg disampel >20 akun. Sebaliknya
    cluster device fraud asli (DEV-VIC-*/DEV-PT-*/DEV-AGG-*/shared_dev di
    inject_typologies.py) 88.6% <= 20 akun (distribusi: <=1:20711 2-5:18191
    6-20:13629 vs 21-100:4064 >100:2674, dari 59,269 device fraud). Threshold
    ini motong noise pool organik, sisakan sinyal cluster fraud asli.
    """
    if df_device.empty:
        return pd.DataFrame(columns=["account_id", "device_sharing_count"])

    pairs = df_device.drop_duplicates()
    cluster_size = pairs.groupby("device_id")["account_id"].transform("nunique")
    tight = pairs.assign(cluster_size=cluster_size)
    tight = tight[tight["cluster_size"] <= max_cluster_size]

    result = (tight.groupby("account_id")["cluster_size"]
              .max()
              .sub(1)          # exclude diri sendiri dari hitungan
              .clip(lower=0)
              .reset_index()
              .rename(columns={"cluster_size": "device_sharing_count"}))
    return result


def _compute_institution_diversity(df_inst: pd.DataFrame) -> pd.DataFrame:
    """
    n_institutions: jumlah institution_id unik yang disentuh akun (in+out).

    Akun organik institution_id-nya di-assign RANDOM PER TRANSAKSI, independen
    dari account_id, dari set {BANK_A, BANK_B} saja (lihat postprocess.py) ->
    maksimum teoretis 2. Institusi lain (BANK_C..H, lihat _MULTI_BANKS di
    inject_typologies.py) CUMA muncul di skema pencucian lintas-bank yang
    di-inject -> n_institutions > 2 mustahil organik, sinyal pasti (bukan
    threshold empiris spt device_sharing, murni struktur data-generation).
    """
    if df_inst.empty:
        return pd.DataFrame(columns=["account_id", "n_institutions"])

    return (df_inst.drop_duplicates()
            .groupby("account_id")["institution_id"]
            .nunique()
            .reset_index()
            .rename(columns={"institution_id": "n_institutions"}))


def _compute_graph_structural(df_pairs: pd.DataFrame, damping: float = 0.85,
                               pr_iters: int = 30) -> pd.DataFrame:
    """
    PageRank + k-core number dari graf transaksi akun-ke-akun (7-Jul, dorongan
    kejar AUC lebih tinggi — semua 22 fitur sebelumnya 1-hop/langsung dari
    riwayat akun itu sendiri; ini fitur MULTI-HOP pertama, tangkap posisi
    struktural akun di JARINGAN keseluruhan, mis. "hub" penting yg dilewati
    banyak jalur uang meski volume transaksinya sendiri biasa saja).

    Dihitung di graf SEDERHANA dari df_pairs (from_account,to_account) unik
    (mis. cpout_running yg sudah ada, basis unique_recipients) — BUKAN dari
    raw baris transaksi (redundan utk struktur graf, cuma bikin lambat tanpa
    nambah informasi: PageRank/k-core soal ADA-TIDAKNYA hubungan, bukan
    berapa kali).

    - PageRank: power iteration di scipy sparse (skalabel, hindari networkx
      yg berat utk graf puluhan juta edge). Tangani dangling node (out-degree
      0) dgn redistribusi massa merata, bukan dibuang.
    - k-core: peeling algorithm divektorisasi (proses semua node berambang
      degree<=k per putaran via np.add.at, bukan loop Python per-node) —
      O(E) praktis, bukan O(N^2).
    """
    if df_pairs.empty:
        return pd.DataFrame(columns=["account_id", "pagerank", "kcore_number"])

    import scipy.sparse as sp

    accounts = pd.unique(pd.concat([df_pairs["from_account"], df_pairs["to_account"]],
                                    ignore_index=True))
    acc_to_idx = {a: i for i, a in enumerate(accounts)}
    n = len(accounts)

    row = df_pairs["from_account"].map(acc_to_idx).to_numpy()
    col = df_pairs["to_account"].map(acc_to_idx).to_numpy()
    data = np.ones(len(row), dtype=np.float64)
    A = sp.csr_matrix((data, (row, col)), shape=(n, n))

    # --- PageRank (directed, power iteration) ---
    out_deg = np.asarray(A.sum(axis=1)).flatten()
    dangling = out_deg == 0
    out_deg_safe = np.where(dangling, 1.0, out_deg)
    A_norm = (sp.diags(1.0 / out_deg_safe) @ A).tocsr()

    r = np.full(n, 1.0 / n)
    for _ in range(pr_iters):
        dangling_mass = r[dangling].sum() if dangling.any() else 0.0
        r = damping * (A_norm.T @ r) + damping * dangling_mass / n + (1 - damping) / n
    pagerank = r

    # --- k-core (undirected, peeling vektorisasi) ---
    A_undirected = A.maximum(A.T).tocsr()
    degree = np.asarray(A_undirected.sum(axis=1)).flatten().astype(np.int64)
    core_number = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)
    remaining_degree = degree.copy()
    indptr, indices = A_undirected.indptr, A_undirected.indices

    k = 0
    n_active = n
    while n_active > 0:
        to_remove = np.where(active & (remaining_degree <= k))[0]
        if len(to_remove) == 0:
            k += 1
            continue
        while len(to_remove) > 0:
            core_number[to_remove] = k
            active[to_remove] = False
            n_active -= len(to_remove)
            starts, ends = indptr[to_remove], indptr[to_remove + 1]
            chunks = [indices[s:e] for s, e in zip(starts, ends)]
            all_neighbors = np.concatenate(chunks) if chunks else np.empty(0, dtype=indices.dtype)
            if len(all_neighbors):
                np.add.at(remaining_degree, all_neighbors, -1)
            to_remove = np.where(active & (remaining_degree <= k))[0]
        k += 1

    return pd.DataFrame({
        "account_id": accounts,
        "pagerank": pagerank,
        "kcore_number": core_number.astype(np.float64),
    })


# ── Core merge ─────────────────────────────────────────────────────────────────

def _merge_in_out(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    df = df_in.merge(df_out, on="account_id", how="outer")
    df = df.fillna(0)

    for col in ["in_amount_sum", "out_amount_sum", "max_single_tx",
                "max_single_tx_out", "illicit_count", "illicit_count_out"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    df["max_single_tx"]   = df[["max_single_tx", "max_single_tx_out"]].max(axis=1)
    df["night_tx_count"]  = df["night_tx_count"] + df["night_tx_count_out"]
    df["total_tx"]        = df["total_tx"] + df["total_tx_out"]
    df["illicit_count"]   = df["illicit_count"] + df["illicit_count_out"]
    df = df.drop(columns=["max_single_tx_out", "night_tx_count_out",
                           "total_tx_out", "illicit_count_out"])

    for col in ["in_degree", "out_degree", "unique_senders", "unique_recipients",
                "night_tx_count", "total_tx"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Baseline derived features
    df["degree_ratio"]     = df["out_degree"] / (df["in_degree"] + 1)
    df["amount_ratio"]     = df["out_amount_sum"] / (df["in_amount_sum"] + 1)
    df["night_tx_ratio"]   = df["night_tx_count"] / df["total_tx"].clip(lower=1)
    df["avg_amount_in"]    = df["in_amount_sum"] / (df["in_degree"] + 1)
    df["avg_amount_out"]   = df["out_amount_sum"] / (df["out_degree"] + 1)
    df["is_laundering_label"] = (df["illicit_count"] > 0).astype(int)

    # round_amount_ratio & structuring_score — sudah dari SQL_OUT, fillna 0 sudah di atas
    for col in ["round_amount_ratio", "structuring_score"]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].astype(float)

    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features(
    engine=None,
    account_ids: list = None,
    limit: int = 100_000,
) -> pd.DataFrame:
    """
    Extract 20-feature account-level DataFrame for training or inference.

    Parameters
    ----------
    engine : sqlalchemy.Engine, optional
    account_ids : list, optional
        If None, samples 80% illicit + 20% licit up to *limit* accounts.
    limit : int

    Returns
    -------
    pd.DataFrame  — columns: account_id + 20 features + is_laundering_label
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(
            f"Gagal konek DB: {exc}. Cek DATABASE_URL={DATABASE_URL!r}"
        ) from exc

    if account_ids is not None:
        return _extract_batch(engine, account_ids)

    illicit_limit = int(limit * 0.8)
    licit_limit   = limit - illicit_limit

    try:
        with engine.connect() as conn:
            illicit_accounts = conn.execute(text("""
                SELECT DISTINCT account_id FROM (
                    SELECT from_account AS account_id FROM transactions
                    WHERE is_laundering = 1
                    UNION
                    SELECT to_account FROM transactions
                    WHERE is_laundering = 1
                ) t ORDER BY account_id LIMIT :lim
            """), {"lim": illicit_limit}).scalars().all()

            licit_accounts = conn.execute(text("""
                SELECT account_id FROM (
                    SELECT from_account AS account_id FROM transactions
                    UNION
                    SELECT to_account FROM transactions
                ) t
                WHERE account_id != ALL(:exclude)
                ORDER BY RANDOM() LIMIT :lim
            """), {"exclude": list(illicit_accounts), "lim": licit_limit}).scalars().all()
    except Exception as exc:
        raise RuntimeError(f"Gagal sampling accounts: {exc}") from exc

    all_ids = list(illicit_accounts) + list(licit_accounts)
    print(f"[FEATURES] Sampling {len(illicit_accounts)} illicit + "
          f"{len(licit_accounts)} licit = {len(all_ids)} accounts")
    return _extract_batch(engine, all_ids)


def _extract_batch(engine, account_ids: list) -> pd.DataFrame:
    """Run all queries in batches of 10K, merge into final 20-feature DataFrame."""
    batch_size = 10_000
    in_frames, out_frames = [], []
    channel_frames, ts_frames = [], []   # hhi_frames dihapus (fix HHI 3-Jul)

    n_accounts = len(account_ids)
    n_batches = max(1, -(-n_accounts // batch_size))  # ceil div
    t_start = time.time()

    for batch_num, i in enumerate(range(0, n_accounts, batch_size), start=1):
        chunk = account_ids[i: i + batch_size]
        t_batch0 = time.time()
        try:
            with engine.connect() as conn:
                # Fix insiden OOM 4-Jul: _SQL_TS_FILTERED (window function per
                # akun) tetap butuh Sort walau sudah pakai index composite
                # idx_tx_from_ts/idx_tx_to_ts — pola WHERE ... = ANY(banyak id)
                # bikin Postgres tak bisa jamin urutan gabungan dari Nested Loop
                # index scan, jadi tetap sort manual. Kalau satu batch kebetulan
                # berisi akun "hub" (rekening kolektor, transaksi sangat banyak
                # — pola mule network), partition-nya besar & bisa OOM di
                # work_mem default (biasanya cuma 4MB). Naikkan work_mem KHUSUS
                # koneksi ini (bukan global Postgres) sebagai lapis pengaman.
                conn.execute(text("SET work_mem = '256MB'"))
                in_rows      = conn.execute(_SQL_IN_FILTERED,      {"ids": chunk}).mappings().all()
                out_rows     = conn.execute(_SQL_OUT_FILTERED,     {"ids": chunk}).mappings().all()
                channel_rows = conn.execute(_SQL_CHANNEL_FILTERED, {"ids": chunk}).mappings().all()
                ts_rows      = conn.execute(_SQL_TS_FILTERED,      {"ids": chunk}).mappings().all()
        except Exception as exc:
            raise RuntimeError(f"Gagal query batch offset={i}: {exc}") from exc

        if in_rows:      in_frames.append(pd.DataFrame(in_rows))
        if out_rows:     out_frames.append(pd.DataFrame(out_rows))
        if channel_rows: channel_frames.append(pd.DataFrame(channel_rows))
        if ts_rows:      ts_frames.append(pd.DataFrame(ts_rows))

        # Progress real-time (fix 4-Jul): sebelumnya 235 batch tanpa info
        # sama sekali — proses TERLIHAT macet padahal jalan normal, cuma
        # tiap batch jalankan 4 query berat (salah satunya window function)
        # ke tabel ratusan juta baris. Sekarang: batch ke-berapa, %, akun
        # terproses, waktu batch ini, elapsed, dan ETA sisa waktu.
        elapsed = time.time() - t_start
        batch_time = time.time() - t_batch0
        avg_per_batch = elapsed / batch_num
        eta_sec = avg_per_batch * (n_batches - batch_num)
        pct = batch_num / n_batches * 100
        print(
            f"[FEATURES] Batch {batch_num}/{n_batches} ({pct:.1f}%) | "
            f"{min(i + batch_size, n_accounts):,}/{n_accounts:,} akun | "
            f"batch={batch_time:.1f}s elapsed={elapsed/60:.1f}m ETA={eta_sec/60:.1f}m",
            flush=True,
        )

    df_in  = pd.concat(in_frames,  ignore_index=True) if in_frames  else pd.DataFrame()
    df_out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    if df_in.empty and df_out.empty:
        logger.warning("[FEATURES] No data returned — empty DataFrame with schema")
        return pd.DataFrame(columns=_SCHEMA_COLS)

    # 13 baseline features
    df = _merge_in_out(df_in, df_out)

    # counterparty_hhi — fix 3-Jul (finding #4): pakai definisi KANONIK
    # (feature_defs.counterparty_hhi = 1/unique_recipients). Dulu TRUE HHI via
    # _SQL_HHI_FILTERED (join mahal) yang BEDA dgn tgn_dataset/feature_store →
    # train/serve skew. Sekarang satu definisi + hemat 1 query berat.
    # Guard: batch inbound-only bisa bikin df_out kosong → kolom unique_recipients
    # tak ada; fallback ke 0 (HHI=1.0, konsisten dgn 1/max(0,1)).
    u_recv = (df["unique_recipients"] if "unique_recipients" in df.columns
              else pd.Series(0.0, index=df.index))
    df["counterparty_hhi"] = counterparty_hhi(u_recv.clip(lower=1))
    df["counterparty_hhi"] = df["counterparty_hhi"].fillna(0.0)

    # channel_entropy
    df_chan = pd.concat(channel_frames, ignore_index=True) if channel_frames else pd.DataFrame()
    df_ent  = _compute_channel_entropy(df_chan)
    if not df_ent.empty:
        df = df.merge(df_ent, on="account_id", how="left")
    else:
        df["channel_entropy"] = 0.0
    df["channel_entropy"] = df["channel_entropy"].fillna(0.0)

    # burst_ratio, inter_tx_std, dormancy_days
    df_ts  = pd.concat(ts_frames,  ignore_index=True) if ts_frames  else pd.DataFrame()
    df_tmp = _compute_temporal_features(df_ts)
    if not df_tmp.empty:
        df = df.merge(df_tmp, on="account_id", how="left")
    else:
        df["burst_ratio"]    = 0.0
        df["inter_tx_std"]   = 0.0
        df["dormancy_days"]  = 0.0
    for col in ["burst_ratio", "inter_tx_std", "dormancy_days"]:
        df[col] = df[col].fillna(0.0)

    # Final cleanup
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Pastikan semua kolom schema ada
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            df[col] = 0

    print(f"[FEATURES] Extracted {len(df)} accounts, "
          f"{df['is_laundering_label'].sum()} illicit, "
          f"{len(FEATURE_COLS)} features")
    return df


def _run_with_heartbeat(label: str, fn, interval: int = 30):
    """
    Jalankan fn() sambil print heartbeat tiap `interval` detik.
    Fix (4-Jul): query bulk (IN/OUT/CHANNEL) itu SATU query full-scan yang
    butuh menit-menitan tanpa progress apapun dari psycopg2/SQLAlchemy —
    tanpa ini, terminal cuma diam total dan terlihat seperti macet (padahal
    sedang genuine disk I/O). Heartbeat ini cuma print elapsed time, BUKAN
    progress asli (Postgres tak expose progress utk plain SELECT).
    """
    stop_event = threading.Event()
    t_start = time.time()

    def _heartbeat():
        while not stop_event.wait(interval):
            print(f"[FEATURES-BULK]   ... {label} masih jalan (elapsed={time.time()-t_start:.0f}s)",
                  flush=True)

    hb_thread = threading.Thread(target=_heartbeat, daemon=True)
    hb_thread.start()
    try:
        return fn()
    finally:
        stop_event.set()
        hb_thread.join(timeout=1)


def extract_features_for_accounts(engine, account_ids: list) -> pd.DataFrame:
    """Public inference API — same as _extract_batch."""
    if not account_ids:
        return pd.DataFrame(columns=_SCHEMA_COLS)
    return _extract_batch(engine, account_ids)


def extract_features_bulk(
    engine=None,
    ts_chunk_size: int = 50_000,
    checkpoint_path: str = None,
) -> pd.DataFrame:
    """
    Ekstrak fitur utk SEMUA akun sekaligus dlm satu pass (dipakai retrain full-data).

    Fix arsitektur (4-Jul, rewrite ke-2 stlh fix OOM): retrain_xgboost.py versi
    lama panggil extract_features() per-chunk 10rb akun x 235 batch. Query
    _SQL_OUT_FILTERED & _SQL_CHANNEL_FILTERED, WALAU sudah di-ANALYZE, tetap
    dieksekusi Postgres sbg Parallel Seq Scan (scan PENUH 177 juta baris transactions)
    krn selektivitas ANY(10rb id) sudah >2% dari tabel — planner scan-sekali
    lebih murah drpd 10rb index probe acak. Ini scr matematis BENAR per-query,
    tapi krn diulang 235x, total biaya scan penuh dibayar 235x (~24-30 jam
    proyeksi, vs. cukup 1x kalau semua akun diproses sekaligus).
    Fix: pakai _SQL_IN/_SQL_OUT/_SQL_CHANNEL (unfiltered) — scan tabel SEKALI
    utk SEMUA akun. _SQL_TS_FILTERED TETAP di-chunk (ts_chunk_size) krn dia
    BUKAN full-scan (LATERAL+LIMIT per akun) — biayanya proporsional jumlah
    akun, bukan jumlah pemanggilan, jadi chunking di situ murni utk ukuran
    query & checkpoint, bukan utk menghindari scan berulang.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)

    t0 = time.time()

    # Checkpoint pre-network (12-Jul, insiden tx_timestamp<=numeric): blok
    # IN/OUT/CHANNEL/TS di bawah makan ~2 JAM untuk 2,1 juta akun dan TIDAK
    # dilindungi checkpoint di level ini (cuma TS punya checkpoint internal,
    # yang dihapus begitu TS loop selesai). Kalau bug muncul SETELAH TS selesai
    # tapi SEBELUM fungsi return (persis insiden ini), seluruh 2 jam itu akan
    # terulang dari nol tiap rerun. Simpan df hasil IN/OUT/CHANNEL/TS ke sini
    # SEBELUM masuk ke first_seen+network, supaya rerun bisa skip langsung.
    #
    # Fix (12-Jul, sesi sama): checkpoint ini disimpan berbasis data Postgres
    # PARTIAL (79.5%, load sempat dihentikan). Kalau nanti data di-reload penuh
    # & retrain_xgboost.py dijalankan ULANG, checkpoint lama bisa KEPAKAI DIAM²
    # dan balikin hasil dari data lama tanpa error apapun (silent staleness,
    # kelas bug sama spt insiden sampling_sig di train_tgn.py/train_dyg.py).
    # Fix: simpan row-count table (estimasi cepat via pg_class.reltuples, bukan
    # COUNT(*) yg mahal di 144rb baris) sbg signature, validasi saat load —
    # toleransi 1% (reltuples cuma estimasi/ANALYZE, wajar sedikit drift tanpa
    # perubahan data nyata; reload data nambah ~20% jauh di atas noise itu).
    with engine.connect() as conn:
        row_count_sig = conn.execute(text(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = 'transactions'"
        )).scalar()

    prenetwork_ckpt = (checkpoint_path + ".prenetwork") if checkpoint_path else None
    df = None
    if prenetwork_ckpt and os.path.exists(prenetwork_ckpt):
        try:
            ckpt_obj = pd.read_pickle(prenetwork_ckpt)
            ckpt_sig = ckpt_obj.get("row_count_sig") if isinstance(ckpt_obj, dict) else None
            if ckpt_sig is not None and row_count_sig and abs(ckpt_sig - row_count_sig) / row_count_sig <= 0.01:
                df = ckpt_obj["df"]
                print(f"[FEATURES-BULK] Checkpoint pre-network ditemukan ({len(df):,} akun, "
                      f"row_count_sig cocok: {ckpt_sig:,.0f}~{row_count_sig:,.0f}) -> skip "
                      f"query IN/OUT/CHANNEL/TS (hemat ~2 jam), lanjut ke first_seen+network.",
                      flush=True)
            else:
                print(f"[FEATURES-BULK] WARNING: checkpoint pre-network STALE (row_count "
                      f"checkpoint={ckpt_sig!r} vs tabel sekarang={row_count_sig!r}, data "
                      f"transactions sudah berubah sejak checkpoint dibuat) — diabaikan, "
                      f"mulai fresh dari awal.", flush=True)
        except Exception as exc:
            print(f"[FEATURES-BULK] WARNING: checkpoint pre-network corrupt/tak terbaca "
                  f"({exc}) — mulai fresh dari awal.", flush=True)
            df = None

    if df is None:
        def _do_in():
            with engine.connect() as conn:
                conn.execute(text("SET work_mem = '256MB'"))
                return conn.execute(_SQL_IN).mappings().all()

        print("[FEATURES-BULK] Query IN (unfiltered, 1x full scan, benchmark ~9m) ...", flush=True)
        in_rows = _run_with_heartbeat("Query IN", _do_in)
        df_in = pd.DataFrame(in_rows)
        print(f"[FEATURES-BULK] IN selesai ({time.time()-t0:.1f}s), {len(df_in):,} akun", flush=True)

        def _do_out():
            with engine.connect() as conn:
                conn.execute(text("SET work_mem = '256MB'"))
                return conn.execute(_SQL_OUT).mappings().all()

        t1 = time.time()
        print("[FEATURES-BULK] Query OUT (unfiltered, 1x full scan, benchmark ~11m) ...", flush=True)
        out_rows = _run_with_heartbeat("Query OUT", _do_out)
        df_out = pd.DataFrame(out_rows)
        print(f"[FEATURES-BULK] OUT selesai ({time.time()-t1:.1f}s), {len(df_out):,} akun", flush=True)

        if df_in.empty and df_out.empty:
            logger.warning("[FEATURES-BULK] No data returned — empty DataFrame with schema")
            return pd.DataFrame(columns=_SCHEMA_COLS)

        df = _merge_in_out(df_in, df_out)

        u_recv = (df["unique_recipients"] if "unique_recipients" in df.columns
                  else pd.Series(0.0, index=df.index))
        df["counterparty_hhi"] = counterparty_hhi(u_recv.clip(lower=1))
        df["counterparty_hhi"] = df["counterparty_hhi"].fillna(0.0)

        def _do_channel():
            with engine.connect() as conn:
                conn.execute(text("SET work_mem = '256MB'"))
                return conn.execute(_SQL_CHANNEL).mappings().all()

        t2 = time.time()
        print("[FEATURES-BULK] Query CHANNEL (unfiltered, 1x full scan, benchmark ~3-4m) ...", flush=True)
        channel_rows = _run_with_heartbeat("Query CHANNEL", _do_channel)
        df_chan = pd.DataFrame(channel_rows)
        print(f"[FEATURES-BULK] CHANNEL selesai ({time.time()-t2:.1f}s)", flush=True)

        df_ent = _compute_channel_entropy(df_chan)
        if not df_ent.empty:
            df = df.merge(df_ent, on="account_id", how="left")
        else:
            df["channel_entropy"] = 0.0
        df["channel_entropy"] = df["channel_entropy"].fillna(0.0)

        # --- TS (temporal): satu2nya bagian yg tetap di-chunk, dgn checkpoint/resume ---
        #
        # Fix MemoryError (5-Jul): versi awal nyimpan RAW baris timestamp (bukan
        # ringkasan) utk SEMUA akun di `ts_frames`, lalu tiap chunk re-concat +
        # re-pickle SELURUH data yang sudah terkumpul (bukan cuma data baru) —
        # itu 2 kesalahan sekaligus: (1) O(n^2) kerja berulang (jelas dari waktu
        # chunk yang makin lambat: 307s -> 873s), (2) _compute_temporal_features
        # baru dipanggil SEKALI di akhir atas SEMUA raw timestamp (bisa ratusan
        # juta baris krn tiap akun sampai 500 baris x 2 arah) -> OOM.
        # Fix: panggil _compute_temporal_features PER CHUNK (spt _extract_batch
        # lama) -> yang disimpan cuma ringkasan 3 kolom per akun (kecil, sebanding
        # ukuran akhir df fitur), BUKAN raw timestamp.
        account_ids = df["account_id"].tolist()
        ts_summary_frames = []
        done_ids: set = set()
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Baca defensif (fix 5-Jul): kalau proses crash PERSIS di tengah
            # nulis checkpoint (mis. MemoryError saat to_pickle), file bisa
            # kepotong/corrupt. Daripada crash lagi pas resume, anggap tidak ada
            # checkpoint & mulai fresh (aman, cuma re-proses chunk yg belum
            # sempat ke-checkpoint dgn benar).
            try:
                prev = pd.read_pickle(checkpoint_path)
                ts_summary_frames.append(prev)
                done_ids = set(prev["account_id"])
                print(f"[FEATURES-BULK] TS checkpoint ditemukan: {len(done_ids):,} akun "
                      f"sudah diproses — lanjut dari situ.", flush=True)
            except Exception as exc:
                print(f"[FEATURES-BULK] WARNING: checkpoint corrupt/tak terbaca ({exc}) "
                      f"— mulai fresh dari awal.", flush=True)

        remaining_ids = [a for a in account_ids if a not in done_ids]
        n_ts_chunks = max(1, -(-len(remaining_ids) // ts_chunk_size))
        t3 = time.time()
        for c, i in enumerate(range(0, len(remaining_ids), ts_chunk_size), start=1):
            chunk = remaining_ids[i:i + ts_chunk_size]
            t_c0 = time.time()
            with engine.connect() as conn:
                conn.execute(text("SET work_mem = '256MB'"))
                ts_rows = conn.execute(_SQL_TS_FILTERED, {"ids": chunk}).mappings().all()
            df_ts_chunk = pd.DataFrame(ts_rows) if ts_rows else pd.DataFrame(columns=["account_id", "tx_timestamp"])
            df_tmp_chunk = _compute_temporal_features(df_ts_chunk)  # ringkas SEGERA, buang raw
            ts_summary_frames.append(df_tmp_chunk)
            if checkpoint_path:
                combined = pd.concat(ts_summary_frames, ignore_index=True) if ts_summary_frames else pd.DataFrame()
                # Tulis atomik (fix 5-Jul): tulis ke file .tmp dulu baru os.replace
                # (atomic rename) — kalau proses mati/crash PERSIS di tengah nulis,
                # checkpoint LAMA yang masih valid tidak ikut kepotong/corrupt.
                tmp_path = checkpoint_path + ".tmp"
                combined.to_pickle(tmp_path)
                os.replace(tmp_path, checkpoint_path)
            elapsed = time.time() - t3
            eta = (elapsed / c) * (n_ts_chunks - c)
            print(f"[FEATURES-BULK] TS chunk {c}/{n_ts_chunks} | {len(chunk):,} akun | "
                  f"chunk={time.time()-t_c0:.1f}s elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m",
                  flush=True)

        df_tmp = pd.concat(ts_summary_frames, ignore_index=True) if ts_summary_frames else pd.DataFrame()
        if not df_tmp.empty:
            df = df.merge(df_tmp, on="account_id", how="left")
        else:
            df["burst_ratio"] = 0.0
            df["inter_tx_std"] = 0.0
            df["dormancy_days"] = 0.0
        for col in ["burst_ratio", "inter_tx_std", "dormancy_days"]:
            df[col] = df[col].fillna(0.0)

        df = df.replace([np.inf, -np.inf], 0).fillna(0)

        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        if prenetwork_ckpt:
            tmp_path = prenetwork_ckpt + ".tmp"
            pd.to_pickle({"df": df, "row_count_sig": row_count_sig}, tmp_path)
            os.replace(tmp_path, prenetwork_ckpt)
            print(f"[FEATURES-BULK] Checkpoint pre-network tersimpan ({len(df):,} akun, "
                  f"row_count_sig={row_count_sig:,.0f}) -> {prenetwork_ckpt}", flush=True)

    # --- first_seen + fitur network CAUSAL (7-Jul, sync Postgres pasca-leak) ---
    # PENTING: blok ini HARUS jalan SEBELUM loop _SCHEMA_COLS di bawah — kalau
    # sebaliknya, device_sharing_count/n_institutions/pagerank/kcore_number
    # (bagian dari FEATURE_COLS) keburu diisi 0 duluan oleh loop itu, lalu
    # merge df_network di sini bikin kolom duplikat bersuffix _x/_y (nama
    # sama, bukan overwrite).
    # first_seen dipakai train_xgboost utk split temporal-inductive (70/15/15
    # by node count, SAMA rasio & metode dgn ml/tgn_dataset.temporal_inductive_
    # split) — konsisten dgn TGN/DyGFormer, bukan lagi train_test_split acak.
    print("[FEATURES-BULK] Query first_seen per akun (dasar split temporal)...", flush=True)
    with engine.connect() as conn:
        conn.execute(text("SET work_mem = '256MB'"))
        first_seen_rows = conn.execute(_SQL_FIRST_SEEN).mappings().all()
    df_first_seen = pd.DataFrame(first_seen_rows)
    # Fix (12-Jul, insiden tx_timestamp<=numeric): astype("int64") // 10**9
    # ASUMSI resolusi datetime64[ns] (ns -> s). psycopg2 balikin datetime
    # microsecond-precision, dan pandas 2.x kadang infer dtype datetime64[us]
    # (bukan [ns]) dari situ -> int64 sudah dalam MIKRODETIK, dibagi 1e9 jadi
    # 1000x kekecilan (val_cutoff_ts=1,772,596 padahal harusnya ~1,772,596,000
    # detik sejak epoch). Fix: hitung selisih ke epoch via Timedelta, resolusi-
    # agnostic, tak peduli unit internal datetime64-nya.
    df_first_seen["first_seen"] = (
        pd.to_datetime(df_first_seen["first_seen"]) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta(seconds=1)
    df = df.merge(df_first_seen.rename(columns={"first_seen": "first_seen_ts"}),
                   on="account_id", how="left")
    df["first_seen_ts"] = df["first_seen_ts"].fillna(df["first_seen_ts"].min())

    order = df["first_seen_ts"].argsort().values
    n = len(order)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    val_cutoff_ts = float(df["first_seen_ts"].values[order[min(n_train + n_val, n) - 1]])
    print(f"[FEATURES-BULK] Cutoff temporal (70/15/15 by first_seen): "
          f"val_cutoff_ts={val_cutoff_ts:.0f}", flush=True)

    print("[FEATURES-BULK] Query fitur NETWORK causal (device/institution/pagerank/kcore, "
          "tx_timestamp <= val_cutoff_ts)...", flush=True)
    df_network = extract_network_features_causal(engine, val_cutoff_ts)
    df = df.merge(df_network, on="account_id", how="left")
    for col in ["device_sharing_count", "n_institutions", "pagerank", "kcore_number"]:
        df[col] = df[col].fillna(0.0)

    # Fix (7-Jul): loop _SCHEMA_COLS dipindah ke SINI (setelah merge network
    # feature di atas) — lihat catatan di atas kenapa urutan ini penting.
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            df[col] = 0

    print(f"[FEATURES-BULK] SELESAI total {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f}m) | "
          f"{len(df):,} akun, {int(df['is_laundering_label'].sum()):,} illicit", flush=True)
    return df
