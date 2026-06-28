"""
MuleRadar Phase 3 — Feature Extraction.
Extract account-level features from PostgreSQL for ML model training & inference.

13 fitur baseline + 7 fitur behavioral (total 20):
  Behavioral baru: burst_ratio, inter_tx_std, dormancy_days, counterparty_hhi,
                   channel_entropy, structuring_score, round_amount_ratio
"""

import logging
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

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

# ── SQL: counterparty_hhi (Herfindahl Index penerima outbound) ─────────────────
# HHI = 1.0 → semua transaksi ke rekening yang sama (sangat suspicious)
# HHI → 0   → distribusi merata ke banyak rekening
_SQL_HHI_FILTERED = text("""
    SELECT t.from_account AS account_id,
           SUM(POWER(t.cnt::float / t2.total::float, 2)) AS counterparty_hhi
    FROM (
        SELECT from_account, to_account, COUNT(*) AS cnt
        FROM transactions
        WHERE from_account = ANY(:ids)
        GROUP BY from_account, to_account
    ) t
    JOIN (
        SELECT from_account, COUNT(*) AS total
        FROM transactions
        WHERE from_account = ANY(:ids)
        GROUP BY from_account
    ) t2 ON t.from_account = t2.from_account
    GROUP BY t.from_account
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
_SQL_TS_FILTERED = text("""
    SELECT account_id, tx_timestamp FROM (
        SELECT from_account AS account_id, tx_timestamp,
               ROW_NUMBER() OVER (PARTITION BY from_account ORDER BY tx_timestamp DESC) AS rn
        FROM transactions WHERE from_account = ANY(:ids)
    ) t WHERE rn <= 500
    UNION ALL
    SELECT account_id, tx_timestamp FROM (
        SELECT to_account AS account_id, tx_timestamp,
               ROW_NUMBER() OVER (PARTITION BY to_account ORDER BY tx_timestamp DESC) AS rn
        FROM transactions WHERE to_account = ANY(:ids)
    ) t WHERE rn <= 500
""")

# ── Feature column definitions ─────────────────────────────────────────────────
# 13 baseline + 7 behavioral = 20 total
# HARUS identik dengan ml/eval_ablation.py, ml/tgn_dataset.py, detection/model.py
FEATURE_COLS = [
    # 13 baseline
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
    # 7 behavioral
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
]

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
    hhi_frames, channel_frames, ts_frames = [], [], []

    for i in range(0, len(account_ids), batch_size):
        chunk = account_ids[i: i + batch_size]
        try:
            with engine.connect() as conn:
                in_rows      = conn.execute(_SQL_IN_FILTERED,      {"ids": chunk}).mappings().all()
                out_rows     = conn.execute(_SQL_OUT_FILTERED,     {"ids": chunk}).mappings().all()
                hhi_rows     = conn.execute(_SQL_HHI_FILTERED,     {"ids": chunk}).mappings().all()
                channel_rows = conn.execute(_SQL_CHANNEL_FILTERED, {"ids": chunk}).mappings().all()
                ts_rows      = conn.execute(_SQL_TS_FILTERED,      {"ids": chunk}).mappings().all()
        except Exception as exc:
            raise RuntimeError(f"Gagal query batch offset={i}: {exc}") from exc

        if in_rows:      in_frames.append(pd.DataFrame(in_rows))
        if out_rows:     out_frames.append(pd.DataFrame(out_rows))
        if hhi_rows:     hhi_frames.append(pd.DataFrame(hhi_rows))
        if channel_rows: channel_frames.append(pd.DataFrame(channel_rows))
        if ts_rows:      ts_frames.append(pd.DataFrame(ts_rows))

    df_in  = pd.concat(in_frames,  ignore_index=True) if in_frames  else pd.DataFrame()
    df_out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    if df_in.empty and df_out.empty:
        logger.warning("[FEATURES] No data returned — empty DataFrame with schema")
        return pd.DataFrame(columns=_SCHEMA_COLS)

    # 13 baseline features
    df = _merge_in_out(df_in, df_out)

    # counterparty_hhi
    df_hhi = pd.concat(hhi_frames, ignore_index=True) if hhi_frames else pd.DataFrame()
    if not df_hhi.empty:
        df_hhi["counterparty_hhi"] = df_hhi["counterparty_hhi"].astype(float)
        df = df.merge(df_hhi[["account_id", "counterparty_hhi"]], on="account_id", how="left")
    else:
        df["counterparty_hhi"] = 0.0
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


def extract_features_for_accounts(engine, account_ids: list) -> pd.DataFrame:
    """Public inference API — same as _extract_batch."""
    if not account_ids:
        return pd.DataFrame(columns=_SCHEMA_COLS)
    return _extract_batch(engine, account_ids)
