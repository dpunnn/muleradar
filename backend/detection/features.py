"""
MuleRadar Phase 3 — Feature Extraction.
Extract account-level features from PostgreSQL for ML model training & inference.

Production-hardened: validasi koneksi, handle inf/nan, output schema konsisten.
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

# Dua query terpisah — merge di pandas, hindari UNION ALL nested yang crash PG
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
           SUM(is_laundering) AS illicit_count_out
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
           SUM(is_laundering) AS illicit_count_out
    FROM transactions
    WHERE from_account = ANY(:ids)
    GROUP BY from_account
""")

# 13 fitur model — HARUS identik dengan detection/model.py & ensemble.py.
# CATATAN: illicit_count & night_tx_count TIDAK boleh masuk sini (label leakage /
# bukan fitur model). illicit_count hanya dipakai untuk turunkan is_laundering_label.
FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
]

# Schema lengkap output: account_id + 13 fitur + label
_SCHEMA_COLS = ["account_id"] + FEATURE_COLS + ["is_laundering_label"]


def _merge_in_out(df_in: pd.DataFrame, df_out: pd.DataFrame) -> pd.DataFrame:
    """Merge inbound + outbound DataFrames on account_id, compute derived features."""
    df = df_in.merge(df_out, on="account_id", how="outer")
    df = df.fillna(0)

    # Cast Decimal (dari PostgreSQL NUMERIC) ke float
    for col in ["in_amount_sum", "out_amount_sum", "max_single_tx",
                "max_single_tx_out", "illicit_count", "illicit_count_out"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Combine overlapping columns
    df["max_single_tx"] = df[["max_single_tx", "max_single_tx_out"]].max(axis=1)
    df["night_tx_count"] = df["night_tx_count"] + df["night_tx_count_out"]
    df["total_tx"] = df["total_tx"] + df["total_tx_out"]
    df["illicit_count"] = df["illicit_count"] + df["illicit_count_out"]
    df = df.drop(columns=["max_single_tx_out", "night_tx_count_out",
                           "total_tx_out", "illicit_count_out"])

    # Cast kolom integer dari SQL NUMERIC/BIGINT yang bisa jadi Decimal
    for col in ["in_degree", "out_degree", "unique_senders", "unique_recipients",
                "night_tx_count", "total_tx"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    # Derived features (division safe: +1 dan clip(lower=1))
    df["degree_ratio"]     = df["out_degree"] / (df["in_degree"] + 1)
    df["amount_ratio"]     = df["out_amount_sum"] / (df["in_amount_sum"] + 1)
    df["night_tx_ratio"]   = df["night_tx_count"] / df["total_tx"].clip(lower=1)
    df["avg_amount_in"]    = df["in_amount_sum"] / (df["in_degree"] + 1)
    df["avg_amount_out"]   = df["out_amount_sum"] / (df["out_degree"] + 1)
    df["is_laundering_label"] = (df["illicit_count"] > 0).astype(int)

    # Cleanup: ganti inf/-inf/NaN dengan 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    return df


def extract_features(
    engine=None,
    account_ids: list = None,
    limit: int = 100_000,
) -> pd.DataFrame:
    """
    Extract account-level features for model training.

    Parameters
    ----------
    engine : sqlalchemy.Engine, optional
        Existing engine. Created from DATABASE_URL if None.
    account_ids : list, optional
        Specific accounts to extract. If None, samples ~80% illicit + 20% random
        licit accounts (up to *limit* total).
    limit : int
        Max accounts to return when account_ids is None.

    Returns
    -------
    pd.DataFrame
        One row per account with all features + is_laundering_label.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)

    # Validasi koneksi DB
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(
            f"Gagal konek DB: {exc}. Cek DATABASE_URL={DATABASE_URL!r}"
        ) from exc

    if account_ids is not None:
        return _extract_batch(engine, account_ids)

    # --- Sampling strategy: 80% illicit, 20% random licit ---
    illicit_limit = int(limit * 0.8)
    licit_limit = limit - illicit_limit

    try:
        with engine.connect() as conn:
            # Accounts that participated in at least one illicit tx
            illicit_accounts = conn.execute(text("""
                SELECT DISTINCT account_id FROM (
                    SELECT from_account AS account_id FROM transactions
                    WHERE is_laundering = 1
                    UNION
                    SELECT to_account FROM transactions
                    WHERE is_laundering = 1
                ) t
                ORDER BY account_id
                LIMIT :lim
            """), {"lim": illicit_limit}).scalars().all()

            # Random licit accounts (not in illicit set)
            licit_accounts = conn.execute(text("""
                SELECT account_id FROM (
                    SELECT from_account AS account_id FROM transactions
                    UNION
                    SELECT to_account FROM transactions
                ) t
                WHERE account_id != ALL(:exclude)
                ORDER BY RANDOM()
                LIMIT :lim
            """), {"exclude": list(illicit_accounts), "lim": licit_limit}).scalars().all()
    except Exception as exc:
        raise RuntimeError(
            f"Gagal konek DB saat sampling accounts: {exc}"
        ) from exc

    all_ids = list(illicit_accounts) + list(licit_accounts)
    print(f"[FEATURES] Sampling {len(illicit_accounts)} illicit + "
          f"{len(licit_accounts)} licit = {len(all_ids)} accounts")

    return _extract_batch(engine, all_ids)


def _extract_batch(engine, account_ids: list) -> pd.DataFrame:
    """Run two separate queries (inbound + outbound) and merge in pandas."""
    batch_size = 10_000
    in_frames: list[pd.DataFrame] = []
    out_frames: list[pd.DataFrame] = []

    for i in range(0, len(account_ids), batch_size):
        chunk = account_ids[i: i + batch_size]
        try:
            with engine.connect() as conn:
                in_rows  = conn.execute(_SQL_IN_FILTERED,  {"ids": chunk}).mappings().all()
                out_rows = conn.execute(_SQL_OUT_FILTERED, {"ids": chunk}).mappings().all()
        except Exception as exc:
            raise RuntimeError(
                f"Gagal konek DB saat query batch offset={i}: {exc}"
            ) from exc
        if in_rows:
            in_frames.append(pd.DataFrame(in_rows))
        if out_rows:
            out_frames.append(pd.DataFrame(out_rows))

    df_in  = pd.concat(in_frames,  ignore_index=True) if in_frames  else pd.DataFrame()
    df_out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    if df_in.empty and df_out.empty:
        logger.warning("[FEATURES] No data returned — returning empty DataFrame with schema")
        return pd.DataFrame(columns=_SCHEMA_COLS)

    df = _merge_in_out(df_in, df_out)

    # Validasi output: pastikan semua kolom schema ada
    for col in _SCHEMA_COLS:
        if col not in df.columns:
            df[col] = 0

    print(f"[FEATURES] Extracted {len(df)} accounts, "
          f"{df['is_laundering_label'].sum()} illicit")
    return df


def extract_features_for_accounts(
    engine, account_ids: list
) -> pd.DataFrame:
    """
    Extract features for specific accounts (inference mode).
    Identical to _extract_batch but public API.

    Parameters
    ----------
    engine : sqlalchemy.Engine
    account_ids : list
        Account IDs to extract features for.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame (no label column guaranteed to be meaningful).
    """
    if not account_ids:
        return pd.DataFrame(columns=_SCHEMA_COLS)
    return _extract_batch(engine, account_ids)
