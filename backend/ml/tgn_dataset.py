"""
TGN Dataset: load transaksi dari CSV -> PyG-compatible dict.
Support chunked loading untuk file besar (179M+ rows).

Cara pakai:
    from ml.tgn_dataset import load_temporal_dataset
    data = load_temporal_dataset(max_rows=5_000_000)
"""

import os
import sys
import math
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_STRUCTURING_BANDS = [
    (9_500, 10_000), (95_000, 100_000), (950_000, 1_000_000),
    (9_500_000, 10_000_000), (95_000_000, 100_000_000), (450_000_000, 500_000_000),
]

logger = logging.getLogger(__name__)

CSV_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                 "transactions_hi_injected.csv")
)

CHANNEL_MAP = {"mobile": 0, "atm": 1, "internet": 2, "teller": 3, "qris": 4}

# --- Penyelarasan mata uang: AMLWorld (IBM) multi-currency -> semua ke IDR ---
# Kurs referensi tetap (proxy) agar nominal sebanding lintas-rekening & konsisten
# dengan konteks Rupiah. Typologi Indonesia hasil injeksi sudah "IDR" (faktor 1.0).
CURRENCY_TO_IDR = {
    "IDR": 1.0,
    "US Dollar": 16000.0, "Euro": 17500.0, "UK Pound": 20500.0,
    "Swiss Franc": 18000.0, "Canadian Dollar": 11800.0, "Australian Dollar": 10700.0,
    "Yuan": 2250.0, "Yen": 105.0, "Rupee": 190.0, "Ruble": 175.0,
    "Brazil Real": 2900.0, "Saudi Riyal": 4270.0, "Mexican Peso": 800.0,
    "Shekel": 4400.0, "Bitcoin": 1_000_000_000.0,
}


def harmonize_amount_to_idr(chunk: pd.DataFrame) -> pd.DataFrame:
    """Konversi kolom amount ke IDR berdasarkan currency (in-place, mengembalikan chunk).
    Mata uang tak dikenal / kosong dianggap sudah IDR (faktor 1.0)."""
    if "currency" in chunk.columns:
        rate = chunk["currency"].map(CURRENCY_TO_IDR).fillna(1.0)
        chunk["amount"] = chunk["amount"].astype(float) * rate.to_numpy()
    return chunk


def _merge_running(running, batch_dfs, sum_cols, max_cols):
    """Merge batch of chunk-level agg DataFrames into running totals."""
    batch = pd.concat(batch_dfs).groupby(level=0).agg(
        {c: "sum" for c in sum_cols} | {c: "max" for c in max_cols}
    )
    if running is None:
        return batch
    # sum cols: add with fill_value=0 untuk akun yang baru muncul
    result = running.reindex(running.index.union(batch.index), fill_value=0)
    batch_r = batch.reindex(result.index, fill_value=0)
    for c in sum_cols:
        result[c] += batch_r[c]
    for c in max_cols:
        result[c] = np.maximum(result[c].values, batch_r[c].values)
    return result


def compute_node_features_full(csv_path: str, chunk_size: int = 1_000_000) -> pd.DataFrame:
    """
    Full scan CSV untuk compute per-node features yang akurat dari semua 181M rows.
    Return DataFrame dengan index=account_id dan 20 kolom features (13 baseline + 7 behavioral).
    Hasil di-cache ke .parquet supaya run berikutnya langsung load (~5 detik).
    """
    cache_path = csv_path.replace(".csv", "_node_features.pkl")
    if os.path.exists(cache_path):
        print(f"  [node-feat] Loading from cache: {os.path.basename(cache_path)}")
        return pd.read_pickle(cache_path)

    _IN_SUM  = ["in_degree", "in_amount_sum", "night_in", "illicit_in"]
    _IN_MAX  = ["in_max"]
    _OUT_SUM = ["out_degree", "out_amount_sum", "night_out", "illicit_out",
                "structuring_count", "round_tx_count"]
    _OUT_MAX = ["out_max"]
    MERGE_EVERY = 20   # merge setiap 20 chunk → max 20M baris per intermediate concat

    in_agg = []
    out_agg = []
    in_running = None
    out_running = None
    rows_read = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype={"from_account": str, "to_account": str,
               "amount": float, "is_laundering": float,
               "typology": str, "channel": str},
        parse_dates=["tx_timestamp"],
        low_memory=False,
    )

    for chunk in reader:
        chunk = harmonize_amount_to_idr(chunk)
        chunk["is_laundering"] = chunk["is_laundering"].fillna(0).astype(int)
        chunk["tx_timestamp"] = pd.to_datetime(chunk["tx_timestamp"], errors="coerce")
        chunk["is_night"] = chunk["tx_timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3]).astype(int)
        chunk["is_round"] = (
            (chunk["amount"] > 0) &
            (chunk["amount"].apply(lambda x: abs(x - round(x / 10000) * 10000) < 1.0))
        ).astype(int)
        chunk["is_struct"] = chunk["amount"].apply(
            lambda a: 1 if any(lo <= a < hi for lo, hi in _STRUCTURING_BANDS) else 0
        )

        ib = chunk.groupby("to_account", sort=False).agg(
            in_degree=("amount", "count"),
            in_amount_sum=("amount", "sum"),
            in_max=("amount", "max"),
            night_in=("is_night", "sum"),
            illicit_in=("is_laundering", "sum"),
        )
        in_agg.append(ib)

        ob = chunk.groupby("from_account", sort=False).agg(
            out_degree=("amount", "count"),
            out_amount_sum=("amount", "sum"),
            out_max=("amount", "max"),
            night_out=("is_night", "sum"),
            illicit_out=("is_laundering", "sum"),
            structuring_count=("is_struct", "sum"),
            round_tx_count=("is_round", "sum"),
        )
        out_agg.append(ob)

        rows_read += len(chunk)
        if rows_read % 10_000_000 == 0:
            print(f"  [node-feat] scanned {rows_read:,} rows...", end="\r")

        # Merge setiap MERGE_EVERY chunk supaya intermediate concat tetap kecil
        if len(in_agg) >= MERGE_EVERY:
            in_running  = _merge_running(in_running,  in_agg,  _IN_SUM,  _IN_MAX)
            out_running = _merge_running(out_running, out_agg, _OUT_SUM, _OUT_MAX)
            in_agg  = []
            out_agg = []

    print(f"  [node-feat] scanned {rows_read:,} rows — final merge...")

    # Merge sisa chunk yang belum di-merge
    if in_agg:
        in_running  = _merge_running(in_running,  in_agg,  _IN_SUM,  _IN_MAX)
        out_running = _merge_running(out_running, out_agg, _OUT_SUM, _OUT_MAX)

    in_df  = in_running
    out_df = out_running

    df = in_df.join(out_df, how="outer").fillna(0)

    df["max_single_tx"]    = df[["in_max", "out_max"]].max(axis=1)
    df["total_tx"]         = df["in_degree"] + df["out_degree"]
    df["night_tx_count"]   = df["night_in"] + df["night_out"]
    df["illicit_count"]    = df["illicit_in"] + df["illicit_out"]

    df["degree_ratio"]     = df["out_degree"] / (df["in_degree"] + 1)
    df["amount_ratio"]     = df["out_amount_sum"] / (df["in_amount_sum"] + 1)
    df["night_tx_ratio"]   = df["night_tx_count"] / df["total_tx"].clip(lower=1)
    df["avg_amount_in"]    = df["in_amount_sum"] / (df["in_degree"] + 1)
    df["avg_amount_out"]   = df["out_amount_sum"] / (df["out_degree"] + 1)
    df["is_laundering_label"] = (df["illicit_count"] > 0).astype(int)

    # unique_senders/recipients: approximasi via in_degree/out_degree
    # (exact version butuh memory terlalu besar untuk 181M rows)
    df["unique_senders"]    = df["in_degree"]
    df["unique_recipients"] = df["out_degree"]

    # Behavioral features yang bisa dihitung dari vectorized scan
    _out_deg1 = df["out_degree"].clip(lower=1)
    df["structuring_score"]  = df["structuring_count"] / _out_deg1
    df["round_amount_ratio"] = df["round_tx_count"]    / _out_deg1
    # HHI approximation (distribusi merata): 1/unique_recipients
    df["counterparty_hhi"]   = 1.0 / df["out_degree"].clip(lower=1)
    # Temporal features tidak bisa dihitung dari aggregated scan → placeholder 0.0
    # (dihitung akurat di load_temporal_dataset() via edge_rows timestamps)
    df["burst_ratio"]    = 0.0
    df["inter_tx_std"]   = 0.0
    df["dormancy_days"]  = 0.0
    df["channel_entropy"] = 0.0

    print(f"  [node-feat] {len(df):,} accounts, {df['is_laundering_label'].sum():,} illicit")
    df.to_pickle(cache_path)
    print(f"  [node-feat] Cache saved: {os.path.basename(cache_path)}")
    return df


FEATURE_COLS = [
    # 13 baseline
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders",
    "unique_recipients", "max_single_tx", "night_tx_ratio",
    "avg_amount_in", "avg_amount_out", "total_tx",
    # 7 behavioral
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
]


def load_temporal_dataset(
    csv_path: str = CSV_DEFAULT,
    chunk_size: int = 500_000,
    max_rows: int = None,
    sample_licit_ratio: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Load CSV chunked -> aggregate per-node features + temporal edges.

    Parameters
    ----------
    csv_path : str
        Path to the transactions CSV file.
    chunk_size : int
        Number of rows per chunk for streaming read.
    max_rows : int, optional
        Maximum total rows to load (None = all rows).
    sample_licit_ratio : float
        Fraction of licit rows to sample (0.1 = 10%). All illicit rows kept.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        - node_features: np.ndarray (N, 20)  [13 baseline + 7 behavioral]
        - edge_index: np.ndarray (2, E)
        - edge_attr: np.ndarray (E, 3)  [amount_log_norm, channel_enc, hour]
        - edge_timestamps: np.ndarray (E,) unix timestamp
        - edge_labels: np.ndarray (E,) is_laundering per edge
        - node_labels: np.ndarray (N,) 1 if any edge illicit
        - account_to_idx: dict {account_id: int}
        - split: dict {train/val/test: np.ndarray of edge indices}
        - scaler: StandardScaler fitted on node features
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected training data at this path. "
            f"Use --csv to specify a different location."
        )

    rng = np.random.RandomState(random_seed)

    # ------------------------------------------------------------------
    # PASS 1: Stream CSV, accumulate per-account stats + sampled edges
    # ------------------------------------------------------------------
    logger.info("Pass 1: scanning CSV (chunked, chunk_size=%d) ...", chunk_size)
    print(f"[DATASET] Loading from: {csv_path}")
    print(f"[DATASET] chunk_size={chunk_size}, max_rows={max_rows}, "
          f"sample_licit_ratio={sample_licit_ratio}")

    # Per-account accumulators
    acc_stats = defaultdict(lambda: {
        "in_degree": 0, "out_degree": 0,
        "in_amount_sum": 0.0, "out_amount_sum": 0.0,
        "night_tx_in": 0, "night_tx_out": 0,
        "total_tx_in": 0, "total_tx_out": 0,
        "max_tx": 0.0,
        "senders": set(), "recipients": set(),
        "illicit_count": 0,
        # behavioral counters
        "round_tx_count": 0, "structuring_count": 0,
        "channel_counts": defaultdict(int),
    })

    # Sampled edges for graph construction
    edge_rows = []  # list of (from_acc, to_acc, amount, channel, timestamp, is_laund)

    rows_read = 0
    illicit_kept = 0
    licit_sampled = 0

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype={
                "from_account": str, "to_account": str,
                "amount": float, "is_laundering": int,
            },
            parse_dates=["tx_timestamp"],
            low_memory=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open CSV: {e}") from e

    for chunk_idx, chunk in enumerate(reader):
        if max_rows is not None and rows_read >= max_rows:
            break

        if max_rows is not None:
            remaining = max_rows - rows_read
            chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)

        chunk = harmonize_amount_to_idr(chunk)   # selaraskan amount -> IDR

        # Separate illicit vs licit
        illicit_mask = chunk["is_laundering"] == 1
        illicit_rows = chunk[illicit_mask]
        licit_rows = chunk[~illicit_mask]

        # Sample licit rows
        n_licit_sample = max(1, int(len(licit_rows) * sample_licit_ratio))
        if n_licit_sample < len(licit_rows):
            licit_sample = licit_rows.sample(
                n=n_licit_sample, random_state=rng.randint(0, 2**31)
            )
        else:
            licit_sample = licit_rows

        # Combine sampled rows
        sampled = pd.concat([illicit_rows, licit_sample], ignore_index=True)
        illicit_kept += len(illicit_rows)
        licit_sampled += len(licit_sample)

        # Update per-account stats from ALL rows (not just sampled)
        # to get accurate node features
        for _, row in chunk.iterrows():
            from_acc = str(row["from_account"])
            to_acc = str(row["to_account"])
            amount = float(row["amount"])
            is_laund = int(row["is_laundering"])

            # Determine night hour
            ts = row["tx_timestamp"]
            if pd.notna(ts):
                hour = ts.hour if hasattr(ts, "hour") else 0
            else:
                hour = 0
            is_night = 1 if hour in (22, 23, 0, 1, 2, 3) else 0

            # Behavioral flags
            is_round = 1 if (amount > 0 and abs(amount - round(amount / 10000) * 10000) < 1.0) else 0
            _STRUCT_BANDS = [(9_500,10_000),(95_000,100_000),(950_000,1_000_000),
                             (9_500_000,10_000_000),(95_000_000,100_000_000),(450_000_000,500_000_000)]
            is_struct = 1 if any(lo <= amount < hi for lo, hi in _STRUCT_BANDS) else 0
            channel_str = str(row.get("channel", "internet") or "internet").lower()

            # From account (outbound)
            s_from = acc_stats[from_acc]
            s_from["out_degree"] += 1
            s_from["out_amount_sum"] += amount
            s_from["total_tx_out"] += 1
            s_from["night_tx_out"] += is_night
            s_from["recipients"].add(to_acc)
            s_from["max_tx"] = max(s_from["max_tx"], amount)
            s_from["illicit_count"] += is_laund
            s_from["round_tx_count"] += is_round
            s_from["structuring_count"] += is_struct
            s_from["channel_counts"][channel_str] += 1

            # To account (inbound)
            s_to = acc_stats[to_acc]
            s_to["in_degree"] += 1
            s_to["in_amount_sum"] += amount
            s_to["total_tx_in"] += 1
            s_to["night_tx_in"] += is_night
            s_to["senders"].add(from_acc)
            s_to["max_tx"] = max(s_to["max_tx"], amount)
            s_to["illicit_count"] += is_laund
            s_to["channel_counts"][channel_str] += 1

        # Collect sampled edges
        for _, row in sampled.iterrows():
            ts = row["tx_timestamp"]
            if pd.notna(ts):
                unix_ts = ts.timestamp() if hasattr(ts, "timestamp") else 0.0
                hour = ts.hour if hasattr(ts, "hour") else 0
            else:
                unix_ts = 0.0
                hour = 0

            channel_str = str(row.get("channel", "internet")).lower()
            channel_enc = CHANNEL_MAP.get(channel_str, 2)

            edge_rows.append((
                str(row["from_account"]),
                str(row["to_account"]),
                float(row["amount"]),
                channel_enc,
                unix_ts,
                hour,
                int(row["is_laundering"]),
            ))

        if (chunk_idx + 1) % 10 == 0:
            print(f"  ... processed {rows_read:,} rows "
                  f"({illicit_kept:,} illicit, {licit_sampled:,} licit sampled)")

    print(f"[DATASET] Scan complete: {rows_read:,} rows total")
    print(f"[DATASET] Edges kept: {len(edge_rows):,} "
          f"({illicit_kept:,} illicit + {licit_sampled:,} licit)")
    print(f"[DATASET] Unique accounts: {len(acc_stats):,}")

    if len(edge_rows) == 0:
        raise ValueError("No edges found in dataset. Check CSV file content.")

    # ------------------------------------------------------------------
    # PASS 1 was iterrow-heavy; for large datasets, use vectorized version
    # This is a tradeoff: iterrows is memory-safe but slow
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Build account index
    # ------------------------------------------------------------------
    all_accounts = sorted(acc_stats.keys())
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    # ------------------------------------------------------------------
    # Compute temporal behavioral features dari sampled edge_rows
    # (burst_ratio, inter_tx_std, dormancy_days via two-pointer per account)
    # ------------------------------------------------------------------
    logger.info("Computing temporal behavioral features from %d sampled edges...", len(edge_rows))
    _acc_ts = defaultdict(list)
    for (frm, to, _amt, _ch, unix_ts, _hr, _il) in edge_rows:
        if unix_ts > 0:
            _acc_ts[frm].append(unix_ts)
            _acc_ts[to].append(unix_ts)

    _acc_temporal = {}  # account_id → (burst_ratio, inter_tx_std, dormancy_days)
    for _acc, _tss in _acc_ts.items():
        _tss.sort()
        _n = len(_tss)
        if _n < 2:
            _acc_temporal[_acc] = (float(_n > 0), 0.0, 0.0)
            continue
        _gaps = [_tss[i+1] - _tss[i] for i in range(_n - 1)]
        _mean_gap = sum(_gaps) / len(_gaps)
        _inter_tx_std = (sum((_g - _mean_gap) ** 2 for _g in _gaps) / len(_gaps)) ** 0.5
        _dormancy_days = max(_gaps) / 86400.0
        # burst_ratio: two-pointer, 1-hour window
        _left = 0
        _max_burst = 1
        for _r in range(_n):
            while _tss[_r] - _tss[_left] > 3600:
                _left += 1
            _max_burst = max(_max_burst, _r - _left + 1)
        _acc_temporal[_acc] = (float(_max_burst) / _n, _inter_tx_std, _dormancy_days)
    del _acc_ts

    # ------------------------------------------------------------------
    # Build node features (N, 20) matching FEATURE_COLS
    # ------------------------------------------------------------------
    node_features_raw = np.zeros((num_nodes, len(FEATURE_COLS)), dtype=np.float32)
    node_labels = np.zeros(num_nodes, dtype=np.int64)

    for acc, idx in account_to_idx.items():
        s = acc_stats[acc]
        in_deg = s["in_degree"]
        out_deg = s["out_degree"]
        in_amt = s["in_amount_sum"]
        out_amt = s["out_amount_sum"]
        total_tx = s["total_tx_in"] + s["total_tx_out"]
        night_tx = s["night_tx_in"] + s["night_tx_out"]

        # Behavioral features
        _burst, _std, _dorm = _acc_temporal.get(acc, (0.0, 0.0, 0.0))
        _u_recv = max(len(s["recipients"]), 1)
        _counterparty_hhi = 1.0 / _u_recv   # approx: 1/unique_recipients
        _chan = s["channel_counts"]
        _chan_total = sum(_chan.values()) or 1
        _channel_entropy = -sum(
            (c / _chan_total) * math.log2(c / _chan_total + 1e-12)
            for c in _chan.values()
        ) if _chan else 0.0
        _out_deg1 = max(out_deg, 1)
        _structuring_score  = s["structuring_count"] / _out_deg1
        _round_amount_ratio = s["round_tx_count"] / _out_deg1

        node_features_raw[idx] = [
            in_deg,                                         # in_degree
            out_deg,                                        # out_degree
            out_deg / (in_deg + 1),                         # degree_ratio
            in_amt,                                         # in_amount_sum
            out_amt,                                        # out_amount_sum
            out_amt / (in_amt + 1),                         # amount_ratio
            len(s["senders"]),                              # unique_senders
            len(s["recipients"]),                           # unique_recipients
            s["max_tx"],                                    # max_single_tx
            night_tx / max(total_tx, 1),                    # night_tx_ratio
            in_amt / (in_deg + 1),                          # avg_amount_in
            out_amt / (out_deg + 1),                        # avg_amount_out
            total_tx,                                       # total_tx
            _burst,                                         # burst_ratio
            _std,                                           # inter_tx_std
            _dorm,                                          # dormancy_days
            _counterparty_hhi,                              # counterparty_hhi
            _channel_entropy,                               # channel_entropy
            _structuring_score,                             # structuring_score
            _round_amount_ratio,                            # round_amount_ratio
        ]

        if s["illicit_count"] > 0:
            node_labels[idx] = 1

    # Free memory from sets in acc_stats
    del acc_stats

    # Normalize node features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features_raw).astype(np.float32)
    del node_features_raw

    # ------------------------------------------------------------------
    # Build edge arrays
    # ------------------------------------------------------------------
    num_edges = len(edge_rows)
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_attr = np.zeros((num_edges, 3), dtype=np.float32)
    edge_timestamps = np.zeros(num_edges, dtype=np.float64)
    edge_labels = np.zeros(num_edges, dtype=np.int64)

    for i, (from_acc, to_acc, amount, channel, unix_ts, hour, is_laund) in enumerate(edge_rows):
        edge_index[0, i] = account_to_idx[from_acc]
        edge_index[1, i] = account_to_idx[to_acc]
        edge_attr[i, 0] = np.log1p(amount)  # log-normalized amount
        edge_attr[i, 1] = channel            # channel encoding
        edge_attr[i, 2] = hour               # hour of day
        edge_timestamps[i] = unix_ts
        edge_labels[i] = is_laund

    del edge_rows  # free memory

    # Normalize edge amount (column 0) with StandardScaler
    amount_scaler = StandardScaler()
    edge_attr[:, 0] = amount_scaler.fit_transform(
        edge_attr[:, 0].reshape(-1, 1)
    ).ravel()

    # ------------------------------------------------------------------
    # Temporal split by timestamp (60/20/20)
    # ------------------------------------------------------------------
    sorted_idx = np.argsort(edge_timestamps)
    n_train = int(0.6 * num_edges)
    n_val = int(0.2 * num_edges)

    split = {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train: n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }

    train_illicit = edge_labels[split["train"]].sum()
    val_illicit = edge_labels[split["val"]].sum()
    test_illicit = edge_labels[split["test"]].sum()

    print(f"[DATASET] Split sizes: "
          f"train={len(split['train']):,} ({train_illicit:,} illicit), "
          f"val={len(split['val']):,} ({val_illicit:,} illicit), "
          f"test={len(split['test']):,} ({test_illicit:,} illicit)")
    print(f"[DATASET] Node features shape: {node_features.shape}")
    print(f"[DATASET] Edge index shape: {edge_index.shape}")

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_timestamps": edge_timestamps,
        "edge_labels": edge_labels,
        "node_labels": node_labels,
        "account_to_idx": account_to_idx,
        "split": split,
        "scaler": scaler,
    }


def load_temporal_dataset_fast(
    csv_path: str = CSV_DEFAULT,
    chunk_size: int = 500_000,
    max_rows: int = None,
    sample_licit_ratio: float = 0.005,
    sample_illicit_ratio: float = 0.05,
    random_seed: int = 42,
) -> dict:
    """
    Vectorized version: scan seluruh CSV dengan chunked sampling.
    sample_licit_ratio=0.005  → ~690K licit dari 138M
    sample_illicit_ratio=0.05 → ~2M illicit dari 41M
    Total ~2.7M edges, ~75% illicit — cukup untuk training GPU.

    Same return format as load_temporal_dataset().
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rng = np.random.RandomState(random_seed)
    print(f"[DATASET-FAST] Loading from: {csv_path}")

    chunks = []
    rows_read = 0
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype={"from_account": str, "to_account": str,
               "amount": float, "is_laundering": float,
               "typology": str, "channel": str},
        parse_dates=["tx_timestamp"],
        low_memory=False,
    )

    for chunk in reader:
        if max_rows is not None and rows_read >= max_rows:
            break
        rows_read += len(chunk)

        chunk["is_laundering"] = chunk["is_laundering"].fillna(0).astype(int)

        illicit = chunk[chunk["is_laundering"] == 1]
        licit   = chunk[chunk["is_laundering"] == 0]

        # Sample illicit
        if sample_illicit_ratio < 1.0 and len(illicit) > 0:
            n_ill = max(1, int(len(illicit) * sample_illicit_ratio))
            if n_ill < len(illicit):
                illicit = illicit.sample(n=n_ill, random_state=rng.randint(0, 2**31))

        # Sample licit
        if len(licit) > 0:
            n_lic = max(1, int(len(licit) * sample_licit_ratio))
            if n_lic < len(licit):
                licit = licit.sample(n=n_lic, random_state=rng.randint(0, 2**31))

        chunks.append(pd.concat([illicit, licit], ignore_index=True))

        if rows_read % 5_000_000 == 0:
            print(f"  scanned {rows_read:,} rows...", end="\r")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    illicit_count = df["is_laundering"].sum()
    print(f"[DATASET-FAST] Loaded {len(df):,} edges from {rows_read:,} rows "
          f"({illicit_count:,} illicit, {illicit_count/len(df)*100:.1f}%)")

    # Build account index dari sampled edges
    all_accounts_sampled = sorted(pd.unique(
        pd.concat([df["from_account"], df["to_account"]])
    ).tolist())
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts_sampled)}
    num_nodes = len(all_accounts_sampled)

    # Map edges
    src = df["from_account"].map(account_to_idx).values.astype(np.int64)
    dst = df["to_account"].map(account_to_idx).values.astype(np.int64)
    edge_index = np.stack([src, dst])

    # Pastikan tx_timestamp bertipe datetime
    df["tx_timestamp"] = pd.to_datetime(df["tx_timestamp"], errors="coerce")

    # Edge attributes
    amounts_log = np.log1p(df["amount"].values).astype(np.float32)
    channels = df["channel"].map(
        lambda x: CHANNEL_MAP.get(str(x).lower(), 2)
    ).values.astype(np.float32) if "channel" in df.columns else np.full(len(df), 2, dtype=np.float32)
    hours = df["tx_timestamp"].dt.hour.fillna(0).values.astype(np.float32)

    amount_scaler = StandardScaler()
    amounts_log = amount_scaler.fit_transform(amounts_log.reshape(-1, 1)).ravel()

    edge_attr = np.stack([amounts_log, channels, hours], axis=1).astype(np.float32)
    edge_timestamps = df["tx_timestamp"].apply(
        lambda x: x.timestamp() if pd.notna(x) else 0.0
    ).values.astype(np.float64)
    edge_labels = df["is_laundering"].values.astype(np.int64)

    # Node features: full scan CSV untuk akurasi (bukan dari sampled edges)
    print("[DATASET-FAST] Computing node features (full CSV scan)...")
    node_feat_df = compute_node_features_full(csv_path)

    # Align node features ke urutan all_accounts_sampled (20 fitur = FEATURE_COLS global)
    feat_aligned = node_feat_df.reindex(all_accounts_sampled)[FEATURE_COLS].fillna(0).astype(np.float32)
    node_features_raw = feat_aligned.values

    # Node labels dari full scan
    node_labels = np.zeros(num_nodes, dtype=np.int64)
    if "is_laundering_label" in node_feat_df.columns:
        lbl = node_feat_df.reindex(all_accounts_sampled)["is_laundering_label"].fillna(0).astype(int)
        node_labels = lbl.values

    scaler = StandardScaler()
    node_features = scaler.fit_transform(node_features_raw).astype(np.float32)

    illicit_nodes = int(node_labels.sum())

    # Temporal split
    sorted_idx = np.argsort(edge_timestamps)
    n_train = int(0.6 * len(edge_timestamps))
    n_val = int(0.2 * len(edge_timestamps))

    split = {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train: n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }

    print(f"[DATASET-FAST] Nodes: {num_nodes:,}, Edges: {len(df):,}")
    print(f"[DATASET-FAST] Node labels: {node_labels.sum():,} illicit / {num_nodes:,}")
    print(f"[DATASET-FAST] Split: train={len(split['train']):,}, "
          f"val={len(split['val']):,}, test={len(split['test']):,}")

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_timestamps": edge_timestamps,
        "edge_labels": edge_labels,
        "node_labels": node_labels,
        "account_to_idx": account_to_idx,
        "split": split,
        "scaler": scaler,
    }
