"""
Post-processing AMLWorld CSV ke format MuleRadar.
Support chunked processing untuk file 15GB+ tanpa OOM.

Production-hardened: validasi input file, validasi kolom, ringkasan rows.

Two-pass approach:
  Pass 1: scan semua chunk → dapat global min/max timestamp
  Pass 2: proses per chunk → apply transformasi → tulis output streaming

Cara pakai:
  python postprocess.py --input D:/data-muleradar/LI-Large_Trans.csv --output ../processed/transactions_li_full.csv
  python postprocess.py --input D:/data-muleradar/LI-Large_Trans.csv --output ../processed/transactions_li_full.csv --chunk-size 200000
  python postprocess.py --input D:/data-muleradar/LI-Large_Trans.csv --output ../processed/transactions_li_full.csv --sample 1000000
"""

import argparse
import os
import random
import sys
import uuid

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

CHUNK_SIZE_DEFAULT = 500_000

SYNTHAML_RENAME = {
    "Account":           "from_account",
    "Account.1":         "to_account",
    "Amount Paid":       "amount",
    "Payment Currency":  "currency",
    "Payment Format":    "payment_format",
    "Is Laundering":     "is_laundering",
    "Timestamp":         "tx_timestamp",
}

CHANNELS    = ["mobile", "atm", "internet", "teller", "qris"]
CHAN_WEIGHTS = [0.55, 0.15, 0.15, 0.05, 0.10]
INSTITUTIONS = ["BANK_A", "BANK_B"]
INST_WEIGHTS  = [0.7, 0.3]


# Distribusi jam transaksi realistis Indonesia (mobile banking, ATM, teller)
_HOUR_WEIGHTS = [
    0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 00-05: sangat sepi
    1.0, 2.0, 3.5, 5.0, 5.5, 5.0,   # 06-11: pagi, mulai ramai jam 9
    4.5, 4.0, 4.5, 4.0, 3.5, 3.0,   # 12-17: siang, istirahat turun jam 12
    3.5, 5.0, 5.5, 4.5, 3.0, 1.5,   # 18-23: malam, puncak jam 19-20
]
_HOUR_WEIGHTS_ARR = np.array(_HOUR_WEIGHTS) / sum(_HOUR_WEIGHTS)


def _add_time_noise(chunk: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Tambah waktu realistis ke transaksi yang hanya punya tanggal (midnight)."""
    midnight = (chunk["tx_timestamp"].dt.hour == 0) & \
               (chunk["tx_timestamp"].dt.minute == 0) & \
               (chunk["tx_timestamp"].dt.second == 0)
    n = midnight.sum()
    if n == 0:
        return chunk
    hours   = rng.choice(24, size=n, p=_HOUR_WEIGHTS_ARR)
    minutes = rng.integers(0, 60, size=n)
    seconds = rng.integers(0, 60, size=n)
    offset  = pd.to_timedelta(hours * 3600 + minutes * 60 + seconds, unit="s")
    chunk.loc[midnight, "tx_timestamp"] = chunk.loc[midnight, "tx_timestamp"] + offset
    return chunk


def _transform_chunk(chunk: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk = chunk.rename(columns=SYNTHAML_RENAME)

    chunk["tx_id"]        = [f"TX-{uuid.uuid4().hex[:12].upper()}" for _ in range(len(chunk))]
    chunk["from_account"] = chunk["from_account"].astype(str).str.strip()
    chunk["to_account"]   = chunk["to_account"].astype(str).str.strip()
    chunk["amount"]       = pd.to_numeric(chunk["amount"], errors="coerce").fillna(0)
    chunk["is_laundering"]= pd.to_numeric(chunk["is_laundering"], errors="coerce").fillna(0).astype(int)
    chunk["tx_timestamp"] = pd.to_datetime(chunk["tx_timestamp"], format="mixed", errors="coerce")
    chunk["typology"]     = None

    # device_id: fraud rekening lebih sering share device
    fraud_accs = set(chunk[chunk["is_laundering"] == 1]["from_account"].unique())
    n_devices  = max(1, int(len(chunk) * 0.001))
    dev_pool   = [f"DEV-{i:06d}" for i in range(n_devices)]
    shared_dev = f"DEV-SHARED-{rng.integers(1000, 9999)}"

    def assign_device(acc):
        if acc in fraud_accs and rng.random() < 0.15:
            return shared_dev
        return dev_pool[hash(acc) % len(dev_pool)]

    chunk["device_id"]      = chunk["from_account"].apply(assign_device)
    chunk["institution_id"] = rng.choice(INSTITUTIONS, size=len(chunk), p=INST_WEIGHTS)
    chunk["channel"]        = rng.choice(CHANNELS, size=len(chunk), p=CHAN_WEIGHTS)
    chunk = _add_time_noise(chunk, rng)

    final_cols = [
        "tx_id", "from_account", "to_account", "amount", "currency",
        "channel", "payment_format", "tx_timestamp", "device_id",
        "institution_id", "is_laundering", "typology",
    ]
    return chunk[final_cols]


def _compress_timestamps(chunk: pd.DataFrame, ts_min, ts_range_s, target_start, target_range_s) -> pd.DataFrame:
    elapsed = (chunk["tx_timestamp"] - ts_min).dt.total_seconds()
    chunk["tx_timestamp"] = target_start + pd.to_timedelta(
        elapsed / ts_range_s * target_range_s, unit="s"
    )
    return chunk


def _inject_judol(n_players=500, n_collectors=3, n_tx=8000, base_ts=None) -> pd.DataFrame:
    if base_ts is None:
        base_ts = datetime(2024, 1, 1)
    players    = [f"JUDOL-PLAYER-{i:04d}" for i in range(n_players)]
    collectors = [f"JUDOL-COLL-{i:02d}" for i in range(n_collectors)]
    crypto_out = "JUDOL-CRYPTO-OUT-01"
    shared_dev = f"DEV-JUDOL-{random.randint(1000,9999)}"
    rows = []
    for _ in range(n_tx):
        hour = random.choices(range(24),
            weights=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,4,5,6,7,8,6], k=1)[0]
        ts = base_ts + timedelta(days=random.randint(0,89), hours=hour, minutes=random.randint(0,59))
        rows.append({"tx_id": f"JDL-{uuid.uuid4().hex[:10].upper()}",
            "from_account": random.choice(players), "to_account": random.choice(collectors),
            "amount": round(random.uniform(50_000, 500_000)/1000)*1000,
            "currency": "IDR", "channel": "mobile", "payment_format": "Transfer",
            "tx_timestamp": ts, "device_id": shared_dev if random.random()<0.3 else f"DEV-PLY-{random.randint(1,200):04d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "judol"})
    for col in collectors:
        for day in range(0, 90, 3):
            ts = base_ts + timedelta(days=day, hours=random.randint(1,4))
            rows.append({"tx_id": f"JDL-OUT-{uuid.uuid4().hex[:10].upper()}",
                "from_account": col, "to_account": crypto_out,
                "amount": round(random.uniform(5_000_000, 30_000_000)),
                "currency": "IDR", "channel": "internet", "payment_format": "Transfer",
                "tx_timestamp": ts, "device_id": shared_dev,
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "judol"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--sample",     type=int, default=None,
                        help="Proses hanya N baris pertama (untuk testing)")
    args = parser.parse_args()

    rng        = np.random.default_rng(SEED)
    chunk_size = args.chunk_size
    nrows      = args.sample

    # ── VALIDASI INPUT ────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file tidak ditemukan: {args.input}")
        sys.exit(1)

    # Validasi kolom AMLWorld yang di-rename ada di file
    expected_cols = list(SYNTHAML_RENAME.keys())
    try:
        peek = pd.read_csv(args.input, nrows=5)
        peek.columns = [c.strip() for c in peek.columns]
    except Exception as exc:
        print(f"[ERROR] Gagal baca CSV: {exc}")
        sys.exit(1)
    missing_cols = [c for c in expected_cols if c not in peek.columns]
    if missing_cols:
        print(f"[WARN] Kolom AMLWorld yang diharapkan TIDAK ADA: {missing_cols}")
        print(f"       Kolom yang ada di CSV: {list(peek.columns)}")
        print(f"       Proses mungkin gagal saat rename. Lanjutkan dengan hati-hati.")

    # ── PASS 1: cari global min/max timestamp ──────────────────────────────
    print(f"[PASS 1] Scan timestamp dari {args.input}...")
    ts_min = pd.Timestamp.max
    ts_max = pd.Timestamp.min
    total_rows = 0

    for chunk in pd.read_csv(args.input, usecols=["Timestamp"], chunksize=chunk_size, nrows=nrows):
        ts = pd.to_datetime(chunk["Timestamp"], format="mixed", errors="coerce").dropna()
        if len(ts):
            ts_min = min(ts_min, ts.min())
            ts_max = max(ts_max, ts.max())
        total_rows += len(chunk)
        print(f"  scanned {total_rows:,} rows...", end="\r")

    print(f"\n  Total rows: {total_rows:,}")
    print(f"  Timestamp range: {ts_min} → {ts_max}")

    target_end    = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    target_start  = target_end - pd.Timedelta(days=88)
    ts_range_s    = (ts_max - ts_min).total_seconds()
    target_range_s= (target_end - target_start).total_seconds()

    # ── PASS 2: transform + compress + tulis output ────────────────────────
    print(f"\n[PASS 2] Transform + compress → {args.output}...")
    written = 0
    rows_dropped = 0
    first_chunk = True

    for chunk in pd.read_csv(args.input, chunksize=chunk_size, nrows=nrows):
        rows_in_chunk = len(chunk)
        chunk = _transform_chunk(chunk, rng)
        rows_after_transform = len(chunk)
        chunk = chunk.dropna(subset=["tx_timestamp"])
        rows_after_drop = len(chunk)
        rows_dropped += (rows_in_chunk - rows_after_drop)
        chunk = _compress_timestamps(chunk, ts_min, ts_range_s, target_start, target_range_s)
        chunk = chunk.sort_values("tx_timestamp")

        chunk.to_csv(args.output, mode="w" if first_chunk else "a",
                     header=first_chunk, index=False)
        first_chunk = False
        written += len(chunk)
        print(f"  written {written:,} rows...", end="\r")

    # ── Append injeksi judol ───────────────────────────────────────────────
    print(f"\n[INJECT] Menambahkan judol deposit ring...")
    judol = _inject_judol()
    judol = _compress_timestamps(judol, ts_min, ts_range_s, target_start, target_range_s)
    judol.to_csv(args.output, mode="a", header=False, index=False)
    written += len(judol)

    print(f"\n=== SELESAI ===")
    print(f"Total rows input   : {total_rows:,}")
    print(f"Total rows ditulis : {written:,} (termasuk {len(judol):,} injeksi judol)")
    if rows_dropped > 0:
        print(f"Rows dropped       : {rows_dropped:,} (timestamp invalid)")
    print(f"Output             : {args.output}")
    print(f"Timestamp range    : {target_start.date()} → {target_end.date()}")


if __name__ == "__main__":
    main()
