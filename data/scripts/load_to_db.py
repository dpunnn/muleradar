"""
Load processed transactions CSV ke PostgreSQL.
Support chunked reading untuk file besar tanpa OOM.

Production-hardened: validasi kolom CSV, skip counter, chunk-level error recovery.

Cara pakai:
  python load_to_db.py --input ../processed/transactions_li_full.csv
  python load_to_db.py --input ../processed/transactions_li_full.csv --chunk-size 10000
"""

import argparse
import os
import sys
import traceback

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar"

CHUNK_SIZE_DEFAULT = 50_000

# Kolom wajib di CSV input
REQUIRED_COLS = ["tx_id", "from_account", "to_account", "amount", "tx_timestamp", "is_laundering"]


def get_engine():
    return create_engine(DATABASE_URL)


def _upsert_accounts(df: pd.DataFrame, conn):
    from_accs = df[["from_account", "institution_id"]].rename(columns={"from_account": "account_id"})
    to_accs   = df[["to_account",   "institution_id"]].rename(columns={"to_account":   "account_id"})
    accounts  = pd.concat([from_accs, to_accs]).drop_duplicates("account_id")
    accounts["account_type"] = "personal"
    accounts["is_active"]    = True

    # ON CONFLICT DO NOTHING — aman untuk chunked upsert
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy import Table, MetaData
    meta  = MetaData()
    table = Table("accounts", meta, autoload_with=conn)
    stmt  = pg_insert(table).values(accounts.to_dict("records"))
    stmt  = stmt.on_conflict_do_nothing(index_elements=["account_id"])
    conn.execute(stmt)


def _upsert_devices(df: pd.DataFrame, conn):
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy import Table, MetaData

    meta = MetaData()

    devices = df[["device_id"]].drop_duplicates()
    devices["device_type"] = "mobile"
    dev_table = Table("devices", meta, autoload_with=conn)
    stmt = pg_insert(dev_table).values(devices.to_dict("records"))
    stmt = stmt.on_conflict_do_nothing(index_elements=["device_id"])
    conn.execute(stmt)

    acc_dev = df[["from_account", "device_id"]].rename(
        columns={"from_account": "account_id"}
    ).drop_duplicates()
    ad_table = Table("account_devices", meta, autoload_with=conn)
    stmt2 = pg_insert(ad_table).values(acc_dev.to_dict("records"))
    stmt2 = stmt2.on_conflict_do_nothing()
    conn.execute(stmt2)


def _insert_transactions(df: pd.DataFrame, engine, chunk_size: int):
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    from sqlalchemy import Table, MetaData

    tx = df[[
        "tx_id", "from_account", "to_account", "amount", "currency",
        "channel", "payment_format", "tx_timestamp", "device_id",
        "institution_id", "is_laundering", "typology",
    ]].copy()
    tx["tx_timestamp"] = pd.to_datetime(tx["tx_timestamp"], errors="coerce", format="mixed")
    tx["is_laundering"] = tx["is_laundering"].fillna(0).astype(int)
    tx = tx.dropna(subset=["tx_id", "from_account", "to_account"])

    meta  = MetaData()
    table = Table("transactions", meta, autoload_with=engine)

    records = tx.to_dict("records")
    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            batch = records[i: i + chunk_size]
            stmt  = pg_insert(table).values(batch)
            stmt  = stmt.on_conflict_do_nothing(index_elements=["tx_id"])
            conn.execute(stmt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT)
    parser.add_argument("--truncate",    action="store_true", default=False,
                        help="Truncate tabel sebelum load (default: False)")
    parser.add_argument("--no-truncate",  action="store_true", default=False,
                        help="Alias eksplisit untuk skip truncate")
    parser.add_argument("--illicit-only", action="store_true", default=False,
                        help="Hanya load baris is_laundering=1 (untuk append illicit baru)")
    args = parser.parse_args()

    engine = get_engine()

    print("[1/3] Connecting ke database...")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("      Connected OK")

    if args.truncate and not args.no_truncate:
        print("[2/3] Truncate tabel lama...")
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE accounts, devices, account_devices, transactions CASCADE"))
        print("      Truncated OK")

    # Validasi kolom wajib di chunk pertama
    print(f"[PRE] Validasi kolom CSV...")
    try:
        first_chunk = pd.read_csv(args.input, nrows=5)
    except Exception as exc:
        print(f"[ERROR] Gagal baca CSV: {exc}")
        sys.exit(1)
    missing_cols = [c for c in REQUIRED_COLS if c not in first_chunk.columns]
    if missing_cols:
        print(f"[ERROR] Kolom wajib tidak ditemukan di CSV: {missing_cols}")
        print(f"        Kolom yang ada: {list(first_chunk.columns)}")
        sys.exit(1)
    print(f"      Kolom OK: {REQUIRED_COLS}")

    print(f"[3/3] Load CSV chunked dari {args.input} (chunk={args.chunk_size:,})...")
    total_tx = 0
    total_skipped = 0
    chunk_num = 0
    failed_chunks = 0

    for chunk in pd.read_csv(args.input, chunksize=args.chunk_size):
        chunk_num += 1

        if args.illicit_only:
            chunk = chunk[chunk["is_laundering"] == 1]
            if len(chunk) == 0:
                continue

        # Hitung baris invalid: amount non-numeric atau tx_id null
        rows_before = len(chunk)
        chunk["amount"] = pd.to_numeric(chunk["amount"], errors="coerce")
        invalid_mask = chunk["tx_id"].isna() | chunk["amount"].isna()
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            chunk = chunk[~invalid_mask]
            total_skipped += n_invalid

        try:
            # Upsert accounts + devices per chunk (ON CONFLICT DO NOTHING)
            with engine.begin() as conn:
                _upsert_accounts(chunk, conn)
                _upsert_devices(chunk, conn)

            # Insert transactions
            _insert_transactions(chunk, engine, chunk_size=500)
        except Exception as exc:
            failed_chunks += 1
            print(f"\n[WARN] Chunk {chunk_num} gagal: {exc}")
            traceback.print_exc()
            continue

        total_tx += len(chunk)
        print(f"  chunk {chunk_num}: {total_tx:,} transaksi loaded...", end="\r")

    print(f"\n\n=== LOAD SELESAI ===")
    if total_skipped > 0:
        print(f"Skipped {total_skipped} invalid rows (amount non-numeric / tx_id null)")
    if failed_chunks > 0:
        print(f"[WARN] {failed_chunks} chunk(s) gagal diproses")
    with engine.connect() as conn:
        tx_count  = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
        acc_count = conn.execute(text("SELECT COUNT(*) FROM accounts")).scalar()
        illicit   = conn.execute(text("SELECT COUNT(*) FROM transactions WHERE is_laundering=1")).scalar()
        print(f"Transactions : {tx_count:,}")
        print(f"Accounts     : {acc_count:,}")
        if tx_count > 0:
            print(f"Illicit      : {illicit:,} ({illicit/tx_count*100:.1f}%)")
        else:
            print(f"Illicit      : {illicit:,}")


if __name__ == "__main__":
    main()
