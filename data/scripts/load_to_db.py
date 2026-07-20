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
# Ukuran sub-batch satu perintah INSERT (lihat catatan di pemanggilnya).
INSERT_BATCH = int(os.getenv("LOAD_INSERT_BATCH", "5000"))

# Kolom wajib di CSV input
REQUIRED_COLS = ["tx_id", "from_account", "to_account", "amount", "tx_timestamp", "is_laundering"]


def get_engine():
    """Engine dgn TIMEOUT & KEEPALIVE eksplisit (fix 20-Jul).

    MASALAH YG DIPERBAIKI: sebelumnya engine dibuat tanpa timeout apa pun.
    Saat Postgres mati di tengah load berjam-jam (kejadian nyata 20-Jul —
    Docker Desktop crash, port 5432 MASIH kelihatan terbuka krn port-forwarder
    Docker tetap mendengarkan, tapi server di belakangnya tak merespons),
    psycopg2 MENUNGGU SELAMANYA. Gejalanya menipu: progres berhenti tanpa
    error, tak bisa dibedakan dari "sedang lambat". Load menggantung >1 jam.

    Sekarang: gagal CEPAT & JELAS.
      connect_timeout   : batasi tunggu saat MEMBUKA koneksi
      keepalives*       : deteksi koneksi mati diam-diam (TCP half-open)
      statement_timeout : satu perintah tak boleh menggantung tanpa batas
                          (10 menit — longgar utk INSERT batch besar, tapi
                          tetap ADA batasnya)
      pool_pre_ping     : cek koneksi sebelum dipakai; kalau basi, buat baru
    """
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        connect_args={
            "connect_timeout": 15,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
            "options": "-c statement_timeout=600000",   # 10 menit
        },
    )


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
    # Fix (20-Jul, scan produksi): NaN->None SEBELUM to_dict — sama dgn
    # _insert_transactions (baris ~100). institution_id yg NaN kalau tidak
    # dikonversi akan ter-insert sbg STRING "NaN" ke accounts.institution_id
    # (bukan SQL NULL). Kelas bug identik yg sudah difix di transaksi, tapi
    # 2 fungsi upsert ini sebelumnya terlewat.
    accounts = accounts.astype(object).where(accounts.notna(), None)
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
    # Fix (20-Jul): NaN->None, lihat catatan di _upsert_accounts.
    devices = devices.astype(object).where(devices.notna(), None)
    stmt = pg_insert(dev_table).values(devices.to_dict("records"))
    stmt = stmt.on_conflict_do_nothing(index_elements=["device_id"])
    conn.execute(stmt)

    acc_dev = df[["from_account", "device_id"]].rename(
        columns={"from_account": "account_id"}
    ).drop_duplicates()
    acc_dev = acc_dev.astype(object).where(acc_dev.notna(), None)
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

    # Fix (17-Jul, root cause investigation): DataFrame.to_dict("records")
    # TIDAK mengubah NaN jadi None — kolom string kosong (mis. "typology"
    # utk baris tanpa tipologi ter-injeksi) tetap float('nan'). psycopg2/
    # SQLAlchemy lalu men-serialize float NaN itu jadi STRING LITERAL "NaN"
    # saat insert ke kolom VARCHAR, bukan SQL NULL. Akibatnya kolom
    # transactions.typology berisi "NaN" (string) utk 100% baris yg
    # harusnya NULL — ditemukan 17-Jul saat cek alert real-time demo di UI
    # (typologi tampil "NaN", bukan kosong). `.where(tx.notna(), None)`
    # konversi eksplisit SEMUA NaN (bukan cuma typology) jadi None SEBELUM
    # to_dict(), supaya psycopg2 insert SQL NULL yg benar.
    tx = tx.astype(object).where(tx.notna(), None)
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
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Lewati N baris DATA pertama (header tetap dibaca). "
                             "Untuk MELANJUTKAN load yang pernah terpotong tanpa "
                             "harus membaca ulang bagian yang sudah masuk.")
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

    # Resume load terpotong (fix 20-Jul): pakai skiprows INTEGER + names=
    # eksplisit. Kenapa begitu — bukan `skiprows=range(...)`: untuk lompatan
    # puluhan juta baris, bentuk list-like membuat pandas memateralisasi
    # koleksi raksasa di memori (bisa OOM). Bentuk INTEGER ditangani langsung
    # oleh parser C tanpa alokasi, tapi ia MEMBUANG baris header juga —
    # karena itu nama kolom diambil dulu lalu diserahkan lewat `names=`.
    read_kwargs = {"chunksize": args.chunk_size}
    if args.skip_rows > 0:
        cols = list(pd.read_csv(args.input, nrows=0).columns)
        read_kwargs["skiprows"] = args.skip_rows + 1   # +1 = baris header
        read_kwargs["names"] = cols
        print(f"[RESUME] Lewati {args.skip_rows:,} baris data pertama "
              f"(lanjut dari baris ke-{args.skip_rows + 1:,}).")
        print(f"         Baris yang sudah ada tetap aman: INSERT pakai "
              f"ON CONFLICT (tx_id) DO NOTHING.")

    print(f"[3/3] Load CSV chunked dari {args.input} (chunk={args.chunk_size:,})...")
    total_tx = 0
    total_skipped = 0
    chunk_num = 0
    failed_chunks = 0

    for chunk in pd.read_csv(args.input, **read_kwargs):
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
            # Sub-batch INSERT (fix 20-Jul): dulu di-hardcode 500 -> satu chunk
            # 50.000 baris jadi 100 perintah INSERT terpisah = 100 round-trip
            # ke DB per chunk. Dinaikkan ke 5.000 (10 round-trip) — jauh lebih
            # sedikit bolak-balik, tapi tetap cukup kecil agar satu perintah
            # tak jadi raksasa & masih muat di statement_timeout.
            _insert_transactions(chunk, engine, chunk_size=INSERT_BATCH)
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
