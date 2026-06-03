"""
Kafka Producer — MuleRadar Transaction Stream

Dua mode:
  replay   : baca transaksi historis dari PostgreSQL → publish ke Kafka
             (mensimulasikan transaksi yang sudah ada seolah baru masuk)
  simulate : generate transaksi sintetis baru tanpa henti → publish ke Kafka
             (demo real-time stream ke juri)

Cara pakai:
  python producer.py --mode replay --batch-size 100 --delay 0.5
  python producer.py --mode simulate --delay 0.1
"""

import os
import sys
import json
import time
import uuid
import random
import argparse
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TX        = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
DATABASE_URL    = os.getenv("DATABASE_URL", "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")

CHANNELS     = ["mobile", "atm", "internet", "teller", "qris"]
INSTITUTIONS = ["BANK_A", "BANK_B"]


def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        acks="all",
        retries=3,
    )


def _row_to_event(row: dict) -> dict:
    """Konversi row DB/dict ke event JSON yang akan di-publish."""
    return {
        "tx_id":          row.get("tx_id", f"TX-{uuid.uuid4().hex[:12].upper()}"),
        "from_account":   str(row["from_account"]),
        "to_account":     str(row["to_account"]),
        "amount":         float(row["amount"]),
        "currency":       row.get("currency", "IDR"),
        "channel":        row.get("channel", "mobile"),
        "payment_format": row.get("payment_format", "Transfer"),
        "tx_timestamp":   str(row.get("tx_timestamp", datetime.utcnow())),
        "device_id":      str(row.get("device_id", "")),
        "institution_id": row.get("institution_id", "BANK_A"),
        "is_laundering":  int(row.get("is_laundering", 0)),
        "typology":       row.get("typology", None),
    }


# ── MODE 1: Replay dari PostgreSQL ──────────────────────────────────────────

def run_replay(producer: KafkaProducer, batch_size: int, delay: float):
    """
    Baca transaksi dari PostgreSQL berurutan (order by tx_timestamp),
    publish ke Kafka satu per satu dengan delay antar batch.
    """
    engine = create_engine(DATABASE_URL)
    print(f"[replay] Koneksi ke PostgreSQL...")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
    print(f"[replay] Total transaksi: {total:,} | batch={batch_size} | delay={delay}s")

    offset = 0
    published = 0

    while offset < total:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT * FROM transactions ORDER BY tx_timestamp LIMIT :lim OFFSET :off"
            ), {"lim": batch_size, "off": offset}).mappings().all()

        for row in rows:
            event = _row_to_event(dict(row))
            producer.send(TOPIC_TX, value=event)
            published += 1

        producer.flush()
        offset += batch_size
        print(f"  published {published:,}/{total:,} transaksi", end="\r")

        if delay > 0:
            time.sleep(delay)

    print(f"\n[replay] Selesai — {published:,} transaksi dipublish ke topic '{TOPIC_TX}'")


# ── MODE 2: Simulate transaksi baru ─────────────────────────────────────────

def _generate_tx() -> dict:
    """Generate satu transaksi sintetis random."""
    account_pool = [f"ACC-{i:06d}" for i in range(1000)]
    is_fraud = random.random() < 0.03  # 3% fraud rate

    if is_fraud:
        # Pola judol: banyak kecil ke satu kolektor
        from_acc = f"JUDOL-PLAYER-{random.randint(0, 499):04d}"
        to_acc   = f"JUDOL-COLL-{random.randint(0, 2):02d}"
        amount   = round(random.uniform(50_000, 500_000) / 1000) * 1000
        channel  = "mobile"
        typology = "judol"
    else:
        from_acc = random.choice(account_pool)
        to_acc   = random.choice(account_pool)
        while to_acc == from_acc:
            to_acc = random.choice(account_pool)
        amount   = round(random.uniform(10_000, 50_000_000))
        channel  = random.choice(CHANNELS)
        typology = None

    return {
        "tx_id":          f"SIM-{uuid.uuid4().hex[:12].upper()}",
        "from_account":   from_acc,
        "to_account":     to_acc,
        "amount":         amount,
        "currency":       "IDR",
        "channel":        channel,
        "payment_format": "Transfer",
        "tx_timestamp":   datetime.utcnow().isoformat(),
        "device_id":      f"DEV-{random.randint(0, 9999):06d}",
        "institution_id": random.choice(INSTITUTIONS),
        "is_laundering":  1 if is_fraud else 0,
        "typology":       typology,
    }


def run_simulate(producer: KafkaProducer, delay: float):
    """Generate transaksi sintetis tanpa henti dan publish ke Kafka."""
    print(f"[simulate] Streaming ke topic '{TOPIC_TX}' | delay={delay}s | Ctrl+C untuk stop")
    published = 0
    try:
        while True:
            tx = _generate_tx()
            producer.send(TOPIC_TX, value=tx)
            published += 1
            if published % 100 == 0:
                producer.flush()
                print(f"  published {published:,} transaksi simulasi", end="\r")
            time.sleep(delay)
    except KeyboardInterrupt:
        producer.flush()
        print(f"\n[simulate] Dihentikan — {published:,} transaksi dipublish")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       choices=["replay", "simulate"], default="replay")
    parser.add_argument("--batch-size", type=int,   default=100)
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Detik antara batch (replay) atau antar transaksi (simulate)")
    args = parser.parse_args()

    print(f"[producer] Connecting ke Kafka {KAFKA_BOOTSTRAP}...")
    producer = _make_producer()
    print(f"[producer] Connected. Mode: {args.mode}")

    if args.mode == "replay":
        run_replay(producer, args.batch_size, args.delay)
    else:
        run_simulate(producer, args.delay)

    producer.close()


if __name__ == "__main__":
    main()
