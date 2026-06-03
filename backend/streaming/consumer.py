"""
Kafka Consumer — MuleRadar Real-Time AI Detection Pipeline (production-grade).

Flow per transaksi masuk (FAST PATH):
  Kafka 'transactions.raw'
    → RealtimeScorer: update feature store (Redis) → XGBoost + streaming signals
    → risk score + decision (NONE/ALERT/ESCALATE/FREEZE) — TANPA label
    → MERGE ke Neo4j (incremental)
    → alert/freeze → PostgreSQL + Kafka 'alerts.generated' + audit trail

Deep re-scoring (TGN ensemble) jalan di SLOW PATH (run_detection terjadwal).

Cara pakai:
  python consumer.py
  python consumer.py --batch-size 50 --flush-interval 5
"""

import os
import sys
import json
import time
import uuid
import argparse
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from realtime_scorer import RealtimeScorer

KAFKA_BOOTSTRAP  = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_TX         = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions.raw")
TOPIC_ALERTS     = os.getenv("KAFKA_TOPIC_ALERTS", "alerts.generated")
DATABASE_URL     = os.getenv("DATABASE_URL", "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
NEO4J_URI        = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER       = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD   = os.getenv("NEO4J_PASSWORD", "muleradar_neo4j")


def _make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC_TX,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="muleradar-detection",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000,
    )


def _make_alert_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )


def _merge_to_neo4j(batch: list[dict], driver):
    """MERGE batch transaksi ke Neo4j secara incremental (streaming-safe)."""
    with driver.session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {account_id: row.from_account})
            MERGE (b:Account {account_id: row.to_account})
            CREATE (a)-[:TRANSFER {
                tx_id:         row.tx_id,
                amount:        row.amount,
                tx_timestamp:  row.tx_timestamp,
                channel:       row.channel,
                device_id:     row.device_id,
                is_laundering: row.is_laundering
            }]->(b)
            """,
            rows=batch,
        )


def _save_alert(tx: dict, scored: dict, engine, alert_producer: KafkaProducer):
    """
    Simpan alert dari hasil RealtimeScorer ke PostgreSQL + publish Kafka.
    Status awal:
      FREEZE   → status FROZEN (auto-freeze typology ilegal jelas, configurable)
      ESCALATE → status NEW (eskalasi ke analis manusia)
      ALERT    → status NEW
    Semua tindakan tercatat di rule_triggered (audit trail).
    """
    decision = scored["decision"]
    status = "FROZEN" if decision == "FREEZE" else "NEW"
    reasons = ", ".join(scored["reasons"]) if scored["reasons"] else "model_score"

    alert = {
        "alert_id":       f"ALT-{uuid.uuid4().hex[:12].upper()}",
        "account_id":     scored["account_id"],
        "tx_id":          tx.get("tx_id"),
        "typology":       tx.get("typology") or "AI_DETECTED",
        "risk_score":     scored["risk_score"],
        "rule_triggered": f"[{decision}] base={scored['base_score']} | {reasons}",
        "severity":       scored["risk_level"],
        "status":         status,
        "created_at":     datetime.utcnow().isoformat(),
    }

    alert_producer.send(TOPIC_ALERTS, value=alert)

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alerts (alert_id, account_id, tx_id, typology,
                    risk_score, rule_triggered, severity, status, created_at)
                VALUES (:alert_id, :account_id, :tx_id, :typology,
                    :risk_score, :rule_triggered, :severity, :status, :created_at)
                ON CONFLICT (alert_id) DO NOTHING
            """), alert)
    except Exception as e:
        print(f"  [warn] Gagal insert alert ke DB: {e}")

    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",     type=int,   default=100,
                        help="Jumlah pesan dikumpulkan sebelum di-push ke Neo4j")
    parser.add_argument("--flush-interval", type=float, default=5.0,
                        help="Detik max tunggu sebelum flush batch walaupun belum penuh")
    args = parser.parse_args()

    print(f"[consumer] Connecting ke Kafka {KAFKA_BOOTSTRAP}...")
    consumer       = _make_consumer()
    alert_producer = _make_alert_producer()
    driver         = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    engine         = create_engine(DATABASE_URL)

    print("[consumer] Loading RealtimeScorer (XGBoost + Redis feature store)...")
    scorer = RealtimeScorer()
    if not scorer.store.ping():
        print("[consumer] WARNING: Redis tidak terjangkau — feature store mati")

    print(f"[consumer] Listening topic '{TOPIC_TX}' | batch={args.batch_size} | flush={args.flush_interval}s")
    print("[consumer] Ctrl+C untuk stop\n")

    batch        = []
    last_flush   = time.time()
    total_tx     = 0
    total_alerts = 0
    total_freeze = 0

    try:
        while True:
            for msg in consumer:
                tx = msg.value
                batch.append(tx)

                # FAST PATH: real-time AI scoring (update store + XGBoost + signals)
                scored = scorer.score(tx)
                if scored["decision"] != "NONE":
                    decision = _save_alert(tx, scored, engine, alert_producer)
                    total_alerts += 1
                    if decision == "FREEZE":
                        total_freeze += 1
                        print(f"\n  🧊 AUTO-FREEZE {scored['account_id']} "
                              f"risk={scored['risk_score']} ({', '.join(scored['reasons'])})")

                # Flush ke Neo4j kalau batch penuh atau timeout
                now = time.time()
                if len(batch) >= args.batch_size or (now - last_flush) >= args.flush_interval:
                    _merge_to_neo4j(batch, driver)
                    total_tx += len(batch)
                    print(f"  flushed {len(batch)} tx → Neo4j | total={total_tx:,} | "
                          f"alerts={total_alerts:,} | frozen={total_freeze:,}", end="\r")
                    batch      = []
                    last_flush = now

            # Flush sisa batch setelah consumer timeout
            if batch:
                _merge_to_neo4j(batch, driver)
                total_tx += len(batch)
                batch     = []
                last_flush= time.time()

    except KeyboardInterrupt:
        if batch:
            _merge_to_neo4j(batch, driver)
            total_tx += len(batch)

        alert_producer.flush()
        consumer.close()
        driver.close()
        print(f"\n[consumer] Dihentikan — {total_tx:,} tx diproses, "
              f"{total_alerts:,} alerts, {total_freeze:,} auto-freeze")


if __name__ == "__main__":
    main()
