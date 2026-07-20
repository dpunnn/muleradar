"""
Uji idempotensi jalur streaming produksi (17-Jul).

Membuktikan (bukan klaim): kirim transaksi yg SAMA (tx_id sama) 3x ke Kafka
-> hasil akhir HARUS:
  - Neo4j: TEPAT 1 edge utk tx_id itu (MERGE, bukan CREATE)
  - Redis: total_tx akun pengirim TEPAT +1 (dedup guard, bukan +3)
  - alert_id deterministik (sha1 tx_id) -> kalau alert terpicu, ON CONFLICT dedup

Prasyarat: consumer.py SEDANG JALAN (proses topic transactions.raw).
Cara pakai: python test_idempotency.py
"""

import json
import time
import hashlib

import redis
from kafka import KafkaProducer
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text

TXID = "TEST-IDEM-0001"
SENDER = "TEST-SENDER-A"
COLLECTOR = "TEST-COLLECTOR-Z"

r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
driver = GraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "muleradar_neo4j"))
engine = create_engine("postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    acks="all",
)

print("[test] Bersihkan state lama utk akun/tx test...")
for acc in (SENDER, COLLECTOR):
    for suf in ("", ":dev", ":in_cp", ":out_cp", ":txwin", ":inwin", ":chan"):
        r.delete(f"acct:{acc}{suf}")
r.delete(f"proc:{TXID}")
with driver.session() as s:
    # Scoped ke akun test (bukan global by tx_id -> hindari full-scan 176M
    # edge yg butuh index transfer_txid yg masih populating).
    s.run("MATCH (a:Account {account_id:$s})-[t:TRANSFER]->() DELETE t", s=SENDER)
alert_id = "ALT-" + hashlib.sha1(TXID.encode()).hexdigest()[:12].upper()
with engine.begin() as conn:
    conn.execute(text("DELETE FROM alerts WHERE tx_id = :id OR alert_id = :aid"),
                 {"id": TXID, "aid": alert_id})

tx = {
    "tx_id": TXID, "from_account": SENDER, "to_account": COLLECTOR,
    "amount": 5_000_000, "currency": "IDR", "channel": "mobile",
    "payment_format": "Transfer", "tx_timestamp": "2026-07-17 10:00:00",
    "device_id": "DEV-TEST01", "institution_id": "BANK_A",
    "is_laundering": 1, "typology": "judol",
}

print(f"[test] Kirim tx {TXID} sebanyak 3x (harus di-dedup jadi 1)...")
for i in range(3):
    producer.send("transactions.raw", value=tx)
producer.flush()

print("[test] Tunggu consumer proses + flush (12 detik)...")
time.sleep(12)

# --- Verifikasi ---
with driver.session() as s:
    # Scoped ke akun test (bukan global by tx_id) -> cepat, tak butuh index.
    n_edges = s.run("MATCH (a:Account {account_id:$s})-[t:TRANSFER {tx_id:$id}]->() "
                    "RETURN count(t) AS n", s=SENDER, id=TXID).single()["n"]
total_tx = r.hget(f"acct:{SENDER}", "total_tx")
fan_in = r.scard(f"acct:{COLLECTOR}:in_cp")
with engine.connect() as conn:
    n_alerts = conn.execute(text("SELECT COUNT(*) FROM alerts WHERE tx_id = :id"),
                            {"id": TXID}).scalar()

print("\n=== HASIL ===")
print(f"Neo4j edge utk tx {TXID}     : {n_edges}  (HARUS 1 kalau MERGE idempoten)")
print(f"Redis total_tx {SENDER}      : {total_tx}  (HARUS 1 kalau dedup guard jalan, bukan 3)")
print(f"Redis fan_in {COLLECTOR}     : {fan_in}  (HARUS 1)")
print(f"Postgres alert utk tx        : {n_alerts}  (HARUS <=1, deterministic alert_id)")

ok = (n_edges == 1 and str(total_tx) == "1" and fan_in == 1 and n_alerts <= 1)
print(f"\n{'LULUS - idempoten' if ok else 'GAGAL - masih ada duplikasi'}")

driver.close()
