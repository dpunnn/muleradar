"""
Kafka Consumer — MuleRadar Real-Time AI Detection Pipeline (production-grade).

Flow per transaksi masuk (FAST PATH):
  Kafka 'transactions.raw'
    → RealtimeScorer: update feature store (Redis) → model base + streaming signals
      (base model DEFAULT = TGN-streaming sejak 20-Jul/Phase 4.8; SCORER_MODE=xgb
       utk balik ke XGBoost. Fallback otomatis ke XGBoost kalau model TGN hilang)
    → risk score + decision (NONE/ALERT/ESCALATE/FREEZE) — TANPA label
    → MERGE ke Neo4j (incremental)
    → alert/freeze → PostgreSQL + Kafka 'alerts.generated' + audit trail

Deep re-scoring (ensemble TGN+DyGFormer) jalan di SLOW PATH (run_detection
terjadwal) — lihat Phase 4.9 utk rencana alert batch pakai ensemble.

Cara pakai:
  python consumer.py
  python consumer.py --batch-size 50 --flush-interval 5
"""

import os
import sys
import json
import time
import uuid
import hashlib
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


DEDUP_TTL_S = int(os.getenv("DEDUP_TTL_S", "86400"))  # 24 jam


def _make_consumer() -> KafkaConsumer:
    # Fix (17-Jul, audit produksi delivery semantics):
    # - enable_auto_commit=False: SEBELUMNYA True (commit offset pakai timer
    #   5 detik, TANPA peduli proses sukses) -> pesan gagal proses tetap
    #   ke-commit -> HILANG permanen. Utk AML, transaksi hilang = fraud
    #   lolos tak terpantau. Sekarang MANUAL commit HANYA setelah flush
    #   Neo4j sukses (at-least-once) -> tidak ada transaksi hilang.
    # - auto_offset_reset="earliest": SEBELUMNYA "latest" (cold start /
    #   offset out-of-range -> LEWATI semua transaksi yg masuk saat consumer
    #   mati). "earliest" -> tak ada window transaksi terlewat. (Utk group yg
    #   sudah py committed offset, tetap lanjut dari situ, bukan rewind 144jt.)
    return KafkaConsumer(
        TOPIC_TX,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id="muleradar-detection",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=1000,
    )


def _dedup_claim(scorer, tx_id) -> bool:
    """
    Idempotensi FeatureStore utk at-least-once (fix 17-Jul, audit produksi).
    Redis FeatureStore pakai HINCRBY (out_degree/total_tx/amount_sum) yg TAK
    idempoten -> replay tx sama saat retry = double-count state. `SET NX EX`
    = atomic claim: True cuma utk tx_id yg BARU pertama kali dilihat. tx yg
    sudah pernah -> False -> consumer panggil score(apply_update=False) ->
    lewati update state. TTL 24h -> memori bounded (bukan tumbuh selamanya).
    Fail-open: kalau Redis error, anggap baru (jangan blokir stream).
    """
    if not tx_id:
        return True
    try:
        return bool(scorer.store.r.set(f"proc:{tx_id}", 1, nx=True, ex=DEDUP_TTL_S))
    except Exception:
        return True


def _make_alert_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )


def _merge_to_neo4j(batch: list[dict], driver):
    """
    MERGE batch transaksi ke Neo4j secara incremental (streaming-safe).

    Fix (17-Jul, audit produksi): edge sebelumnya pakai CREATE -> replay/
    redelivery Kafka (normal saat rebalance/scale) bikin edge TRANSFER
    DUPLIKAT (tx_id sama) -> fan_in/degree membengkak palsu -> korupsi
    sinyal yg jadi andalan scorer. Sekarang MERGE by tx_id (idempoten):
    edge yg sama cuma dibuat SEKALI. WAJIB ada index TRANSFER.tx_id
    (dibuat di main(), lihat _ensure_indexes) — tanpa index, MERGE by
    property = full-scan 176M edge = fatal.
    """
    with driver.session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (a:Account {account_id: row.from_account})
            MERGE (b:Account {account_id: row.to_account})
            MERGE (a)-[t:TRANSFER {tx_id: row.tx_id}]->(b)
            ON CREATE SET t.amount        = row.amount,
                          t.tx_timestamp  = row.tx_timestamp,
                          t.channel       = row.channel,
                          t.device_id     = row.device_id,
                          t.is_laundering = row.is_laundering
            """,
            rows=batch,
        )


def _ensure_indexes(driver):
    """
    Fix (17-Jul, audit produksi): index utk MERGE edge by tx_id (idempotensi
    di _merge_to_neo4j). Tanpa ini MERGE (a)-[:TRANSFER {tx_id}]->(b) =
    full-scan 176M edge tiap batch. IF NOT EXISTS -> aman dipanggil berkali2.
    """
    try:
        with driver.session() as session:
            session.run(
                "CREATE INDEX transfer_txid IF NOT EXISTS FOR ()-[r:TRANSFER]-() ON (r.tx_id)"
            )
        print("[consumer] Index TRANSFER.tx_id dipastikan ada (utk MERGE idempoten).")
    except Exception as e:
        print(f"[consumer] WARNING: gagal buat index TRANSFER.tx_id: {e}")


def _safe_merge_to_neo4j(batch: list[dict], driver) -> bool:
    """
    Fix (17-Jul, audit kesiapan produksi): _merge_to_neo4j() SEBELUMNYA
    dipanggil TANPA try/except di 3 tempat berbeda di main() — Neo4j
    connection drop/timeout SEKALI SAJA bikin consumer crash total (main
    loop cuma tangkap KeyboardInterrupt). Ini PERSIS jenis bug yg baru
    ditemukan+difix di realtime_scorer.py (feature mismatch), tapi di
    fungsi lain yg belum disentuh saat itu.

    CATATAN JUJUR: ini BUKAN Dead Letter Queue asli (Phase 15.1, belum
    dibangun) — batch yg gagal merge di sini DIBUANG (log error, lanjut),
    bukan diparkir utk retry/tinjau ulang. Cukup utk cegah crash total
    sekarang; DLQ genuine tetap perlu dikerjakan terpisah.
    """
    try:
        _merge_to_neo4j(batch, driver)
        return True
    except Exception as e:
        # Wording fix (QC e2e 20-Jul): batch TIDAK dibuang di sini — return
        # False bikin flush_and_commit MENAHAN batch (offset tak di-commit)
        # utk di-retry loop berikutnya. Batch baru benar2 dibuang HANYA kalau
        # menumpuk > max_batch (cegah OOM), di main() cabang [CRITICAL].
        # (Solusi tuntas = DLQ, Phase 15.1.)
        print(f"\n  [warn] Gagal merge {len(batch)} tx ke Neo4j (batch DITAHAN "
              f"utk retry, offset tak di-commit; dibuang hanya jika menumpuk "
              f"lewat cap anti-OOM — lihat Phase 15.1): {e}")
        return False


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

    # Fix (17-Jul, ditemukan saat cek alert demo-collector di UI): kolom
    # transactions.typology di DB ternyata berisi STRING LITERAL "NaN"
    # (bukan NULL asli - kemungkinan artefak pandas fillna/CSV round-trip
    # saat load data, root cause terpisah blm diselidiki tuntas) utk 100%
    # baris (144 juta). "NaN" itu truthy di Python, jadi `tx.get(...) or
    # "AI_DETECTED"` sebelumnya TIDAK PERNAH fallback -> alert baru dari
    # jalur real-time selalu tampil typology "NaN" di UI. Alert BATCH lama
    # (31rb+) tidak kena krn typology-nya dihitung dari rules.py, bukan
    # disalin dari kolom transactions.typology ini.
    raw_typology = tx.get("typology")
    typology = raw_typology if raw_typology and str(raw_typology).strip().lower() != "nan" else "AI_DETECTED"

    # Fix (17-Jul, audit produksi at-least-once): alert_id sebelumnya
    # uuid4() ACAK tiap panggil -> kalau tx diproses ulang (redelivery Kafka
    # saat rebalance / at-least-once retry), alert_id beda -> ON CONFLICT
    # DO NOTHING TIDAK dedup -> alert DUPLIKAT di Postgres. Sekarang
    # DETERMINISTIK dari tx_id (1 alert per transaksi, semantik yg benar):
    # replay tx yg sama -> alert_id sama -> ON CONFLICT beneran dedup.
    tx_id = tx.get("tx_id") or uuid.uuid4().hex
    alert_id = "ALT-" + hashlib.sha1(str(tx_id).encode()).hexdigest()[:12].upper()

    alert = {
        "alert_id":       alert_id,
        "account_id":     scored["account_id"],
        "tx_id":          tx.get("tx_id"),
        "typology":       typology,
        # detection_layer (fix 6.7, 20-Jul): jalur real-time = fusion
        # XGBoost + sinyal perilaku (fan_in/velocity), sumber deteksinya
        # MODEL, jadi masuk "ML_ENSEMBLE" (salah satu dari 5 sumber deteksi,
        # lihat detection/rules.py DETECTION_LAYER). Bukan salah satu rule
        # layer batch (AML_CORE/TYPOLOGY_ID/STATISTICAL/GRAPH_MOTIF).
        "detection_layer": "ML_ENSEMBLE",
        "risk_score":     scored["risk_score"],
        "rule_triggered": f"[{decision}] base={scored['base_score']} | {reasons}",
        "severity":       scored["risk_level"],
        "status":         status,
        "created_at":     datetime.utcnow().isoformat(),
    }

    # Fix (17-Jul, audit kesiapan produksi): send() sebelumnya TIDAK
    # dibungkus try/except — kalau broker Kafka bermasalah sesaat, ini
    # crash SELURUH consumer (main loop cuma tangkap KeyboardInterrupt).
    # Insert Postgres tetap jalan walau publish Kafka gagal (alert tetap
    # tersimpan, cuma topic 'alerts.generated' yg tak kebagian event ini).
    try:
        alert_producer.send(TOPIC_ALERTS, value=alert)
    except Exception as e:
        print(f"  [warn] Gagal publish alert ke Kafka: {e}")

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO alerts (alert_id, account_id, tx_id, typology,
                    detection_layer, risk_score, rule_triggered, severity,
                    status, created_at)
                VALUES (:alert_id, :account_id, :tx_id, :typology,
                    :detection_layer, :risk_score, :rule_triggered, :severity,
                    :status, :created_at)
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

    _ensure_indexes(driver)

    print("[consumer] Loading RealtimeScorer (Redis feature store)...")
    scorer = RealtimeScorer()
    # Cetak mode base-model yg BENAR-BENAR aktif (bukan asumsi) — kalau TGN
    # gagal load, scorer sudah fallback ke xgb & ini menunjukkannya jujur.
    print(f"[consumer] Base model aktif: {scorer.mode.upper()}"
          f"{' (TGN-streaming, memory-state di Redis)' if scorer.mode == 'tgn' else ' (XGBoost)'}")
    if not scorer.store.ping():
        print("[consumer] WARNING: Redis tidak terjangkau - feature store mati")

    print(f"[consumer] Listening topic '{TOPIC_TX}' | batch={args.batch_size} | flush={args.flush_interval}s "
          f"| at-least-once (manual commit)")
    print("[consumer] Ctrl+C untuk stop\n")

    # Cap memori: kalau Neo4j down lama, batch tak boleh tumbuh tak terbatas
    # (kita PERNAH kena insiden OOM). Di atas cap, buang tertua (BUKAN DLQ
    # asli - Phase 15.1) supaya proses tak mati.
    max_batch = args.batch_size * 20

    def flush_and_commit(b) -> bool:
        """Flush batch ke Neo4j (MERGE idempoten) lalu commit offset Kafka
        HANYA kalau flush sukses -> at-least-once, tidak ada transaksi
        hilang. Return True kalau aman clear batch."""
        if not b:
            return True
        if not _safe_merge_to_neo4j(b, driver):
            return False
        try:
            consumer.commit()   # commit HANYA setelah Neo4j sukses
            return True
        except Exception as e:
            print(f"\n  [warn] Gagal commit offset Kafka: {e}")
            return False

    batch        = []
    last_flush   = time.time()
    total_tx     = 0
    total_alerts = 0
    total_freeze = 0

    try:
        while True:
            for msg in consumer:
                tx = msg.value
                txid = tx.get("tx_id")
                batch.append(tx)

                # Dedup guard (fix produksi): apply update state HANYA sekali
                # per tx_id -> aman utk at-least-once (replay tak double-count).
                # CATATAN JUJUR: ada window mikro-detik antara SET proc dan
                # store.update() di mana crash bisa bikin 1 tx tak ter-apply
                # ke state (undercount fan_in akun itu by 1) — probabilitas
                # sangat kecil, dampak self-heal di tx berikutnya. Jauh lebih
                # baik drpd at-most-once (data loss) sebelumnya.
                is_new = _dedup_claim(scorer, txid)

                # FAST PATH: real-time scoring. try/except -> gagal skor tak
                # matikan consumer (tx tetap masuk batch Neo4j).
                try:
                    scored = scorer.score(tx, apply_update=is_new)
                except Exception as e:
                    print(f"\n  [warn] Gagal scoring tx {txid}: {e}")
                    scored = None

                if scored and scored["decision"] != "NONE":
                    decision = _save_alert(tx, scored, engine, alert_producer)
                    total_alerts += 1
                    if decision == "FREEZE":
                        total_freeze += 1
                        print(f"\n  [FREEZE] AUTO-FREEZE {scored['account_id']} "
                              f"risk={scored['risk_score']} ({', '.join(scored['reasons'])})")

                # Flush + commit kalau batch penuh atau timeout
                now = time.time()
                if len(batch) >= args.batch_size or (now - last_flush) >= args.flush_interval:
                    if flush_and_commit(batch):
                        total_tx += len(batch)
                        print(f"  flushed {len(batch)} tx -> Neo4j + committed | total={total_tx:,} | "
                              f"alerts={total_alerts:,} | frozen={total_freeze:,}", end="\r")
                        batch = []
                    elif len(batch) > max_batch:
                        drop = len(batch) - max_batch
                        print(f"\n  [CRITICAL] Neo4j/commit gagal lama — buang {drop} tx tertua "
                              f"(BUKAN DLQ, lihat Phase 15.1) cegah OOM")
                        batch = batch[drop:]
                    last_flush = now

            # consumer_timeout (1s tanpa pesan baru): flush sisa + commit
            if batch and flush_and_commit(batch):
                total_tx += len(batch)
                batch = []
            last_flush = time.time()

    except KeyboardInterrupt:
        flush_and_commit(batch)
        alert_producer.flush()
        consumer.close()
        driver.close()
        print(f"\n[consumer] Dihentikan - {total_tx:,} tx diproses, "
              f"{total_alerts:,} alerts, {total_freeze:,} auto-freeze")


if __name__ == "__main__":
    main()
