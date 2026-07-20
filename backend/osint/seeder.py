"""
Phase 4.5.6 — Seeder: hasil OSINT → PostgreSQL + Neo4j + trigger PPR → Alert.

Peran seeder menghubungkan modul OSINT ke inti graph MuleRadar:

  1. Upsert rekening hasil ekstraksi ke osint_accounts (append sumber_url,
     update shared_count, ambil confidence tertinggi).
  2. Seed tiap rekening sebagai node :Account:OsintAccount di Neo4j. Rekening
     judol = calon `to_account` di transaction graph, jadi node ini menjadi
     starting point PPR untuk menemukan money mule yang terhubung.
  3. Jalankan PPR dari node OSINT baru → temukan akun transaksi yang terhubung.
  4. Buat Alert HIGH (typology='judol_osint', status='NEW') untuk ditinjau analis.

PENTING (training loop): rekening OSINT TIDAK langsung masuk training_pool.
Alur: OSINT → Alert (via PPR) → analis konfirmasi → training_pool (source=OSINT).
Human-in-the-loop menjaga kualitas label. Seeder berhenti di pembuatan alert.

Import modul graph di-guard agar seeder tetap bisa persist ke PostgreSQL
meski konteks eksekusi tidak punya akses driver Neo4j (mis. unit test PG-only).
"""

import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "muleradar_neo4j")

# Skor risiko default alert yang berasal dari OSINT (rekening sudah ada di
# blocklist judol → prior tinggi). >= SEVERITY_HIGH di alerts.py (0.8).
OSINT_ALERT_RISK = 0.90
PPR_TOP_K = 20

# Guarded imports — graph stack.
try:
    from neo4j import GraphDatabase  # type: ignore
    _NEO4J_LIB = True
except ImportError:
    _NEO4J_LIB = False

try:
    from graph.analytics import run_ppr  # type: ignore
    _PPR_AVAILABLE = True
except Exception:
    _PPR_AVAILABLE = False

from osint.network import risk_from_site_count


def get_driver():
    if not _NEO4J_LIB:
        return None
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# -----------------------------------------------------------------
# 1. Persist ke PostgreSQL
# -----------------------------------------------------------------

def persist_accounts(accounts: list, source_url: str, engine) -> int:
    """
    Upsert rekening ke osint_accounts.

    `accounts` : list objek dengan atribut .rekening .bank .account_type
                 .confidence (output extractor.extract).
    Untuk rekening yang sudah ada: tambahkan source_url ke array sumber_url
    (bila belum ada), naikkan shared_count = cardinality(sumber_url),
    perbarui last_seen, dan ambil confidence tertinggi.
    """
    if not accounts:
        return 0
    inserted_or_updated = 0
    now = datetime.utcnow()
    with engine.begin() as conn:
        for acc in accounts:
            conn.execute(
                text("""
                    INSERT INTO osint_accounts
                        (rekening, bank, account_type, sumber_url, shared_count,
                         confidence, seeded_to_graph, first_seen, last_seen)
                    VALUES
                        (:rek, :bank, :atype, ARRAY[:url], 1,
                         :conf, FALSE, :now, :now)
                    ON CONFLICT (rekening) DO UPDATE SET
                        sumber_url = CASE
                            WHEN :url = ANY(osint_accounts.sumber_url)
                                THEN osint_accounts.sumber_url
                            ELSE array_append(osint_accounts.sumber_url, :url)
                        END,
                        shared_count = cardinality(CASE
                            WHEN :url = ANY(osint_accounts.sumber_url)
                                THEN osint_accounts.sumber_url
                            ELSE array_append(osint_accounts.sumber_url, :url)
                        END),
                        confidence = GREATEST(osint_accounts.confidence, :conf),
                        last_seen = :now
                """),
                {
                    "rek": acc.rekening,
                    "bank": acc.bank,
                    "atype": getattr(acc, "account_type", "bank"),
                    "url": source_url,
                    "conf": float(getattr(acc, "confidence", 1.0)),
                    "now": now,
                },
            )
            inserted_or_updated += 1

    # Perbarui rekening_count di osint_sites untuk situs ini.
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE osint_sites SET rekening_count = :cnt WHERE url = :url
            """),
            {"cnt": len(accounts), "url": source_url},
        )
    return inserted_or_updated


# -----------------------------------------------------------------
# 2. Seed ke Neo4j
# -----------------------------------------------------------------

def lookup_risk_levels(rekening_list: list[str], engine) -> dict[str, str]:
    """
    Ambil shared_count TERKINI dari osint_accounts untuk tiap rekening lalu
    konversi ke risk_level (HIGH/MED/LOW) via risk_from_site_count — satu
    sumber kebenaran yang sama dipakai network.py & exporter.py.

    Rekening yang belum ada di osint_accounts (mis. dipanggil sebelum
    persist_accounts) dianggap baru terlihat di 1 situs → LOW.
    """
    if not rekening_list:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT rekening, shared_count FROM osint_accounts WHERE rekening = ANY(:reks)"),
            {"reks": rekening_list},
        ).fetchall()
    counts = {r[0]: r[1] for r in rows}
    return {rek: risk_from_site_count(counts.get(rek, 1)) for rek in rekening_list}


def seed_to_neo4j(rekening_risk: dict[str, str], driver) -> int:
    """
    MERGE tiap rekening sebagai node :Account dengan label tambahan :OsintAccount.

    `rekening_risk`: {rekening: risk_level} — risk_level PER REKENING (dari
    lookup_risk_levels), BUKAN satu nilai dipukul rata untuk seluruh batch.
    Rekening yang muncul di 3+ situs (jaringan bandar) harus tampil HIGH di
    Graph Explorer; yang baru 1 situs jangan ikut ditandai HIGH juga.

    Node yang sudah ada di transaction graph (karena pernah bertransaksi) hanya
    ditambahi label + properti OSINT — TIDAK menghapus data transaksi.
    """
    if driver is None or not rekening_risk:
        return 0
    seeded = 0
    with driver.session() as session:
        for rek, risk in rekening_risk.items():
            session.run(
                """
                MERGE (a:Account {account_id: $rek})
                SET a:OsintAccount,
                    a.source = 'osint',
                    a.osint_risk_level = $risk,
                    a.osint_seen_at = datetime()
                """,
                rek=rek, risk=risk,
            )
            seeded += 1
    return seeded


def _mark_seeded(rekening_list: list[str], engine) -> None:
    if not rekening_list:
        return
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE osint_accounts SET seeded_to_graph = TRUE
                WHERE rekening = ANY(:reks)
            """),
            {"reks": rekening_list},
        )


# -----------------------------------------------------------------
# 3. Trigger PPR + 4. Buat Alert
# -----------------------------------------------------------------

def trigger_ppr_alerts(rekening_list: list[str], driver, engine) -> int:
    """
    Untuk tiap rekening OSINT, jalankan PPR di transaction graph → temukan akun
    terhubung. Buat satu alert HIGH untuk rekening OSINT itu sendiri, plus alert
    untuk akun ber-skor tertinggi yang terhubung (kandidat money mule).

    Return: jumlah alert dibuat.
    """
    if driver is None or not _PPR_AVAILABLE:
        # Tetap buat alert untuk rekening OSINT (tanpa ekspansi jaringan).
        return _insert_osint_alerts(rekening_list, {}, engine)

    connected: dict[str, float] = {}
    for rek in rekening_list:
        neighbors = run_ppr(driver, seed_node=rek, top_k=PPR_TOP_K)
        for acc_id, score in neighbors.items():
            # Ambil skor tertinggi bila akun muncul dari beberapa seed.
            if score > connected.get(acc_id, 0.0):
                connected[acc_id] = score

    return _insert_osint_alerts(rekening_list, connected, engine)


def _insert_osint_alerts(
    seed_rekening: list[str],
    connected: dict[str, float],
    engine,
) -> int:
    """
    INSERT alert ke tabel alerts (pola sama dengan consumer.py).
    - Rekening OSINT seed: risk OSINT_ALERT_RISK, severity HIGH.
    - Akun terhubung via PPR: risk = OSINT_ALERT_RISK * ppr_score (min 0.5).

    Dedup: satu account_id hanya boleh punya SATU alert judol_osint yang
    masih terbuka (status bukan FP/CLOSED). Tanpa ini, rekening yang sama
    akan berulang kali menghasilkan alert baru setiap kali muncul lagi di
    crawl/sync berikutnya (rekening OSINT sering ditemukan di banyak situs
    dan banyak sesi crawl) — membanjiri antrean analis dengan kasus kembar,
    bertentangan dengan tujuan utama MuleRadar (mengurangi alert fatigue).
    """
    now = datetime.utcnow().isoformat()

    def _row(account_id: str, risk: float, note: str) -> dict:
        return {
            "alert_id": f"ALT-{uuid.uuid4().hex[:12]}",
            "account_id": account_id,
            "tx_id": None,
            "typology": "judol_osint",
            "risk_score": round(float(risk), 4),
            "rule_triggered": note,
            "severity": "HIGH" if risk >= 0.8 else "MEDIUM",
            "status": "NEW",
            "created_at": now,
        }

    rows: list[dict] = []
    for rek in seed_rekening:
        rows.append(_row(
            rek, OSINT_ALERT_RISK,
            "OSINT: rekening ditemukan di situs judol (Kominfo blocklist)",
        ))
    for acc_id, score in connected.items():
        if acc_id in seed_rekening:
            continue
        risk = max(0.5, OSINT_ALERT_RISK * score)
        rows.append(_row(
            acc_id, risk,
            f"OSINT-PPR: terhubung ke rekening judol (skor {score:.3f})",
        ))

    if not rows:
        return 0

    candidate_ids = list({row["account_id"] for row in rows})
    with engine.connect() as conn:
        existing = conn.execute(
            text("""
                SELECT DISTINCT account_id FROM alerts
                WHERE account_id = ANY(:ids) AND typology = 'judol_osint'
                  AND status NOT IN ('FP', 'CLOSED')
            """),
            {"ids": candidate_ids},
        ).fetchall()
    already_open = {r[0] for r in existing}
    rows = [row for row in rows if row["account_id"] not in already_open]
    if not rows:
        return 0

    created = 0
    with engine.begin() as conn:
        for row in rows:
            result = conn.execute(
                text("""
                    INSERT INTO alerts (alert_id, account_id, tx_id, typology,
                        risk_score, rule_triggered, severity, status, created_at)
                    VALUES (:alert_id, :account_id, :tx_id, :typology,
                        :risk_score, :rule_triggered, :severity, :status, :created_at)
                    ON CONFLICT (alert_id) DO NOTHING
                """),
                row,
            )
            created += result.rowcount or 0
    return created


# -----------------------------------------------------------------
# Orkestrasi penuh: dipanggil per situs oleh pipeline crawl→extract→seed
# -----------------------------------------------------------------

def seed_site_results(source_url: str, accounts: list, engine=None, driver=None) -> dict:
    """
    Satu situs: persist rekening → seed Neo4j → PPR → alert.
    `accounts` = output extractor.extract(html, screenshot).
    """
    own_engine = engine is None
    if own_engine:
        engine = create_engine(DATABASE_URL)
    own_driver = driver is None
    if own_driver:
        driver = get_driver()

    try:
        persisted = persist_accounts(accounts, source_url, engine)
        rekening_list = [a.rekening for a in accounts]
        risk_map = lookup_risk_levels(rekening_list, engine)
        seeded = seed_to_neo4j(risk_map, driver)
        _mark_seeded(rekening_list, engine)
        alerts = trigger_ppr_alerts(rekening_list, driver, engine)
        return {
            "url": source_url,
            "accounts_persisted": persisted,
            "seeded_to_graph": seeded,
            "alerts_created": alerts,
        }
    finally:
        if own_driver and driver is not None:
            driver.close()


if __name__ == "__main__":
    # Smoke test tanpa crawler: butuh objek rekening dummy.
    from dataclasses import dataclass

    @dataclass
    class _Acc:
        rekening: str
        bank: str
        account_type: str = "bank"
        confidence: float = 1.0

    sample_accounts = [_Acc("1234567890", "BCA"), _Acc("0987654321", "BNI")]
    print("[seeder] sample:", seed_site_results("https://sample.example/deposit", sample_accounts))
