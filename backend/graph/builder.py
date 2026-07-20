"""
Graph builder: load transaksi dari PostgreSQL → Neo4j property graph.
Node: Account {account_id}
Edge: TRANSFER {amount, tx_timestamp, channel, device_id, is_laundering}
"""

import logging
import os
import pandas as pd
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "muleradar_neo4j")

GDS_GRAPH_NAME = "transactionGraph"


def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def load_transactions(db_url: str = DATABASE_URL, days: int = None) -> pd.DataFrame:
    """
    Query transaksi dari PostgreSQL.
    days=None  → load semua data (default, Neo4j disk-based tidak ada RAM limit)
    days=N     → sliding window N hari terakhir (untuk real-time detection)
    """
    engine = create_engine(db_url)
    # tx_id WAJIB di-SELECT (fix 20-Jul): load_graph_to_neo4j sekarang MERGE
    # edge by tx_id — tanpa tx_id asli, semua edge tx_id="" akan kolaps jadi
    # SATU edge (MERGE collapse). Dulu aman krn pakai CREATE (tak butuh key),
    # sekarang tidak.
    if days is not None:
        query = f"""
            SELECT tx_id, from_account, to_account, amount, tx_timestamp, channel, device_id, is_laundering
            FROM transactions
            WHERE tx_timestamp >= (SELECT MAX(tx_timestamp) FROM transactions) - INTERVAL '{days} days'
        """
    else:
        query = """
            SELECT tx_id, from_account, to_account, amount, tx_timestamp, channel, device_id, is_laundering
            FROM transactions
        """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def _ensure_index(driver) -> None:
    """Constraint Account.account_id + index TRANSFER.tx_id sebelum MERGE.

    index transfer_txid (fix 20-Jul): WAJIB utk MERGE edge by tx_id — tanpa
    ini MERGE-by-property = full-scan graph tiap edge (fatal di skala 176M).
    IF NOT EXISTS -> aman dipanggil tiap load.
    """
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT account_id_unique IF NOT EXISTS
            FOR (a:Account) REQUIRE a.account_id IS UNIQUE
        """)
        session.run("""
            CREATE INDEX transfer_txid IF NOT EXISTS
            FOR ()-[r:TRANSFER]-() ON (r.tx_id)
        """)


def load_graph_to_neo4j(df: pd.DataFrame, driver, batch_size: int = 5000,
                         full_reload: bool = False) -> dict:
    """
    Load/update transaction DataFrame ke Neo4j sebagai property graph.

    full_reload=False (default, streaming-safe):
        MERGE node + MERGE edge by tx_id → GENUINELY idempoten, aman dipanggil
        berulang thd data overlap (backfill/resync/retry), tidak duplikat edge.
        Cocok untuk Kafka consumer yang push transaksi baru secara incremental.

    full_reload=True (initial bulk load):
        DETACH DELETE semua dulu, lalu insert ulang.
        Dipakai saat load dataset pertama kali atau reset graph.

    Fix (20-Jul, scan produksi): SEBELUMNYA `CREATE (a)-[:TRANSFER {...}]->(b)`
    dan docstring KELIRU klaim "aman dipanggil berulang" — CREATE TIDAK
    idempoten: panggil ulang thd tx yg sama (mis. reload overlap, retry
    setelah error) -> edge DUPLIKAT -> fan_in/degree/pagerank membengkak
    palsu. Sekarang `MERGE ...{tx_id}... ON CREATE SET` — sama dgn perbaikan
    yg sudah dilakukan di streaming/consumer.py (blocker #2 audit C.2).
    WAJIB ada index transfer_txid (dibuat di _ensure_index, IF NOT EXISTS)
    supaya MERGE-by-property tidak full-scan.
    """
    if df.empty:
        return {"nodes": 0, "edges": 0}

    df = df.copy()
    df["tx_timestamp"]  = df["tx_timestamp"].astype(str)
    df["amount"]        = df["amount"].astype(float)
    df["is_laundering"] = df["is_laundering"].fillna(0).astype(int)
    df["device_id"]     = df["device_id"].fillna("").astype(str)
    df["tx_id"]         = df.get("tx_id", pd.Series([""] * len(df))).fillna("").astype(str)

    # Guard (QC 20-Jul): MERGE by tx_id -> baris tx_id KOSONG akan KOLAPS jadi
    # satu edge (semua "" dianggap tx_id sama). load_transactions/load_neo4j
    # sudah SELECT tx_id (PK NOT NULL) jadi normal-nya tak kejadian, TAPI kalau
    # caller lain kirim df tanpa tx_id, buang baris itu + warn — jangan diam2
    # korupsi graph. Lebih baik kehilangan baris cacat + log drpd 1 super-edge.
    n_before = len(df)
    df = df[df["tx_id"].str.strip() != ""]
    n_dropped = n_before - len(df)
    if n_dropped:
        logging.getLogger(__name__).warning(
            "load_graph_to_neo4j: %d baris tx_id KOSONG dibuang (cegah kolaps "
            "MERGE) — caller harus sertakan tx_id.", n_dropped)
    if df.empty:
        return {"nodes": 0, "edges": 0}

    _ensure_index(driver)

    with driver.session() as session:
        if full_reload:
            while True:
                result = session.run(
                    "MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS deleted"
                ).single()
                if result["deleted"] == 0:
                    break

        # Drop GDS projection dulu supaya tidak stale setelah data berubah
        drop_gds_projection(driver)

        records = df.to_dict("records")
        for i in range(0, len(records), batch_size):
            batch = records[i: i + batch_size]
            session.run(
                """
                UNWIND $rows AS row
                MERGE (a:Account {account_id: row.from_account})
                MERGE (b:Account {account_id: row.to_account})
                MERGE (a)-[t:TRANSFER {tx_id: row.tx_id}]->(b)
                ON CREATE SET
                    t.amount = row.amount,
                    t.tx_timestamp = row.tx_timestamp,
                    t.channel = row.channel,
                    t.device_id = row.device_id,
                    t.is_laundering = row.is_laundering
                """,
                rows=batch,
            )

        result = session.run(
            """
            MATCH (n:Account) WITH count(n) AS nodes
            MATCH ()-[r:TRANSFER]->() WITH nodes, count(r) AS edges
            RETURN nodes, edges
            """
        ).single()

    return {"nodes": result["nodes"], "edges": result["edges"]}


def ensure_gds_projection(driver, graph_name: str = GDS_GRAPH_NAME) -> None:
    """Buat GDS in-memory graph projection jika belum ada."""
    with driver.session() as session:
        exists = session.run(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists",
            name=graph_name,
        ).single()["exists"]

        if not exists:
            session.run(
                """
                CALL gds.graph.project($name, 'Account', 'TRANSFER',
                  { relationshipProperties: ['amount', 'is_laundering'] })
                """,
                name=graph_name,
            )


def drop_gds_projection(driver, graph_name: str = GDS_GRAPH_NAME) -> None:
    with driver.session() as session:
        exists = session.run(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists",
            name=graph_name,
        ).single()["exists"]
        if exists:
            session.run("CALL gds.graph.drop($name)", name=graph_name)


def graph_stats(driver) -> dict:
    """Return statistik dasar graph dari Neo4j."""
    with driver.session() as session:
        base = session.run(
            """
            MATCH (n:Account) WITH count(n) AS nodes
            MATCH ()-[r:TRANSFER]->() WITH nodes, count(r) AS edges
            RETURN nodes, edges
            """
        ).single()

        nodes = base["nodes"]
        edges = base["edges"]
        density = round(edges / (nodes * (nodes - 1)), 6) if nodes > 1 else 0.0

        try:
            ensure_gds_projection(driver)
            wcc = session.run(
                f"""
                CALL gds.wcc.stats('{GDS_GRAPH_NAME}')
                YIELD componentCount, componentDistribution
                RETURN componentCount,
                       componentDistribution.max AS largest_component_size
                """
            ).single()
            wcc_count = wcc["componentCount"]
            largest = wcc["largest_component_size"]
        except Exception:
            # Fix (20-Jul): log dulu — WCC di graph 176M edge bisa OOM/timeout,
            # jangan diam-diam laporkan 0 komponen seolah graph kosong.
            logging.getLogger(__name__).warning(
                "graph_stats: GDS WCC gagal (graph besar/OOM?) — wcc dilaporkan 0",
                exc_info=True)
            wcc_count = 0
            largest = 0

    return {
        "nodes": nodes,
        "edges": edges,
        "density": density,
        "weakly_connected_components": wcc_count,
        "largest_component_size": largest,
    }
