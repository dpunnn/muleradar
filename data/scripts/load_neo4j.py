"""
Bulk load transaksi dari PostgreSQL → Neo4j secara chunked.
Aman untuk dataset besar (176M+ rows) tanpa OOM.

Cara pakai:
  python load_neo4j.py
  python load_neo4j.py --chunk-size 50000 --full-reload
  python load_neo4j.py --illicit-only
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL     = os.getenv("DATABASE_URL",      "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
NEO4J_URI        = os.getenv("NEO4J_URI",         "bolt://localhost:7687")
NEO4J_USER       = os.getenv("NEO4J_USER",        "neo4j")
NEO4J_PASS       = os.getenv("NEO4J_PASSWORD",    "muleradar_neo4j")
NEO4J_CONTAINER  = os.getenv("NEO4J_CONTAINER",   "muleradar_neo4j")
# Path bind mount Neo4j di host — sesuaikan jika berbeda
NEO4J_DATA_HOST  = os.getenv("NEO4J_DATA_HOST",   r"D:\muleradar-data\neo4j\data")

CHUNK_DEFAULT = 50_000

_MERGE_CYPHER = """
UNWIND $rows AS row
MERGE (a:Account {account_id: row.from_account})
MERGE (b:Account {account_id: row.to_account})
CREATE (a)-[:TRANSFER {
    tx_id: row.tx_id,
    amount: row.amount,
    tx_timestamp: row.tx_timestamp,
    channel: row.channel,
    device_id: row.device_id,
    is_laundering: row.is_laundering
}]->(b)
"""


def get_total(engine, illicit_only: bool) -> int:
    where = "WHERE is_laundering = 1" if illicit_only else ""
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM transactions {where}")).scalar()


def stream_from_pg(engine, chunk_size: int, illicit_only: bool):
    where = "WHERE is_laundering = 1" if illicit_only else ""
    query = f"""
        SELECT tx_id, from_account, to_account, amount, tx_timestamp,
               channel, device_id, is_laundering
        FROM transactions {where}
        ORDER BY tx_timestamp
    """
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(text(query))
        cols   = list(result.keys())
        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            import pandas as pd
            yield pd.DataFrame(rows, columns=cols)


def _prep_chunk(chunk) -> list:
    chunk = chunk.copy()
    chunk["tx_timestamp"]  = chunk["tx_timestamp"].astype(str)
    chunk["amount"]        = chunk["amount"].astype(float)
    chunk["is_laundering"] = chunk["is_laundering"].fillna(0).astype(int)
    chunk["device_id"]     = chunk["device_id"].fillna("").astype(str)
    chunk["tx_id"]         = chunk["tx_id"].fillna("").astype(str)
    return chunk.to_dict("records")


def _clear_neo4j_via_docker(driver_factory):
    """
    Clear Neo4j dengan cara aman: stop container → hapus data dir → start ulang.
    Jauh lebih reliable dibanding Cypher DELETE untuk dataset besar.
    """
    from neo4j import GraphDatabase

    print("      Stopping Neo4j container...")
    subprocess.run(["docker", "stop", NEO4J_CONTAINER], check=True, capture_output=True)

    # Hapus hanya subdirektori graph data, bukan seluruh data dir
    for subdir in ["databases", "transactions"]:
        path = os.path.join(NEO4J_DATA_HOST, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"      Cleared {subdir}/")

    print("      Starting Neo4j container...")
    subprocess.run(["docker", "start", NEO4J_CONTAINER], check=True, capture_output=True)

    print("      Waiting for Neo4j to be ready", end="", flush=True)
    for _ in range(60):
        time.sleep(5)
        print(".", end="", flush=True)
        try:
            driver = driver_factory()
            driver.verify_connectivity()
            driver.close()
            print(" OK")
            return
        except Exception:
            pass
    raise RuntimeError("Neo4j tidak ready setelah 300 detik — cek Docker Desktop")


def bulk_load(chunk_size: int, full_reload: bool, illicit_only: bool):
    from neo4j import GraphDatabase
    from graph.builder import _ensure_index, drop_gds_projection

    def make_driver():
        return GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASS),
            connection_acquisition_timeout=300,
            max_connection_lifetime=7200,
        )

    # keepalives cegah koneksi streaming PG putus di Windows
    engine = create_engine(
        DATABASE_URL,
        connect_args={
            "keepalives": 1,
            "keepalives_idle": 60,
            "keepalives_interval": 10,
            "keepalives_count": 10,
        },
    )

    print("[1/3] Connecting...")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("      PostgreSQL OK")

    driver = make_driver()
    driver.verify_connectivity()
    print("      Neo4j OK")

    total = get_total(engine, illicit_only)
    print(f"\n[2/3] Total rows to load: {total:,}")

    if full_reload:
        print("      Full reload — clearing via Docker...")
        driver.close()
        _clear_neo4j_via_docker(make_driver)
        # Buat driver baru setelah Neo4j restart
        driver = make_driver()

    # Setup index dan GDS sekali di awal
    _ensure_index(driver)
    drop_gds_projection(driver)

    print(f"\n[3/3] Loading ke Neo4j (chunk={chunk_size:,})...")
    loaded = 0
    chunk_num = 0
    t0 = time.time()

    for chunk in stream_from_pg(engine, chunk_size, illicit_only):
        chunk_num += 1
        records = _prep_chunk(chunk)

        with driver.session() as session:
            session.run(_MERGE_CYPHER, rows=records)

        loaded += len(chunk)
        elapsed = time.time() - t0
        rate    = loaded / elapsed if elapsed > 0 else 0
        eta     = (total - loaded) / rate if rate > 0 else 0
        print(f"  chunk {chunk_num}: {loaded:,}/{total:,} "
              f"({loaded/total*100:.1f}%) "
              f"rate={rate:.0f}/s ETA={eta/3600:.1f}h", end="\r")

    print(f"\n\n=== LOAD SELESAI ===")
    print(f"Total loaded : {loaded:,} rows")
    try:
        with driver.session() as session:
            r = session.run(
                "MATCH (n:Account) WITH count(n) AS nodes "
                "MATCH ()-[r:TRANSFER]->() WITH nodes, count(r) AS edges "
                "RETURN nodes, edges"
            ).single()
            print(f"Neo4j Nodes : {r['nodes']:,}")
            print(f"Neo4j Edges : {r['edges']:,}")
    except Exception as e:
        print(f"(Stats query skipped: {e})")

    driver.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size",  type=int, default=CHUNK_DEFAULT)
    parser.add_argument("--full-reload", action="store_true", default=False,
                        help="Clear Neo4j via Docker restart, lalu load ulang dari awal")
    parser.add_argument("--illicit-only", action="store_true", default=False,
                        help="Hanya load transaksi illicit")
    args = parser.parse_args()

    bulk_load(args.chunk_size, args.full_reload, args.illicit_only)


if __name__ == "__main__":
    main()
