"""
Phase 5.1 — REST endpoints untuk Graph Explorer (Halaman 3).

Endpoint:
    GET /graph/overview                  -> stats (nodes, edges, clusters)
    GET /graph/cluster/{cluster_id}       -> nodes + edges dalam cluster
    GET /graph/node/{account_id}/neighbors -> tetangga 2 hop
    GET /graph/node/{account_id}/ppr      -> PPR score dari node ini
    GET /graph/node/{account_id}/flags    -> transaction flags (structuring dll)

Reuse fungsi graph/analytics.py (run_ppr, find_clusters, get_transaction_flags)
dan graph/builder.py (get_driver) yang SUDAH ada & teruji — tidak menulis
ulang logic Neo4j dari nol.
"""

import os
import sys

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from graph.analytics import run_ppr, find_clusters, get_transaction_flags
from graph.builder import get_driver

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/graph", tags=["graph"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = get_driver()
    return _driver


@router.get("/overview")
def graph_overview():
    """Statistik graph secara umum — total node, edge, cluster terdeteksi."""
    try:
        with _get_driver().session() as session:
            n_nodes = session.run("MATCH (n:Account) RETURN count(n) AS c").single()["c"]
            n_edges = session.run("MATCH ()-[r:TRANSFER]->() RETURN count(r) AS c").single()["c"]
    except Exception:
        raise HTTPException(503, "Graph service (Neo4j) tidak tersedia saat ini")

    clusters = find_clusters(_get_driver(), min_size=2)
    return {
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "total_clusters": len(clusters),
        "clusters_high_risk": sum(1 for c in clusters if c["risk_level"] == "HIGH"),
        "density": (n_edges / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0,
    }


@router.get("/clusters")
def list_clusters(min_size: int = Query(2, ge=2)):
    """Daftar cluster/jaringan mencurigakan terdeteksi (Union-Find di illicit subgraph)."""
    clusters = find_clusters(_get_driver(), min_size=min_size)
    # jangan kirim seluruh nodes-list di listing (bisa besar) — cukup ringkasan
    return {
        "items": [
            {k: v for k, v in c.items() if k != "nodes"} | {"sample_nodes": c["nodes"][:5]}
            for c in clusters
        ]
    }


@router.get("/cluster/{cluster_id}")
def cluster_detail(cluster_id: str, min_size: int = Query(2, ge=2)):
    """Detail satu cluster: semua node + edge di dalamnya (utk render Graph Explorer)."""
    clusters = find_clusters(_get_driver(), min_size=min_size)
    match = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
    if not match:
        raise HTTPException(404, f"Cluster {cluster_id} tidak ditemukan")

    nodes = match["nodes"]
    try:
        with _get_driver().session() as session:
            edges = session.run(
                """
                MATCH (a:Account)-[r:TRANSFER]->(b:Account)
                WHERE a.account_id IN $nodes AND b.account_id IN $nodes
                RETURN a.account_id AS src, b.account_id AS dst, r.amount AS amount,
                       r.is_laundering AS is_laundering
                LIMIT 2000
                """,
                nodes=nodes,
            ).data()
    except Exception:
        raise HTTPException(503, "Graph service (Neo4j) tidak tersedia saat ini")

    return {
        "cluster_id": cluster_id,
        "risk_level": match["risk_level"],
        "size": match["size"],
        "nodes": nodes,
        "edges": edges,
    }


@router.get("/node/{account_id}/neighbors")
def node_neighbors(account_id: str, hops: int = Query(1, ge=1, le=2), limit: int = Query(50, le=500)):
    """
    Tetangga 1-2 hop dari satu akun — utk tombol 'Expand N Hop' di Graph Explorer.

    Fix (QC 15-Jul): versi awal edge-query cuma ambil edge LANGSUNG dari
    seed, jadi utk hops=2 node 2-hop-nya "mengambang" tanpa garis penghubung
    (edge antar 1-hop<->2-hop tidak ikut kebawa). Fix: setelah dapat semua
    account_id tetangga (seed+1hop+2hop), query ulang SEMUA edge yg
    menghubungkan node2 itu satu sama lain — pola sama persis dgn
    cluster_detail() di bawah, yg sudah benar dari awal.
    """
    try:
        with _get_driver().session() as session:
            exists = session.run(
                "MATCH (n:Account {account_id: $id}) RETURN count(n) AS c", id=account_id
            ).single()["c"]
            if not exists:
                raise HTTPException(404, f"Akun {account_id} tidak ditemukan di graph")

            rows = session.run(
                f"""
                MATCH (seed:Account {{account_id: $id}})-[r:TRANSFER*1..{hops}]-(nbr:Account)
                WHERE nbr.account_id <> $id
                RETURN DISTINCT nbr.account_id AS account_id,
                       nbr.risk_score AS risk_score
                LIMIT $limit
                """,
                id=account_id,
                limit=limit,
            ).data()

            all_ids = [account_id] + [r["account_id"] for r in rows]
            # Directed match (bukan undirected) -> src/dst tetap benar arah
            # aliran dana, dan tiap edge cuma muncul sekali (sama pola dgn
            # cluster_detail() di bawah).
            edges = session.run(
                """
                MATCH (a:Account)-[r:TRANSFER]->(b:Account)
                WHERE a.account_id IN $ids AND b.account_id IN $ids
                RETURN a.account_id AS src, b.account_id AS dst, r.amount AS amount,
                       r.is_laundering AS is_laundering
                LIMIT $limit
                """,
                ids=all_ids,
                limit=limit,
            ).data()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "Graph service (Neo4j) tidak tersedia saat ini")

    return {"seed": account_id, "neighbors": rows, "edges": edges}


@router.get("/node/{account_id}/ppr")
def node_ppr(account_id: str, top_k: int = Query(20, le=100)):
    """Personalized PageRank dari satu akun — highlight node berisiko terdekat."""
    scores = run_ppr(_get_driver(), account_id, top_k=top_k)
    if not scores:
        return {"seed": account_id, "scores": {}, "note": "Akun tidak ditemukan atau tak punya tetangga"}
    return {"seed": account_id, "scores": scores}


@router.get("/node/{account_id}/flags")
def node_flags(account_id: str, limit: int = Query(500, le=2000)):
    """Transaction flags (structuring, rapid cash-out, dll) utk panel detail node."""
    with _engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT from_account, to_account, amount, tx_timestamp, channel
                FROM transactions
                WHERE from_account = :id OR to_account = :id
                ORDER BY tx_timestamp DESC
                LIMIT :limit
            """),
            {"id": account_id, "limit": limit},
        ).mappings().all()

    if not rows:
        return {"account_id": account_id, "flags": [], "note": "Tak ada transaksi ditemukan"}

    df = pd.DataFrame([dict(r) for r in rows])
    flags = get_transaction_flags(account_id, df)
    return {"account_id": account_id, "flags": flags}
