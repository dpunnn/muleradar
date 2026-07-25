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

    node_ids = match["nodes"]
    full_size = match["size"]
    try:
        with _get_driver().session() as session:
            # Graf-A3: bawa properti MODEL per-node (risk_score = risk index model,
            # risk_level = severity, tipologi, degree) supaya panel detail tampil
            # skor model asli — bukan lagi "PPR proxy" & degree "—".
            node_rows = session.run(
                """
                MATCH (a:Account) WHERE a.account_id IN $nodes
                RETURN a.account_id AS account_id,
                       coalesce(a.risk_score, 0.0) AS risk_score,
                       coalesce(a.risk_level, 'LOW') AS risk_level,
                       coalesce(a.typology, 'UNKNOWN') AS typology,
                       coalesce(a.in_degree, 0) AS in_degree,
                       coalesce(a.out_degree, 0) AS out_degree
                """,
                nodes=node_ids,
            ).data()
            # ambil SEMUA edge internal cluster utk rekonstruksi konektivitas.
            # cap tinggi (bukan 20k) — kalau dipotong, edge seed bisa kebuang &
            # BFS mentok. Cluster terbesar ~40k edge, 300k aman.
            all_edges = session.run(
                """
                MATCH (a:Account)-[r:TRANSFER]->(b:Account)
                WHERE a.account_id IN $nodes AND b.account_id IN $nodes
                RETURN a.account_id AS src, b.account_id AS dst, r.amount AS amount,
                       r.is_laundering AS is_laundering
                LIMIT 300000
                """,
                nodes=node_ids,
            ).data()
    except Exception:
        raise HTTPException(503, "Graph service (Neo4j) tidak tersedia saat ini")

    # Graf-A3 FIX (25-Jul): cluster besar bisa ribuan node. JANGAN cherry-pick
    # top-N by risk_score — node paling berisiko sering TIDAK saling terhubung
    # langsung (terhubung lewat perantara di luar sample), jadi edge terinduksi
    # nyaris kosong -> node melayang tanpa garis. Solusi: BFS dari seed berisiko
    # tertinggi mengikuti edge nyata, tumbuh ke tetangga risk-tinggi dulu, sampai
    # RENDER_N node TERHUBUNG. Subgraph terinduksinya dijamin padat & tersambung.
    RENDER_N = 120
    prop = {n["account_id"]: n for n in node_rows}
    if full_size <= RENDER_N:
        render_nodes = sorted(node_rows, key=lambda n: n["risk_score"], reverse=True)
        render_set = set(prop)
    else:
        from collections import defaultdict, deque
        adj = defaultdict(set)
        for e in all_edges:
            adj[e["src"]].add(e["dst"])
            adj[e["dst"]].add(e["src"])
        # Seed = node degree tertinggi (hub sejati = "collector") — dijamin punya
        # tetangga, jadi BFS tumbuh padat. (risk tertinggi bisa degree kecil ->
        # BFS mentok di 1 node.) Fallback ke risk kalau adjacency kosong.
        seed = max(
            node_rows,
            key=lambda n: len(adj.get(n["account_id"], ())) * 1000 + n["risk_score"],
        )["account_id"]
        seen, order, dq = {seed}, [], deque([seed])
        while dq and len(order) < RENDER_N:
            x = dq.popleft()
            order.append(x)
            for nb in sorted(adj[x], key=lambda k: prop.get(k, {}).get("risk_score", 0), reverse=True):
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        render_set = set(order)
        render_nodes = [prop[a] for a in order]  # seed (paling berisiko) di depan

    render_edges = [e for e in all_edges if e["src"] in render_set and e["dst"] in render_set][:2000]

    return {
        "cluster_id": cluster_id,
        "risk_level": match["risk_level"],
        "size": full_size,
        "nodes": render_nodes,
        "edges": render_edges,
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
