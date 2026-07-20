"""
Analytics report untuk screenshot PDF — PPR-style ranking + fraud clusters.
Pakai Cypher bounded (TANPA GDS projection) supaya aman di graph 176M edge.

Jalankan: python analytics_report.py
"""

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
from builder import get_driver


def top_risky_from_seed(driver, max_seeds: int = 1):
    """
    PPR-style: pilih seed = node illicit dengan illicit out-degree tertinggi,
    lalu ranking tetangga 1-2 hop berdasarkan jumlah koneksi illicit.
    Bounded via Cypher — tanpa GDS.
    """
    with driver.session() as s:
        # Cari seed: collector dengan illicit in-degree tinggi
        seed = s.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->(b:Account)
            WITH b.account_id AS acc, count(r) AS illicit_in
            RETURN acc, illicit_in
            ORDER BY illicit_in DESC
            LIMIT 1
        """).single()

        if not seed:
            print("  (tidak ada node illicit)")
            return

        seed_id = seed["acc"]
        print(f"\n  SEED (collector berisiko tertinggi): {seed_id} "
              f"({seed['illicit_in']} illicit masuk)")

        # Top pengirim ke seed (PPR-style: kontributor langsung)
        rows = s.run("""
            MATCH (a:Account)-[r:TRANSFER]->(seed:Account {account_id: $sid})
            WITH a.account_id AS sender,
                 count(r) AS tx_count,
                 sum(CASE WHEN r.is_laundering = 1 THEN 1 ELSE 0 END) AS illicit_count,
                 sum(r.amount) AS total_amount
            RETURN sender, tx_count, illicit_count, total_amount
            ORDER BY illicit_count DESC, total_amount DESC
            LIMIT 10
        """, sid=seed_id).data()

        print("\n  TOP 10 REKENING BERISIKO (kontributor ke seed):")
        print(f"  {'#':>2} {'account_id':<14} {'tx':>4} {'illicit':>8} {'total_amount':>16}")
        for i, r in enumerate(rows, 1):
            amt = float(r["total_amount"]) if r["total_amount"] else 0
            print(f"  {i:>2} {r['sender']:<14} {r['tx_count']:>4} "
                  f"{r['illicit_count']:>8} Rp{amt:>14,.0f}")


def fraud_clusters(driver, sample_edges: int = 2000, top_n: int = 10):
    """
    Cluster detection via connected components di Python (bounded).
    Ambil sampel illicit edge → bangun adjacency → temukan komponen terhubung.
    Tanpa GDS — aman.
    """
    with driver.session() as s:
        edges = s.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->(b:Account)
            RETURN a.account_id AS src, b.account_id AS dst
            LIMIT $lim
        """, lim=sample_edges).data()

    # Union-Find untuk connected components
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    edge_count = defaultdict(int)
    for e in edges:
        union(e["src"], e["dst"])

    # Hitung ukuran komponen
    comp_nodes = defaultdict(set)
    for e in edges:
        root = find(e["src"])
        comp_nodes[root].add(e["src"])
        comp_nodes[root].add(e["dst"])
        edge_count[root] += 1

    clusters = sorted(
        [(root, len(nodes), edge_count[root]) for root, nodes in comp_nodes.items()],
        key=lambda x: x[1], reverse=True
    )[:top_n]

    print(f"\n  FRAUD CLUSTERS (dari {len(edges):,} illicit edge sampel):")
    print(f"  {'cluster':<10} {'nodes':>7} {'illicit_edges':>14} {'risk':>8}")
    for i, (root, nsize, ecount) in enumerate(clusters):
        risk = "HIGH" if nsize >= 20 else ("MEDIUM" if nsize >= 5 else "LOW")
        print(f"  C{i:04d}     {nsize:>7,} {ecount:>14,} {risk:>8}")


def main():
    print("=" * 64)
    print("MuleRadar — Analytics Report (PPR-style + Fraud Clusters)")
    print("=" * 64)

    driver = get_driver()
    try:
        top_risky_from_seed(driver)
        fraud_clusters(driver)
    finally:
        driver.close()

    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
