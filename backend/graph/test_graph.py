"""
Test graph engine: PostgreSQL → Neo4j → GDS analytics.
Jalankan setelah docker-compose up -d db neo4j
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from builder import (
    get_driver,
    load_transactions,
    load_graph_to_neo4j,
    graph_stats,
    ensure_gds_projection,
)
from analytics import run_ppr, find_clusters, get_transaction_flags

SEP = "=" * 60

print(SEP)
print("MuleRadar — Graph Engine Test (Neo4j)")
print(SEP)

# 1. Load transaksi dari PostgreSQL
print("\n[1/5] Load transaksi dari PostgreSQL (semua data)...")
df = load_transactions()  # days=None = full load, tidak ada RAM limit di Neo4j
print(f"      {len(df):,} transaksi dimuat")
if df.empty:
    print("      ERROR: tidak ada data. Pastikan load_to_db.py sudah dijalankan.")
    sys.exit(1)

# 2. Load ke Neo4j
print("\n[2/5] Load ke Neo4j (full_reload=True, initial load)...")
driver = get_driver()
result = load_graph_to_neo4j(df, driver, full_reload=True)
print(f"      Nodes: {result['nodes']:,} | Edges: {result['edges']:,}")

# 3. Graph stats (termasuk GDS WCC)
print("\n[3/5] Graph stats dari Neo4j...")
ensure_gds_projection(driver)
stats = graph_stats(driver)
for k, v in stats.items():
    print(f"      {k}: {v}")

# 4. PPR dari seed node fraud
print("\n[4/5] Personalized PageRank dari fraud seed node...")
illicit_accounts = df[df["is_laundering"] == 1]["from_account"].unique()
if len(illicit_accounts) == 0:
    print("      Tidak ada node illicit di data, skip PPR.")
else:
    seed = illicit_accounts[0]
    print(f"      Seed node: {seed}")
    ppr_scores = run_ppr(driver, seed_node=seed, alpha=0.85, top_k=10)
    print(f"      Top {len(ppr_scores)} node berisiko:")
    for account, score in ppr_scores.items():
        print(f"        {account}: {score:.6f}")

# 5. Cluster detection
print("\n[5/5] Cluster detection (WCC)...")
clusters = find_clusters(driver, min_size=2)
print(f"      Total clusters: {len(clusters)}")
for c in clusters[:5]:
    print(
        f"      {c['cluster_id']} | size={c['size']} | "
        f"illicit_ratio={c['illicit_ratio']:.1%} | risk={c['risk_level']}"
    )

driver.close()
print(f"\n{SEP}")
print("Test selesai.")
print(SEP)
