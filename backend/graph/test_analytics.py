"""
Test analytics.py: PPR, find_clusters, get_transaction_flags.
Jalankan dari folder backend/graph/:  python test_analytics.py
Butuh PostgreSQL running (docker-compose up -d).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from builder import load_graph, graph_stats
from analytics import run_ppr, find_clusters, get_transaction_flags
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")

print("=" * 60)
print("Memuat graph dari PostgreSQL...")
G = load_graph()
stats = graph_stats(G)
print(f"Graph loaded: {stats}")

# ---- PPR ----
print("\n[1] PPR — Personalized PageRank")
if G.number_of_nodes() > 0:
    # Ambil node pertama sebagai seed (idealnya node yang diketahui fraud)
    seed = list(G.nodes())[0]
    print(f"  Seed node: {seed}")
    ppr_scores = run_ppr(G, seed_node=seed, alpha=0.85, top_k=10)
    print(f"  Top {len(ppr_scores)} node terdekat ke seed:")
    for i, (node, score) in enumerate(ppr_scores.items(), 1):
        print(f"    {i:2d}. {node:<20s}  score={score:.6f}")
else:
    print("  Graph kosong — skip PPR")

# ---- Clusters ----
print("\n[2] find_clusters — Weakly Connected Components")
clusters = find_clusters(G, min_size=2)
print(f"  Total cluster (size >= 2): {len(clusters)}")
if clusters:
    print(f"  5 cluster terbesar:")
    for c in clusters[:5]:
        print(
            f"    {c['cluster_id']}  size={c['size']:4d}  "
            f"edges={c['edge_count']:5d}  illicit={c['illicit_edge_count']:4d}  "
            f"ratio={c['illicit_ratio']:.2%}  risk={c['risk_level']}"
        )

# ---- Transaction Flags ----
print("\n[3] get_transaction_flags — Behavioral Anomaly Detection")
engine = create_engine(DATABASE_URL)
with engine.connect() as conn:
    df = pd.read_sql(
        "SELECT from_account, to_account, amount, tx_timestamp, channel FROM transactions LIMIT 50000",
        conn,
    )
print(f"  DataFrame loaded: {len(df)} rows")

if not df.empty and G.number_of_nodes() > 0:
    # Pilih account dengan banyak transaksi sebagai contoh
    account_counts = pd.concat([
        df["from_account"].value_counts(),
        df["to_account"].value_counts()
    ]).groupby(level=0).sum()
    target_account = account_counts.idxmax()
    print(f"  Account with most transactions: {target_account} ({account_counts.max()} tx)")

    flags = get_transaction_flags(G, account_id=target_account, df=df)
    if flags:
        print(f"  {len(flags)} flags detected:")
        for f in flags:
            print(f"    [{f['severity']}] {f['flag_type']}: {f['detail']}")
    else:
        print("  Tidak ada flag terdeteksi untuk account ini")
else:
    print("  Skip — data kosong")

print("\n" + "=" * 60)
print("Test selesai.")
