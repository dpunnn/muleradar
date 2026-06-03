"""
Test analytics.py dengan data sintetik — tidak butuh PostgreSQL.
Jalankan: python test_analytics_offline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import networkx as nx
import pandas as pd
from datetime import datetime, timezone, timedelta
from analytics import run_ppr, find_clusters, get_transaction_flags

print("=" * 60)
print("TEST ANALYTICS.PY — offline mode (data sintetik)")
print("=" * 60)

# ---------------------------------------------------------------
# Buat graph sintetik: 2 cluster
# Cluster A: 5 node, beberapa edge is_laundering=True (mule chain)
# Cluster B: 3 node, bersih
# ---------------------------------------------------------------

G = nx.DiGraph()

# Cluster A — fraud ring
edges_A = [
    ("ACC_001", "ACC_002", {"amount": 4_900_000, "is_laundering": True,  "channel": "QRIS"}),
    ("ACC_002", "ACC_003", {"amount": 4_800_000, "is_laundering": True,  "channel": "TRANSFER"}),
    ("ACC_003", "ACC_004", {"amount": 4_700_000, "is_laundering": True,  "channel": "ATM"}),
    ("ACC_004", "ACC_005", {"amount": 4_600_000, "is_laundering": False, "channel": "MOBILE"}),
    ("ACC_001", "ACC_005", {"amount": 3_000_000, "is_laundering": True,  "channel": "QRIS"}),
]
# Cluster B — bersih
edges_B = [
    ("ACC_010", "ACC_011", {"amount": 200_000, "is_laundering": False, "channel": "TRANSFER"}),
    ("ACC_011", "ACC_012", {"amount": 150_000, "is_laundering": False, "channel": "TRANSFER"}),
]

for u, v, attr in edges_A + edges_B:
    G.add_edge(u, v, **attr)

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- [1] PPR ----
print("\n[1] PPR dari seed ACC_001 (fraud origin)")
ppr = run_ppr(G, seed_node="ACC_001", alpha=0.85, top_k=10)
for i, (node, score) in enumerate(ppr.items(), 1):
    print(f"  {i}. {node}  score={score:.6f}")
assert "ACC_002" in ppr, "ACC_002 harus muncul di PPR dari ACC_001"
print("  PASS: ACC_002 ada di top PPR")

# ---- [2] Clusters ----
print("\n[2] find_clusters")
clusters = find_clusters(G, min_size=2)
print(f"  {len(clusters)} clusters found")
for c in clusters:
    print(
        f"  {c['cluster_id']}  size={c['size']}  illicit_ratio={c['illicit_ratio']:.0%}  "
        f"risk={c['risk_level']}"
    )
assert any(c["risk_level"] == "HIGH" for c in clusters), "Harus ada 1 cluster HIGH"
print("  PASS: ada cluster HIGH (fraud ring)")

# ---- [3] Transaction Flags ----
print("\n[3] get_transaction_flags untuk ACC_002")

base_time = datetime(2025, 3, 15, 2, 30, tzinfo=timezone.utc)  # 02:30 = late night

rows = [
    # Structuring: 4 transaksi mendekati Rp 5 juta
    {"from_account": "ACC_002", "to_account": "X", "amount": 4_990_000, "tx_timestamp": base_time + timedelta(minutes=0),  "channel": "QRIS"},
    {"from_account": "ACC_002", "to_account": "X", "amount": 4_980_000, "tx_timestamp": base_time + timedelta(minutes=5),  "channel": "TRANSFER"},
    {"from_account": "ACC_002", "to_account": "X", "amount": 4_970_000, "tx_timestamp": base_time + timedelta(minutes=10), "channel": "ATM"},
    {"from_account": "ACC_002", "to_account": "X", "amount": 4_960_000, "tx_timestamp": base_time + timedelta(minutes=15), "channel": "MOBILE"},
    # Rapid cash-out: masuk Rp5M, keluar Rp4.9M dalam 3 menit
    {"from_account": "ACC_001", "to_account": "ACC_002", "amount": 5_000_000, "tx_timestamp": base_time + timedelta(minutes=20), "channel": "TRANSFER"},
    {"from_account": "ACC_002", "to_account": "ACC_003", "amount": 4_900_000, "tx_timestamp": base_time + timedelta(minutes=23), "channel": "QRIS"},
    # Extra untuk frequency anomaly
    {"from_account": "ACC_002", "to_account": "X2", "amount": 100_000, "tx_timestamp": base_time + timedelta(minutes=25), "channel": "QRIS"},
    {"from_account": "ACC_002", "to_account": "X2", "amount": 100_000, "tx_timestamp": base_time + timedelta(minutes=30), "channel": "QRIS"},
    {"from_account": "ACC_002", "to_account": "X2", "amount": 100_000, "tx_timestamp": base_time + timedelta(minutes=35), "channel": "TRANSFER"},
    {"from_account": "ACC_002", "to_account": "X2", "amount": 100_000, "tx_timestamp": base_time + timedelta(minutes=40), "channel": "ATM"},
]
df = pd.DataFrame(rows)

flags = get_transaction_flags(G, account_id="ACC_002", df=df)
print(f"  {len(flags)} flags terdeteksi:")
for f in flags:
    print(f"  [{f['severity']:6s}] {f['flag_type']}: {f['detail']}")

flag_types = {f["flag_type"] for f in flags}
assert "STRUCTURING" in flag_types,     "Harus detect STRUCTURING"
assert "TIMING_ANOMALY" in flag_types,  "Harus detect TIMING_ANOMALY"
assert "RAPID_CASHOUT" in flag_types,   "Harus detect RAPID_CASHOUT"
assert "CHANNEL_SWITCH" in flag_types,  "Harus detect CHANNEL_SWITCH"
print("  PASS: STRUCTURING + TIMING_ANOMALY + RAPID_CASHOUT + CHANNEL_SWITCH semua terdeteksi")

print("\n" + "=" * 60)
print("Semua test PASS. analytics.py siap.")
print("=" * 60)
