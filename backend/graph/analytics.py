"""
Graph analytics: PPR, cluster detection, transaction flags.
PPR via Cypher ego-network (GDS-free); WCC via Union-Find pada illicit subgraph;
transaction flags via pandas DataFrame.
"""

from __future__ import annotations

import pandas as pd
from datetime import timedelta
from neo4j import Driver

# PERINGATAN: ensure_gds_projection memproyeksi full graph ke GDS in-memory
# — JANGAN dipakai pada graph 176M edge (OOM, Neo4j heap 4G).
# Hanya untuk subgraph kecil. Dipertahankan karena mungkin dipakai modul lain.
try:
    from .builder import GDS_GRAPH_NAME, ensure_gds_projection
except ImportError:
    from builder import GDS_GRAPH_NAME, ensure_gds_projection

# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

STRUCTURING_THRESHOLDS = [500_000, 1_000_000, 5_000_000, 25_000_000, 100_000_000]
STRUCTURING_RATIO_LOW = 0.85
STRUCTURING_RATIO_HIGH = 0.999

ANOMALY_HOUR_START = 0
ANOMALY_HOUR_END = 4

RAPID_CASHOUT_MINUTES = 10

FREQUENCY_ANOMALY_THRESHOLD = 8

RISK_HIGH = 0.15
RISK_MEDIUM = 0.03


# -----------------------------------------------------------------
# 1. Personalized PageRank — Scoped ego-network via Cypher (GDS-free)
# -----------------------------------------------------------------

def run_ppr(
    driver: Driver,
    seed_node: str,
    alpha: float = 0.85,
    top_k: int = 20,
    max_hops: int = 2,
) -> dict[str, float]:
    """
    Scoped ego-network PPR approximation via Cypher (GDS-free, no OOM).

    Tidak memproyeksi full graph ke GDS. Sebagai gantinya, traversal
    bounded dari seed_node, meranking tetangga berdasarkan frekuensi +
    bobot koneksi (proxy PPR). Aman untuk graph 176M edge.

    Returns {account_id: score} dinormalisasi 0-1, top_k entries, seed excluded.
    """
    try:
        with driver.session() as session:
            seed_exists = session.run(
                "MATCH (n:Account {account_id: $id}) RETURN count(n) AS c",
                id=seed_node,
            ).single()["c"]

            if not seed_exists:
                return {}

            rows = session.run(
                """
                MATCH (seed:Account {account_id: $seed_id})-[r:TRANSFER]-(neighbor:Account)
                WITH neighbor,
                     count(r) AS direct_links,
                     sum(r.amount) AS total_amount,
                     sum(CASE WHEN r.is_laundering = 1 THEN 1 ELSE 0 END) AS illicit_links
                WHERE neighbor.account_id <> $seed_id
                RETURN neighbor.account_id AS account_id,
                       direct_links, total_amount, illicit_links,
                       (direct_links * 1.0 + illicit_links * 5.0) AS score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                seed_id=seed_node,
                top_k=top_k,
            ).data()

        if not rows:
            return {}

        max_score = max(row["score"] for row in rows)
        if max_score == 0:
            return {}

        return {
            row["account_id"]: round(row["score"] / max_score, 6)
            for row in rows
        }

    except Exception:
        return {}


# -----------------------------------------------------------------
# 2. Cluster / Component Analysis — Union-Find pada illicit subgraph
# -----------------------------------------------------------------

def find_clusters(
    driver: Driver,
    min_size: int = 2,
    max_illicit_edges: int = 5000,
) -> list[dict]:
    """
    Connected components pada illicit subgraph (Union-Find, GDS-free, bounded, no OOM).

    Tidak memproyeksi full graph ke GDS. Sebagai gantinya, ambil sampel
    illicit edges (LIMIT max_illicit_edges) via Cypher, lalu bangun
    connected components di Python menggunakan Union-Find. Aman untuk
    graph 176M edge.

    Returns list of cluster dicts sorted by size descending, filter size>=min_size.
    risk_level: HIGH jika size>=20, MEDIUM jika >=5, LOW otherwise.
    """
    try:
        with driver.session() as session:
            rows = session.run(
                """
                MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->(b:Account)
                RETURN a.account_id AS src, b.account_id AS dst
                LIMIT $lim
                """,
                lim=max_illicit_edges,
            ).data()

        if not rows:
            return []

        # Union-Find (path compression + union by rank)
        parent: dict[str, str] = {}
        rank: dict[str, int] = {}

        def find(x: str) -> str:
            if parent.setdefault(x, x) != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            rank.setdefault(rx, 0)
            rank.setdefault(ry, 0)
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        illicit_edge_counts: dict[str, int] = {}

        for row in rows:
            src, dst = row["src"], row["dst"]
            union(src, dst)
            root = find(src)
            illicit_edge_counts[root] = illicit_edge_counts.get(root, 0) + 1

        # Kelompokkan node per root
        components: dict[str, list[str]] = {}
        for node in parent:
            root = find(node)
            components.setdefault(root, []).append(node)

        # Re-map illicit_edge_counts ke root setelah path compression penuh
        final_illicit: dict[str, int] = {}
        for old_root, count in illicit_edge_counts.items():
            new_root = find(old_root)
            final_illicit[new_root] = final_illicit.get(new_root, 0) + count

        result = []
        for idx, (root, nodes) in enumerate(
            sorted(components.items(), key=lambda kv: len(kv[1]), reverse=True)
        ):
            size = len(nodes)
            if size < min_size:
                continue

            illicit_count = final_illicit.get(root, 0)

            if size >= 20:
                risk_level = "HIGH"
            elif size >= 5:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            result.append({
                "cluster_id": f"C{idx:04d}",
                "nodes": sorted(nodes),
                "size": size,
                "illicit_edge_count": illicit_count,
                "risk_level": risk_level,
            })

        return result

    except Exception:
        return []


# -----------------------------------------------------------------
# 3. Transaction Flags — pandas DataFrame
# -----------------------------------------------------------------

def get_transaction_flags(
    account_id: str,
    df: pd.DataFrame,
) -> list[dict]:
    """
    Detect behavioral anomalies untuk account_id dari DataFrame transaksi.
    df kolom: from_account, to_account, amount, tx_timestamp, channel.
    Returns list of {flag_type, severity, detail, evidence_rows}.
    """
    flags: list[dict] = []

    if not pd.api.types.is_datetime64_any_dtype(df["tx_timestamp"]):
        df = df.copy()
        df["tx_timestamp"] = pd.to_datetime(df["tx_timestamp"], utc=True)

    sent = df[df["from_account"] == account_id].copy()
    received = df[df["to_account"] == account_id].copy()
    all_tx = pd.concat([sent, received]).sort_values("tx_timestamp")

    if all_tx.empty:
        return flags

    # --- Flag 1: Structuring ---
    for threshold in STRUCTURING_THRESHOLDS:
        band_low = threshold * STRUCTURING_RATIO_LOW
        band_high = threshold * STRUCTURING_RATIO_HIGH
        suspicious = sent[sent["amount"].between(band_low, band_high)]
        if len(suspicious) >= 3:
            flags.append({
                "flag_type": "STRUCTURING",
                "severity": "HIGH",
                "detail": (
                    f"{len(suspicious)} transaksi antara "
                    f"Rp{band_low:,.0f}–Rp{threshold:,.0f}"
                ),
                "evidence_rows": suspicious["tx_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()[:5],
            })

    # --- Flag 2: Timing Anomaly ---
    late_night = all_tx[
        all_tx["tx_timestamp"].dt.hour.between(ANOMALY_HOUR_START, ANOMALY_HOUR_END)
    ]
    if len(late_night) >= 2:
        flags.append({
            "flag_type": "TIMING_ANOMALY",
            "severity": "MEDIUM",
            "detail": (
                f"{len(late_night)} transaksi antara "
                f"{ANOMALY_HOUR_START:02d}:00–{ANOMALY_HOUR_END:02d}:59"
            ),
            "evidence_rows": late_night["tx_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()[:5],
        })

    # --- Flag 3: Rapid Cash-Out ---
    rapid_events = []
    for _, inbound in received.iterrows():
        window_end = inbound["tx_timestamp"] + timedelta(minutes=RAPID_CASHOUT_MINUTES)
        outbound_in_window = sent[
            (sent["tx_timestamp"] >= inbound["tx_timestamp"]) &
            (sent["tx_timestamp"] <= window_end)
        ]
        for _, outbound in outbound_in_window.iterrows():
            gap_seconds = (outbound["tx_timestamp"] - inbound["tx_timestamp"]).total_seconds()
            rapid_events.append({
                "in_time": inbound["tx_timestamp"],
                "out_time": outbound["tx_timestamp"],
                "gap_seconds": int(gap_seconds),
                "in_amount": inbound["amount"],
                "out_amount": outbound["amount"],
            })

    if rapid_events:
        fastest = min(rapid_events, key=lambda x: x["gap_seconds"])
        flags.append({
            "flag_type": "RAPID_CASHOUT",
            "severity": "HIGH",
            "detail": (
                f"{len(rapid_events)} kejadian cash-out cepat; "
                f"tercepat {fastest['gap_seconds']}s "
                f"(masuk Rp{fastest['in_amount']:,.0f} → keluar Rp{fastest['out_amount']:,.0f})"
            ),
            "evidence_rows": [e["in_time"].strftime("%Y-%m-%d %H:%M:%S") for e in rapid_events[:5]],
        })

    # --- Flag 4: Channel Switch ---
    if len(all_tx) >= 3:
        channels = all_tx["channel"].dropna().tolist()
        unique_channels = set(channels)
        if len(unique_channels) >= 3:
            flags.append({
                "flag_type": "CHANNEL_SWITCH",
                "severity": "MEDIUM",
                "detail": f"Menggunakan {len(unique_channels)} channel berbeda: {', '.join(sorted(unique_channels))}",
                "evidence_rows": all_tx["tx_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()[:3],
            })
        elif len(unique_channels) == 2:
            all_tx_sorted = all_tx.sort_values("tx_timestamp")
            for i in range(len(all_tx_sorted) - 1):
                row_a = all_tx_sorted.iloc[i]
                row_b = all_tx_sorted.iloc[i + 1]
                if (
                    row_a["channel"] != row_b["channel"]
                    and (row_b["tx_timestamp"] - row_a["tx_timestamp"]).total_seconds() < 1800
                ):
                    flags.append({
                        "flag_type": "CHANNEL_SWITCH",
                        "severity": "LOW",
                        "detail": f"Pindah channel {row_a['channel']}→{row_b['channel']} dalam waktu singkat",
                        "evidence_rows": [
                            row_a["tx_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                            row_b["tx_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                        ],
                    })
                    break

    # --- Flag 5: Frequency Anomaly ---
    if len(all_tx) >= 2:
        timestamps = all_tx.sort_values("tx_timestamp")["tx_timestamp"].tolist()
        window = timedelta(hours=1)
        max_count = 0
        max_window_start = None

        for ts in timestamps:
            count = sum(1 for t in timestamps if ts <= t < ts + window)
            if count > max_count:
                max_count = count
                max_window_start = ts

        if max_count >= FREQUENCY_ANOMALY_THRESHOLD:
            flags.append({
                "flag_type": "FREQUENCY_ANOMALY",
                "severity": "HIGH",
                "detail": f"{max_count} transaksi dalam 1 jam (mulai {max_window_start.strftime('%Y-%m-%d %H:%M')})",
                "evidence_rows": [
                    t.strftime("%Y-%m-%d %H:%M:%S")
                    for t in timestamps
                    if max_window_start <= t < max_window_start + window
                ][:5],
            })

    return flags
