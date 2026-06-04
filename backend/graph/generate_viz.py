"""
Generate vis.js HTML visualization dari Neo4j graph.
Output: outputs/graph_viz.html
Jalankan SETELAH docker-compose up -d: python generate_viz.py
"""

import os
import sys
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

from builder import get_driver

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "graph_viz.html")


def load_viz_data(driver, limit_illicit: int = 350,
                  limit_clean: int = 150) -> tuple[list, list, list, list]:
    """
    Query Neo4j untuk visualisasi subgraph FOKUS (bukan seluruh graph).
    Ambil sampel illicit edge + sedikit clean edge sebagai konteks.
    Degree dihitung HANYA untuk node yang terlibat (cepat, tidak scan 176M edge).
    Returns: (illicit_edges, clean_edges, degree_data, illicit_deg_data)
    """
    with driver.session() as session:
        # Sampel illicit edges (bounded — fokus untuk screenshot)
        illicit_result = session.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->(b:Account)
            RETURN a.account_id AS from_acc, b.account_id AS to_acc,
                   r.amount AS amount, r.channel AS channel,
                   r.is_laundering AS is_laundering
            LIMIT $limit
        """, limit=limit_illicit).data()

        # Sample clean edges untuk konteks
        clean_result = session.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 0}]->(b:Account)
            RETURN a.account_id AS from_acc, b.account_id AS to_acc,
                   r.amount AS amount, r.channel AS channel,
                   r.is_laundering AS is_laundering
            LIMIT $limit
        """, limit=limit_clean).data()

        # Kumpulkan node yang terlibat saja
        involved = set()
        for e in illicit_result + clean_result:
            involved.add(e["from_acc"])
            involved.add(e["to_acc"])
        involved = list(involved)

        # Degree per node — HANYA untuk node terlibat (cepat)
        degree_result = session.run("""
            MATCH (a:Account)-[r:TRANSFER]->()
            WHERE a.account_id IN $nodes
            WITH a.account_id AS acc, count(r) AS out_deg
            RETURN acc, out_deg
        """, nodes=involved).data()

        illicit_deg = session.run("""
            MATCH (a:Account)-[r:TRANSFER {is_laundering: 1}]->()
            WHERE a.account_id IN $nodes
            WITH a.account_id AS acc, count(r) AS illicit_out
            RETURN acc, illicit_out
        """, nodes=involved).data()

    return illicit_result, clean_result, degree_result, illicit_deg


def _node_color(illicit_out: int, total_out: int) -> str:
    if total_out == 0:
        return "#4CAF50"
    ratio = illicit_out / total_out
    if ratio >= 0.6:
        return "#E53935"
    elif ratio >= 0.2:
        return "#FFB300"
    return "#4CAF50"


def _node_size(degree: int) -> int:
    return max(15, min(45, 10 + degree * 3))


def build_html(illicit_edges: list, clean_edges: list,
               degree_data: list, illicit_deg_data: list) -> str:

    # Build lookup dicts
    degree_map   = {r["acc"]: r["out_deg"]     for r in degree_data}
    ill_deg_map  = {r["acc"]: r["illicit_out"] for r in illicit_deg_data}

    # Kumpulkan semua node yang terlibat
    all_nodes = set()
    for e in illicit_edges + clean_edges:
        all_nodes.add(e["from_acc"])
        all_nodes.add(e["to_acc"])

    # Build node JS
    nodes_js = []
    for node in all_nodes:
        total_out  = degree_map.get(node, 0)
        illicit_out= ill_deg_map.get(node, 0)
        color      = _node_color(illicit_out, total_out)
        size       = _node_size(total_out)
        nodes_js.append(
            f'{{id: "{node}", label: "{node}", '
            f'color: "{color}", size: {size}, '
            f'title: "Out-degree: {total_out} | Illicit out: {illicit_out}"}}'
        )

    # Build edge JS
    edges_js = []
    for e in illicit_edges:
        amount  = e.get("amount", 0) or 0
        channel = e.get("channel", "?") or "?"
        edges_js.append(
            f'{{from: "{e["from_acc"]}", to: "{e["to_acc"]}", '
            f'color: {{color: "#E53935", opacity: 0.8}}, '
            f'width: 3, arrows: "to", '
            f'title: "Rp{amount:,.0f} | {channel}"}}'
        )
    for e in clean_edges:
        amount  = e.get("amount", 0) or 0
        channel = e.get("channel", "?") or "?"
        edges_js.append(
            f'{{from: "{e["from_acc"]}", to: "{e["to_acc"]}", '
            f'color: {{color: "#90A4AE", opacity: 0.5}}, '
            f'width: 1, arrows: "to", '
            f'title: "Rp{amount:,.0f} | {channel}"}}'
        )

    nodes_str      = ",\n    ".join(nodes_js)
    edges_str      = ",\n    ".join(edges_js)
    total_edges    = len(illicit_edges) + len(clean_edges)
    illicit_pct    = len(illicit_edges) / total_edges * 100 if total_edges > 0 else 0
    gen_ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MuleRadar — Graph Investigation Workbench</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0d1117; font-family: 'Segoe UI', sans-serif; color: #e6edf3; }}
    #header {{ padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d;
               display: flex; align-items: center; justify-content: space-between; }}
    #header h1 {{ font-size: 18px; font-weight: 600; color: #58a6ff; }}
    #header h1 span {{ color: #f0883e; }}
    #stats {{ display: flex; gap: 16px; font-size: 13px; color: #8b949e; }}
    #stats b {{ color: #e6edf3; }}
    #mynetwork {{ width: 100%; height: calc(100vh - 60px); background: #0d1117; }}
    .legend {{ position: absolute; top: 80px; right: 20px; background: #161b22;
               border: 1px solid #30363d; border-radius: 8px; padding: 12px 16px; font-size: 12px; }}
    .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
  </style>
</head>
<body>
  <div id="header">
    <h1>Mule<span>Radar</span> — Graph Investigation Workbench</h1>
    <div id="stats">
      <span>Tampil: <b>{len(all_nodes)}</b> node / <b>{total_edges}</b> edge (sampel)</span>
      <span>Illicit: <b style="color:#E53935">{illicit_pct:.1f}%</b></span>
      <span>Graph penuh: <b>~176 jt edge</b></span>
      <span>Source: <b>Neo4j (live query)</b></span>
      <span>Generated: <b>{gen_ts}</b></span>
    </div>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="dot" style="background:#E53935"></div> High Risk (&ge;60% illicit)</div>
    <div class="legend-item"><div class="dot" style="background:#FFB300"></div> Medium Risk (20-60%)</div>
    <div class="legend-item"><div class="dot" style="background:#4CAF50"></div> Low Risk (&lt;20%)</div>
    <div class="legend-item" style="margin-top:8px">
      <div style="width:20px;height:2px;background:#E53935;border-radius:2px"></div> Illicit Edge
    </div>
    <div class="legend-item">
      <div style="width:20px;height:2px;background:#90A4AE;border-radius:2px"></div> Clean Edge
    </div>
  </div>
  <div style="position:absolute; bottom:16px; left:20px; max-width:520px;
              background:#161b22; border:1px solid #30363d; border-radius:8px;
              padding:10px 14px; font-size:11px; color:#8b949e; line-height:1.5;">
    <b style="color:#58a6ff">Asal data:</b> subgraph sampel ini di-<b>query langsung dari Neo4j</b>
    (graph ~176 juta edge, dibangun dari dataset <b>AMLWorld</b> + injeksi typologi Indonesia).
    Bukan data statis buatan tangan. Query Cypher:<br>
    <code style="color:#7ee787">MATCH (a:Account)-[r:TRANSFER {{is_laundering:1}}]-&gt;(b:Account) RETURN ... LIMIT N</code>
  </div>
  <div id="mynetwork"></div>
  <script>
    var nodes = new vis.DataSet([{nodes_str}]);
    var edges = new vis.DataSet([{edges_str}]);
    var container = document.getElementById('mynetwork');
    var options = {{
      layout: {{ improvedLayout: true }},
      physics: {{
        enabled: true,
        solver: "forceAtlas2Based",
        forceAtlas2Based: {{
          gravitationalConstant: -80, centralGravity: 0.005,
          springLength: 120, springConstant: 0.06,
          damping: 0.3, avoidOverlap: 0.9
        }},
        stabilization: {{ iterations: 250, updateInterval: 10 }},
        maxVelocity: 80, minVelocity: 0.4
      }},
      interaction: {{ hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true }},
      nodes: {{ shape: "dot", font: {{ color: "#e6edf3", size: 11 }}, borderWidth: 1, borderWidthSelected: 3 }},
      edges: {{ smooth: {{ type: "continuous" }} }}
    }};
    new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
  </script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("MuleRadar — Graph Visualization (Neo4j)")
    print("=" * 60)

    driver = get_driver()
    print("  Querying Neo4j...")

    illicit_edges, clean_edges, degree_data, illicit_deg_data = load_viz_data(driver)
    driver.close()

    print(f"  Illicit edges: {len(illicit_edges):,}")
    print(f"  Clean edges  : {len(clean_edges):,}")

    html = build_html(illicit_edges, clean_edges, degree_data, illicit_deg_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nOutput: {OUTPUT_FILE}")
    print("Buka file HTML di browser untuk melihat graph.")
    print("=" * 60)
