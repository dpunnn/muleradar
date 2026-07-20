"""
Populate alert utk Layer 3 (Statistical Anomaly) & Layer 4 (Graph Motif)
— fix 20-Jul, scan produksi. Kedua layer SUDAH dibangun di rules.py &
di-wire ke run_detection.py, tapi run_detection penuh berat (feature
extraction + train XGBoost di 144M baris). Script ini menjalankan HANYA
2 layer itu lalu insert via generate_alerts (predictions kosong -> hanya
alert rule-only yg dibuat), supaya filter "Sumber Deteksi: Statistik /
Graph Motif" di UI genuinely punya data.

Cara pakai:
  cd backend && python populate_layer34.py
"""

import os
import sys
import time

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detection.rules import check_statistical_anomaly, check_graph_motifs
from detection.alerts import generate_alerts

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


def main():
    engine = create_engine(DATABASE_URL)

    print("[1/3] Layer 3 — Statistical Anomaly (FANOUT/FANIN, z-score)...")
    t0 = time.time()
    stat_rules = check_statistical_anomaly(engine)
    print(f"      {len(stat_rules):,} hits ({time.time()-t0:.1f}s)")

    # Suspect utk Layer 4: hits statistik + account_id yg SUDAH punya alert
    # (mereka memang sudah dicurigai) — supaya join graph-motif scoped & cepat.
    with engine.connect() as conn:
        existing = [r[0] for r in conn.execute(text(
            "SELECT DISTINCT account_id FROM alerts WHERE account_id IS NOT NULL LIMIT 5000"
        ))]
    suspects = list({r["account_id"] for r in stat_rules} | set(existing))
    print(f"[2/3] Layer 4 — Graph Motif (3-hop cycle) atas {len(suspects):,} suspect...")
    t0 = time.time()
    motif_rules = check_graph_motifs(engine, suspect_accounts=suspects)
    print(f"      {len(motif_rules):,} hits ({time.time()-t0:.1f}s)")

    rules = stat_rules + motif_rules
    if not rules:
        print("Tidak ada hit dari Layer 3/4 — tidak ada yg di-insert.")
        return

    # predictions kosong -> generate_alerts hanya proses rule-only accounts.
    empty_preds = pd.DataFrame(columns=["account_id", "risk_score", "risk_level"])
    print(f"[3/3] Insert alert dari {len(rules):,} rule hits...")
    inserted = generate_alerts(engine, empty_preds, rules)
    print(f"      Selesai. {inserted:,} alert diproses (ON CONFLICT dedup).")

    # Ringkas distribusi detection_layer sesudahnya
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT detection_layer, COUNT(*) FROM alerts GROUP BY detection_layer ORDER BY 2 DESC"
        )).all()
    print("\n=== detection_layer sesudah populate ===")
    for layer, cnt in rows:
        print(f"  {layer or '(null)':14s}: {cnt:,}")


if __name__ == "__main__":
    main()
