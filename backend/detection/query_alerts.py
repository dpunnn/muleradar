"""
Query ringkasan alerts dari PostgreSQL untuk screenshot PDF.
Cepat — pakai index, tidak scan graph.

Jalankan: python query_alerts.py
"""

import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


def main():
    engine = create_engine(DATABASE_URL)
    print("=" * 60)
    print("MuleRadar — Alert Summary (PostgreSQL)")
    print("=" * 60)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM alerts")).scalar()
        print(f"\n  TOTAL ALERTS: {total:,}")

        # Breakdown per typology
        print("\n  BREAKDOWN PER TYPOLOGY:")
        print(f"  {'typology':<22} {'count':>8} {'avg_risk':>10}")
        rows = conn.execute(text("""
            SELECT typology, COUNT(*) AS cnt, AVG(risk_score) AS avg_risk
            FROM alerts
            GROUP BY typology
            ORDER BY cnt DESC
        """)).fetchall()
        for r in rows:
            avg = float(r[2]) if r[2] is not None else 0.0
            print(f"  {str(r[0]):<22} {r[1]:>8,} {avg:>10.4f}")

        # Breakdown per severity
        print("\n  BREAKDOWN PER SEVERITY:")
        print(f"  {'severity':<12} {'count':>8}")
        rows = conn.execute(text("""
            SELECT severity, COUNT(*) AS cnt
            FROM alerts
            GROUP BY severity
            ORDER BY cnt DESC
        """)).fetchall()
        for r in rows:
            print(f"  {str(r[0]):<12} {r[1]:>8,}")

        # Breakdown per status
        print("\n  BREAKDOWN PER STATUS:")
        print(f"  {'status':<12} {'count':>8}")
        rows = conn.execute(text("""
            SELECT status, COUNT(*) AS cnt
            FROM alerts
            GROUP BY status
            ORDER BY cnt DESC
        """)).fetchall()
        for r in rows:
            print(f"  {str(r[0]):<12} {r[1]:>8,}")

        # Top 10 alert berisiko tertinggi
        print("\n  TOP 10 ALERT RISIKO TERTINGGI:")
        print(f"  {'account_id':<14} {'typology':<18} {'risk':>7} {'severity':>9}")
        rows = conn.execute(text("""
            SELECT account_id, typology, risk_score, severity
            FROM alerts
            ORDER BY risk_score DESC
            LIMIT 10
        """)).fetchall()
        for r in rows:
            risk = float(r[2]) if r[2] is not None else 0.0
            print(f"  {str(r[0]):<14} {str(r[1]):<18} {risk:>7.4f} {str(r[3]):>9}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
