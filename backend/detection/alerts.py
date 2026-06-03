"""
MuleRadar Phase 3 — Alert Generation & Persistence.
Combine model predictions + rule hits -> INSERT into PostgreSQL alerts table.
"""

import os
import uuid
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

RISK_THRESHOLD = 0.5   # minimum risk_score to generate an alert
SEVERITY_HIGH = 0.8    # >= ini → HIGH
SEVERITY_MEDIUM = 0.5  # >= ini → MEDIUM


def _severity_from_score(risk_score: float) -> str:
    """Single source of truth: severity SELALU diturunkan dari risk_score."""
    if risk_score >= SEVERITY_HIGH:
        return "HIGH"
    if risk_score >= SEVERITY_MEDIUM:
        return "MEDIUM"
    return "LOW"


def generate_alerts(
    engine,
    predictions_df: pd.DataFrame,
    rules_list: list[dict],
) -> int:
    """
    Combine model predictions + rule hits and INSERT into the alerts table.

    Parameters
    ----------
    engine : sqlalchemy.Engine
        Database engine for INSERT.
    predictions_df : pd.DataFrame
        Columns: account_id, risk_score, risk_level.
    rules_list : list[dict]
        Output from check_typology_rules().

    Returns
    -------
    int
        Number of alerts inserted.
    """
    # Index rules by account_id for fast lookup
    rules_by_account: dict[str, list] = {}
    for rule in rules_list:
        acc = rule["account_id"]
        if acc not in rules_by_account:
            rules_by_account[acc] = []
        rules_by_account[acc].append(rule)

    # Filter predictions above threshold
    high_risk = predictions_df[
        predictions_df["risk_score"] >= RISK_THRESHOLD
    ].copy()

    # Add rule-only accounts that may not appear in high_risk
    rule_only_accounts = (
        set(rules_by_account.keys())
        - set(high_risk["account_id"].tolist())
    )
    if rule_only_accounts:
        extra_rows = []
        for acc in rule_only_accounts:
            rules = rules_by_account[acc]
            max_conf = max(r["confidence"] for r in rules)
            if max_conf >= 0.5:
                extra_rows.append({
                    "account_id": acc,
                    "risk_score": max_conf * 0.8,  # rule-only, slight discount
                    "risk_level": "HIGH" if max_conf >= 0.8 else "MEDIUM",
                })
        if extra_rows:
            high_risk = pd.concat(
                [high_risk, pd.DataFrame(extra_rows)], ignore_index=True
            )

    if high_risk.empty:
        print("No alerts to generate")
        return 0

    # Build alert records
    alerts = []
    for _, row in high_risk.iterrows():
        account_id = row["account_id"]
        risk_score = float(row["risk_score"])

        matched_rules = rules_by_account.get(account_id, [])
        typology = (
            matched_rules[0]["typology"] if matched_rules else "UNKNOWN"
        )
        rule_triggered = (
            "; ".join(r["rule_triggered"] for r in matched_rules)
            if matched_rules
            else ""
        )

        # Severity SELALU dari risk_score (konsisten, tidak ada lagi risk 1.0→MEDIUM)
        severity = _severity_from_score(risk_score)

        alerts.append({
            "alert_id": f"ALT-{uuid.uuid4().hex[:12].upper()}",
            "account_id": account_id,
            "tx_id": None,
            "cluster_id": None,
            "typology": typology,
            "risk_score": round(risk_score, 4),
            "rule_triggered": rule_triggered[:2000],
            "severity": severity,
            "node_count": None,
            "status": "NEW",
            "created_at": datetime.utcnow(),
        })

    # Batch insert with ON CONFLICT DO NOTHING (idempotent)
    inserted = 0
    batch_size = 500
    with engine.begin() as conn:
        for i in range(0, len(alerts), batch_size):
            batch = alerts[i: i + batch_size]
            conn.execute(
                text("""
                    INSERT INTO alerts
                        (alert_id, account_id, tx_id, cluster_id, typology,
                         risk_score, rule_triggered, severity, node_count,
                         status, created_at)
                    VALUES
                        (:alert_id, :account_id, :tx_id, :cluster_id,
                         :typology, :risk_score, :rule_triggered, :severity,
                         :node_count, :status, :created_at)
                    ON CONFLICT (alert_id) DO NOTHING
                """),
                batch,
            )
            inserted += len(batch)

    print(f"Inserted {inserted} alerts (threshold={RISK_THRESHOLD})")
    return inserted
