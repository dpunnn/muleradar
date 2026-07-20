"""
MuleRadar Phase 3 — Alert Generation & Persistence.
Combine model predictions + rule hits -> INSERT into PostgreSQL alerts table.
"""

import hashlib
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

try:
    from detection.rules import detection_layer_for
except ImportError:
    from rules import detection_layer_for

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


def _deterministic_alert_id(account_id: str, typology: str, run_key: str) -> str:
    """Alert_id deterministik (fix 20-Jul, ganti uuid4 acak).

    SEBELUMNYA `ALT-{uuid4}` acak -> `ON CONFLICT (alert_id) DO NOTHING`
    MANDUL (UUID nyaris tak pernah collide), jadi run_detection.py berkala
    thd akun yg TETAP di atas threshold bikin alert DUPLIKAT menumpuk utk
    risiko yg SAMA. Sekarang sha1(account+typology+run_key): dalam satu
    run_key (default: tanggal run) akun+typology -> SATU alert_id (dedup
    efektif via ON CONFLICT); run_key hari berikutnya -> alert baru (wajar,
    periode deteksi baru). BUKAN dari risk_score (itu bisa berubah tipis
    antar-run utk akun sama -> malah gagal dedup).
    """
    raw = f"{account_id}|{(typology or '').upper()}|{run_key}"
    return "ALT-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12].upper()


def generate_alerts(
    engine,
    predictions_df: pd.DataFrame,
    rules_list: list[dict],
    run_key: str = None,
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
    # run_key: default tanggal hari ini (UTC) — semua alert dari SATU run
    # detection harian berbagi run_key yg sama -> dedup per (akun,typology,hari).
    if run_key is None:
        run_key = datetime.utcnow().date().isoformat()

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
        # Fix 6.7 (20-Jul): alert tanpa rule match = hasil skor MODEL ML.
        # Relabel dari "UNKNOWN" (membingungkan, kelihatan spt error) jadi
        # "ML_ENSEMBLE" eksplisit — sumber deteksi paling canggih kita tak
        # lagi tampil paling "tak jelas" ke analis.
        typology = (
            matched_rules[0]["typology"] if matched_rules else "ML_ENSEMBLE"
        )
        detection_layer = detection_layer_for(typology)
        rule_triggered = (
            "; ".join(r["rule_triggered"] for r in matched_rules)
            if matched_rules
            else ""
        )

        # Severity SELALU dari risk_score (konsisten, tidak ada lagi risk 1.0→MEDIUM)
        severity = _severity_from_score(risk_score)

        alerts.append({
            "alert_id": _deterministic_alert_id(account_id, typology, run_key),
            "account_id": account_id,
            "tx_id": None,
            "cluster_id": None,
            "typology": typology,
            "detection_layer": detection_layer,
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
                         detection_layer, risk_score, rule_triggered, severity,
                         node_count, status, created_at)
                    VALUES
                        (:alert_id, :account_id, :tx_id, :cluster_id,
                         :typology, :detection_layer, :risk_score,
                         :rule_triggered, :severity, :node_count, :status,
                         :created_at)
                    ON CONFLICT (alert_id) DO NOTHING
                """),
                batch,
            )
            inserted += len(batch)

    print(f"Inserted {inserted} alerts (threshold={RISK_THRESHOLD})")
    return inserted
