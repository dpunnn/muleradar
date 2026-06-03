"""
MuleRadar Phase 3 — Detection Rules.

Dua layer deteksi:
  1. AML Core Rules  — pola money laundering dari AMLWorld (sistem utama)
  2. Indonesia Typology Rules — 7 tipologi Indonesia sebagai layer tambahan
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


# ===========================================================================
# LAYER 1 — AML CORE RULES (sistem utama, pola dari AMLWorld)
# ===========================================================================

def check_aml_core_rules(
    engine=None, db_url=None, limit_per_rule: int = 5000
) -> list[dict]:
    """
    Deteksi pola money laundering dasar:
    STRUCTURING, FAN_OUT, LAYERING, CYCLE.
    Ini adalah sistem deteksi utama MuleRadar berbasis AMLWorld benchmark.
    """
    if engine is None:
        engine = create_engine(db_url or DATABASE_URL)

    results: list[dict] = []

    # Rule A1: STRUCTURING
    try:
        rows = engine.connect().execute(text("""
            SELECT from_account AS account_id,
                   COUNT(*) FILTER (WHERE amount BETWEEN 425000 AND 499999)   AS n_500k,
                   COUNT(*) FILTER (WHERE amount BETWEEN 850000 AND 999999)   AS n_1m,
                   COUNT(*) FILTER (WHERE amount BETWEEN 4250000 AND 4999999) AS n_5m,
                   COUNT(*) FILTER (WHERE amount BETWEEN 21250000 AND 24999999) AS n_25m
            FROM transactions
            GROUP BY from_account
            HAVING COUNT(*) FILTER (WHERE amount BETWEEN 425000   AND 499999)   >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 850000   AND 999999)   >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 4250000  AND 4999999)  >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 21250000 AND 24999999) >= 3
            ORDER BY (
                COUNT(*) FILTER (WHERE amount BETWEEN 425000 AND 499999) +
                COUNT(*) FILTER (WHERE amount BETWEEN 850000 AND 999999)
            ) DESC
            LIMIT :lim
        """), {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            peak = max(int(r["n_500k"]), int(r["n_1m"]), int(r["n_5m"]), int(r["n_25m"]))
            results.append({
                "account_id": r["account_id"],
                "typology": "STRUCTURING",
                "rule_triggered": f"n_500k={r['n_500k']}, n_1m={r['n_1m']}, n_5m={r['n_5m']}",
                "confidence": round(min(0.90, 0.5 + (peak / 20) * 0.4), 4),
                "meta": {"n_500k": int(r["n_500k"]), "n_1m": int(r["n_1m"]),
                         "n_5m": int(r["n_5m"]), "n_25m": int(r["n_25m"])},
            })
        print(f"  [AML] STRUCTURING    : {len(rows):,} hits")
    except Exception as e:
        print(f"  [AML] STRUCTURING failed: {e}")

    # Rule A2: FAN_OUT
    try:
        rows = engine.connect().execute(text("""
            SELECT from_account AS account_id,
                   COUNT(DISTINCT to_account) AS recipient_count,
                   COUNT(*) AS tx_count,
                   SUM(amount) AS total_out
            FROM transactions
            GROUP BY from_account
            HAVING COUNT(DISTINCT to_account) >= 20 AND COUNT(*) >= 20
            ORDER BY recipient_count DESC
            LIMIT :lim
        """), {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            rc = float(r["recipient_count"])
            results.append({
                "account_id": r["account_id"],
                "typology": "FAN_OUT",
                "rule_triggered": f"recipient_count={int(rc)}, tx_count={r['tx_count']}",
                "confidence": round(min(0.85, 0.4 + (rc / 100) * 0.45), 4),
                "meta": {"recipient_count": int(rc), "tx_count": int(r["tx_count"]),
                         "total_out": float(r["total_out"])},
            })
        print(f"  [AML] FAN_OUT        : {len(rows):,} hits")
    except Exception as e:
        print(f"  [AML] FAN_OUT failed: {e}")

    # Rule A3: LAYERING (dua query terpisah, merge di Python)
    try:
        import pandas as pd
        with engine.connect() as conn:
            df_in  = pd.DataFrame(conn.execute(text("""
                SELECT to_account AS account_id, SUM(amount) AS total_in, COUNT(*) AS in_cnt
                FROM transactions GROUP BY to_account
                HAVING SUM(amount) > 1000000 AND COUNT(*) >= 3
            """)).mappings().all())
            df_out = pd.DataFrame(conn.execute(text("""
                SELECT from_account AS account_id, SUM(amount) AS total_out, COUNT(*) AS out_cnt
                FROM transactions GROUP BY from_account HAVING COUNT(*) >= 3
            """)).mappings().all())
        if not df_in.empty and not df_out.empty:
            df_in["total_in"]   = df_in["total_in"].astype(float)
            df_out["total_out"] = df_out["total_out"].astype(float)
            m = df_in.merge(df_out, on="account_id", how="inner")
            m["passthrough_ratio"] = m["total_out"] / m["total_in"]
            layer = m[m["passthrough_ratio"].between(0.80, 1.05)].nlargest(limit_per_rule, "total_in")
            for _, r in layer.iterrows():
                pt = float(r["passthrough_ratio"])
                results.append({
                    "account_id": r["account_id"],
                    "typology": "LAYERING",
                    "rule_triggered": f"passthrough_ratio={pt:.3f}, in={int(r['in_cnt'])}, out={int(r['out_cnt'])}",
                    "confidence": round(min(0.80, 0.45 + (1.0 - abs(pt - 0.95)) * 0.35), 4),
                    "meta": {"passthrough_ratio": round(pt, 4),
                             "in_cnt": int(r["in_cnt"]), "out_cnt": int(r["out_cnt"])},
                })
            print(f"  [AML] LAYERING       : {len(layer):,} hits")
        else:
            print("  [AML] LAYERING       : 0 hits")
    except Exception as e:
        print(f"  [AML] LAYERING failed: {e}")

    # Rule A4: CYCLE (A→B→A, proxy via direct mutual transfers)
    try:
        rows = engine.connect().execute(text("""
            SELECT DISTINCT t1.from_account AS account_id
            FROM transactions t1
            JOIN transactions t2
              ON t1.to_account   = t2.from_account
             AND t2.to_account   = t1.from_account
             AND t1.from_account <> t1.to_account
            LIMIT :lim
        """), {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            results.append({
                "account_id": r["account_id"],
                "typology": "CYCLE",
                "rule_triggered": "CYCLE: mutual transfer (A→B→A)",
                "confidence": 0.75,
                "meta": {},
            })
        print(f"  [AML] CYCLE          : {len(rows):,} hits")
    except Exception as e:
        print(f"  [AML] CYCLE failed: {e}")

    print(f"  [AML CORE TOTAL] {len(results):,} hits\n")
    return results


# ===========================================================================
# LAYER 2 — INDONESIA TYPOLOGY RULES (layer tambahan)
# ===========================================================================

def check_typology_rules(
    engine=None, db_url=None, limit_per_rule: int = 5000
) -> list[dict]:
    """
    Deteksi 7 tipologi Indonesia sebagai layer tambahan di atas AML core:
    JUDOL_RING, QRIS_RING, DORMANT_ACTIVATION.
    """
    if engine is None:
        engine = create_engine(db_url or DATABASE_URL)

    results: list[dict] = []

    # ------------------------------------------------------------------
    # Rule 1: JUDOL_RING
    # ------------------------------------------------------------------
    try:
        sql_judol = text("""
            SELECT to_account AS account_id,
                   COUNT(DISTINCT from_account) AS sender_count,
                   COUNT(*) AS tx_count,
                   AVG(amount) AS avg_amount,
                   SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                        IN (22,23,0,1,2,3) THEN 1 ELSE 0 END)::float
                        / COUNT(*) AS night_ratio
            FROM transactions
            WHERE amount < 500000
            GROUP BY to_account
            HAVING COUNT(DISTINCT from_account) >= 30
               AND COUNT(*) >= 50
               AND SUM(CASE WHEN EXTRACT(HOUR FROM tx_timestamp)
                        IN (22,23,0,1,2,3) THEN 1 ELSE 0 END)::float
                        / COUNT(*) >= 0.3
            ORDER BY sender_count DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_judol, {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            sender_count = float(r["sender_count"])
            night_ratio = float(r["night_ratio"])
            conf = min(0.95, 0.5 + (sender_count / 200) * 0.3 + night_ratio * 0.2)
            results.append({
                "account_id": r["account_id"],
                "typology": "JUDOL_RING",
                "rule_triggered": "JUDOL_RING: many small senders + night dominance",
                "confidence": round(conf, 4),
                "meta": {
                    "sender_count": int(sender_count),
                    "tx_count": int(r["tx_count"]),
                    "avg_amount": float(r["avg_amount"]),
                    "night_ratio": round(night_ratio, 4),
                },
            })
        print(f"[RULE] JUDOL_RING: {len(rows)} hits")
    except Exception as e:
        print(f"[RULE] JUDOL_RING failed: {e}")

    # ------------------------------------------------------------------
    # Rule 2: QRIS_RING
    # ------------------------------------------------------------------
    try:
        sql_qris = text("""
            SELECT to_account AS account_id,
                   COUNT(DISTINCT from_account) AS merchant_count,
                   COUNT(*) AS tx_count,
                   SUM(amount) AS total_received
            FROM transactions
            WHERE channel = 'qris'
            GROUP BY to_account
            HAVING COUNT(DISTINCT from_account) >= 10 AND COUNT(*) >= 30
            ORDER BY merchant_count DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_qris, {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            merchant_count = float(r["merchant_count"])
            conf = min(0.90, 0.5 + (merchant_count / 100) * 0.4)
            results.append({
                "account_id": r["account_id"],
                "typology": "QRIS_RING",
                "rule_triggered": "QRIS_RING: many QRIS senders to one receiver",
                "confidence": round(conf, 4),
                "meta": {
                    "merchant_count": int(merchant_count),
                    "tx_count": int(r["tx_count"]),
                    "total_received": float(r["total_received"]),
                },
            })
        print(f"[RULE] QRIS_RING: {len(rows)} hits")
    except Exception as e:
        print(f"[RULE] QRIS_RING failed: {e}")

    # ------------------------------------------------------------------
    # Rule 3: DORMANT_ACTIVATION — dua query terpisah, merge di Python
    # ------------------------------------------------------------------
    try:
        import pandas as pd

        # Ambil max timestamp dataset sekali
        with engine.connect() as conn:
            max_ts = conn.execute(text("SELECT MAX(tx_timestamp) FROM transactions")).scalar()

        sql_sent = text("""
            SELECT from_account AS account_id,
                   MIN(tx_timestamp) AS first_sent,
                   MAX(tx_timestamp) AS last_sent,
                   COUNT(*) AS sent_total,
                   COUNT(*) FILTER (WHERE tx_timestamp > :cutoff) AS sent_recent
            FROM transactions
            GROUP BY from_account
        """)
        sql_recv = text("""
            SELECT to_account AS account_id,
                   MIN(tx_timestamp) AS first_recv,
                   MAX(tx_timestamp) AS last_recv,
                   COUNT(*) AS recv_total,
                   COUNT(*) FILTER (WHERE tx_timestamp > :cutoff) AS recv_recent
            FROM transactions
            GROUP BY to_account
        """)

        cutoff = max_ts - __import__('datetime').timedelta(days=60)
        with engine.connect() as conn:
            df_sent = pd.DataFrame(conn.execute(sql_sent, {"cutoff": cutoff}).mappings().all())
            df_recv = pd.DataFrame(conn.execute(sql_recv, {"cutoff": cutoff}).mappings().all())

        if not df_sent.empty and not df_recv.empty:
            m = df_sent.merge(df_recv, on="account_id", how="outer")
            # Isi numeric columns saja — jangan fillna timestamp columns
            for col in ["sent_total", "recv_total", "sent_recent", "recv_recent"]:
                m[col] = m[col].fillna(0).astype(float)
            m["first_tx"] = m[["first_sent", "first_recv"]].min(axis=1)
            m["last_tx"]  = m[["last_sent",  "last_recv"]].max(axis=1)
            m["total_tx"]     = m["sent_total"] + m["recv_total"]
            m["recent_cnt"]   = m["sent_recent"] + m["recv_recent"]
            m["span_days"]    = (m["last_tx"] - m["first_tx"]).dt.total_seconds() / 86400

            dormant = m[
                (m["span_days"] >= 180) &
                (m["recent_cnt"] >= 5) &
                (m["total_tx"] <= 100)
            ].nlargest(limit_per_rule, "recent_cnt")

            for _, r in dormant.iterrows():
                conf = min(0.80, 0.45 + (float(r["recent_cnt"]) / 30) * 0.35)
                results.append({
                    "account_id": r["account_id"],
                    "typology": "DORMANT_ACTIVATION",
                    "rule_triggered": f"gap={r['span_days']:.0f}d, recent={int(r['recent_cnt'])}, total={int(r['total_tx'])}",
                    "confidence": round(conf, 4),
                    "meta": {
                        "span_days": round(float(r["span_days"]), 1),
                        "total_tx": int(r["total_tx"]),
                        "recent_cnt": int(r["recent_cnt"]),
                    },
                })
            print(f"[RULE] DORMANT_ACTIVATION: {len(dormant)} hits")
        else:
            print("[RULE] DORMANT_ACTIVATION: 0 hits (empty data)")
    except Exception as e:
        print(f"[RULE] DORMANT_ACTIVATION failed: {e}")

    # ------------------------------------------------------------------
    # Rule 4: STRUCTURING
    # ------------------------------------------------------------------
    try:
        sql_struct = text("""
            SELECT from_account AS account_id,
                   COUNT(*) FILTER (WHERE amount BETWEEN 425000 AND 499999) AS n_500k,
                   COUNT(*) FILTER (WHERE amount BETWEEN 850000 AND 999999) AS n_1m,
                   COUNT(*) FILTER (WHERE amount BETWEEN 4250000 AND 4999999) AS n_5m,
                   COUNT(*) FILTER (WHERE amount BETWEEN 21250000 AND 24999999) AS n_25m
            FROM transactions
            GROUP BY from_account
            HAVING COUNT(*) FILTER (WHERE amount BETWEEN 425000 AND 499999) >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 850000 AND 999999) >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 4250000 AND 4999999) >= 3
                OR COUNT(*) FILTER (WHERE amount BETWEEN 21250000 AND 24999999) >= 3
            ORDER BY (
                COUNT(*) FILTER (WHERE amount BETWEEN 425000 AND 499999) +
                COUNT(*) FILTER (WHERE amount BETWEEN 850000 AND 999999)
            ) DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_struct, {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            n_500k = int(r["n_500k"])
            n_1m = int(r["n_1m"])
            n_5m = int(r["n_5m"])
            n_25m = int(r["n_25m"])
            peak = max(n_500k, n_1m, n_5m, n_25m)
            conf = min(0.90, 0.5 + (peak / 20) * 0.4)
            results.append({
                "account_id": r["account_id"],
                "typology": "STRUCTURING",
                "rule_triggered": "STRUCTURING: repeated amounts just below thresholds",
                "confidence": round(conf, 4),
                "meta": {
                    "n_500k": n_500k,
                    "n_1m": n_1m,
                    "n_5m": n_5m,
                    "n_25m": n_25m,
                },
            })
        print(f"[RULE] STRUCTURING: {len(rows)} hits")
    except Exception as e:
        print(f"[RULE] STRUCTURING failed: {e}")

    # ------------------------------------------------------------------
    # Rule 5: FAN_OUT
    # ------------------------------------------------------------------
    try:
        sql_fanout = text("""
            SELECT from_account AS account_id,
                   COUNT(DISTINCT to_account) AS recipient_count,
                   COUNT(*) AS tx_count,
                   SUM(amount) AS total_out
            FROM transactions
            GROUP BY from_account
            HAVING COUNT(DISTINCT to_account) >= 20 AND COUNT(*) >= 20
            ORDER BY recipient_count DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_fanout, {"lim": limit_per_rule}).mappings().all()
        for r in rows:
            recipient_count = float(r["recipient_count"])
            conf = min(0.85, 0.4 + (recipient_count / 100) * 0.45)
            results.append({
                "account_id": r["account_id"],
                "typology": "FAN_OUT",
                "rule_triggered": "FAN_OUT: single sender to many recipients",
                "confidence": round(conf, 4),
                "meta": {
                    "recipient_count": int(recipient_count),
                    "tx_count": int(r["tx_count"]),
                    "total_out": float(r["total_out"]),
                },
            })
        print(f"[RULE] FAN_OUT: {len(rows)} hits")
    except Exception as e:
        print(f"[RULE] FAN_OUT failed: {e}")

    # ------------------------------------------------------------------
    # Rule 6: LAYERING — dua query terpisah, merge di Python
    # (UNION ALL nested pada 176M rows crash PostgreSQL)
    # ------------------------------------------------------------------
    try:
        sql_in = text("""
            SELECT to_account AS account_id,
                   SUM(amount) AS total_in,
                   COUNT(*) AS in_cnt
            FROM transactions
            GROUP BY to_account
            HAVING SUM(amount) > 1000000 AND COUNT(*) >= 3
        """)
        sql_out = text("""
            SELECT from_account AS account_id,
                   SUM(amount) AS total_out,
                   COUNT(*) AS out_cnt
            FROM transactions
            GROUP BY from_account
            HAVING COUNT(*) >= 3
        """)
        import pandas as pd
        with engine.connect() as conn:
            df_in  = pd.DataFrame(conn.execute(sql_in).mappings().all())
            df_out = pd.DataFrame(conn.execute(sql_out).mappings().all())

        if not df_in.empty and not df_out.empty:
            df_in["total_in"]   = df_in["total_in"].astype(float)
            df_out["total_out"] = df_out["total_out"].astype(float)
            merged = df_in.merge(df_out, on="account_id", how="inner")
            merged["passthrough_ratio"] = merged["total_out"] / merged["total_in"]
            layer = merged[
                merged["passthrough_ratio"].between(0.80, 1.05)
            ].nlargest(limit_per_rule, "total_in")

            for _, r in layer.iterrows():
                pt = float(r["passthrough_ratio"])
                conf = min(0.80, 0.45 + (1.0 - abs(pt - 0.95)) * 0.35)
                results.append({
                    "account_id": r["account_id"],
                    "typology": "LAYERING",
                    "rule_triggered": "LAYERING: high pass-through ratio (in ~ out)",
                    "confidence": round(conf, 4),
                    "meta": {
                        "total_in": float(r["total_in"]),
                        "total_out": float(r["total_out"]),
                        "in_cnt": int(r["in_cnt"]),
                        "out_cnt": int(r["out_cnt"]),
                        "passthrough_ratio": round(pt, 4),
                    },
                })
            print(f"[RULE] LAYERING: {len(layer)} hits")
        else:
            print("[RULE] LAYERING: 0 hits (empty data)")
    except Exception as e:
        print(f"[RULE] LAYERING failed: {e}")

    print(f"  [TYPOLOGY TOTAL] {len(results):,} hits\n")
    return results


# ===========================================================================
# LAYER 3 — STATISTICAL ANOMALY DETECTION (adaptive, no hardcoded thresholds)
# ===========================================================================

def check_statistical_anomaly(
    engine=None, db_url=None, limit_per_rule: int = 5000
) -> list[dict]:
    """
    Deteksi outlier statistik adaptif (z-score / percentile) — tidak pakai
    threshold hardcoded. Flag akun di atas P99 populasi untuk:
      - FANOUT_ANOMALY  : jumlah penerima unik jauh di atas normal
      - FANIN_ANOMALY   : jumlah pengirim unik jauh di atas normal (collector)
    Confidence skala dengan z-score (makin ekstrem makin tinggi).
    """
    if engine is None:
        engine = create_engine(db_url or DATABASE_URL)

    results: list[dict] = []

    # FANOUT_ANOMALY: distribusi COUNT(DISTINCT to_account) per from_account
    try:
        sql_fanout = text("""
            WITH fanout AS (
                SELECT from_account AS acc, COUNT(DISTINCT to_account) AS metric
                FROM transactions
                GROUP BY from_account
            ),
            stat AS (
                SELECT AVG(metric) AS mu, STDDEV_POP(metric) AS sd,
                       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY metric) AS p99
                FROM fanout
            )
            SELECT f.acc, f.metric, s.mu, s.sd, s.p99,
                   (f.metric - s.mu) / NULLIF(s.sd, 0) AS zscore
            FROM fanout f CROSS JOIN stat s
            WHERE f.metric > s.p99 AND f.metric >= 5
            ORDER BY zscore DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_fanout, {"lim": limit_per_rule}).mappings().all()

        p99_fanout = float(rows[0]["p99"]) if rows else 0.0
        for r in rows:
            metric = int(r["metric"])
            mu = float(r["mu"])
            p99 = float(r["p99"])
            zscore = float(r["zscore"]) if r["zscore"] is not None else 0.0
            conf = round(min(0.95, 0.5 + min(zscore / 10, 1.0) * 0.45), 4)
            results.append({
                "account_id": r["acc"],
                "typology": "FANOUT_ANOMALY",
                "rule_triggered": f"metric={metric} (P99={p99:.0f}, z={zscore:.1f})",
                "confidence": conf,
                "meta": {
                    "metric": metric,
                    "p99": round(p99, 2),
                    "zscore": round(zscore, 2),
                    "population_mean": round(mu, 2),
                },
            })
        print(f"  [STAT] FANOUT_ANOMALY: {len(rows)} hits (P99={p99_fanout:.0f})")
    except Exception as e:
        print(f"  [STAT] FANOUT_ANOMALY failed: {e}")

    # FANIN_ANOMALY: distribusi COUNT(DISTINCT from_account) per to_account
    try:
        sql_fanin = text("""
            WITH fanin AS (
                SELECT to_account AS acc, COUNT(DISTINCT from_account) AS metric
                FROM transactions
                GROUP BY to_account
            ),
            stat AS (
                SELECT AVG(metric) AS mu, STDDEV_POP(metric) AS sd,
                       PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY metric) AS p99
                FROM fanin
            )
            SELECT f.acc, f.metric, s.mu, s.sd, s.p99,
                   (f.metric - s.mu) / NULLIF(s.sd, 0) AS zscore
            FROM fanin f CROSS JOIN stat s
            WHERE f.metric > s.p99 AND f.metric >= 5
            ORDER BY zscore DESC
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_fanin, {"lim": limit_per_rule}).mappings().all()

        p99_fanin = float(rows[0]["p99"]) if rows else 0.0
        for r in rows:
            metric = int(r["metric"])
            mu = float(r["mu"])
            p99 = float(r["p99"])
            zscore = float(r["zscore"]) if r["zscore"] is not None else 0.0
            conf = round(min(0.95, 0.5 + min(zscore / 10, 1.0) * 0.45), 4)
            results.append({
                "account_id": r["acc"],
                "typology": "FANIN_ANOMALY",
                "rule_triggered": f"metric={metric} (P99={p99:.0f}, z={zscore:.1f})",
                "confidence": conf,
                "meta": {
                    "metric": metric,
                    "p99": round(p99, 2),
                    "zscore": round(zscore, 2),
                    "population_mean": round(mu, 2),
                },
            })
        print(f"  [STAT] FANIN_ANOMALY: {len(rows)} hits (P99={p99_fanin:.0f})")
    except Exception as e:
        print(f"  [STAT] FANIN_ANOMALY failed: {e}")

    print(f"  [STAT TOTAL] {len(results):,} hits\n")
    return results


# ===========================================================================
# LAYER 4 — GRAPH MOTIF DETECTION (scoped 3-hop cycle, suspect-only)
# ===========================================================================

def check_graph_motifs(
    engine=None, db_url=None, suspect_accounts: list = None,
    limit_per_rule: int = 1000
) -> list[dict]:
    """
    Deteksi motif CYCLE 3-hop (A->B->C->A) HANYA di antara akun suspect.
    Scoped supaya join cepat — bukan full-graph scan.
    suspect_accounts: list account_id dari rules lain. Jika None/empty → return [].
    """
    if not suspect_accounts:
        print("  [MOTIF] CYCLE_MOTIF: skipped (no suspect accounts provided)")
        return []

    if engine is None:
        engine = create_engine(db_url or DATABASE_URL)

    # Batasi ke 5000 suspect pertama supaya join aman
    suspects = list(suspect_accounts)[:5000]

    results: list[dict] = []

    try:
        sql_cycle = text("""
            SELECT DISTINCT t1.from_account AS account_id
            FROM transactions t1
            JOIN transactions t2 ON t1.to_account = t2.from_account
            JOIN transactions t3 ON t2.to_account = t3.from_account
                                 AND t3.to_account = t1.from_account
            WHERE t1.from_account = ANY(:suspects)
              AND t1.from_account <> t2.from_account
              AND t2.from_account <> t3.from_account
              AND t1.from_account <> t2.to_account
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql_cycle, {"suspects": suspects, "lim": limit_per_rule}).mappings().all()

        for r in rows:
            results.append({
                "account_id": r["account_id"],
                "typology": "CYCLE_MOTIF",
                "rule_triggered": "CYCLE_MOTIF: 3-hop circular flow (A->B->C->A)",
                "confidence": 0.85,
                "meta": {},
            })
        print(f"  [MOTIF] CYCLE_MOTIF: {len(rows)} hits (dari {len(suspects)} suspect)")
    except Exception as e:
        print(f"  [MOTIF] CYCLE_MOTIF failed: {e}")

    print(f"  [MOTIF TOTAL] {len(results):,} hits\n")
    return results
