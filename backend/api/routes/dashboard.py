"""
Phase 5.6 — REST endpoints untuk Dashboard Overview (Halaman 1).

Endpoint:
    GET /dashboard/overview           -> KPI cards (alert aktif, cluster, dana
                                          berisiko, kasus pending)
    GET /dashboard/typology-breakdown -> count per typologi (bar chart)
    GET /dashboard/risk-trend         -> time series 30 hari (line chart)
    GET /dashboard/heatmap            -> alert activity jam vs hari
    GET /dashboard/top-accounts       -> top 10 rekening paling berisiko

Semua angka dihitung LANGSUNG dari tabel alerts/transactions asli — tidak
ada data dummy/hardcode.
"""

import logging
import os
import sys

from fastapi import APIRouter, Query
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from graph.analytics import find_clusters
from graph.builder import get_driver

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = get_driver()
    return _driver


@router.get("/overview")
def overview():
    """KPI cards utama Dashboard Overview."""
    # Fix (15-Jul): cluster TIDAK disimpan di kolom alerts.cluster_id (selalu
    # NULL - cluster dihitung dinamis dari graph Neo4j, bukan per-alert).
    # Query lama SELECT COUNT(DISTINCT cluster_id) karena itu selalu balik 0.
    # Reuse find_clusters() yg sama dipakai graph.py (union-find di subgraph
    # illicit) supaya angka konsisten dgn Graph Explorer.
    try:
        n_clusters = len(find_clusters(_get_driver(), min_size=2))
    except Exception:
        # Fix (20-Jul): log dulu — n_clusters=None (Neo4j gagal) beda dari
        # n_clusters=0 (memang tak ada cluster). Tanpa log, KPI "0 cluster"
        # bisa berarti Neo4j down, bukan graph bersih.
        logger.exception("overview: find_clusters gagal — n_clusters=None (Neo4j down?)")
        n_clusters = None

    with _engine.connect() as conn:
        n_active_alerts = conn.execute(text(
            "SELECT COUNT(*) FROM alerts WHERE status IN ('NEW', 'IN_REVIEW')"
        )).scalar()
        # Fix (QC 15-Jul): INNER JOIN sebelumnya diam-diam exclude alert dgn
        # tx_id NULL (alert dari pola behavioral, bukan transaksi spesifik -
        # kolom tx_id nullable di schema). LEFT JOIN supaya alert begitu
        # tetap terhitung (amount-nya cuma 0, bukan hilang dari agregat).
        dana_berisiko = conn.execute(text("""
            SELECT COALESCE(SUM(t.amount), 0)
            FROM alerts a LEFT JOIN transactions t ON a.tx_id = t.tx_id
            WHERE a.status IN ('NEW', 'IN_REVIEW')
        """)).scalar()
        n_pending_case = conn.execute(text(
            "SELECT COUNT(*) FROM cases WHERE status NOT IN ('CLOSED', 'CONFIRM', 'FP')"
        )).scalar()
        n_total_alerts = conn.execute(text("SELECT COUNT(*) FROM alerts")).scalar()

    return {
        "alert_aktif": n_active_alerts,
        "cluster_aktif": n_clusters,
        "dana_berisiko_idr": float(dana_berisiko),
        "kasus_pending": n_pending_case,
        "total_alert_sepanjang_waktu": n_total_alerts,
    }


@router.get("/typology-breakdown")
def typology_breakdown():
    """Jumlah alert per typologi — buat bar chart."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(typology, 'unknown') AS typology, COUNT(*) AS count
            FROM alerts
            GROUP BY typology
            ORDER BY count DESC
        """)).mappings().all()
    return {"items": [dict(r) for r in rows]}


@router.get("/status-distribution")
def status_distribution():
    """Distribusi status alert — buat donut chart."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT status, COUNT(*) AS count FROM alerts GROUP BY status ORDER BY count DESC
        """)).mappings().all()
    return {"items": [dict(r) for r in rows]}


@router.get("/risk-trend")
def risk_trend(days: int = Query(30, ge=1, le=365)):
    """Time series jumlah alert per hari, N hari terakhir — buat line chart."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DATE(created_at) AS date, COUNT(*) AS count,
                   ROUND(AVG(risk_score)::numeric, 4) AS avg_risk_score
            FROM alerts
            WHERE created_at >= NOW() - (:days || ' days')::interval
            GROUP BY DATE(created_at)
            ORDER BY date
        """), {"days": days}).mappings().all()
    return {"items": [dict(r) for r in rows]}


@router.get("/heatmap")
def heatmap():
    """Alert activity jam (0-23) vs hari-dalam-minggu (0=Minggu) — buat heatmap."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT EXTRACT(DOW FROM created_at)::int AS day_of_week,
                   EXTRACT(HOUR FROM created_at)::int AS hour,
                   COUNT(*) AS count
            FROM alerts
            GROUP BY day_of_week, hour
            ORDER BY day_of_week, hour
        """)).mappings().all()
    return {"items": [dict(r) for r in rows]}


@router.get("/top-accounts")
def top_accounts(limit: int = Query(10, ge=1, le=100)):
    """Top N rekening dengan risk_score tertinggi — buat tabel di dashboard."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT account_id, MAX(risk_score) AS max_risk_score,
                   COUNT(*) AS alert_count,
                   STRING_AGG(DISTINCT typology, ', ') AS typologies
            FROM alerts
            WHERE account_id IS NOT NULL
            GROUP BY account_id
            ORDER BY max_risk_score DESC, alert_count DESC
            LIMIT :limit
        """), {"limit": limit}).mappings().all()
    return {"items": [dict(r) for r in rows]}
