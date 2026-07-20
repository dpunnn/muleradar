"""
Phase 5.2 — REST endpoints untuk Alert List & Alert Detail.

Endpoint:
    GET   /alerts                  -> list alert (filter status/typology/risk/tanggal, pagination)
    GET   /alerts/{alert_id}       -> detail satu alert
    PATCH /alerts/{alert_id}/status -> ubah status NEW|IN_REVIEW|CONFIRM|FP
    POST  /alerts/{alert_id}/assign -> assign ke analyst (upsert ke tabel cases)

Pola sama dgn api/routes/osint.py: SQLAlchemy engine module-level, query
mentah via text() + mappings(), tanpa ORM (konsisten dgn seluruh backend).
"""

import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/alerts", tags=["alerts"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

_VALID_STATUS = {"NEW", "IN_REVIEW", "CONFIRM", "FP", "CLOSED"}
_VALID_SEVERITY = {"HIGH", "MEDIUM", "LOW"}
# Sumber deteksi (fix 6.7, 20-Jul) — dimensi TERPISAH dari typology.
_VALID_DETECTION_LAYER = {
    "AML_CORE", "TYPOLOGY_ID", "STATISTICAL", "GRAPH_MOTIF", "ML_ENSEMBLE",
}


@router.get("")
def list_alerts(
    status: Optional[str] = Query(None, description="Filter: NEW/IN_REVIEW/CONFIRM/FP/CLOSED"),
    typology: Optional[str] = Query(None, description="Filter typologi, mis. judol/structuring"),
    detection_layer: Optional[str] = Query(None, description="Filter sumber deteksi: AML_CORE/TYPOLOGY_ID/STATISTICAL/GRAPH_MOTIF/ML_ENSEMBLE"),
    severity: Optional[str] = Query(None, description="Filter: HIGH/MEDIUM/LOW"),
    account_id: Optional[str] = Query(None, description="Filter alert milik satu akun spesifik"),
    min_risk: Optional[float] = Query(None, ge=0, le=1, description="Risk score minimum"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List alert dgn filter opsional, urut terbaru dulu."""
    # Fix (QC 15-Jul): status/severity dulu diteruskan mentah ke query tanpa
    # validasi — input salah ketik (mis. ?severity=CRITICAL) diam-diam balik
    # list kosong, bukan error jelas. Validasi thd enum yg sama dgn kolom DB.
    if status is not None and status not in _VALID_STATUS:
        raise HTTPException(422, f"status harus salah satu dari {sorted(_VALID_STATUS)}")
    if severity is not None and severity not in _VALID_SEVERITY:
        raise HTTPException(422, f"severity harus salah satu dari {sorted(_VALID_SEVERITY)}")
    if detection_layer is not None and detection_layer not in _VALID_DETECTION_LAYER:
        raise HTTPException(422, f"detection_layer harus salah satu dari {sorted(_VALID_DETECTION_LAYER)}")

    where = []
    params: dict = {"limit": limit, "offset": offset}
    if status:
        where.append("status = :status")
        params["status"] = status
    if typology:
        where.append("typology = :typology")
        params["typology"] = typology
    if detection_layer:
        where.append("detection_layer = :detection_layer")
        params["detection_layer"] = detection_layer
    if severity:
        where.append("severity = :severity")
        params["severity"] = severity
    if account_id:
        where.append("account_id = :account_id")
        params["account_id"] = account_id
    if min_risk is not None:
        where.append("risk_score >= :min_risk")
        params["min_risk"] = min_risk
    if date_from:
        where.append("created_at >= :date_from")
        params["date_from"] = date_from
    if date_to:
        where.append("created_at <= :date_to")
        params["date_to"] = date_to
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with _engine.connect() as conn:
        total = conn.execute(
            text(f"SELECT COUNT(*) FROM alerts {where_sql}"), params
        ).scalar()
        rows = conn.execute(
            text(f"""
                SELECT alert_id, account_id, tx_id, cluster_id, typology,
                       detection_layer, risk_score, rule_triggered, severity,
                       node_count, status, created_at
                FROM alerts
                {where_sql}
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).mappings().all()

    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


@router.get("/{alert_id}")
def alert_detail(alert_id: str):
    """Detail satu alert + info case terkait (kalau sudah ada)."""
    with _engine.connect() as conn:
        alert = conn.execute(
            text("SELECT * FROM alerts WHERE alert_id = :id"), {"id": alert_id}
        ).mappings().first()
        if not alert:
            raise HTTPException(404, f"Alert {alert_id} tidak ditemukan")

        case = conn.execute(
            text("SELECT * FROM cases WHERE alert_id = :id ORDER BY created_at DESC LIMIT 1"),
            {"id": alert_id},
        ).mappings().first()

    result = dict(alert)
    result["case"] = dict(case) if case else None
    return result


class StatusUpdate(BaseModel):
    status: str


@router.patch("/{alert_id}/status")
def update_status(alert_id: str, body: StatusUpdate):
    """Ubah status alert. Valid: NEW, IN_REVIEW, CONFIRM, FP, CLOSED."""
    if body.status not in _VALID_STATUS:
        raise HTTPException(422, f"Status harus salah satu dari {sorted(_VALID_STATUS)}")

    with _engine.begin() as conn:
        result = conn.execute(
            text("UPDATE alerts SET status = :status WHERE alert_id = :id"),
            {"status": body.status, "id": alert_id},
        )
        if result.rowcount == 0:
            raise HTTPException(404, f"Alert {alert_id} tidak ditemukan")

    return {"alert_id": alert_id, "status": body.status}


class AssignBody(BaseModel):
    assigned_to: str
    notes: Optional[str] = None


@router.post("/{alert_id}/assign")
def assign_alert(alert_id: str, body: AssignBody):
    """
    Assign alert ke analyst — upsert ke tabel `cases` (buat case baru kalau
    belum ada utk alert ini, update assigned_to kalau sudah ada).
    """
    with _engine.begin() as conn:
        alert = conn.execute(
            text("SELECT alert_id FROM alerts WHERE alert_id = :id"), {"id": alert_id}
        ).mappings().first()
        if not alert:
            raise HTTPException(404, f"Alert {alert_id} tidak ditemukan")

        existing = conn.execute(
            text("SELECT case_id, assigned_to FROM cases WHERE alert_id = :id ORDER BY created_at DESC LIMIT 1"),
            {"id": alert_id},
        ).mappings().first()

        # Fix (QC 15-Jul): sebelumnya SELALU insert event ASSIGNED, termasuk
        # kalau re-assign ke orang yg sama -> audit trail penuh baris
        # duplikat identik. Cuma log kalau assigned_to BENAR-BENAR berubah.
        already_same = bool(existing) and existing["assigned_to"] == body.assigned_to

        is_new_case = existing is None

        if existing:
            conn.execute(
                text("""
                    UPDATE cases SET assigned_to = :assigned_to,
                           notes = COALESCE(:notes, notes), updated_at = NOW()
                    WHERE case_id = :case_id
                """),
                {"assigned_to": body.assigned_to, "notes": body.notes, "case_id": existing["case_id"]},
            )
            case_id = existing["case_id"]
        else:
            row = conn.execute(
                text("""
                    INSERT INTO cases (alert_id, assigned_to, status, notes)
                    VALUES (:alert_id, :assigned_to, 'NEW', :notes)
                    RETURNING case_id
                """),
                {"alert_id": alert_id, "assigned_to": body.assigned_to, "notes": body.notes},
            ).mappings().first()
            case_id = row["case_id"]

        conn.execute(
            text("UPDATE alerts SET status = 'IN_REVIEW' WHERE alert_id = :id AND status = 'NEW'"),
            {"id": alert_id},
        )

        # Fix (QC ronde 2): jalur ini bisa bikin case BARU (spt create_case
        # di cases.py) tapi sebelumnya cuma log ASSIGNED, tak pernah
        # CASE_CREATED — audit trail jadi tidak lengkap dibanding case yg
        # dibuat lewat cases.py. Samakan: log CASE_CREATED dulu kalau baru.
        if is_new_case:
            conn.execute(
                text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                        VALUES (:id, 'CASE_CREATED', :actor, :detail)"""),
                {"id": case_id, "actor": body.assigned_to or "System", "detail": f"Case dibuat dari alert {alert_id}"},
            )

        if not already_same:
            conn.execute(
                text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                        VALUES (:id, 'ASSIGNED', :actor, :detail)"""),
                {"id": case_id, "actor": body.assigned_to, "detail": f"Case di-assign ke {body.assigned_to}"},
            )

    return {"alert_id": alert_id, "case_id": case_id, "assigned_to": body.assigned_to}
