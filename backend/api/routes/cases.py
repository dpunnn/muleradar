"""
Phase 5.3 — REST endpoints untuk Case Detail (Halaman 4).

Endpoint:
    POST  /cases              -> buat case dari alert_id
    GET   /cases               -> list semua case (filter status)
    GET   /cases/{case_id}     -> detail case + audit trail (alert terkait,
                                   transaction flags akun terkait)
    PATCH /cases/{case_id}     -> update notes/status/assigned_to
"""

import os
import sys
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import create_engine, text

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from graph.analytics import get_transaction_flags, find_clusters
from graph.builder import get_driver

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/cases", tags=["cases"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

_VALID_STATUS = {"NEW", "IN_REVIEW", "CONFIRM", "FP", "CLOSED"}

_driver = None


def _get_driver():
    global _driver
    if _driver is None:
        _driver = get_driver()
    return _driver


class CreateCaseBody(BaseModel):
    alert_id: str
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


@router.post("")
def create_case(body: CreateCaseBody):
    """Buat case baru dari sebuah alert. 409 kalau alert itu sudah py case."""
    with _engine.begin() as conn:
        alert = conn.execute(
            text("SELECT alert_id FROM alerts WHERE alert_id = :id"), {"id": body.alert_id}
        ).mappings().first()
        if not alert:
            raise HTTPException(404, f"Alert {body.alert_id} tidak ditemukan")

        # Fix (QC 15-Jul): cegah case duplikat kalau endpoint ini dipanggil
        # 2x utk alert yg sama (schema tak punya UNIQUE constraint di
        # alert_id) — beda dgn assign_alert() di alerts.py yg sudah upsert.
        existing = conn.execute(
            text("SELECT case_id FROM cases WHERE alert_id = :id LIMIT 1"),
            {"id": body.alert_id},
        ).mappings().first()
        if existing:
            raise HTTPException(
                409, f"Alert {body.alert_id} sudah punya case #{existing['case_id']}"
            )

        row = conn.execute(
            text("""
                INSERT INTO cases (alert_id, assigned_to, status, notes)
                VALUES (:alert_id, :assigned_to, 'NEW', :notes)
                RETURNING case_id, alert_id, assigned_to, status, notes, created_at, updated_at
            """),
            {"alert_id": body.alert_id, "assigned_to": body.assigned_to, "notes": body.notes},
        ).mappings().first()

        conn.execute(
            text("UPDATE alerts SET status = 'IN_REVIEW' WHERE alert_id = :id AND status = 'NEW'"),
            {"id": body.alert_id},
        )

        detail = f"Case dibuat dari alert {body.alert_id}"
        if body.assigned_to:
            detail += f", di-assign ke {body.assigned_to}"
        conn.execute(
            text("""
                INSERT INTO audit_log (case_id, event_type, actor, detail)
                VALUES (:case_id, 'CASE_CREATED', :actor, :detail)
            """),
            {"case_id": row["case_id"], "actor": body.assigned_to or "System", "detail": detail},
        )

    return dict(row)


@router.get("")
def list_cases(
    status: Optional[str] = Query(None),
    assigned_to: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List semua case, filter opsional by status/assigned_to."""
    # Fix (QC 15-Jul): list_alerts sudah validasi status thd _VALID_STATUS
    # (422 kalau salah), list_cases belum — ?status=TYPO diam-diam balik
    # 0 hasil. Samakan perilakunya.
    if status and status not in _VALID_STATUS:
        raise HTTPException(422, f"status harus salah satu dari {sorted(_VALID_STATUS)}")

    where = []
    params: dict = {"limit": limit, "offset": offset}
    if status:
        where.append("c.status = :status")
        params["status"] = status
    if assigned_to:
        where.append("c.assigned_to = :assigned_to")
        params["assigned_to"] = assigned_to
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    with _engine.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM cases c {where_sql}"), params).scalar()
        rows = conn.execute(
            text(f"""
                SELECT c.case_id, c.alert_id, c.assigned_to, c.status, c.notes,
                       c.created_at, c.updated_at,
                       a.account_id, a.typology, a.risk_score, a.severity
                FROM cases c
                JOIN alerts a ON c.alert_id = a.alert_id
                {where_sql}
                ORDER BY c.created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            params,
        ).mappings().all()

    return {"total": total, "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


@router.get("/{case_id}")
def case_detail(case_id: int):
    """
    Detail case lengkap: info case, alert terkait, transaction flags akun
    terkait, institusi + cluster asli (Neo4j), dana terlibat (sum transaksi
    di-flag), dan audit trail (riwayat event asli dari tabel audit_log —
    BUKAN timeline fabrikasi, cuma nyatat apa yg benar2 pernah terjadi).
    """
    with _engine.connect() as conn:
        case = conn.execute(
            text("SELECT * FROM cases WHERE case_id = :id"), {"id": case_id}
        ).mappings().first()
        if not case:
            raise HTTPException(404, f"Case {case_id} tidak ditemukan")

        alert = conn.execute(
            text("SELECT * FROM alerts WHERE alert_id = :id"), {"id": case["alert_id"]}
        ).mappings().first()

        flags = []
        institusi = None
        dana_terlibat = 0.0
        if alert and alert["account_id"]:
            institusi = conn.execute(
                text("SELECT institution_id FROM accounts WHERE account_id = :id"),
                {"id": alert["account_id"]},
            ).scalar()

            tx_rows = conn.execute(
                text("""
                    SELECT from_account, to_account, amount, tx_timestamp, channel
                    FROM transactions
                    WHERE from_account = :id OR to_account = :id
                    ORDER BY tx_timestamp DESC
                    LIMIT 500
                """),
                {"id": alert["account_id"]},
            ).mappings().all()
            if tx_rows:
                df = pd.DataFrame([dict(r) for r in tx_rows])
                flags = get_transaction_flags(alert["account_id"], df)
                dana_terlibat = float(df["amount"].sum())

        audit_trail = conn.execute(
            text("""
                SELECT log_id, event_type, actor, detail, created_at
                FROM audit_log WHERE case_id = :id ORDER BY created_at ASC
            """),
            {"id": case_id},
        ).mappings().all()

    cluster_info = None
    if alert and alert["account_id"]:
        try:
            clusters = find_clusters(_get_driver(), min_size=2)
            match = next((c for c in clusters if alert["account_id"] in c["nodes"]), None)
            if match:
                cluster_info = {"cluster_id": match["cluster_id"], "size": match["size"], "risk_level": match["risk_level"]}
        except Exception:
            cluster_info = None

    return {
        "case": dict(case),
        "alert": dict(alert) if alert else None,
        "transaction_flags": flags,
        "institusi": institusi,
        "cluster": cluster_info,
        "dana_terlibat_idr": dana_terlibat,
        "audit_trail": [dict(r) for r in audit_trail],
    }


class UpdateCaseBody(BaseModel):
    status: Optional[str] = None
    assigned_to: Optional[str] = None
    notes: Optional[str] = None


@router.patch("/{case_id}")
def update_case(case_id: int, body: UpdateCaseBody):
    """Update status/assigned_to/notes suatu case."""
    if body.status is not None and body.status not in _VALID_STATUS:
        raise HTTPException(422, f"Status harus salah satu dari {sorted(_VALID_STATUS)}")

    fields = []
    params: dict = {"id": case_id}
    if body.status is not None:
        fields.append("status = :status")
        params["status"] = body.status
    if body.assigned_to is not None:
        fields.append("assigned_to = :assigned_to")
        params["assigned_to"] = body.assigned_to
    if body.notes is not None:
        fields.append("notes = :notes")
        params["notes"] = body.notes
    if not fields:
        raise HTTPException(422, "Tidak ada field yang diupdate")
    fields.append("updated_at = NOW()")

    with _engine.begin() as conn:
        before = conn.execute(
            text("SELECT status, assigned_to, notes FROM cases WHERE case_id = :id"), {"id": case_id}
        ).mappings().first()
        if not before:
            raise HTTPException(404, f"Case {case_id} tidak ditemukan")

        row = conn.execute(
            text(f"UPDATE cases SET {', '.join(fields)} WHERE case_id = :id RETURNING *"),
            params,
        ).mappings().first()

        if body.status in ("CONFIRM", "FP", "CLOSED"):
            conn.execute(
                text("UPDATE alerts SET status = :status WHERE alert_id = :alert_id"),
                {"status": body.status, "alert_id": row["alert_id"]},
            )

        actor = row["assigned_to"] or "System"
        # Fix (15-Jul): audit_log cuma catat event yg BENAR-BENAR berubah,
        # bukan asumsikan semua field di body — supaya timeline jujur.
        if body.status is not None and body.status != before["status"]:
            conn.execute(
                text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                        VALUES (:id, 'STATUS_CHANGED', :actor, :detail)"""),
                {"id": case_id, "actor": actor, "detail": f"Status diubah -> {body.status}"},
            )
        if body.assigned_to is not None and body.assigned_to != before["assigned_to"]:
            conn.execute(
                text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                        VALUES (:id, 'ASSIGNED', :actor, :detail)"""),
                {"id": case_id, "actor": body.assigned_to, "detail": f"Case di-assign ke {body.assigned_to}"},
            )
        if body.notes is not None and body.notes != (before["notes"] or ""):
            conn.execute(
                text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                        VALUES (:id, 'NOTE_ADDED', :actor, 'Catatan investigasi ditambahkan')"""),
                {"id": case_id, "actor": actor},
            )

    return dict(row)


class EscalateBody(BaseModel):
    note: Optional[str] = None


@router.post("/{case_id}/escalate")
def escalate_case(case_id: int, body: EscalateBody):
    """
    Eskalasi case (mis. ke compliance/legal) — TIDAK mengubah status enum
    (skema _VALID_STATUS tak punya nilai ESCALATED), tapi mencatat event
    ESCALATED asli ke audit_log supaya tombol "Eskalasi" di frontend genuinely
    melakukan sesuatu, bukan cuma alias dari simpan-catatan (fix QC 15-Jul).
    """
    with _engine.begin() as conn:
        case = conn.execute(
            text("SELECT case_id, assigned_to FROM cases WHERE case_id = :id"), {"id": case_id}
        ).mappings().first()
        if not case:
            raise HTTPException(404, f"Case {case_id} tidak ditemukan")

        detail = "Case dieskalasi" + (f" — {body.note}" if body.note else "")
        conn.execute(
            text("""INSERT INTO audit_log (case_id, event_type, actor, detail)
                    VALUES (:id, 'ESCALATED', :actor, :detail)"""),
            {"id": case_id, "actor": case["assigned_to"] or "System", "detail": detail},
        )

    return {"case_id": case_id, "escalated": True}
