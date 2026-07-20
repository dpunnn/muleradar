"""
Phase 5.5 + Fase 7 — REST endpoints untuk LLM Copilot (panel di Case Detail).

Endpoint:
    GET  /copilot/typology/{name} -> penjelasan + red flags tipologi (statis)
    POST /copilot/summary         -> generate case summary via SARNarrator
    POST /copilot/ltkm            -> generate draft LTKM via SARNarrator

Menyambungkan backend/llm/sar_narrator.py (SUDAH ADA & SELESAI, Fase 7.0)
ke API — bukan menulis ulang logic LLM. Default provider "template"
(rule-based, tanpa LLM, SELALU tersedia) kecuali LLM_PROVIDER di-set di env.
"""

import os
import re
import sys

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
from sqlalchemy import create_engine, text

# account_id valid: alfanumerik + hyphen/underscore, maks 64 char (fix 20-Jul,
# scan produksi). Selain mencegah query aneh, ini SANITASI prompt-injection:
# account_id mengalir ke prompt LLM (kalau LLM_PROVIDER di-set) via
# _build_prompt -> string dgn newline/instruksi jahat ("...\n\nIgnore previous
# instructions...") bisa membelokkan output LTKM. Pola ketat = tak ada
# karakter kontrol/whitespace yg bisa dipakai injeksi.
_ACCOUNT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from llm.sar_narrator import SARNarrator, PPATK_REF
from detection.features import extract_features_for_accounts

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/copilot", tags=["copilot"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)
_narrator = SARNarrator()  # provider dari env LLM_PROVIDER, default "template"


@router.get("/typology/{name}")
def typology_info(name: str):
    """Penjelasan singkat + referensi PPATK utk satu tipologi (statis, no LLM)."""
    key = name.upper()
    if key not in PPATK_REF:
        raise HTTPException(404, f"Tipologi '{name}' tak dikenal. Pilihan: {sorted(PPATK_REF)}")
    ref, desc = PPATK_REF[key]
    return {"typology": key, "ppatk_ref": ref, "description": desc}


def _get_account_context(account_id: str) -> dict:
    """Ambil risk_score/typology/rule dari alert TERBARU + fitur asli akun."""
    with _engine.connect() as conn:
        alert = conn.execute(
            text("""
                SELECT typology, risk_score, rule_triggered
                FROM alerts WHERE account_id = :id
                ORDER BY created_at DESC LIMIT 1
            """),
            {"id": account_id},
        ).mappings().first()

    feats_df = extract_features_for_accounts(_engine, [account_id])
    features = {} if feats_df.empty else feats_df.iloc[0].to_dict()

    return {
        "risk_score": float(alert["risk_score"]) if alert else 0.0,
        "typology": (alert["typology"] or "UNKNOWN").upper() if alert else "UNKNOWN",
        "rules": [alert["rule_triggered"]] if alert and alert["rule_triggered"] else [],
        "features": features,
    }


class AccountBody(BaseModel):
    account_id: str

    @field_validator("account_id")
    @classmethod
    def _validate_account_id(cls, v: str) -> str:
        if not _ACCOUNT_ID_RE.match(v or ""):
            raise ValueError(
                "account_id tidak valid (hanya A-Z a-z 0-9 _ - , maks 64 char)")
        return v


@router.post("/summary")
def generate_summary(body: AccountBody):
    """
    Ringkasan CEPAT (bukan LLM) utk tombol 'Generate Summary' — beda dgn
    /copilot/ltkm yg narasi formal panjang siap-kirim-PPATK.

    Fix (QC 15-Jul): sebelumnya endpoint ini manggil _narrator.narrate() yg
    SAMA PERSIS dgn /copilot/ltkm — hasilnya identik, kalau juri klik dua
    tombol itu di demo bakal ketahuan sama. narrate() SENDIRI memang
    didesain generate narasi ber-gaya LTKM (lihat docstring sar_narrator.py),
    jadi bukan cocok jadi "ringkasan cepat". Summary di sini dibuat langsung
    dari data terstruktur (deterministic, tanpa panggil LLM) — genuinely
    beda: 1-2 kalimat quick-glance, bukan paragraf formal.
    """
    ctx = _get_account_context(body.account_id)
    ppatk_ref, ppatk_desc = PPATK_REF.get(ctx["typology"], PPATK_REF["UNKNOWN"])
    severity = "TINGGI" if ctx["risk_score"] >= 0.8 else "SEDANG" if ctx["risk_score"] >= 0.5 else "RENDAH"
    rule_str = "; ".join(ctx["rules"]) if ctx["rules"] else "tidak ada rule spesifik"

    summary = (
        f"Akun {body.account_id} — risiko {severity} (skor {ctx['risk_score']:.4f}), "
        f"terindikasi {ppatk_desc} ({ppatk_ref}). Rule terpicu: {rule_str}."
    )
    return {
        "account_id": body.account_id,
        "provider": "deterministic",
        "summary": summary,
    }


@router.post("/ltkm")
def generate_ltkm(body: AccountBody):
    """Generate draft LTKM siap kirim PPATK — dipakai tombol 'Draft LTKM'."""
    ctx = _get_account_context(body.account_id)
    narrative = _narrator.narrate(
        account_id=body.account_id,
        risk_score=ctx["risk_score"],
        typology=ctx["typology"],
        features=ctx["features"],
        rules=ctx["rules"],
    )
    return {
        "account_id": body.account_id,
        "provider": _narrator.provider,
        "ltkm_draft": narrative,
        "note": "Submit langsung ke PPATK via GRIPS — roadmap, belum otomatis.",
    }
