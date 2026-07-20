"""
Phase 4.5.9 — Manajemen API key untuk endpoint watchlist OSINT.

Jalan di TIER TERPUSAT (infra MuleRadar). Setiap bank klien mendapat satu API
key untuk mengonsumsi GET /osint/watchlist (dipakai watchlist_consumer.py di
sisi bank). Ini otentikasi SERVICE-TO-SERVICE (machine), bukan JWT sesi
analis (Phase 11) — dua hal yang sengaja dipisah karena beda aktor & beda
siklus hidup credential.

Key HANYA disimpan sebagai hash SHA-256 — plaintext ditampilkan SATU KALI
saat issue_key() dipanggil, sama seperti pola personal access token
GitHub/Stripe. SHA-256 cukup di sini (bukan bcrypt/argon2) karena key adalah
token acak berentropi tinggi (32 byte random), bukan password pilihan
manusia yang rentan brute-force kamus.
"""

import hashlib
import os
import secrets
import sys
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def issue_key(bank_id: str, engine=None) -> str:
    """
    Terbitkan API key baru untuk bank_id. Return PLAINTEXT sekali — simpan
    sekarang di sisi bank (mis. sebagai MULERADAR_API_KEY di .env mereka),
    tidak bisa ditampilkan ulang setelah ini (hanya hash yang tersimpan).
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)
    plaintext = f"mr_{secrets.token_urlsafe(32)}"
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO osint_api_keys (bank_id, key_hash, is_active, created_at)
                VALUES (:bank_id, :hash, TRUE, :now)
            """),
            {"bank_id": bank_id, "hash": _hash_key(plaintext), "now": datetime.utcnow()},
        )
    return plaintext


def authenticate(plaintext: str, engine=None) -> str | None:
    """Return bank_id kalau key valid & aktif, None kalau tidak — dipakai endpoint /osint/watchlist."""
    if not plaintext:
        return None
    if engine is None:
        engine = create_engine(DATABASE_URL)
    key_hash = _hash_key(plaintext)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT bank_id FROM osint_api_keys WHERE key_hash = :hash AND is_active = TRUE"),
            {"hash": key_hash},
        ).first()
        if row is None:
            return None
        conn.execute(
            text("UPDATE osint_api_keys SET last_used_at = :now WHERE key_hash = :hash"),
            {"now": datetime.utcnow(), "hash": key_hash},
        )
    return row[0]


def revoke_key(bank_id: str, engine=None) -> int:
    """Nonaktifkan semua key aktif milik bank_id. Return jumlah key yang di-revoke."""
    if engine is None:
        engine = create_engine(DATABASE_URL)
    with engine.begin() as conn:
        result = conn.execute(
            text("UPDATE osint_api_keys SET is_active = FALSE WHERE bank_id = :bank_id AND is_active = TRUE"),
            {"bank_id": bank_id},
        )
    return result.rowcount or 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m osint.api_keys <bank_id>", file=sys.stderr)
        sys.exit(1)
    bank_id = sys.argv[1]
    key = issue_key(bank_id)
    print(f"API key untuk bank_id='{bank_id}' (SIMPAN SEKARANG — tidak akan ditampilkan lagi):")
    print(key)
