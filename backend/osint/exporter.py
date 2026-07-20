"""
Phase 4.5.9 — Exporter: watchlist rekening publik untuk konsumsi bank on-premise.

Jalan di TIER TERPUSAT (infra MuleRadar). Expose isi osint_accounts yang
sudah diverifikasi PUBLIK (rekening yang ditemukan di situs judol blocklist
Kominfo) — TIDAK ADA data nasabah/transaksi bank di modul ini sama sekali.
Konsumen: watchlist_consumer.py yang jalan di infra bank (tier on-premise).

Field yang SENGAJA tidak diekspos ke feed ini: screenshot_path, daftar
sumber_url mentah — itu bukti forensik internal, tetap bisa diquery analis
lewat GET /osint/accounts/{rekening} di sisi terpusat. Bank hanya perlu
nomor rekening + level risiko untuk seed graph, bukan detail crawl.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from osint.network import risk_from_site_count

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)


def get_watchlist(since: datetime | None, engine=None, limit: int = 5000) -> list[dict]:
    """
    Rekening OSINT dengan last_seen >= `since` (atau semua kalau since=None),
    diurutkan last_seen ASC menaik — supaya konsumen bisa majukan cursor
    dengan aman dari last_seen item TERAKHIR yang berhasil diproses, walau
    proses terhenti di tengah batch (tidak melompati sisa data yang belum
    sempat diambil).

    SENGAJA pakai >= bukan > : seeder.persist_accounts() menangkap satu
    `now()` untuk SELURUH rekening dari satu situs (bukan per-rekening),
    jadi banyak baris osint_accounts bisa punya last_seen IDENTIK. Kalau
    LIMIT kebetulan memotong tepat di tengah grup timestamp yang sama dan
    query pakai `>`, rekening di sisi lain potongan itu TIDAK AKAN PERNAH
    match lagi di query berikutnya — hilang permanen dari watchlist bank.
    Dengan `>=`, item tepat di batas cursor mungkin ke-fetch ulang sesekali,
    tapi itu harmless: seed_to_neo4j() pakai MERGE (idempoten) dan
    trigger_ppr_alerts() sudah dedup alert yang masih terbuka (seeder.py).
    Redundant-tapi-aman jauh lebih baik daripada silent data loss di sistem
    deteksi fraud.

    risk_level dihitung ON-THE-FLY dari shared_count (risk_from_site_count),
    BUKAN dibaca dari osint_networks — tabel itu snapshot periodik dari
    network.detect() yang bisa sedikit basi; shared_count di osint_accounts
    selalu terkini (di-update tiap kali seeder.persist_accounts jalan).
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)

    clause = "WHERE last_seen >= :since" if since is not None else ""
    params: dict = {"limit": limit}
    if since is not None:
        params["since"] = since

    with engine.connect() as conn:
        rows = conn.execute(
            text(f"""
                SELECT rekening, bank, account_type, shared_count, confidence, last_seen
                FROM osint_accounts
                {clause}
                ORDER BY last_seen ASC
                LIMIT :limit
            """),
            params,
        ).mappings().all()

    items = []
    for r in rows:
        item = dict(r)
        item["risk_level"] = risk_from_site_count(item["shared_count"])
        item["last_seen"] = item["last_seen"].isoformat()
        items.append(item)
    return items
