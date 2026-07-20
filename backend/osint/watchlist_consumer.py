"""
Phase 4.5.9 — Watchlist consumer: jalan di infra BANK (tier on-premise).

Polling berkala ke GET /osint/watchlist milik MuleRadar (tier terpusat), lalu
seed rekening baru ke Neo4j LOKAL bank + jalankan PPR + buat alert TERHADAP
graph transaksi nasabah milik bank sendiri. Fungsi seed_to_neo4j() dan
trigger_ppr_alerts() dari seeder.py sudah generic (tidak peduli siapa yang
memanggil) — modul ini hanya menyediakan sumber data (watchlist API, bukan
crawl langsung) dan menjalankannya terhadap DATABASE_URL/NEO4J_URI LOKAL bank.

Data yang masuk lewat modul ini HANYA watchlist publik (nomor rekening +
risk_level) dari MuleRadar. Tidak ada data nasabah yang pernah dikirim
KELUAR dari bank — arah aliran selalu MuleRadar → bank, sesuai keputusan
arsitektur "zero data egress" di PIPELINE.txt Phase 4.5.

Konfigurasi (env, SISI BANK):
    MULERADAR_WATCHLIST_URL  default: http://localhost:8000/osint/watchlist
    MULERADAR_API_KEY        wajib — diterbitkan via osint/api_keys.py::issue_key
                              di sisi MuleRadar, dikirim ke bank via kanal aman
    MULERADAR_CONSUMER_ID    default: "default" (untuk multi-instance di 1 bank)
    (DATABASE_URL, NEO4J_URI, dst — env yang sama dipakai modul lain, tapi
     di deployment bank harus menunjuk ke infra LOKAL bank, bukan MuleRadar)
"""

import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from osint import seeder

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)
WATCHLIST_URL = os.getenv("MULERADAR_WATCHLIST_URL", "http://localhost:8000/osint/watchlist")
API_KEY = os.getenv("MULERADAR_API_KEY", "")
CONSUMER_ID = os.getenv("MULERADAR_CONSUMER_ID", "default")
FETCH_TIMEOUT = int(os.getenv("MULERADAR_WATCHLIST_TIMEOUT", "30"))


def _get_cursor(engine) -> datetime | None:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT last_synced_at FROM osint_watchlist_sync WHERE consumer_id = :cid"),
            {"cid": CONSUMER_ID},
        ).first()
    return row[0] if row and row[0] else None


def _set_cursor(engine, ts: datetime) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO osint_watchlist_sync (consumer_id, last_synced_at, updated_at)
                VALUES (:cid, :ts, :now)
                ON CONFLICT (consumer_id) DO UPDATE SET
                    last_synced_at = EXCLUDED.last_synced_at,
                    updated_at = EXCLUDED.updated_at
            """),
            {"cid": CONSUMER_ID, "ts": ts, "now": datetime.utcnow()},
        )


def fetch_watchlist(since: datetime | None) -> dict:
    """GET ke endpoint watchlist MuleRadar. Melempar exception kalau API key kosong/invalid."""
    if not API_KEY:
        raise RuntimeError(
            "MULERADAR_API_KEY belum diisi — minta API key ke operator MuleRadar "
            "(diterbitkan via osint/api_keys.py::issue_key di sisi mereka)"
        )
    params = {"since": since.isoformat()} if since is not None else {}
    resp = requests.get(
        WATCHLIST_URL,
        params=params,
        timeout=FETCH_TIMEOUT,
        headers={"X-API-Key": API_KEY},
    )
    resp.raise_for_status()
    return resp.json()


def sync_once(engine=None, driver=None) -> dict:
    """
    Satu siklus sync: fetch watchlist baru → seed ke Neo4j lokal bank →
    PPR + alert lokal → majukan cursor. Idempoten: seed_to_neo4j() pakai
    MERGE (aman diulang), trigger_ppr_alerts() sudah dedup alert per akun
    (lihat seeder.py) jadi rekening yang sama tidak membanjiri alert baru.
    """
    own_engine = engine is None
    if own_engine:
        engine = create_engine(DATABASE_URL)
    own_driver = driver is None
    if own_driver:
        driver = seeder.get_driver()

    try:
        since = _get_cursor(engine)
        payload = fetch_watchlist(since)
        items = payload.get("items", [])
        if not items:
            return {"fetched": 0, "seeded": 0, "alerts_created": 0}

        risk_map = {item["rekening"]: item["risk_level"] for item in items}
        rekening_list = list(risk_map.keys())

        seeded = seeder.seed_to_neo4j(risk_map, driver)
        alerts = seeder.trigger_ppr_alerts(rekening_list, driver, engine)

        latest = max(datetime.fromisoformat(item["last_seen"]) for item in items)
        _set_cursor(engine, latest)

        return {"fetched": len(items), "seeded": seeded, "alerts_created": alerts}
    finally:
        if own_driver and driver is not None:
            driver.close()


if __name__ == "__main__":
    try:
        result = sync_once()
    except Exception as exc:
        print(f"[watchlist_consumer] gagal sync: {exc}", file=sys.stderr)
        sys.exit(1)
    print("[watchlist_consumer] hasil sync:", result)
