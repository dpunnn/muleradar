"""
Phase 4.5.7 — REST endpoints untuk OSINT Intelligence Module.
Phase 4.5.9 — + endpoint watchlist untuk konsumsi bank (tier terpusat).

Endpoint:
    POST /osint/crawl          → trigger crawl batch (background) crawl→extract→seed
    GET  /osint/status         → status worker + ukuran queue
    GET  /osint/queue          → daftar URL di queue + progress
    GET  /osint/accounts       → daftar rekening ditemukan
    GET  /osint/accounts/{rek} → detail rekening + situs + screenshot bukti
    GET  /osint/networks       → daftar jaringan bandar terdeteksi
    POST /osint/seed/{rek}     → seed satu rekening ke Neo4j graph
    POST /osint/seed-all       → seed semua rekening belum-di-graph sekaligus
    GET  /osint/watchlist      → export watchlist publik untuk watchlist_consumer.py
                                  sisi bank (auth API key, BUKAN JWT dashboard)

Modul crawl (Playwright) dijalankan sebagai BackgroundTask agar request tidak
blocking. Bila Playwright belum terpasang, /osint/crawl mengembalikan 503 jelas.
"""

import asyncio
import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from sqlalchemy import create_engine, text

from osint import api_keys, crawler, exporter, extractor, network, seeder

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/osint", tags=["osint"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)


# -----------------------------------------------------------------
# Pipeline orchestration (crawl → extract → seed) untuk background task
# -----------------------------------------------------------------

def _run_crawl_pipeline(workers: int) -> None:
    """
    Proses satu batch queue: crawl tiap URL, ekstrak rekening, seed ke graph.
    Dijalankan di thread background (BackgroundTasks) sehingga boleh blocking.
    """
    driver = seeder.get_driver()

    def _on_result(result: "crawler.CrawlResult") -> None:
        if result.status != "DONE" or not result.html_content:
            return
        accounts = extractor.extract(result.html_content, result.screenshot_path)
        if accounts:
            seeder.seed_site_results(result.url, accounts, engine=_engine, driver=driver)

    try:
        asyncio.run(crawler.run_pool(workers=workers, once=True, on_result=_on_result))
        # Setelah batch selesai, refresh deteksi jaringan bandar.
        network.detect(_engine)
    finally:
        if driver is not None:
            driver.close()


@router.post("/crawl")
def trigger_crawl(background_tasks: BackgroundTasks,
                  workers: int = Query(10, ge=1, le=50)):
    """Trigger manual satu batch crawl (background). Butuh Playwright terpasang."""
    if not crawler.playwright_available():
        raise HTTPException(
            status_code=503,
            detail="Playwright belum terpasang. Jalankan: pip install playwright "
                   "&& python -m playwright install chromium",
        )
    background_tasks.add_task(_run_crawl_pipeline, workers)
    return {"status": "started", "workers": workers,
            "message": "Crawl batch berjalan di background (crawl→extract→seed)."}


@router.get("/status")
def osint_status():
    """Ukuran queue per status + ketersediaan Playwright/OCR."""
    status = crawler.get_status(_engine)
    status["ocr_available"] = extractor.ocr_available()
    return status


@router.get("/queue")
def list_queue(status: str | None = Query(None, description="Filter: PENDING/DONE/FAILED/SKIP"),
               limit: int = Query(100, ge=1, le=1000),
               offset: int = Query(0, ge=0)):
    """Daftar URL di queue beserta progress crawl."""
    clause = "WHERE status = :status" if status else ""
    params = {"limit": limit, "offset": offset}
    if status:
        params["status"] = status
    with _engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT url, priority, status, attempts, queued_at, crawled_at
            FROM osint_queue
            {clause}
            ORDER BY priority ASC, queued_at ASC
            LIMIT :limit OFFSET :offset
        """), params).mappings().all()
    return {"count": len(rows), "items": [dict(r) for r in rows]}


@router.get("/accounts")
def list_accounts(min_shared: int = Query(1, ge=1, description="Filter shared_count minimum"),
                  limit: int = Query(100, ge=1, le=1000),
                  offset: int = Query(0, ge=0)):
    """Daftar rekening ditemukan, terurut dari yang paling banyak dipakai lintas situs."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT rekening, bank, account_type, shared_count, confidence,
                   seeded_to_graph, first_seen, last_seen
            FROM osint_accounts
            WHERE shared_count >= :min_shared
            ORDER BY shared_count DESC, last_seen DESC
            LIMIT :limit OFFSET :offset
        """), {"min_shared": min_shared, "limit": limit, "offset": offset}).mappings().all()
    return {"count": len(rows), "items": [dict(r) for r in rows]}


@router.get("/accounts/{rekening}")
def account_detail(rekening: str):
    """Detail satu rekening: bank, situs sumber, screenshot bukti pertama."""
    with _engine.connect() as conn:
        acc = conn.execute(text("""
            SELECT rekening, bank, account_type, sumber_url, shared_count,
                   confidence, seeded_to_graph, first_seen, last_seen
            FROM osint_accounts WHERE rekening = :rek
        """), {"rek": rekening}).mappings().first()
        if acc is None:
            raise HTTPException(status_code=404, detail="Rekening tidak ditemukan")

        acc = dict(acc)
        sites = list(acc.get("sumber_url") or [])
        screenshots = []
        if sites:
            shot_rows = conn.execute(text("""
                SELECT url, screenshot_path, http_status, crawled_at
                FROM osint_sites WHERE url = ANY(:urls)
            """), {"urls": sites}).mappings().all()
            screenshots = [dict(r) for r in shot_rows]

    acc["sites"] = screenshots
    return acc


@router.get("/networks")
def list_networks(refresh: bool = Query(False, description="Jalankan ulang deteksi jaringan"),
                  risk: str | None = Query(None, description="Filter: HIGH/MED")):
    """Daftar jaringan bandar (rekening yang dipakai lintas situs)."""
    summary = None
    if refresh:
        summary = network.detect(_engine)
    clause = "WHERE risk_level = :risk" if risk else ""
    params = {"risk": risk} if risk else {}
    with _engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT network_id, rekening_list, site_list, risk_level, detected_at
            FROM osint_networks
            {clause}
            ORDER BY (risk_level = 'HIGH') DESC, cardinality(site_list) DESC
        """), params).mappings().all()
    return {"refreshed": summary, "count": len(rows),
            "items": [dict(r) for r in rows]}


@router.post("/seed/{rekening}")
def seed_one(rekening: str):
    """Seed satu rekening ke Neo4j graph secara manual + trigger PPR/alert."""
    with _engine.connect() as conn:
        exists = conn.execute(text(
            "SELECT 1 FROM osint_accounts WHERE rekening = :rek"
        ), {"rek": rekening}).first()
    if exists is None:
        raise HTTPException(status_code=404, detail="Rekening tidak ditemukan")

    driver = seeder.get_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Driver Neo4j tidak tersedia")
    try:
        risk_map = seeder.lookup_risk_levels([rekening], _engine)
        seeded = seeder.seed_to_neo4j(risk_map, driver)
        seeder._mark_seeded([rekening], _engine)
        alerts = seeder.trigger_ppr_alerts([rekening], driver, _engine)
    finally:
        driver.close()
    return {"rekening": rekening, "seeded": seeded, "alerts_created": alerts}


@router.post("/seed-all")
def seed_all():
    """Seed semua rekening yang belum masuk graph sekaligus."""
    with _engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT rekening FROM osint_accounts WHERE seeded_to_graph = FALSE
        """)).fetchall()
    rekening_list = [r[0] for r in rows]
    if not rekening_list:
        return {"seeded": 0, "alerts_created": 0, "message": "Tidak ada rekening baru."}

    driver = seeder.get_driver()
    if driver is None:
        raise HTTPException(status_code=503, detail="Driver Neo4j tidak tersedia")
    try:
        risk_map = seeder.lookup_risk_levels(rekening_list, _engine)
        seeded = seeder.seed_to_neo4j(risk_map, driver)
        seeder._mark_seeded(rekening_list, _engine)
        alerts = seeder.trigger_ppr_alerts(rekening_list, driver, _engine)
    finally:
        driver.close()
    return {"seeded": seeded, "alerts_created": alerts}


@router.get("/watchlist")
def watchlist(
    since: str | None = Query(None, description="ISO 8601 timestamp; hanya rekening baru/updated setelahnya"),
    limit: int = Query(5000, ge=1, le=20000),
    x_api_key: str = Header(..., alias="X-API-Key", description="API key per-bank (osint/api_keys.py)"),
):
    """
    Export watchlist rekening publik untuk watchlist_consumer.py di sisi bank
    (tier on-premise). Endpoint SERVICE-TO-SERVICE — otentikasi via API key
    per-bank, BUKAN JWT sesi analis (Phase 11), karena konsumennya proses
    machine-to-machine, bukan user login ke dashboard.
    """
    bank_id = api_keys.authenticate(x_api_key, _engine)
    if bank_id is None:
        raise HTTPException(status_code=401, detail="API key tidak valid atau tidak aktif")

    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(status_code=400, detail="Parameter since harus format ISO 8601")

    items = exporter.get_watchlist(since_dt, _engine, limit)
    return {
        "bank_id": bank_id,
        "server_time": datetime.utcnow().isoformat(),
        "count": len(items),
        "items": items,
    }
