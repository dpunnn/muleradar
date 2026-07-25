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
import time
import logging
import threading
from datetime import datetime

logger = logging.getLogger("osint.crawl")

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query
from sqlalchemy import create_engine, text

from osint import api_keys, crawler, exporter, extractor, network, seeder

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

router = APIRouter(prefix="/osint", tags=["osint"])
_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# --- State crawler continuous (tombol ON/OFF di UI, Phase 4.5.12) ---
# Bukan 24/7 paksa: user kontrol via UI. ON -> thread loop konsumsi queue;
# OFF -> _stop_event.set(), worker selesai batch berjalan lalu berhenti.
_crawl_thread: threading.Thread | None = None
_stop_event = threading.Event()
_start_lock = threading.Lock()   # cegah dua /start bikin dua thread (QC fix 21-Jul)
_crawl_state = {"running": False, "started_at": None, "batches_done": 0,
                "last_batch_at": None, "seed_failures": 0}


def _safe_seed(url: str, accounts: list, driver) -> bool:
    """Seed hasil crawl ke graph — TAHAN GAGAL (QC fix 21-Jul). Kegagalan Neo4j
    / seed pada SATU situs tak boleh mematikan thread crawl (mirror prinsip
    resilience consumer, Phase 15.1). Rekening tetap tercatat di Postgres oleh
    extractor+_finalize; kalau seed gagal, bisa di-seed ulang nanti via
    /osint/seed-all. Return True kalau seed sukses."""
    try:
        seeder.seed_site_results(url, accounts, engine=_engine, driver=driver)
        return True
    except Exception as e:
        _crawl_state["seed_failures"] = _crawl_state.get("seed_failures", 0) + 1
        logger.warning("seed_site_results gagal utk %s (%s) — lanjut crawl, "
                       "rekening bisa di-seed ulang via /osint/seed-all", url, e)
        return False


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
        if result.status != "DONE":
            return
        # Satu kesatuan: rekening dari AGENT (register->deposit) diutamakan;
        # fallback ke ekstraksi pasif HTML kalau agent tak jalan/kosong.
        accounts = result.agent_accounts or (
            extractor.extract(result.html_content, result.screenshot_path)
            if result.html_content else [])
        if accounts:
            _safe_seed(result.url, accounts, driver)

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


# -----------------------------------------------------------------
# Crawler CONTINUOUS — tombol ON/OFF (Phase 4.5.12)
# -----------------------------------------------------------------

def _continuous_crawl_loop(workers: int, poll_interval: int = 10) -> None:
    """Loop hingga _stop_event di-set: proses batch queue, lalu tunggu queue
    terisi lagi (dari kominfo_sync). Reuse run_pool(once=True) per batch supaya
    stop bersifat GRACEFUL (berhenti di antara batch, tak memutus crawl aktif)."""
    driver = seeder.get_driver()

    def _on_result(result: "crawler.CrawlResult") -> None:
        if result.status != "DONE":
            return
        # Satu kesatuan: rekening dari AGENT (register->deposit) diutamakan;
        # fallback ke ekstraksi pasif HTML kalau agent tak jalan/kosong.
        accounts = result.agent_accounts or (
            extractor.extract(result.html_content, result.screenshot_path)
            if result.html_content else [])
        if accounts:
            _safe_seed(result.url, accounts, driver)

    try:
        while not _stop_event.is_set():
            with _engine.connect() as conn:
                pending = conn.execute(
                    text("SELECT count(*) FROM osint_queue WHERE status='PENDING'")
                ).scalar() or 0
            if pending == 0:
                # queue kosong — tunggu (cek stop tiap detik supaya OFF responsif)
                for _ in range(poll_interval):
                    if _stop_event.is_set():
                        break
                    time.sleep(1)
                continue
            asyncio.run(crawler.run_pool(workers=workers, once=True, on_result=_on_result))
            # network.detect di-guard (QC fix 21-Jul): kegagalan deteksi jaringan
            # bandar (Postgres) tak boleh mematikan thread crawl continuous.
            try:
                network.detect(_engine)
            except Exception as e:
                logger.warning("network.detect gagal (%s) — lanjut crawl", e)
            _crawl_state["batches_done"] += 1
            _crawl_state["last_batch_at"] = datetime.utcnow().isoformat()
    except Exception as e:  # jangan biarkan thread mati diam-diam
        _crawl_state["error"] = str(e)[:200]
    finally:
        if driver is not None:
            driver.close()
        _crawl_state["running"] = False


@router.post("/start")
def start_crawl(workers: int = Query(5, ge=1, le=20)):
    """Tombol ON: mulai crawler continuous (konsumsi queue terus sampai OFF)."""
    global _crawl_thread
    if not crawler.playwright_available():
        raise HTTPException(status_code=503,
            detail="Playwright belum terpasang. Jalankan: pip install playwright "
                   "&& python -m playwright install chromium")
    # HARDENING KEAMANAN (4.5.12): crawler continuous TANPA proxy = IP asli
    # bocor berulang ke situs judol. Tolak start kalau proxy belum di-set —
    # cegah kebocoran, konsisten dgn prinsip verify-egress 4.5.11. (Crawl batch
    # manual /crawl tetap boleh tanpa proxy utk pengembangan lokal sample.)
    if not crawler.PROXY_SERVER:
        raise HTTPException(status_code=428,
            detail="OSINT_PROXY_SERVER belum di-set. Crawler continuous WAJIB "
                   "lewat proxy (mis. Tor socks5://127.0.0.1:9050) supaya IP asli "
                   "tak bocor ke situs target. Set env lalu restart backend.")
    # Lock (QC fix 21-Jul): tanpa ini dua POST /start hampir bersamaan bisa
    # sama-sama lolos cek is_alive() lalu bikin DUA thread crawl -> dua browser
    # -> boros RAM/OOM + thread pertama jadi orphan (global cuma track terakhir).
    with _start_lock:
        if _crawl_thread is not None and _crawl_thread.is_alive():
            return {"status": "already_running", "state": _crawl_state}
        _stop_event.clear()
        _crawl_state.update(running=True, started_at=datetime.utcnow().isoformat(),
                            batches_done=0, last_batch_at=None, seed_failures=0)
        _crawl_state.pop("error", None)
        _crawl_thread = threading.Thread(target=_continuous_crawl_loop,
                                         args=(workers,), daemon=True)
        _crawl_thread.start()
    return {"status": "started", "mode": "continuous", "workers": workers,
            "message": "Crawler ON — konsumsi queue terus sampai /osint/stop."}


@router.post("/stop")
def stop_crawl():
    """Tombol OFF: hentikan crawler continuous (graceful — selesaikan batch aktif)."""
    if _crawl_thread is None or not _crawl_thread.is_alive():
        return {"status": "not_running"}
    _stop_event.set()
    return {"status": "stopping",
            "message": "Crawler OFF — worker berhenti setelah batch berjalan selesai."}


@router.get("/status")
def osint_status():
    """Ukuran queue per status + ketersediaan Playwright/OCR + state crawler
    ON/OFF (untuk tombol toggle di UI, Phase 4.5.12)."""
    status = crawler.get_status(_engine)
    status["ocr_available"] = extractor.ocr_available()
    # State crawler continuous (ON/OFF) + statistik untuk UI
    running = _crawl_thread is not None and _crawl_thread.is_alive()
    with _engine.connect() as conn:
        done_today = conn.execute(text(
            "SELECT count(*) FROM osint_sites WHERE crawled_at::date = CURRENT_DATE"
        )).scalar() or 0
    status["crawling"] = "ON" if running else "OFF"
    status["crawl_state"] = {**_crawl_state, "running": running}
    status["done_today"] = done_today
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
