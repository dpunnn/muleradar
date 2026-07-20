"""
Phase 4.5.3 — Crawler worker pool (Playwright).
Phase 4.5.10 — + proxy 2-tier, verifikasi egress, klasifikasi error, anti-fingerprint.

Mengambil URL dari osint_queue (PENDING, urut priority) dan meng-crawl-nya
secara CONTINUOUS 24/7 dengan pool worker asyncio (bukan cron). Per situs:
cek aktif → navigate ke halaman deposit/cara-bayar → screenshot sebagai bukti →
kembalikan HTML untuk diekstrak (4.5.4).

AKSES SITUS BLOCKLIST KOMINFO (lihat PIPELINE.txt Phase 4.5.10 untuk detail
diskusi): situs sumber sudah diblokir ISP Indonesia di 3 lapis (DNS, SNI/DPI,
IP). User-agent saja TIDAK CUKUP. Modul ini mendukung:
    Tier 1 (wajib untuk produksi): satu proxy tetap — VPS+WireGuard/SOCKS5
        luar negeri, tembus DNS+SNI+IP block. Env: OSINT_PROXY_SERVER
        (+ USERNAME/PASSWORD kalau perlu auth).
    Tier 2 (opsional, skeleton — TIDAK wajib dipakai, budget-dependent):
        proxy residensial untuk retry kasus yang gagal di Tier 1 dengan
        sinyal PROXY_ERROR/BOT_BLOCKED. Env: OSINT_TIER2_PROXY_SERVER dst.
        Kalau env ini tidak diisi, Tier 2 otomatis nonaktif — tidak ada
        biaya/resource tambahan sampai benar-benar dikonfigurasi.
Tanpa proxy sama sekali (default pengembangan lokal): crawler tetap jalan, cocok untuk
sample_blocklist.txt lokal yang tidak butuh akses situs asli.

Playwright (dan playwright-stealth, opsional) adalah dependency berat.
Import-nya di-guard sehingga modul lain yang meng-import package `osint`
tidak crash bila keduanya belum terpasang.

Instalasi (saat siap dipakai):
    pip install playwright
    python -m playwright install chromium
    pip install playwright-stealth   # opsional, lihat catatan _STEALTH_AVAILABLE
"""

import asyncio
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)
SCREENSHOT_DIR = os.getenv(
    "OSINT_SCREENSHOT_DIR",
    os.path.join(os.path.dirname(__file__), "../../data/osint_screenshots"),
)
NAV_TIMEOUT_MS = int(os.getenv("OSINT_NAV_TIMEOUT_MS", "15000"))
MAX_ATTEMPTS = int(os.getenv("OSINT_MAX_ATTEMPTS", "2"))

# --- Proxy Tier 1 (workhorse — lihat catatan modul di atas) ---
PROXY_SERVER = os.getenv("OSINT_PROXY_SERVER")            # mis. "socks5://1.2.3.4:1080"
PROXY_USERNAME = os.getenv("OSINT_PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("OSINT_PROXY_PASSWORD")

# --- Proxy Tier 2 (opsional, escalation — nonaktif kalau env tidak diisi) ---
TIER2_PROXY_SERVER = os.getenv("OSINT_TIER2_PROXY_SERVER")
TIER2_PROXY_USERNAME = os.getenv("OSINT_TIER2_PROXY_USERNAME")
TIER2_PROXY_PASSWORD = os.getenv("OSINT_TIER2_PROXY_PASSWORD")

# Error yang layak dieskalasi ke Tier 2 (kalau tersedia) — proxy Tier 1 gagal
# konek, atau situs jelas menyajikan halaman challenge/anti-bot.
_ESCALATE_ERROR_TYPES = {"PROXY_ERROR", "BOT_BLOCKED"}

# Layanan cek IP publik untuk verifikasi egress (gratis, tanpa API key).
IP_CHECK_URL = os.getenv(
    "OSINT_IP_CHECK_URL",
    "http://ip-api.com/json/?fields=status,countryCode,query",
)

# Kata kunci halaman yang paling mungkin memuat nomor rekening.
_DEPOSIT_KEYWORDS = ("deposit", "cara-bayar", "cara bayar", "bank", "pembayaran", "setor")

# Penanda umum halaman challenge/anti-bot (Cloudflare dkk). Situs bisa balas
# HTTP 200 tapi isinya interstitial, bukan konten asli — kalau lolos dari
# pengecekan status code, extractor akan "berhasil" tapi dapat nol rekening
# tanpa tahu itu karena diblokir, bukan karena situs memang kosong.
_BOT_BLOCK_MARKERS = (
    "checking your browser",
    "just a moment",
    "cf-browser-verification",
    "enable javascript and cookies to continue",
    "attention required! | cloudflare",
    "ddos protection by",
    "verifying you are human",
)

# Pool UA realistis (rotasi per-context, bukan 1 string statis yang bisa
# di-pattern-match dan makin basi seiring waktu karena Chrome asli auto-update).
_UA_POOL = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
_VIEWPORT_POOL = (
    {"width": 1366, "height": 768},
    {"width": 1920, "height": 1080},
    {"width": 1536, "height": 864},
    {"width": 1440, "height": 900},
    {"width": 1280, "height": 720},
)

# Tempel sebagai init script per context: patch sinyal bot-detection paling
# umum. Playwright/Selenium men-set navigator.webdriver=true secara default —
# ini yang PALING sering dicek sistem anti-bot, lebih berdampak dari UA string.
_WEBDRIVER_PATCH_JS = "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"

# Guarded import — Playwright opsional.
try:
    from playwright.async_api import async_playwright  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

# Guarded import — playwright-stealth opsional (lapis tambahan, lihat
# PIPELINE.txt 4.5.10 untuk alasan kenapa ini tidak wajib).
try:
    from playwright_stealth import stealth_async  # type: ignore
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False


@dataclass
class CrawlResult:
    url: str
    status: str                       # DONE, FAILED, SKIP
    http_status: int | None = None
    html_content: str = ""
    screenshot_path: str | None = None
    error: str | None = None
    error_type: str | None = None     # PROXY_ERROR | TIMEOUT | SITE_ERROR | BOT_BLOCKED
    tier: int = 1                     # 1 = proxy utama, 2 = hasil dari eskalasi Tier 2
    extra_urls: list[str] = field(default_factory=list)  # link deposit yang ditemukan


def playwright_available() -> bool:
    return _PLAYWRIGHT_AVAILABLE


def stealth_available() -> bool:
    return _STEALTH_AVAILABLE


def tier2_configured() -> bool:
    return bool(TIER2_PROXY_SERVER)


def _build_proxy_config(server: str | None, username: str | None, password: str | None) -> dict | None:
    """Bentuk dict `proxy=` untuk Playwright. None kalau server tidak diisi (jalan tanpa proxy)."""
    if not server:
        return None
    cfg: dict = {"server": server}
    if username:
        cfg["username"] = username
    if password:
        cfg["password"] = password
    return cfg


def _classify_error(msg: str) -> str:
    """
    Klasifikasi pesan error Playwright/Chromium supaya ops tahu apa yang harus
    dibenahi: tunnel/proxy (PROXY_ERROR) vs situs memang lambat/mati (TIMEOUT/
    SITE_ERROR). Tanpa ini, semua kegagalan terlihat sama — padahal "proxy
    putus" dan "situs judolnya sudah tutup" butuh tindakan yang beda sekali.
    """
    lowered = msg.lower()
    proxy_markers = (
        "err_proxy_connection_failed", "err_tunnel_connection_failed",
        "err_socks_connection_failed", "err_socks_connection_host_unreachable",
        "err_proxy_auth", "proxy connection",
    )
    if any(m in lowered for m in proxy_markers):
        return "PROXY_ERROR"
    if "timeout" in lowered or "err_timed_out" in lowered or "err_connection_timed_out" in lowered:
        return "TIMEOUT"
    return "SITE_ERROR"


def _detect_bot_block(html: str) -> bool:
    """True kalau HTML mengandung penanda halaman challenge/anti-bot umum."""
    lowered = html.lower()
    return any(marker in lowered for marker in _BOT_BLOCK_MARKERS)


# -----------------------------------------------------------------
# Queue access
# -----------------------------------------------------------------

def claim_batch(engine, limit: int) -> list[str]:
    """
    Ambil `limit` URL PENDING dengan prioritas tertinggi dan tandai IN_PROGRESS
    secara atomik (SKIP LOCKED) agar worker paralel tidak mengambil URL yang sama.
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                WITH picked AS (
                    SELECT url FROM osint_queue
                    WHERE status = 'PENDING'
                    ORDER BY priority ASC, queued_at ASC
                    LIMIT :limit
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE osint_queue q
                SET status = 'IN_PROGRESS', attempts = attempts + 1
                FROM picked
                WHERE q.url = picked.url
                RETURNING q.url
            """),
            {"limit": limit},
        ).fetchall()
    return [r[0] for r in rows]


def _finalize(engine, result: CrawlResult) -> None:
    """Tulis hasil crawl ke osint_queue + osint_sites. FAILED bisa di-requeue."""
    now = datetime.utcnow()
    with engine.begin() as conn:
        # Requeue FAILED bila belum melewati batas percobaan.
        if result.status == "FAILED":
            attempts = conn.execute(
                text("SELECT attempts FROM osint_queue WHERE url = :u"),
                {"u": result.url},
            ).scalar() or MAX_ATTEMPTS
            final_status = "PENDING" if attempts < MAX_ATTEMPTS else "FAILED"
        else:
            final_status = result.status

        conn.execute(
            text("""
                UPDATE osint_queue
                SET status = :status, crawled_at = :now
                WHERE url = :url
            """),
            {"status": final_status, "now": now, "url": result.url},
        )
        conn.execute(
            text("""
                INSERT INTO osint_sites (url, status, http_status, screenshot_path,
                    error_type, crawled_at)
                VALUES (:url, :status, :http_status, :shot, :error_type, :now)
                ON CONFLICT (url) DO UPDATE SET
                    status = EXCLUDED.status,
                    http_status = EXCLUDED.http_status,
                    screenshot_path = EXCLUDED.screenshot_path,
                    error_type = EXCLUDED.error_type,
                    crawled_at = EXCLUDED.crawled_at
            """),
            {
                "url": result.url,
                "status": result.status,
                "http_status": result.http_status,
                "shot": result.screenshot_path,
                "error_type": result.error_type,
                "now": now,
            },
        )


# -----------------------------------------------------------------
# Single-site crawl
# -----------------------------------------------------------------

async def crawl_one(browser, url: str, tier: int = 1) -> CrawlResult:
    """
    Crawl satu URL. Membuat CONTEXT baru per panggilan (bukan reuse context
    global) — supaya tiap request punya fingerprint sendiri (UA + viewport
    acak dari pool, lihat modul docstring), bukan satu identitas statis yang
    dipakai berulang jutaan kali. Context di Playwright ringan (berbagi
    proses browser yang sama), jadi ini bukan overhead besar walau dibuat
    ulang tiap crawl.
    """
    context = await browser.new_context(
        user_agent=random.choice(_UA_POOL),
        viewport=random.choice(_VIEWPORT_POOL),
        ignore_https_errors=True,   # situs judol sering SSL kadaluarsa
    )
    await context.add_init_script(_WEBDRIVER_PATCH_JS)

    page = await context.new_page()
    page.set_default_timeout(NAV_TIMEOUT_MS)
    if _STEALTH_AVAILABLE:
        try:
            await stealth_async(page)
        except Exception:
            pass  # lapis tambahan opsional — kegagalan di sini tidak boleh gugurkan crawl

    try:
        response = await page.goto(url, wait_until="domcontentloaded")
        http_status = response.status if response else None

        # Situs mati / diblokir DNS → SKIP (bukan FAILED, tidak perlu retry).
        if http_status is not None and http_status >= 400:
            return CrawlResult(url=url, status="SKIP", http_status=http_status, tier=tier)

        html = await page.content()
        bot_blocked = _detect_bot_block(html)

        # Coba temukan & ikuti link ke halaman deposit (rekening biasanya di sana).
        extra_urls = await _find_deposit_links(page)

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        shot_path = os.path.join(SCREENSHOT_DIR, _safe_filename(url) + ".png")
        await page.screenshot(path=shot_path, full_page=True)

        return CrawlResult(
            url=url,
            status="DONE",
            http_status=http_status,
            html_content=html,
            screenshot_path=shot_path,
            error_type="BOT_BLOCKED" if bot_blocked else None,
            tier=tier,
            extra_urls=extra_urls,
        )
    except Exception as exc:  # timeout, SSL, proxy, navigation error
        msg = str(exc)[:200]
        return CrawlResult(url=url, status="FAILED", error=msg,
                            error_type=_classify_error(msg), tier=tier)
    finally:
        await page.close()
        await context.close()


async def _find_deposit_links(page) -> list[str]:
    """Kumpulkan href yang teksnya mengandung kata kunci deposit/pembayaran."""
    try:
        hrefs = await page.eval_on_selector_all(
            "a",
            """els => els.map(e => ({href: e.href, text: (e.textContent||'').toLowerCase()}))""",
        )
    except Exception:
        return []
    found = []
    for item in hrefs:
        text_l = item.get("text", "")
        href = item.get("href", "")
        if href and any(kw in text_l for kw in _DEPOSIT_KEYWORDS):
            found.append(href)
    return found[:3]  # cukup beberapa, hindari ledakan URL


def _safe_filename(url: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in url)
    return keep[:120]


# -----------------------------------------------------------------
# Verifikasi egress (proxy benar-benar jalan, bukan diam-diam fallback)
# -----------------------------------------------------------------

async def verify_egress(browser) -> dict:
    """
    Cek IP yang benar-benar dipakai browser untuk keluar internet (lewat
    proxy kalau dikonfigurasi). PENTING: proxy yang salah setting bisa
    diam-diam gagal tanpa exception apapun — request tetap "berhasil" tapi
    sebenarnya lewat IP asli (Indonesia), yang berarti blocklist Kominfo
    TIDAK tembus. Verifikasi ini memastikan operator tahu SEBELUM crawl
    jutaan URL sia-sia karena proxy ternyata tidak jalan.

    Tidak fatal kalau gagal — layanan cek IP eksternal bisa saja unreachable
    tanpa itu berarti proxy-nya sendiri bermasalah, jadi hanya WARNING.
    """
    context = await browser.new_context(ignore_https_errors=True)
    page = await context.new_page()
    try:
        resp = await page.goto(IP_CHECK_URL, timeout=10000)
        data = await resp.json()
        ip = data.get("query")
        country = data.get("countryCode")
        is_indonesia = country == "ID"
        return {
            "ok": data.get("status") == "success" and not is_indonesia,
            "ip": ip,
            "country": country,
            "warning": (
                f"Egress IP masih Indonesia ({ip}) — proxy TIDAK jalan, "
                f"blocklist Kominfo tidak akan tembus" if is_indonesia else None
            ),
        }
    except Exception as exc:
        return {
            "ok": None, "ip": None, "country": None,
            "warning": f"Gagal cek egress IP ({exc}) — layanan cek mungkin unreachable, "
                       f"belum tentu proxy bermasalah",
        }
    finally:
        await page.close()
        await context.close()


# -----------------------------------------------------------------
# Worker pool (continuous)
# -----------------------------------------------------------------

async def run_pool(
    workers: int = 10,
    poll_interval: float = 5.0,
    once: bool = False,
    on_result=None,
) -> None:
    """
    Jalankan pool crawler CONTINUOUS.

    workers       : jumlah crawl paralel (pengembangan laptop ~10, server ~50).
    poll_interval : jeda saat queue kosong sebelum cek lagi (detik).
    once          : True → proses satu batch lalu berhenti (untuk test/manual trigger).
    on_result     : callback(CrawlResult) opsional — dipakai seeder untuk pipeline
                    crawl→extract→seed. Kalau None, hasil hanya ditulis ke DB.

    Proxy Tier 1 dipasang di level browser.launch() (semua context mewarisi
    satu jalur keluar). Tier 2 (kalau dikonfigurasi) adalah BROWSER TERPISAH
    dengan proxy berbeda, dipakai HANYA untuk retry URL yang gagal dengan
    error_type PROXY_ERROR/BOT_BLOCKED di Tier 1 — bukan untuk semua request
    (menjaga biaya proxy residensial tetap proporsional).
    """
    if not _PLAYWRIGHT_AVAILABLE:
        print(
            "[crawler] Playwright belum terpasang. Jalankan:\n"
            "    pip install playwright && python -m playwright install chromium",
            file=sys.stderr,
        )
        return

    engine = create_engine(DATABASE_URL)
    sem = asyncio.Semaphore(workers)
    proxy_tier1 = _build_proxy_config(PROXY_SERVER, PROXY_USERNAME, PROXY_PASSWORD)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, proxy=proxy_tier1)

        if proxy_tier1 is not None:
            egress = await verify_egress(browser)
            if egress["warning"]:
                print(f"[crawler] PERINGATAN EGRESS: {egress['warning']}", file=sys.stderr)
            else:
                print(f"[crawler] Egress terverifikasi: IP={egress['ip']} negara={egress['country']}")
        else:
            print(
                "[crawler] Jalan TANPA proxy (OSINT_PROXY_SERVER kosong) — "
                "situs yang di-blocklist Kominfo via DNS/SNI/IP TIDAK akan terjangkau. "
                "Ini normal untuk pengembangan lokal dengan sample_blocklist.txt.",
                file=sys.stderr,
            )

        browser_tier2 = None
        proxy_tier2 = _build_proxy_config(TIER2_PROXY_SERVER, TIER2_PROXY_USERNAME, TIER2_PROXY_PASSWORD)
        if proxy_tier2 is not None:
            browser_tier2 = await pw.chromium.launch(headless=True, proxy=proxy_tier2)
            print("[crawler] Tier 2 (proxy eskalasi) aktif.")

        async def worker(url: str):
            async with sem:
                result = await crawl_one(browser, url, tier=1)
                if result.error_type in _ESCALATE_ERROR_TYPES and browser_tier2 is not None:
                    result_tier2 = await crawl_one(browser_tier2, url, tier=2)
                    if result_tier2.status == "DONE" and result_tier2.error_type is None:
                        result = result_tier2
                _finalize(engine, result)
                if on_result is not None:
                    on_result(result)

        try:
            while True:
                batch = claim_batch(engine, workers)
                if not batch:
                    if once:
                        break
                    await asyncio.sleep(poll_interval)
                    continue
                await asyncio.gather(*(worker(u) for u in batch))
                if once:
                    break
        finally:
            await browser.close()
            if browser_tier2 is not None:
                await browser_tier2.close()


def get_status(engine=None) -> dict:
    """Ringkasan queue untuk endpoint /osint/status."""
    if engine is None:
        engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT status, COUNT(*) FROM osint_queue GROUP BY status
        """)).fetchall()
    counts = {r[0]: r[1] for r in rows}
    return {
        "pending": counts.get("PENDING", 0),
        "in_progress": counts.get("IN_PROGRESS", 0),
        "done": counts.get("DONE", 0),
        "failed": counts.get("FAILED", 0),
        "skip": counts.get("SKIP", 0),
        "playwright_available": _PLAYWRIGHT_AVAILABLE,
        "stealth_available": _STEALTH_AVAILABLE,
        "proxy_tier1_configured": bool(PROXY_SERVER),
        "proxy_tier2_configured": tier2_configured(),
    }


if __name__ == "__main__":
    # Manual: proses satu batch lalu berhenti.
    asyncio.run(run_pool(workers=int(os.getenv("OSINT_WORKERS", "10")), once=True))
