"""
Phase 4.5.2 — Sinkronisasi Kominfo blocklist → osint_queue.

Kominfo mempublikasikan daftar URL situs judol yang diblokir (TrustPositif).
Tugas modul ini: ambil daftar URL terbaru, masukkan yang belum ada ke
osint_queue dengan prioritas (situs terbaru diblokir = paling depan).

Bukan crawl — hanya mengisi antrean. Crawler (4.5.3) yang mengonsumsi 24/7.
Dijadwalkan 1x/hari (APScheduler / cron eksternal memanggil main()).

Sumber blocklist bersifat configurable via env KOMINFO_BLOCKLIST_URL supaya
mudah diarahkan ke mirror/file lokal saat pengembangan (tanpa akses internet ke TrustPositif).
Format yang didukung: satu URL/domain per baris (plain text), '#' = komentar.
"""

import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)
# Default menunjuk ke file lokal agar pengembangan lokal tidak butuh internet.
# Di produksi arahkan ke feed TrustPositif Kominfo.
BLOCKLIST_URL = os.getenv(
    "KOMINFO_BLOCKLIST_URL",
    os.path.join(os.path.dirname(__file__), "sample_blocklist.txt"),
)
FETCH_TIMEOUT = int(os.getenv("KOMINFO_FETCH_TIMEOUT", "30"))


def fetch_blocklist(source: str = BLOCKLIST_URL) -> list[str]:
    """
    Ambil daftar URL dari `source`. Mendukung http(s):// maupun path file lokal.
    Kembalikan list URL ternormalisasi (ada skema, tanpa duplikat, urut input).
    """
    if source.startswith(("http://", "https://")):
        resp = requests.get(source, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        raw = resp.text
    else:
        if not os.path.exists(source):
            print(f"[kominfo_sync] blocklist tidak ditemukan: {source}", file=sys.stderr)
            return []
        with open(source, "r", encoding="utf-8") as f:
            raw = f.read()

    seen: set[str] = set()
    urls: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url = _normalize_url(line)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _normalize_url(entry: str) -> str:
    """Domain telanjang → https://; buang trailing slash. Return '' kalau invalid."""
    entry = entry.split()[0]  # buang kolom tambahan bila feed berformat tabel
    if not entry:
        return ""
    if not entry.startswith(("http://", "https://")):
        entry = "https://" + entry
    return entry.rstrip("/")


def sync_to_queue(urls: list[str], engine=None) -> dict:
    """
    Masukkan URL baru ke osint_queue. Prioritas berurut sesuai posisi di feed
    (index 0 = paling baru diblokir = priority terkecil = didahulukan crawler).

    URL yang sudah ada di queue di-skip (ON CONFLICT DO NOTHING) — status crawl
    lama tidak direset.
    """
    if engine is None:
        engine = create_engine(DATABASE_URL)
    if not urls:
        return {"fetched": 0, "inserted": 0, "skipped": 0}

    inserted = 0
    with engine.begin() as conn:
        for idx, url in enumerate(urls):
            result = conn.execute(
                text("""
                    INSERT INTO osint_queue (url, priority, status, queued_at)
                    VALUES (:url, :priority, 'PENDING', :queued_at)
                    ON CONFLICT (url) DO NOTHING
                """),
                {"url": url, "priority": idx, "queued_at": datetime.utcnow()},
            )
            inserted += result.rowcount or 0

    return {
        "fetched": len(urls),
        "inserted": inserted,
        "skipped": len(urls) - inserted,
    }


def main() -> dict:
    engine = create_engine(DATABASE_URL)
    print(f"[kominfo_sync] fetch blocklist dari: {BLOCKLIST_URL}")
    urls = fetch_blocklist()
    stats = sync_to_queue(urls, engine)
    print(
        f"[kominfo_sync] selesai — fetched={stats['fetched']} "
        f"inserted={stats['inserted']} skipped={stats['skipped']}"
    )
    return stats


if __name__ == "__main__":
    main()
