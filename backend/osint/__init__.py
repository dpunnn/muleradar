"""
MuleRadar Phase 4.5 — OSINT Intelligence Module.

Proaktif hunting rekening judol dari situs publik (Kominfo blocklist),
ekstrak nomor rekening, deteksi jaringan bandar, dan seed ke Neo4j sebagai
starting node untuk graph tracing.

Alur:
    kominfo_sync  → isi osint_queue dari blocklist (1x/hari)
    crawler       → ambil URL dari queue, screenshot (24/7 worker pool)
    extractor     → regex + OCR rekening dari HTML/gambar
    network       → deteksi rekening yang dipakai lintas situs (bandar)
    seeder        → simpan ke PostgreSQL + node :OsintAccount di Neo4j + trigger PPR

Modul crawler (Playwright) dan OCR (Tesseract) adalah dependency opsional yang
berat; import-nya di-guard agar package tetap bisa dimuat untuk API/testing
tanpa kedua paket tersebut terpasang.
"""

__all__ = [
    "kominfo_sync",
    "extractor",
    "network",
    "seeder",
]
