"""
Phase 4.5.4 — Ekstraksi nomor rekening dari konten situs judol.

Input : HTML (dan opsional path screenshot) dari crawler.
Output : list rekening {rekening, bank, account_type, raw_text, confidence}.

Tantangan utama: situs judol sengaja meng-obfuscate nomor rekening agar lolos
filter otomatis. Contoh: "1 2 3 4 - 5 6 7 - 8 9 0", "BCA : 1234.567.890",
atau menaruh nomor sebagai gambar. Modul ini menangani:
  1. Regex per-bank pada teks yang sudah dinormalisasi (spasi/titik/strip dibuang).
  2. Deteksi e-wallet (GoPay/OVO/DANA) via nomor HP Indonesia.
  3. OCR (Tesseract) untuk rekening dalam bentuk gambar — OPSIONAL & di-guard.

Tesseract opsional. Kalau pytesseract/Pillow tidak ada, OCR dilewati dengan
graceful degradation (confidence path OCR tidak dipakai).
"""

import re
from dataclasses import dataclass

# Panjang nomor rekening umum per bank Indonesia (untuk validasi kasar).
# Sumber: format standar retail masing-masing bank.
_BANK_RULES = {
    "BCA":     (10, 10),   # 10 digit
    "BNI":     (10, 10),   # 10 digit
    "MANDIRI": (13, 13),   # 13 digit
    "BRI":     (15, 15),   # 15 digit (xxxx-xx-xxxxxx-xx-x)
    "BSI":     (10, 10),   # 10 digit
    "DANAMON": (10, 11),
    "CIMB":    (13, 14),
    "PERMATA": (10, 16),
}

# Nama bank + alias yang sering muncul di situs, dipetakan ke kunci kanonik.
_BANK_ALIASES = {
    "bca": "BCA", "klikbca": "BCA",
    "bni": "BNI",
    "mandiri": "MANDIRI", "mndr": "MANDIRI",
    "bri": "BRI", "britama": "BRI",
    "bsi": "BSI", "syariah indonesia": "BSI",
    "danamon": "DANAMON",
    "cimb": "CIMB", "niaga": "CIMB",
    "permata": "PERMATA",
}

_EWALLET_ALIASES = {
    "gopay": "GOPAY", "gojek": "GOPAY",
    "ovo": "OVO",
    "dana": "DANA",
    "shopeepay": "SHOPEEPAY", "shopee": "SHOPEEPAY",
    "linkaja": "LINKAJA",
}

# Deobfuscation: buang pemisah antar-digit (spasi, titik, strip, koma, underscore).
_SEP = re.compile(r"[\s.\-,_]+")
# Kandidat blok digit panjang (setelah normalisasi) 10–18 digit.
_DIGIT_BLOCK = re.compile(r"\d{10,18}")
# Nomor HP Indonesia untuk e-wallet: 08/+62/62, toleran separator antar-digit
# ("0812-3456-7890", "0812 3456 7890") — tiap digit boleh didahului satu pemisah.
_PHONE = re.compile(r"(?:\+62|62|0)8(?:[\s.\-]?\d){7,12}")


@dataclass
class ExtractedAccount:
    rekening: str          # digit ternormalisasi
    bank: str              # kunci kanonik (BCA/…/GOPAY)
    account_type: str      # 'bank' | 'ewallet'
    raw_text: str          # potongan teks sumber (audit)
    confidence: float      # 1.0 regex bersih, <1 hasil OCR/heuristik


# Guarded OCR import.
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


def ocr_available() -> bool:
    return _OCR_AVAILABLE


def _html_to_text(html: str) -> str:
    """Buang tag & script/style, sisakan teks kasar untuk regex."""
    html = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return text


def _nearest_bank(text: str, pos: int, window: int = 40) -> str | None:
    """
    Cari nama bank/e-wallet TERDEKAT sebelum posisi nomor ditemukan.

    Penting: halaman deposit judol sering mendaftarkan banyak bank sekaligus
    ("BCA … BNI … Mandiri … DANA"). Maka kita pilih keyword dengan posisi paling
    dekat ke nomor (indeks terbesar dalam window), bukan yang pertama muncul —
    supaya rekening tidak salah diatribusikan ke bank yang keliru.
    """
    start = max(0, pos - window)
    context = text[start:pos].lower()
    best_canon = None
    best_idx = -1
    for aliases in (_EWALLET_ALIASES, _BANK_ALIASES):
        for alias, canon in aliases.items():
            idx = context.rfind(alias)
            if idx > best_idx:
                best_idx = idx
                best_canon = canon
    return best_canon


def _valid_length(bank: str, digits: str) -> bool:
    rule = _BANK_RULES.get(bank)
    if rule is None:
        return 10 <= len(digits) <= 18   # e-wallet/unknown: terima rentang wajar
    lo, hi = rule
    return lo <= len(digits) <= hi


def extract_from_text(text: str, source_confidence: float = 1.0) -> list[ExtractedAccount]:
    """
    Ekstrak rekening dari teks bebas. Menormalisasi obfuscation lalu mencocokkan
    blok digit dengan nama bank terdekat. Rekening tanpa bank terdekat tetap
    diambil bila panjangnya valid (confidence diturunkan).
    """
    results: list[ExtractedAccount] = []
    seen: set[tuple[str, str]] = set()

    # 1) E-wallet via nomor HP (di teks asli, sebelum normalisasi separator).
    ewallet_digits: set[str] = set()   # untuk mencegah double-capture di bank-pass
    for m in _PHONE.finditer(text):
        raw = m.group(0)
        digits = _normalize_phone(raw)
        if not (10 <= len(digits) <= 13):   # panjang HP Indonesia wajar
            continue
        bank = _nearest_bank(text, m.start())
        # Hanya terima kalau konteks memang menyebut e-wallet, agar tidak
        # menangkap sembarang nomor HP admin/CS di situs.
        if bank not in _EWALLET_ALIASES.values():
            continue
        key = (digits, bank)
        if key in seen:
            continue
        seen.add(key)
        ewallet_digits.add(digits)
        results.append(ExtractedAccount(
            rekening=digits, bank=bank, account_type="ewallet",
            raw_text=raw, confidence=round(source_confidence * 0.9, 3),
        ))

    # 2) Rekening bank: normalisasi separator lalu cari blok digit panjang.
    normalized = _SEP.sub("", text)
    # Blok digit muncul dalam urutan yang sama di `normalized` dan `text`, jadi
    # kita majukan cursor `search_from` agar relokasi anchor tidak salah menunjuk
    # ke kemunculan lebih awal (mis. dua nomor berbagi prefiks 6-digit yang sama).
    search_from = 0
    for m in _DIGIT_BLOCK.finditer(normalized):
        digits = m.group(0)
        anchor = digits[:6]
        pos = _find_loose(text, anchor, search_from)
        if pos < 0:
            pos = _find_loose(text, anchor)   # fallback: cari dari awal
        if pos >= 0:
            search_from = pos + 1
        bank = _nearest_bank(text, pos) if pos >= 0 else None

        # Sudah tertangkap sebagai e-wallet → jangan duplikat sebagai rekening bank.
        if digits in ewallet_digits:
            continue

        if bank and bank in _BANK_RULES:
            if not _valid_length(bank, digits):
                continue
            conf = source_confidence
        elif bank in _EWALLET_ALIASES.values():
            # Nomor e-wallet ter-obfuscate yang lolos _PHONE tapi terdeteksi via
            # blok digit + konteks e-wallet. Klasifikasikan sebagai ewallet.
            if not (10 <= len(digits) <= 13):
                continue
            key = (digits, bank)
            if key in seen:
                continue
            seen.add(key)
            results.append(ExtractedAccount(
                rekening=digits, bank=bank, account_type="ewallet",
                raw_text=digits, confidence=round(source_confidence * 0.8, 3),
            ))
            continue
        else:
            # Tidak ada konteks bank/e-wallet di sekitar angka → SKIP.
            # Situs judol selalu melabeli tujuan transfer ("BCA/DANA/…"), jadi
            # angka telanjang (mis. nomor HP admin, kode voucher) diabaikan
            # demi presisi — hindari node sampah & alert palsu.
            continue

        key = (digits, bank)
        if key in seen:
            continue
        seen.add(key)
        results.append(ExtractedAccount(
            rekening=digits, bank=bank, account_type="bank",
            raw_text=digits, confidence=conf,
        ))

    return results


def _normalize_phone(raw: str) -> str:
    """+62/62/0 → normalisasi ke bentuk 08xxxx (digit saja)."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("62"):
        digits = "0" + digits[2:]
    return digits


def _find_loose(text: str, anchor: str, start: int = 0) -> int:
    """
    Cari posisi `anchor` (6 digit) di teks asli meski disisipi separator,
    dimulai dari indeks `start`. Return index awal, atau -1 kalau tidak ketemu.
    """
    pattern = r"\D*".join(re.escape(c) for c in anchor)
    m = re.compile(pattern).search(text, start)
    return m.start() if m else -1


def extract_from_image(image_path: str) -> list[ExtractedAccount]:
    """OCR gambar → teks → ekstrak. Kosong bila Tesseract tidak tersedia."""
    if not _OCR_AVAILABLE:
        return []
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
    except Exception:
        return []
    # Hasil OCR lebih rawan salah baca → confidence dasar diturunkan.
    return extract_from_text(text, source_confidence=0.7)


def extract(html: str, screenshot_path: str | None = None) -> list[ExtractedAccount]:
    """
    Entry point utama: gabungkan hasil dari HTML dan (opsional) OCR screenshot.
    Dedup lintas-sumber: rekening sama, ambil confidence tertinggi.
    """
    text = _html_to_text(html)
    accounts = extract_from_text(text, source_confidence=1.0)

    if screenshot_path and _OCR_AVAILABLE:
        accounts += extract_from_image(screenshot_path)

    best: dict[str, ExtractedAccount] = {}
    for a in accounts:
        prev = best.get(a.rekening)
        if prev is None or a.confidence > prev.confidence:
            best[a.rekening] = a
    return list(best.values())
