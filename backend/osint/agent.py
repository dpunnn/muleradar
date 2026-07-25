"""
Phase 4.5.12 — AGENTIC DEPOSIT EXPLORER.

Menjawab temuan 4.5.11: situs judol MODERN menyembunyikan nomor rekening di
balik login/form dinamis, jadi crawler pasif dapat 0 rekening. Modul ini
mengadopsi pendekatan GambitHunter: auto-register dummy account -> navigasi
ke halaman deposit -> ekstrak rekening yang DITAMPILKAN situs -> screenshot.

============================ GUARDRAIL LEGAL ============================
Agent HANYA sampai titik situs MENAMPILKAN rekening tujuan (yang normalnya
diisi user SEBELUM transfer manual). Agent TIDAK PERNAH:
  - menyetor/mengirim uang,
  - menekan tombol transfer/bayar/konfirmasi pembayaran,
  - berpartisipasi dalam permainan.
Tidak ada uang berpindah, tidak ada partisipasi judi — murni OSINT untuk
memperoleh rekening mule -> seed graph -> lapor/blokir. Setiap kandidat aksi
disaring _is_forbidden_action() sebelum di-klik. Ini pertahanan berlapis:
heuristik hanya menekan tombol yang cocok whitelist (register/deposit/nav),
DAN menolak apa pun yang cocok blacklist finansial.
========================================================================

LLM (opsional, guarded — pola sama dgn playwright-stealth/OCR): kalau
Ollama tersedia (OSINT_LLM_BASE diisi), _llm_pick_action() dipakai sebagai
FALLBACK saat heuristik buntu (form/alur tak biasa). Tanpa LLM, agent tetap
jalan dengan heuristik murni. LLM multimodal (mis. Qwen3.5:4b) bisa membaca
screenshot halaman deposit untuk menemukan rekening yang di-obfuscate secara
visual — lihat _llm_read_screenshot(). Test LLM ditunda sampai RAM tersedia
(insiden OOM); kode ini dibangun & bisa diverifikasi tanpa LLM jalan.
"""

import os
import random
import string
import logging
from dataclasses import dataclass, field

from . import extractor

logger = logging.getLogger(__name__)

# --- Guardrail: penanda aksi FINANSIAL NYATA — agent tidak pernah menekan ini ---
_FORBIDDEN_ACTION_MARKERS = (
    "transfer", "kirim uang", "bayar sekarang", "konfirmasi transfer",
    "submit payment", "pay now", "withdraw", "tarik dana", "confirm deposit",
    "konfirmasi deposit", "lanjut bayar", "proceed payment", "checkout",
    "confirm payment", "saya sudah transfer", "sudah bayar", "upload bukti",
)

# --- Penanda tombol/link untuk tiap tahap (whitelist heuristik) ---
_REGISTER_MARKERS = ("daftar", "register", "sign up", "signup", "buat akun",
                     "join", "registrasi", "create account")
_LOGIN_MARKERS    = ("login", "masuk", "sign in", "signin", "log in")
_DEPOSIT_MARKERS  = ("deposit", "setor", "isi saldo", "top up", "topup",
                     "cara bayar", "pembayaran", "isi ulang", "add fund")
_SUBMIT_MARKERS   = ("daftar", "register", "submit", "kirim", "buat akun",
                     "sign up", "signup", "next", "lanjut", "continue")

# Field yang JANGAN diisi otomatis (butuh manusia / bukan data akun)
_SKIP_FIELD_MARKERS = ("captcha", "kode", "otp", "referral", "refferal",
                       "kode-referral", "verification", "verifikasi")

# Penanda tombol TUTUP popup/overlay (announcement, promo, age-gate) yg lazim
# muncul saat load & menutupi tombol daftar. Judol sites hampir selalu punya ini.
# CATATAN: jangan taruh bare "x" di sini — akan substring-match "Next"/"Box" dll.
# Glyph 'x'/'×' ditangani lewat pencocokan teks-PERSIS di _dismiss_popups.
_CLOSE_MARKERS = ("close", "tutup", "skip", "lewati", "got it", "saya mengerti",
                  "don't show", "dont show", "jangan tampilkan", "no thanks",
                  "nanti saja", "lain kali", "×", "✕", "✖", "⊗", "✗")


# ------------------------- credential dummy -------------------------
def _rand_user() -> str:
    return "user" + "".join(random.choices(string.ascii_lowercase + string.digits, k=7))

def _rand_pass() -> str:
    # penuhi syarat umum: huruf besar+kecil+angka+simbol
    body = "".join(random.choices(string.ascii_letters + string.digits, k=9))
    return body + "A9!x"

def _rand_email() -> str:
    return _rand_user() + "@" + random.choice(("gmail.com", "mail.com", "outlook.com"))

def _rand_phone() -> str:
    return "08" + "".join(random.choices(string.digits, k=10))

def _make_creds() -> dict:
    u = _rand_user()
    return {"username": u, "password": _rand_pass(),
            "email": u + "@gmail.com", "phone": _rand_phone()}


@dataclass
class ExploreResult:
    """Hasil eksplorasi agentic satu situs (untuk seed graph + audit)."""
    url: str
    registered: bool = False
    reached_deposit: bool = False
    accounts: list = field(default_factory=list)      # list[extractor.ExtractedAccount]
    screenshot_path: str | None = None
    steps: list = field(default_factory=list)         # jejak aksi (audit trail)
    error: str | None = None


def _is_forbidden_action(text: str | None) -> bool:
    """True kalau teks tombol/link menandakan aksi finansial nyata (JANGAN klik)."""
    t = (text or "").lower()
    return any(m in t for m in _FORBIDDEN_ACTION_MARKERS)


def _matches(text: str | None, markers) -> bool:
    t = (text or "").lower()
    return any(m in t for m in markers)


# ------------------------- LLM guarded (opsional) -------------------------
# Aktif hanya kalau OSINT_LLM_BASE diisi (mis. http://localhost:11434/v1).
# Test ditunda sampai RAM tersedia — lihat catatan modul.
_LLM_BASE = os.getenv("OSINT_LLM_BASE", "")
_LLM_MODEL = os.getenv("OSINT_LLM_MODEL", "qwen3.5:4b")
# API key: "ollama" utk lokal (diabaikan Ollama); utk OpenAI/Anthropic isi key asli
# via OSINT_LLM_API_KEY (JANGAN hardcode di repo).
_LLM_API_KEY = os.getenv("OSINT_LLM_API_KEY", "ollama")
# Model punya kemampuan VISION? (mis. qwen2.5-vl). Kalau OFF, jangan panggil
# _llm_read_screenshot — model text-only balas 400 tiap langkah = boros latency.
_LLM_VISION = os.getenv("OSINT_LLM_VISION", "") in ("1", "true", "yes")


def llm_available() -> bool:
    return bool(_LLM_BASE)


def llm_vision_available() -> bool:
    return bool(_LLM_BASE) and _LLM_VISION


async def _llm_pick_action(page_summary: str, goal: str) -> dict | None:
    """FALLBACK saat heuristik buntu: minta LLM pilih aksi berikutnya.

    Return {"action": "click"|"fill"|"stop", "selector": str, "value": str}
    atau None kalau LLM tak tersedia/gagal. Guardrail tetap berlaku: caller
    WAJIB cek _is_forbidden_action pada aksi yg dikembalikan LLM.
    """
    if not _LLM_BASE:
        return None
    try:
        # OpenAI-compatible (Ollama localhost:11434/v1). Import di dalam supaya
        # modul tetap import walau paket openai belum ada (guarded).
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=_LLM_BASE, api_key=_LLM_API_KEY)
        prompt = (
            "Kamu agent OSINT anti-judol. Tujuan: " + goal + ". "
            "JANGAN pernah menekan tombol transfer/bayar/konfirmasi pembayaran. "
            "Dari ringkasan halaman berikut, pilih SATU aksi berikutnya sebagai JSON "
            '{"action":"click|fill|stop","selector":"...","value":"..."}.\n\n'
            + page_summary[:6000]
        )
        resp = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        import json
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.warning("LLM pick_action gagal (%s) — lanjut heuristik", e)
        return None


async def _llm_read_screenshot(screenshot_path: str) -> list:
    """LLM MULTIMODAL baca screenshot halaman deposit untuk rekening yang
    di-obfuscate secara visual (gambar/teks aneh). Return list dict rekening.
    Guarded: kosong kalau LLM/multimodal tak tersedia."""
    if not llm_vision_available() or not screenshot_path or not os.path.isfile(screenshot_path):
        return []
    try:
        import base64
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=_LLM_BASE, api_key=_LLM_API_KEY)
        with open(screenshot_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Sebutkan semua nomor rekening bank / "
                 "nomor e-wallet + nama banknya yang terlihat di gambar ini, "
                 "sebagai JSON list [{\"rekening\":\"\",\"bank\":\"\"}]. "
                 "Kosong kalau tidak ada."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            temperature=0.0,
        )
        import json
        raw = (resp.choices[0].message.content or "").strip()
        # toleran: GPT-4o sering bungkus JSON dgn ```json ... ``` atau prosa.
        # Ambil array [..] pertama.
        s, e = raw.find("["), raw.rfind("]")
        if s < 0 or e < 0:
            return []
        data = json.loads(raw[s:e + 1])
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("LLM read_screenshot gagal (%s)", e)
        return []


# =====================================================================
# ENGINE LLM-DRIVEN (produksi 4.5.12, 21-Jul) — pendekatan GambitHunter.
# Loop: PERSEPSI halaman -> LLM putuskan aksi -> GUARDRAIL -> eksekusi ->
# ekstraksi tiap langkah. Menggantikan heuristik statis yg terbukti 0-rekening
# di situs judol modern (SPA JS-heavy). Heuristik tetap jadi FALLBACK per-langkah
# saat LLM tak tersedia/gagal, jadi modul tetap jalan (degraded) tanpa Ollama.
# =====================================================================

_GOAL = ("Buat akun dummy (daftar), lalu buka halaman DEPOSIT/SETOR sampai "
         "situs MENAMPILKAN nomor rekening bank/e-wallet tujuan setoran. "
         "JANGAN pernah menekan tombol transfer/bayar/konfirmasi pembayaran — "
         "cukup sampai rekening terlihat.")


async def _goto_resilient(page, url: str, steps: list) -> bool:
    """Navigasi tahan-JS (QC fix): coba networkidle (SPA sempat render), fallback
    domcontentloaded + tunggu. Return True kalau halaman ter-load."""
    for attempt, (wu, to) in enumerate([("networkidle", 30000), ("domcontentloaded", 20000)]):
        try:
            await page.goto(url, wait_until=wu, timeout=to)
            if wu == "domcontentloaded":
                try:
                    await page.wait_for_timeout(4000)  # beri JS waktu render
                except Exception:
                    pass
            return True
        except Exception as e:
            if attempt == 1:
                steps.append(f"goto gagal: {str(e)[:60]}")
                return False
    return False


async def _dismiss_popups(page, steps: list, max_pop: int = 4) -> int:
    """Tutup overlay/announcement popup yg menutupi halaman (promo/berita/age-gate).
    Wajib sebelum navigasi: kalau tidak, klik 'Daftar' di belakang overlay percuma
    (temuan 50phbet). Guardrail: tak pernah klik aksi finansial. Return jumlah ditutup."""
    closed = 0
    _EXACT_CLOSE = ("×", "✕", "✖", "⊗", "✗", "x", "close", "tutup", "skip", "lewati")
    for _ in range(max_pop):
        target = None
        # (a) selector KHUSUS tombol-tutup (cepat, kandidat sedikit) — bukan scan
        # semua button yg lambat via Tor.
        try:
            cands = await page.query_selector_all(
                "[aria-label*=close i], [class*=modal-close i], [class*=popup-close i], "
                "[class*=btn-close i], [class*=close-btn i], [class*=dialog] [class*=close i]")
        except Exception:
            cands = []
        for el in (cands or [])[:12]:
            try:
                if not await el.is_visible():
                    continue
                if _is_forbidden_action(await el.get_attribute("aria-label") or ""):
                    continue
                target = el
                break
            except Exception:
                continue
        # (b) fallback: tombol/link yg TEKS-nya PERSIS glyph/kata tutup (cap 60).
        if target is None:
            try:
                txtc = await page.query_selector_all("button, [role=button], a")
            except Exception:
                txtc = []
            for el in (txtc or [])[:60]:
                try:
                    inner = (await el.inner_text() or "").strip().lower()
                    if inner in _EXACT_CLOSE and await el.is_visible():
                        target = el
                        break
                except Exception:
                    continue
        if target is None:
            break
        try:
            await target.click(timeout=2500, force=True)
            closed += 1
            await page.wait_for_timeout(500)
        except Exception:
            break
    # (c) FALLBACK generik utk modal bandel (icon-font close tak ter-selektor,
    # Escape tak mempan — mis. template judol China): sembunyikan overlay besar
    # ber-z-index tinggi yg TAK punya input (announcement/promo). Overlay yg PUNYA
    # input (form register) sengaja DIBIARKAN supaya tak merusak alur daftar.
    try:
        removed = await page.evaluate("""() => {
          let n=0; const vw=innerWidth, vh=innerHeight;
          for (const e of document.querySelectorAll('div,section,aside')){
            const s=getComputedStyle(e);
            if(s.position!=='fixed' && s.position!=='absolute') continue;
            if(s.display==='none' || s.visibility==='hidden') continue;
            const z=parseInt(s.zIndex)||0;
            const r=e.getBoundingClientRect();
            const cover = r.width>vw*0.5 && r.height>vh*0.4;
            if(z>=50 && cover &&
               e.querySelectorAll('input,textarea,select').length===0){
              e.style.display='none'; n++;
            }
          }
          return n;
        }""")
        if removed:
            closed += removed
            await page.wait_for_timeout(400)
    except Exception:
        pass
    if closed:
        steps.append(f"tutup {closed} popup overlay")
    return closed


async def _perceive(page):
    """PERSEPSI halaman: kumpulkan elemen interaktif sbg daftar bernomor +
    handle paralel. LLM mereferensikan by id (bukan CSS rapuh). Handle hanya
    dipakai dalam SATU siklus observasi (jadi stale setelah navigasi -> re-persepsi).
    Tahan navigation-race (QC fix): semua query di-guard."""
    elements, handles = [], []
    # Tag elemen NON-SEMANTIK yg clickable (div/span ber-cursor:pointer atau
    # onclick) — banyak landing judol bikin tombol dari <div> (mis. "GET BONUS"
    # di pespan.live) yg tak tertangkap selector a/button. Tanpa ini LLM lihat
    # halaman "kosong" lalu giveup.
    try:
        await page.evaluate("""() => {
          for (const e of document.querySelectorAll('div,span,li,label')) {
            const s=getComputedStyle(e);
            const txt=(e.innerText||'').trim();
            const clickable = s.cursor==='pointer' || e.hasAttribute('onclick');
            if (clickable && txt.length>0 && txt.length<40 &&
                e.offsetParent!==null && e.clientHeight<130) {
              e.setAttribute('data-osint-clk','1');
            }
          }
        }""")
    except Exception:
        pass
    try:
        raw = await page.query_selector_all(
            "a, button, input, textarea, select, [role=button], [data-osint-clk]")
    except Exception:
        return elements, handles  # execution context destroyed (navigasi) -> kosong
    cand = []           # (prioritas, urutanDOM, handle, meta)
    dup_count = {}      # (tag, label) -> berapa kali muncul (dedupe tombol repetitif)
    for order, el in enumerate(raw):
        try:
            tag = (await el.evaluate("e => e.tagName") or "").lower()
            itype = (await el.get_attribute("type") or "").lower()
            if itype in ("hidden",):
                continue
            # Hanya elemen yg BENAR-BENAR terlihat: banyak SPA judol simpan field
            # template/duplikat modal display:none -> kalau di-fill error "not
            # visible". Filter di sini bikin LLM cuma lihat elemen actionable.
            try:
                if not await el.is_visible():
                    continue
            except Exception:
                continue
            txt = (await el.inner_text() if tag != "input" else "") or ""
            txt = txt.strip()
            val = await el.get_attribute("value") or ""
            ph = await el.get_attribute("placeholder") or ""
            name = await el.get_attribute("name") or await el.get_attribute("id") or ""
            aria = await el.get_attribute("aria-label") or ""
            label = (txt or val or ph or aria or name)[:60]
            is_field = tag in ("input", "textarea", "select")
            # STATE terisi: utk field ambil VALUE saat ini (property .value, bukan
            # attribute — nilai yg diketik user ada di property). Supaya LLM tahu
            # field SUDAH diisi lalu lanjut ke field berikutnya/submit, bukan isi
            # ulang berkali-kali (fix stuck 'isi username 3x' di 3b & 7b).
            filled = False
            if is_field and tag != "select":
                try:
                    cur = await el.evaluate("e => e.value || ''")
                    filled = bool((cur or "").strip())
                except Exception:
                    filled = False
            if not label and not is_field:
                continue  # elemen tak berteks & bukan field -> tak berguna utk LLM
            low = label.lower()
            # PRIORITAS konteks: field form (0) > CTA daftar/deposit/submit (1) >
            # sisanya (2). Supaya elemen bermakna tak terdorong ke luar cap 45 oleh
            # puluhan tombol identik ("Play Now" dsb).
            if is_field:
                prio = 0
            elif (_matches(low, _REGISTER_MARKERS) or _matches(low, _DEPOSIT_MARKERS)
                  or _matches(low, _SUBMIT_MARKERS) or _matches(low, _LOGIN_MARKERS)):
                prio = 1
            else:
                prio = 2
            # dedupe tombol/link identik non-field: simpan maksimal 2
            if not is_field:
                key = (tag, low)
                dup_count[key] = dup_count.get(key, 0) + 1
                if dup_count[key] > 2:
                    continue
            cand.append((prio, order, el,
                         {"tag": tag, "type": itype, "text": label,
                          "name": name.lower()[:30], "filled": filled}))
        except Exception:
            continue
    cand.sort(key=lambda c: (c[0], c[1]))   # stabil: prioritas lalu urutan DOM
    for _, _, el, meta in cand[:45]:         # cap konteks LLM
        meta["id"] = len(elements)
        elements.append(meta)
        handles.append(el)
    return elements, handles


async def _llm_next_action(elements: list, history: list, creds: dict) -> dict | None:
    """LLM putuskan SATU aksi berikutnya dari persepsi halaman. Return dict:
    {action: click|fill|wait|extract|done|giveup, id: int, field: str, reason: str}
    atau None kalau LLM tak tersedia/gagal (caller fallback ke heuristik)."""
    if not _LLM_BASE:
        return None
    try:
        import json
        from openai import OpenAI  # type: ignore
        client = OpenAI(base_url=_LLM_BASE, api_key=_LLM_API_KEY)
        el_lines = "\n".join(
            f"  [{e['id']}] <{e['tag']}{'/'+e['type'] if e['type'] else ''}> "
            f"\"{e['text']}\"{' name='+e['name'] if e['name'] else ''}"
            f"{'  [SUDAH TERISI]' if e.get('filled') else ''}"
            for e in elements) or "  (tak ada elemen interaktif terlihat)"
        hist = "; ".join(history[-6:]) or "(baru mulai)"
        prompt = (
            f"Kamu agent OSINT anti-judol. TUJUAN: {_GOAL}\n"
            f"Kredensial dummy tersedia utk isi form: username, password, email, phone.\n"
            f"LARANGAN MUTLAK: jangan pilih elemen yg berarti transfer/bayar/"
            f"konfirmasi pembayaran/withdraw.\n\n"
            f"Riwayat aksi: {hist}\n\n"
            f"Elemen interaktif di halaman SEKARANG:\n{el_lines}\n\n"
            f"Pilih SATU aksi berikutnya. Keluarkan SATU objek JSON dgn field:\n"
            f'  "action": PILIH TEPAT SATU kata: click ATAU fill ATAU wait ATAU '
            f"extract ATAU done ATAU giveup (bukan daftarnya, cukup satu)\n"
            f'  "id": nomor [n] elemen target sbg integer; -1 kalau wait/done/giveup/extract\n'
            f'  "field": kalau action=fill isi salah satu: username/password/email/phone; '
            f'selain itu ""\n'
            f'  "reason": alasan singkat\n'
            f'Contoh jawaban valid: {{"action":"click","id":5,"field":"","reason":"klik tombol Daftar"}}\n'
            f"Panduan: klik tombol Daftar/Register dulu; TAPI kalau di Riwayat kamu "
            f"SUDAH klik Daftar/Sign up/Register, form pendaftaran kemungkinan sudah "
            f"terbuka — sekarang PRIORITASKAN isi (fill) field input yg ada satu per "
            f"satu, JANGAN klik tombol Daftar lagi. PENTING: field bertanda "
            f"[SUDAH TERISI] JANGAN diisi ulang — pilih field KOSONG berikutnya; kalau "
            f"SEMUA field sudah [SUDAH TERISI], KLIK tombol submit/daftar (jangan fill "
            f"lagi). Sesudah akun jadi cari menu Deposit/Setor; kalau nomor rekening "
            f"kemungkinan sudah tampil pilih extract. "
            f"HALAMAN LANDING BONUS/PROMO (mis. cuma ada tombol 'GET BONUS', 'CLAIM', "
            f"'PLAY NOW', 'ENTER', 'MAINKAN', 'MULAI', 'GET STARTED') = jalan menuju "
            f"form register — KLIK tombol itu untuk lanjut, ITU BUKAN aksi finansial. "
            f"JANGAN giveup selama masih ada tombol/link yg bisa diklik menuju daftar/"
            f"deposit; giveup HANYA kalau benar-benar tak ada elemen actionable sama "
            f"sekali. JANGAN salin contoh mentah — sesuaikan dgn elemen nyata di atas.")
        resp = client.chat.completions.create(
            model=_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        # toleran: ambil blok JSON pertama
        s, e = raw.find("{"), raw.rfind("}")
        act = json.loads(raw[s:e + 1]) if s >= 0 else None
        # Guard model kecil: kadang balas "click|fill|..." (nyalin enum). Ambil
        # kata valid pertama supaya tak jadi 'aksi tak dikenal'.
        if act and isinstance(act.get("action"), str) and "|" in act["action"]:
            for w in ("click", "fill", "wait", "extract", "done", "giveup"):
                if w in act["action"].lower():
                    act["action"] = w
                    break
        return act
    except Exception as e:
        logger.warning("LLM next_action gagal (%s) — fallback heuristik", e)
        return None


async def _exec_action(page, action: dict, handles: list, creds: dict,
                       steps: list) -> str:
    """Eksekusi aksi LLM dgn GUARDRAIL + tahan navigasi. Return status:
    'ok'|'blocked'|'done'|'giveup'|'extract'|'error'."""
    act = (action.get("action") or "").lower()
    if act in ("done", "giveup", "extract"):
        steps.append(f"LLM -> {act}: {action.get('reason','')[:40]}")
        return act
    if act == "wait":
        try:
            await page.wait_for_timeout(2500)
        except Exception:
            pass
        steps.append("LLM -> wait")
        return "ok"

    idx = action.get("id", -1)
    if not isinstance(idx, int) or idx < 0 or idx >= len(handles):
        steps.append(f"LLM -> {act} id invalid ({idx})")
        return "error"
    el = handles[idx]
    try:
        label = (await el.inner_text() or await el.get_attribute("value") or "")[:40]
    except Exception:
        label = ""

    if act == "click":
        # GUARDRAIL: jangan pernah klik aksi finansial (pertahanan kode, bukan
        # cuma prompt — LLM bisa keliru/di-inject).
        if _is_forbidden_action(label):
            steps.append(f"GUARDRAIL blok klik: '{label.strip()}'")
            return "blocked"
        clicked = False
        try:
            await el.click(timeout=6000)
            clicked = True
        except Exception:
            # Fallback anti-overlay/off-screen: scroll ke elemen -> force click ->
            # JS click. Banyak situs judol taruh tombol di balik overlay/animasi.
            try:
                await el.scroll_into_view_if_needed(timeout=3000)
                await el.click(timeout=4000, force=True)
                clicked = True
            except Exception:
                try:
                    await el.evaluate("e => e.click()")
                    clicked = True
                except Exception as e3:
                    steps.append(f"klik gagal '{label.strip()[:20]}': {str(e3)[:35]}")
                    return "error"
        if clicked:
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                await page.wait_for_timeout(2000)
            steps.append(f"klik: '{label.strip()[:30]}'")
            return "ok"

    if act == "fill":
        field = (action.get("field") or "").lower()
        # jangan isi field sensitif (captcha/otp/referral)
        try:
            meta = ((await el.get_attribute("name") or "") +
                    (await el.get_attribute("placeholder") or "")).lower()
        except Exception:
            meta = ""
        if _matches(meta, _SKIP_FIELD_MARKERS):
            steps.append(f"skip field sensitif: {meta[:20]}")
            return "ok"
        value = creds.get(field) or creds["username"]
        try:
            await el.scroll_into_view_if_needed(timeout=2500)
        except Exception:
            pass
        try:
            await el.fill(value, timeout=5000)
            steps.append(f"isi {field or 'field'}={value[:12]}")
            return "ok"
        except Exception:
            # fallback: klik lalu ketik manual (field non-standar/custom widget)
            try:
                await el.click(timeout=3000)
                await page.keyboard.type(value, delay=15)
                steps.append(f"isi(type) {field or 'field'}={value[:12]}")
                return "ok"
            except Exception as e:
                steps.append(f"fill gagal: {str(e)[:35]}")
                return "error"

    steps.append(f"aksi tak dikenal: {act}")
    return "error"


def _visual_to_accounts(visual: list) -> list:
    """Konversi hasil LLM multimodal (list dict) -> ExtractedAccount (QC fix B:
    dulu dibuang). Confidence lebih rendah (0.6) karena via OCR-visual."""
    out = []
    for v in visual or []:
        rek = "".join(ch for ch in str(v.get("rekening", "")) if ch.isdigit())
        if len(rek) < 8:
            continue
        try:
            out.append(extractor.ExtractedAccount(
                rekening=rek, bank=(v.get("bank") or "UNKNOWN").upper(),
                account_type="bank", confidence=0.6))
        except Exception:
            continue
    return out


async def _try_extract(page, screenshot_dir: str, url: str, steps: list):
    """Screenshot bukti + ekstraksi regex; kalau kosong & LLM multimodal ada,
    baca screenshot (QC fix B: hasilnya kini DI-MERGE, bukan dibuang)."""
    import os as _os
    _os.makedirs(screenshot_dir, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in url)[:80]
    shot = _os.path.join(screenshot_dir, f"deposit_{safe}.png")
    try:
        # viewport-only (bukan full_page) supaya cepat di loop; halaman judol sering
        # sangat tinggi -> full_page tiap langkah = lambat via Tor.
        await page.screenshot(path=shot, full_page=False)
    except Exception:
        shot = None
    try:
        html = await page.content()
    except Exception:
        html = ""
    accounts = extractor.extract(html, screenshot_path=shot) if html else []
    if not accounts and shot and llm_vision_available():
        visual = await _llm_read_screenshot(shot)
        merged = _visual_to_accounts(visual)
        if merged:
            steps.append(f"LLM multimodal: +{len(merged)} rekening dari screenshot")
            accounts = merged
    return accounts, shot


async def explore_deposit_llm(browser, url: str, screenshot_dir: str,
                              creds: dict = None, max_steps: int = 14) -> "ExploreResult":
    """LOOP AGENTIC LLM-DRIVEN (inti produksi). Context punya fingerprint acak +
    stealth (QC fix C). Tiap langkah: persepsi -> LLM aksi (fallback heuristik) ->
    guardrail -> eksekusi -> coba ekstraksi. Berhenti saat rekening ketemu /
    done / giveup / max_steps."""
    result = ExploreResult(url=url)
    creds = creds or _make_creds()
    ctx = await browser.new_context(
        user_agent=random.choice(_UA_POOL_AGENT),
        viewport=random.choice(_VIEWPORT_POOL_AGENT),
        ignore_https_errors=True)
    try:
        await ctx.add_init_script(
            "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});")
    except Exception:
        pass
    page = await ctx.new_page()
    if _STEALTH_AGENT:
        try:
            await _stealth_async_agent(page)
        except Exception:
            pass
    try:
        if not await _goto_resilient(page, url, result.steps):
            result.error = "goto gagal"
            return result
        # WAJIB: tutup popup announcement/promo yg menutupi halaman dulu, kalau
        # tidak klik 'Daftar' di belakang overlay percuma (temuan 50phbet).
        await _dismiss_popups(page, result.steps)

        empty_streak = 0        # stuck-detection: halaman 0-elemen berturut
        clicked_labels = []     # anti-toggle: label yg baru sukses diklik
        for step in range(max_steps):
            # 0. Popup bisa muncul lagi tiap navigasi -> tutup dulu (3 langkah awal).
            if step < 3:
                await _dismiss_popups(page, result.steps, max_pop=2)
            # 1. Coba ekstraksi tiap langkah (rekening bisa muncul kapan saja).
            accounts, shot = await _try_extract(page, screenshot_dir, url, result.steps)
            if shot:
                result.screenshot_path = shot
            if accounts:
                result.accounts = accounts
                result.reached_deposit = True
                result.steps.append(f"REKENING ketemu langkah-{step}: {len(accounts)}")
                break

            # 2. Persepsi halaman sekarang.
            elements, handles = await _perceive(page)
            if not elements:
                empty_streak += 1
                # STUCK-DETECTION (fix 21-Jul): 0 elemen 3x berturut = halaman
                # challenge/rate-limit/kosong -> stop, jangan buang max_steps.
                if empty_streak >= 3:
                    result.steps.append(f"stop: 0 elemen {empty_streak}x berturut "
                                        f"(challenge/rate-limit/kosong)")
                    break
                result.steps.append(f"langkah-{step}: 0 elemen, tunggu JS...")
                try:
                    await page.wait_for_timeout(2500)
                except Exception:
                    pass
                continue
            empty_streak = 0

            steps_before = len(result.steps)

            # 3. LLM putuskan aksi; fallback heuristik kalau LLM tak ada.
            action = await _llm_next_action(elements, result.steps, creds)
            if action is None:
                # FALLBACK HEURISTIK per-langkah (degraded, tanpa LLM):
                did = await _heuristic_step(page, elements, handles, creds, result.steps)
                if not did:
                    result.steps.append("heuristik buntu -> stop")
                    break
            else:
                # 4. ANTI-TOGGLE (deterministik): kalau LLM mau klik-ulang elemen
                # yg BARU sukses diklik (mis. tombol 'Sign up' yg toggle modal buka/
                # tutup), jangan toggle lagi — alihkan ke isi-form heuristik supaya
                # progres. Model kecil sering nyangkut di sini walau sudah di-hint.
                _act = action.get("action")
                _idx = action.get("id", -1)
                _lbl = ""
                if _act == "click" and isinstance(_idx, int) and 0 <= _idx < len(elements):
                    _lbl = (elements[_idx]["text"] or "").lower().strip()
                if _lbl and _lbl in clicked_labels[-2:]:
                    result.steps.append(f"anti-toggle: '{_lbl[:22]}' sudah diklik -> isi form")
                    did = await _heuristic_step(page, elements, handles, creds, result.steps)
                    status = "ok" if did else await _exec_action(
                        page, action, handles, creds, result.steps)
                    clicked_labels.append("__form__" if did else _lbl)
                else:
                    # Guardrail + eksekusi normal.
                    status = await _exec_action(page, action, handles, creds, result.steps)
                    if _act == "click" and status == "ok" and _lbl:
                        clicked_labels.append(_lbl)
                if status in ("done", "giveup"):
                    break
                if status == "extract":
                    continue

            # TAB/WINDOW BARU: banyak situs judol buka form register/deposit lewat
            # window.open (temuan 50phbet — form tak render inline). Pindah fokus ke
            # tab terbaru supaya persepsi berikutnya baca form yg benar.
            try:
                if len(ctx.pages) > 1 and ctx.pages[-1] is not page:
                    newp = ctx.pages[-1]
                    # Tunggu SPA tab baru render PENUH sebelum persepsi — kalau
                    # kecepetan, form belum ada -> LLM lihat halaman kosong lalu
                    # giveup (temuan GPT-4o di pespan.live/welcome). networkidle dulu,
                    # fallback domcontentloaded + settle 4s utk JS.
                    try:
                        await newp.wait_for_load_state("networkidle", timeout=12000)
                    except Exception:
                        try:
                            await newp.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            pass
                        try:
                            await newp.wait_for_timeout(4000)
                        except Exception:
                            pass
                    page = newp
                    clicked_labels = []   # reset konteks anti-toggle di halaman baru
                    result.steps.append(f"pindah ke tab baru: {page.url[:38]}")
                    await _dismiss_popups(page, result.steps, max_pop=2)
            except Exception:
                pass

            # STUCK-DETECTION (fix 21-Jul): aksi SAMA berulang (mis. klik tombol
            # splash "REGISTER IN 10 SECONDS" yg tak mengubah halaman). Kalau 3
            # langkah terakhir identik -> stop (tanpa LLM tak bisa varias taktik).
            recent = result.steps[steps_before:]
            sig = recent[-1] if recent else ""
            _repeat = [s for s in result.steps[-6:] if s == sig]
            if sig and len(_repeat) >= 3:
                result.steps.append(f"stop: aksi berulang 3x ('{sig[:35]}') — stuck")
                break

        # ekstraksi final sekali lagi kalau belum ketemu
        if not result.accounts:
            accounts, shot = await _try_extract(page, screenshot_dir, url, result.steps)
            if shot and not result.screenshot_path:
                result.screenshot_path = shot
            result.accounts = accounts
        result.steps.append(f"selesai: {len(result.accounts)} rekening, "
                            f"{step+1} langkah")
        return result
    except Exception as e:
        result.error = str(e)[:200]
        return result
    finally:
        try:
            await page.close(); await ctx.close()
        except Exception:
            pass


async def _heuristic_step(page, elements, handles, creds, steps) -> bool:
    """Satu langkah heuristik (fallback tanpa LLM): prioritas isi password field ->
    submit register -> klik register-open -> klik deposit. Return True kalau ada aksi."""
    # a) ada password field kosong? isi seluruh form (register).
    for i, e in enumerate(elements):
        if e["type"] == "password":
            ok = await _try_register(page, steps)
            return ok or True
    # b) klik tombol register/daftar.
    for i, e in enumerate(elements):
        if _matches(e["text"], _REGISTER_MARKERS) and not _is_forbidden_action(e["text"]):
            try:
                await handles[i].click(timeout=8000)
                await page.wait_for_timeout(2500)
                steps.append(f"[heuristik] klik register: '{e['text'][:25]}'")
                return True
            except Exception:
                continue
    # c) klik menu deposit.
    for i, e in enumerate(elements):
        if _matches(e["text"], _DEPOSIT_MARKERS) and not _is_forbidden_action(e["text"]):
            try:
                await handles[i].click(timeout=8000)
                await page.wait_for_timeout(2500)
                steps.append(f"[heuristik] klik deposit: '{e['text'][:25]}'")
                return True
            except Exception:
                continue
    return False


# --- fingerprint pool + stealth utk context agentic (QC fix C) ---
_UA_POOL_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
_VIEWPORT_POOL_AGENT = ({"width": 1366, "height": 768}, {"width": 1920, "height": 1080},
                        {"width": 1440, "height": 900})
# playwright-stealth 2.x pakai class Stealth().apply_stealth_async (bukan
# stealth_async 1.x). Fix produksi 21-Jul: dukung KEDUA API supaya stealth
# benar-benar aktif (sebelumnya diam-diam mati -> mudah di-fingerprint bot).
try:
    from playwright_stealth import Stealth as _Stealth2  # type: ignore
    _stealth2_inst = _Stealth2()
    async def _stealth_async_agent(page):
        await _stealth2_inst.apply_stealth_async(page)
    _STEALTH_AGENT = True
except ImportError:
    try:
        from playwright_stealth import stealth_async as _stealth_async_agent  # type: ignore
        _STEALTH_AGENT = True
    except ImportError:
        _STEALTH_AGENT = False


# ------------------------- form register (heuristik) -------------------------
async def _try_register(page, steps: list) -> bool:
    """Cari form berisi password field, isi field standar dgn credential dummy,
    lalu submit tombol yang lolos guardrail. Return True kalau submit terjadi."""
    pw = await page.query_selector("input[type=password]")
    if not pw:
        return False

    creds = _make_creds()
    inputs = await page.query_selector_all("input, textarea")
    filled = 0
    for inp in inputs:
        try:
            itype = (await inp.get_attribute("type") or "").lower()
            if itype in ("hidden", "submit", "button", "checkbox", "radio", "file"):
                continue
            meta = " ".join(filter(None, [
                await inp.get_attribute("name"),
                await inp.get_attribute("id"),
                await inp.get_attribute("placeholder"),
            ])).lower()
            if _matches(meta, _SKIP_FIELD_MARKERS):
                continue  # captcha/otp/referral — jangan tebak
            if itype == "password":
                await inp.fill(creds["password"]); filled += 1
            elif itype == "email" or "email" in meta:
                await inp.fill(creds["email"]); filled += 1
            elif any(k in meta for k in ("phone", "hp", "telp", "wa", "nomor")):
                await inp.fill(creds["phone"]); filled += 1
            elif itype in ("text", "") and any(
                    k in meta for k in ("user", "nama", "name", "akun", "account", "login")):
                await inp.fill(creds["username"]); filled += 1
        except Exception:
            continue

    if filled == 0:
        return False
    steps.append(f"isi form register ({filled} field, user={creds['username']})")

    # Submit: pilih tombol yang cocok whitelist register/submit DAN lolos guardrail.
    buttons = await page.query_selector_all("button, input[type=submit], a")
    for btn in buttons:
        try:
            label = (await btn.inner_text() or "") if await btn.evaluate("e => e.tagName") != "INPUT" \
                    else (await btn.get_attribute("value") or "")
            if _is_forbidden_action(label):
                continue  # GUARDRAIL: jangan tekan aksi finansial
            if _matches(label, _SUBMIT_MARKERS) or _matches(label, _REGISTER_MARKERS):
                await btn.click()
                steps.append(f"klik submit register: '{label.strip()[:30]}'")
                await page.wait_for_timeout(2500)
                return True
        except Exception:
            continue
    return False


# ------------------------- navigasi ke deposit -------------------------
async def _goto_deposit(page, steps: list) -> bool:
    """Cari & klik link/tombol menuju halaman deposit (post-login).
    Guardrail: lewati apa pun yang cocok blacklist finansial."""
    candidates = await page.query_selector_all("a, button")
    for el in candidates:
        try:
            label = (await el.inner_text() or "").strip()
            if not label:
                continue
            if _is_forbidden_action(label):
                continue  # GUARDRAIL
            if _matches(label, _DEPOSIT_MARKERS):
                await el.click()
                steps.append(f"klik menu deposit: '{label[:30]}'")
                await page.wait_for_timeout(2500)
                return True
        except Exception:
            continue
    return False


# ------------------------- entry point -------------------------
async def explore_deposit(browser, url: str, screenshot_dir: str,
                          try_register: bool = True) -> ExploreResult:
    """
    ENTRY POINT eksplorasi agentic satu situs judol -> rekening deposit.
    Sejak produksi 4.5.12 (21-Jul): MENDELEGASIKAN ke engine LLM-DRIVEN
    (explore_deposit_llm) — loop persepsi->LLM aksi->guardrail->ekstraksi, dgn
    fallback heuristik per-langkah saat LLM tak tersedia. Menggantikan loop
    heuristik statis lama (terbukti 0-rekening di situs SPA JS-heavy).
    Guardrail no-transfer tetap diberlakukan di _exec_action + _heuristic_step.
    `try_register` disimpan utk kompat tanda tangan (loop selalu coba register
    sbg bagian tujuan)."""
    return await explore_deposit_llm(browser, url, screenshot_dir)
