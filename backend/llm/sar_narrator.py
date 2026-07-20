"""
LLM Layer — SAR Narrative Generation untuk MuleRadar.

Posisi di pipeline:
    Ensemble (XGBoost + TGN + DyGFormer)
        → alerts.py (risk score + rule trigger)
        → SARNarrator (narasi bahasa Indonesia siap kirim ke PPATK)
        → Dashboard / LTKM export

Provider yang didukung (urutan prioritas):
    1. anthropic  — Claude Haiku (cepat, murah, kualitas tinggi)
    2. openai     — GPT-4o-mini (alternatif)
    3. ollama     — llama3 lokal (dev/lokal, tanpa biaya)
    4. template   — fallback rule-based (tidak butuh LLM, selalu tersedia)

Konfigurasi via environment variable:
    LLM_PROVIDER=anthropic|openai|ollama|template
    ANTHROPIC_API_KEY=...
    OPENAI_API_KEY=...
    OLLAMA_BASE_URL=http://localhost:11434   (default)
    OLLAMA_MODEL=llama3                      (default)
"""

import os
import logging
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

# ── Referensi tipologi PPATK ────────────────────────────────────────────────
# Public alias (QC 15-Jul: api/routes/copilot.py butuh import ini dari luar
# modul — underscore-prefix asalnya cuma penanda "internal", tapi bikin
# import luar rapuh/ImportError kalau nama berubah. PPATK_REF = alias publik,
# _PPATK_REF tetap ada demi backward-compat internal file ini sendiri).
_PPATK_REF = {
    "DORMANT_ACTIVATION":  ("Tipologi No. 3 PPATK",  "Dormant Account Activation"),
    "STRUCTURING":         ("Tipologi No. 1 PPATK",  "Structuring / Smurfing"),
    "LAYERING":            ("Tipologi No. 2 PPATK",  "Layering via Rantai Rekening"),
    "FAN_OUT":             ("Tipologi No. 5 PPATK",  "Fan-out / Collector Account"),
    "JUDOL_RING":          ("Tipologi No. 7 PPATK",  "Jaringan Rekening Judol"),
    "QRIS_FRAUD":          ("Tipologi No. 6 PPATK",  "Penyalahgunaan QRIS"),
    "PEP_NETWORK":         ("Tipologi No. 4 PPATK",  "Rekening Terkait PEP"),
    "VENDOR_SHELL":        ("Tipologi No. 8 PPATK",  "Perusahaan Cangkang"),
    "SMURF_LAYERING":      ("Tipologi No. 1 PPATK",  "Smurf Layering"),
    "RAPID_IN_OUT":        ("Tipologi No. 3 PPATK",  "Rapid In-Out / Pass-Through"),
    "UNKNOWN":             ("Tipologi PPATK",         "Transaksi Mencurigakan"),
}
PPATK_REF = _PPATK_REF  # alias publik, lihat komentar di atas

# ── Feature label untuk narasi ──────────────────────────────────────────────
_FEAT_LABELS = {
    "dormancy_days":       "masa dormant",
    "burst_ratio":         "rasio burst transaksi",
    "structuring_score":   "skor structuring",
    "round_amount_ratio":  "rasio jumlah bulat",
    "counterparty_hhi":    "konsentrasi counterparty",
    "channel_entropy":     "entropi channel",
    "inter_tx_std":        "variabilitas antar-transaksi",
    "night_tx_ratio":      "rasio transaksi malam",
    "out_degree":          "jumlah transaksi keluar",
    "in_amount_sum":       "total dana masuk",
    "out_amount_sum":      "total dana keluar",
    "max_single_tx":       "transaksi terbesar",
}


def _format_features(features: dict, top_n: int = 5) -> str:
    """Pilih fitur paling informatif dan format ke teks."""
    priority = [
        "dormancy_days", "burst_ratio", "structuring_score",
        "round_amount_ratio", "night_tx_ratio", "out_degree",
        "in_amount_sum", "out_amount_sum", "max_single_tx",
    ]
    lines = []
    for key in priority:
        if key in features and features[key]:
            val = features[key]
            label = _FEAT_LABELS.get(key, key)
            if key in ("in_amount_sum", "out_amount_sum", "max_single_tx"):
                lines.append(f"- {label}: Rp {val:,.0f}")
            elif key == "dormancy_days":
                lines.append(f"- {label}: {val:.0f} hari")
            elif key in ("burst_ratio", "structuring_score",
                         "round_amount_ratio", "night_tx_ratio"):
                lines.append(f"- {label}: {val:.1%}")
            else:
                lines.append(f"- {label}: {val:.2f}")
        if len(lines) >= top_n:
            break
    return "\n".join(lines) if lines else "- (tidak ada data fitur)"


def _build_prompt(
    account_id: str,
    risk_score: float,
    typology: str,
    features: dict,
    rules: list[str],
) -> str:
    ppatk_ref, ppatk_desc = _PPATK_REF.get(typology, _PPATK_REF["UNKNOWN"])
    feat_text = _format_features(features)
    rules_text = "; ".join(rules[:3]) if rules else "tidak ada rule spesifik"
    severity = "TINGGI" if risk_score >= 0.8 else "SEDANG"

    return textwrap.dedent(f"""
        Kamu adalah compliance analyst AML senior di bank Indonesia.
        Buat narasi laporan LTKM (Laporan Transaksi Keuangan Mencurigakan) singkat
        untuk disampaikan ke PPATK. Gunakan bahasa Indonesia formal dan profesional.
        Panjang: 3-5 kalimat. Sertakan indikator spesifik, bukan pernyataan generik.

        DATA REKENING:
        - ID Rekening  : {account_id}
        - Risk Score   : {risk_score:.4f} (Risiko {severity})
        - Tipologi     : {ppatk_desc} ({ppatk_ref})
        - Rule Trigger : {rules_text}

        INDIKATOR BEHAVIORAL:
        {feat_text}

        Tulis narasi LTKM:
    """).strip()


class SARNarrator:
    """
    Generate narasi SAR/LTKM otomatis menggunakan LLM.

    Cara pakai:
        narrator = SARNarrator()   # auto-detect provider dari env

        narrative = narrator.narrate(
            account_id="ACC_001234",
            risk_score=0.9812,
            typology="DORMANT_ACTIVATION",
            features={"dormancy_days": 247, "burst_ratio": 0.91, ...},
            rules=["dormant >180 hari + burst >3 tx/jam"],
        )

        # Triage: urutkan alert berdasarkan prioritas investigasi
        prioritized = narrator.triage(alerts_list)
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "template")
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("SARNarrator: Anthropic Claude siap")
            except Exception as e:
                logger.warning("Anthropic init gagal: %s — fallback template", e)
                self.provider = "template"

        elif self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("SARNarrator: OpenAI siap")
            except Exception as e:
                logger.warning("OpenAI init gagal: %s — fallback template", e)
                self.provider = "template"

        elif self.provider == "ollama":
            self._ollama_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
            logger.info("SARNarrator: Ollama %s @ %s", self._ollama_model, self._ollama_url)

        else:
            self.provider = "template"
            logger.info("SARNarrator: template (rule-based, tanpa LLM)")

    # ── Public API ──────────────────────────────────────────────────────────

    # Penanda LLM "keluar karakter" / menolak — output begini TIDAK boleh
    # jadi draft LTKM (dokumen hukum). Kalau muncul -> fallback template.
    _REFUSAL_MARKERS = (
        "as an ai", "sebagai ai", "i cannot", "i can't", "saya tidak bisa",
        "i'm unable", "language model", "model bahasa", "as a language",
        "i apologize", "mohon maaf, saya",
    )

    def _validate_llm_output(self, text: str, account_id: str) -> tuple[bool, str]:
        """Guard output LLM sebelum dipakai jadi draft LTKM (fix 20-Jul).

        LTKM = dokumen HUKUM dikirim ke PPATK. Kalau LLM halusinasi / menolak /
        output rusak, itu TIDAK boleh lolos jadi laporan. Kalau gagal salah
        satu cek -> caller fallback ke template deterministik (yg grounded ke
        data asli). Ini BUKAN deteksi halusinasi sempurna (mustahil), tapi
        jaring pengaman thd mode-gagal yg paling nyata & berbahaya.
        """
        if not text or len(text.strip()) < 40:
            return False, "output kosong/terlalu pendek (LLM gagal)"
        if len(text) > 4000:
            return False, "output terlalu panjang (kemungkinan runaway generation)"
        low = text.lower()
        for m in self._REFUSAL_MARKERS:
            if m in low:
                return False, f"LLM keluar-karakter/menolak (penanda: '{m}')"
        # Grounding: narasi ttg rekening spesifik HARUS menyebut ID-nya.
        # Kalau tidak, kemungkinan besar off-topic/halusinasi -> tak layak LTKM.
        if account_id and account_id.lower() not in low:
            return False, "narasi tak menyebut account_id (grounding lemah)"
        return True, ""

    def narrate(
        self,
        account_id: str,
        risk_score: float,
        typology: str,
        features: dict,
        rules: Optional[list[str]] = None,
    ) -> str:
        """
        Generate narasi LTKM untuk satu rekening.

        Returns str — narasi siap pakai (atau template jika LLM error/output
        gagal validasi guard, lihat _validate_llm_output).
        """
        rules = rules or []
        prompt = _build_prompt(account_id, risk_score, typology, features, rules)

        # Template = deterministic, grounded, tak perlu divalidasi.
        if self.provider == "template":
            return self._template_narrate(account_id, risk_score, typology, features, rules)

        try:
            if self.provider == "anthropic":
                out = self._call_anthropic(prompt)
            elif self.provider == "openai":
                out = self._call_openai(prompt)
            elif self.provider == "ollama":
                out = self._call_ollama(prompt)
            else:
                return self._template_narrate(account_id, risk_score, typology, features, rules)
        except Exception as e:
            logger.warning("LLM narrate gagal (%s): %s — pakai template", self.provider, e)
            return self._template_narrate(account_id, risk_score, typology, features, rules)

        # Guard output LLM sebelum dilepas jadi draft LTKM.
        ok, reason = self._validate_llm_output(out, account_id)
        if not ok:
            logger.warning("Output LLM (%s) DITOLAK guard: %s — fallback template",
                           self.provider, reason)
            return self._template_narrate(account_id, risk_score, typology, features, rules)
        return out

    def triage(self, alerts: list[dict]) -> list[dict]:
        """
        Prioritaskan alert berdasarkan urgensi investigasi.

        Kriteria (descending):
          1. severity HIGH lebih dulu
          2. typologi kritis (JUDOL_RING, LAYERING, DORMANT_ACTIVATION)
          3. risk_score tertinggi

        Returns list[dict] terurut, dengan tambahan field `triage_rank`.
        """
        _TYPOLOGY_WEIGHT = {
            "JUDOL_RING": 4, "LAYERING": 4, "DORMANT_ACTIVATION": 3,
            "STRUCTURING": 3, "RAPID_IN_OUT": 3, "FAN_OUT": 2,
            "QRIS_FRAUD": 2, "PEP_NETWORK": 2, "VENDOR_SHELL": 1,
            "SMURF_LAYERING": 2, "UNKNOWN": 0,
        }
        _SEVERITY_WEIGHT = {"HIGH": 10, "MEDIUM": 5, "LOW": 1}

        def _score(alert):
            sev   = _SEVERITY_WEIGHT.get(alert.get("severity", "LOW"), 1)
            typ   = _TYPOLOGY_WEIGHT.get(alert.get("typology", "UNKNOWN"), 0)
            risk  = float(alert.get("risk_score", 0))
            return sev * 10 + typ + risk

        sorted_alerts = sorted(alerts, key=_score, reverse=True)
        for rank, alert in enumerate(sorted_alerts, 1):
            alert["triage_rank"] = rank
        return sorted_alerts

    # ── Provider calls ──────────────────────────────────────────────────────

    def _call_anthropic(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    def _call_openai(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    def _call_ollama(self, prompt: str) -> str:
        import urllib.request
        import json

        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())["response"].strip()

    # ── Template fallback ───────────────────────────────────────────────────

    def _template_narrate(
        self,
        account_id: str,
        risk_score: float,
        typology: str,
        features: dict,
        rules: list[str],
    ) -> str:
        ppatk_ref, ppatk_desc = _PPATK_REF.get(typology, _PPATK_REF["UNKNOWN"])
        severity = "tinggi" if risk_score >= 0.8 else "sedang"

        # Pilih indikator paling menonjol
        indicators = []
        dormancy = features.get("dormancy_days", 0)
        if dormancy > 30:
            indicators.append(f"masa dormant {dormancy:.0f} hari")
        burst = features.get("burst_ratio", 0)
        if burst > 0.5:
            indicators.append(f"rasio burst {burst:.1%}")
        struct = features.get("structuring_score", 0)
        if struct > 0.3:
            indicators.append(f"skor structuring {struct:.1%}")
        night = features.get("night_tx_ratio", 0)
        if night > 0.4:
            indicators.append(f"transaksi malam {night:.1%}")
        in_amt = features.get("in_amount_sum", 0)
        if in_amt > 0:
            indicators.append(f"total dana masuk Rp {in_amt:,.0f}")

        indikator_str = (
            ", ".join(indicators[:3]) if indicators else "pola transaksi tidak wajar"
        )
        rule_str = rules[0] if rules else ppatk_desc

        return (
            f"Rekening {account_id} teridentifikasi memiliki risiko {severity} "
            f"dengan skor {risk_score:.4f} berdasarkan analisis model ensemble MuleRadar. "
            f"Rekening ini menunjukkan indikator {indikator_str}, "
            f"yang konsisten dengan {ppatk_desc} ({ppatk_ref}). "
            f"Rule yang terpicu: {rule_str}. "
            f"Direkomendasikan untuk dilakukan pemeriksaan lebih lanjut dan "
            f"apabila terkonfirmasi, segera laporkan melalui GRIPS PPATK."
        )
