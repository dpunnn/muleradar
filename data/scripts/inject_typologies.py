"""
Inject 7 typologi fraud Indonesia ke atas AMLWorld base data.
Parameter dikalibrasi berdasarkan:
  - PPATK Laporan Tipologi & Laporan Tahunan 2024
  - Bareskrim kasus konkret (PT AST/TDC, judol ring Jan 2026)
  - BI statistik BI-FAST 2024 (rata-rata Rp 2,6 juta/transaksi)
  - FATF Indonesia Mutual Evaluation 2023

Cara pakai:
  python inject_typologies.py --input ../processed/transactions_li.csv \\
                               --output ../processed/transactions_li_injected.csv

Typologi (sesuai proposal):
  1. judol       - Judi online deposit ring (PPATK: 32,1% STR 2024)
  2. scam        - Transfer penipuan berantai
  3. dormant     - Rekening tidur tiba-tiba aktif (>140rb rekening, Rp428,6M)
  4. pep         - PEP layering via perantara (11% STR Jan-Sep 2020)
  5. vendor      - Vendor cangkang (kasus PT AST/TDC: Rp530M, 197 rekening)
  6. bendahara   - Bendahara/APBD korupsi
  7. qris        - QRIS merchant fraud ring (kasus: 15 PT, 21 situs judol)
"""

import argparse
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 99
random.seed(SEED)
np.random.seed(SEED)

BASE_TS = datetime(2022, 1, 1)   # aligned ke AMLWorld timestamp range

# ── Distribusi baseline BI (kalibrasi B) ──────────────────────────────────────
# Sumber: BI-FAST 2024 (Rp 2,6 juta rata-rata, skewed right)
# Normal transaction: log-normal, median ~Rp 500rb, mean ~Rp 2,6 juta
_BI_AMOUNT_MU    = 13.1   # ln(Rp 490rb) ≈ 13.1
_BI_AMOUNT_SIGMA = 1.8    # heavy right tail

# Threshold PPATK (Sumber: PPATK peraturan pelaporan)
_LTKT_THRESHOLD  = 500_000_000   # Rp 500 juta → wajib lapor (cash)
_SUB_THRESHOLD   = 499_000_000   # selalu di bawah ini untuk semua tipologi

# Channel weights (AMLWorld-calibrated, IDR)
_AML_CURRENCIES = ["IDR"]  # semua transaksi IDR untuk typologi Indonesia
_AML_CHANNELS   = ["mobile", "internet", "atm", "qris", "teller"]
_AML_CHAN_W     = [0.546, 0.149, 0.148, 0.097, 0.060]
_AML_FORMATS    = ["ACH", "Cheque", "Credit Card", "Cash", "Bitcoin"]
_AML_FMT_W      = [0.658, 0.183, 0.095, 0.051, 0.013]
_AML_BANKS      = ["BANK_A", "BANK_B"]
_MULTI_BANKS    = ["BANK_A", "BANK_B", "BANK_C", "BANK_D",
                   "BANK_E", "BANK_F", "BANK_G", "BANK_H"]  # 8 bank (kasus vendor)


def _idr_normal_amount() -> float:
    """Sample amount sesuai distribusi transaksi normal Indonesia (log-normal BI-FAST)."""
    return round(np.random.lognormal(_BI_AMOUNT_MU, _BI_AMOUNT_SIGMA) / 1000) * 1000


def _aml_amount() -> float:
    """Sample amount sesuai distribusi AMLWorld illicit (multi-currency, heavy tail)."""
    p = random.random()
    if p < 0.10:   return round(random.uniform(0.001, 100), 3)
    elif p < 0.50: return round(random.uniform(100, 10_000), 2)
    elif p < 0.80: return round(random.uniform(10_000, 200_000), 2)
    elif p < 0.95: return round(random.uniform(200_000, 5_000_000), 2)
    else:          return round(random.uniform(5_000_000, 500_000_000), 2)


def _aml_meta() -> dict:
    return {
        "currency":       random.choices(
            ["US Dollar","Euro","Yuan","Rupee","Yen","Canadian Dollar",
             "Ruble","Swiss Franc","UK Pound","Australian Dollar",
             "Brazil Real","Mexican Peso","Shekel","Saudi Riyal","Bitcoin"],
            [0.38,0.25,0.055,0.039,0.036,0.035,0.034,0.031,0.030,0.029,
             0.021,0.020,0.018,0.014,0.013]
        )[0],
        "channel":        random.choices(_AML_CHANNELS, _AML_CHAN_W)[0],
        "payment_format": random.choices(_AML_FORMATS, _AML_FMT_W)[0],
        "institution_id": random.choice(_AML_BANKS),
    }


def _ts(day_offset: int, hour: int = None, minute: int = None) -> datetime:
    h = hour if hour is not None else random.randint(0, 23)
    m = minute if minute is not None else random.randint(0, 59)
    return BASE_TS + timedelta(days=day_offset, hours=h, minutes=m)


def _txid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


# ── 1. JUDOL — Judi Online Deposit Ring ──────────────────────────────────────
# Parameter: PPATK 2024, Bareskrim Jan 2026 (21 situs, 15 PT, 2 agregator)
# Amount: Rp 10rb–5jt smurfing (PPATK: "deposit dipecah untuk hindari FDS")
# Timing: peak 20:00-02:00 WIB (perilaku judi online)
# Struktur: pemain → PT QRIS fiktif (15) → aggregator (2) → crypto/overseas exit

def inject_judol(
    n_players: int = 5000,
    n_pt_qris: int = 15,       # PT QRIS fiktif (sesuai kasus Bareskrim)
    n_aggregators: int = 2,    # PT penampung akhir (sesuai kasus)
    n_transactions: int = 80000,
) -> pd.DataFrame:
    players     = [f"JUDOL-PLY-{i:05d}" for i in range(n_players)]
    pt_qris     = [f"JUDOL-PT-{i:02d}"  for i in range(n_pt_qris)]
    aggregators = [f"JUDOL-AGG-{i:02d}" for i in range(n_aggregators)]
    crypto_exit = "JUDOL-CRYPTO-EXIT"

    # Jam dengan bobot: peak 20-02 WIB, aktif 24/7
    _judol_hour_weights = [
        5,4,4,4,4,3, 2,2,2,2,2,3,  # 00-11
        3,3,3,4,5,6, 7,8,9,9,8,7   # 12-23
    ]
    rows = []

    # Layer 1: pemain → PT QRIS fiktif (smurfing, amount kecil)
    for _ in range(n_transactions):
        hour = random.choices(range(24), weights=_judol_hour_weights, k=1)[0]
        # Smurfing: mayoritas kecil Rp 10rb–500rb, kadang sampai Rp 5jt
        p = random.random()
        if p < 0.60:    amount = round(random.uniform(10_000, 100_000) / 1000) * 1000
        elif p < 0.90:  amount = round(random.uniform(100_000, 500_000) / 1000) * 1000
        else:           amount = round(random.uniform(500_000, 5_000_000) / 1000) * 1000

        rows.append({
            "tx_id": _txid("JDL"),
            "from_account": random.choice(players),
            "to_account": random.choice(pt_qris),
            "amount": amount,
            "currency": "IDR", "channel": "qris",
            "payment_format": "QRIS",
            "tx_timestamp": _ts(random.randint(0, 364), hour),
            "device_id": f"DEV-PLY-{random.randint(1, n_players // 3):05d}",
            "institution_id": random.choice(_MULTI_BANKS[:4]),
            "is_laundering": 1, "typology": "judol",
        })

    # Layer 2: PT QRIS → aggregator (settlement harian)
    for pt in pt_qris:
        for day in range(0, 365, 1):
            if random.random() < 0.7:   # tidak setiap hari (realistis)
                rows.append({
                    "tx_id": _txid("JDL-PT"),
                    "from_account": pt,
                    "to_account": random.choice(aggregators),
                    "amount": round(random.uniform(3_000_000, 30_000_000)),
                    "currency": "IDR", "channel": "internet",
                    "payment_format": "Transfer",
                    "tx_timestamp": _ts(day, random.randint(22, 23), random.randint(0, 59)),
                    "device_id": f"DEV-PT-{pt[-2:]}",
                    "institution_id": random.choice(_MULTI_BANKS[:4]),
                    "is_laundering": 1, "typology": "judol",
                })

    # Layer 3: aggregator → crypto/overseas (cash-out)
    for agg in aggregators:
        for day in range(0, 365, 3):
            rows.append({
                "tx_id": _txid("JDL-OUT"),
                "from_account": agg,
                "to_account": crypto_exit,
                "amount": round(random.uniform(50_000_000, 300_000_000)),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day, random.randint(1, 4)),
                "device_id": f"DEV-AGG-{agg[-2:]}",
                "institution_id": "BANK_A",
                "is_laundering": 1, "typology": "judol",
            })

    return pd.DataFrame(rows)


# ── 2. SCAM — Transfer Penipuan Berantai ──────────────────────────────────────
# Parameter: PPATK + OJK 2024 (Rp 2,5 triliun kerugian dari 155rb pengaduan)
# Amount: Rp 20rb–5jt per hop (kecil, korban tidak sadar)
# Timing: malam/akhir pekan (korban panik, sulit konfirmasi)
# Velocity: < 24 jam per hop (harus cepat sebelum diblokir)

def inject_scam(n_victims: int = 3000, n_chains: int = 300) -> pd.DataFrame:
    rows = []
    for c in range(n_chains):
        mule1    = f"SCAM-M1-{c:04d}"
        mule2    = f"SCAM-M2-{c:04d}"
        mule3    = f"SCAM-M3-{c:04d}"
        mule4    = f"SCAM-M4-{c:04d}"
        exit_acc = f"SCAM-EXIT-{c % 20:02d}"
        # Malam/akhir pekan
        base_day = random.randint(0, 360)
        is_weekend = (base_day % 7) in [5, 6]
        base_hour  = random.randint(20, 23) if (is_weekend or random.random() < 0.6) else random.randint(10, 16)

        # Tiap korban transfer kecil ke mule1
        n_vic_this_chain = n_victims // n_chains
        if n_vic_this_chain == 0:
            continue
        sum_victims = 0
        for v in range(n_vic_this_chain):
            victim = f"SCAM-VIC-{c:04d}-{v:03d}"
            amount_v = round(random.uniform(20_000, 5_000_000) / 1000) * 1000
            sum_victims += amount_v
            rows.append({
                "tx_id": _txid(f"SCM-V{c}"),
                "from_account": victim, "to_account": mule1,
                "amount": amount_v, "currency": "IDR",
                "channel": random.choice(["mobile", "internet"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(base_day, base_hour, v % 60),
                "device_id": f"DEV-VIC-{v:03d}",
                "institution_id": random.choice(["BANK_A", "BANK_B"]),
                "is_laundering": 1, "typology": "scam",
            })

        # Chain: mule1 → mule2 → mule3 → mule4 → exit (< 24 jam per hop)
        total_in = sum_victims * random.uniform(0.7, 0.9)
        chain = [
            (mule1, mule2, total_in * 0.95, base_day, base_hour + 1),
            (mule2, mule3, total_in * 0.90, base_day, base_hour + 3),
            (mule3, mule4, total_in * 0.85, base_day, base_hour + 6),
            (mule4, exit_acc, total_in * 0.80, base_day, base_hour + 10),
        ]
        for src, dst, amt, d, h in chain:
            rows.append({
                "tx_id": _txid(f"SCM-C{c}"),
                "from_account": src, "to_account": dst,
                "amount": max(10_000, round(amt / max(1, n_vic_this_chain))),
                "currency": "IDR",
                "channel": random.choice(["mobile", "internet", "atm"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(d, min(h, 23)),
                "device_id": f"DEV-SCM-{c:04d}",
                "institution_id": random.choice(["BANK_A", "BANK_B"]),
                "is_laundering": 1, "typology": "scam",
            })

    return pd.DataFrame(rows)


# ── 3. DORMANT — Rekening Tidur Tiba-Tiba Aktif ───────────────────────────────
# Parameter: PPATK (>140rb rekening dormant >10 tahun, total Rp428,6M)
# Gap: 180–365 hari dormant sebelum aktivasi
# Aktivasi: tiba-tiba volume tinggi, tidak ada ramp-up bertahap
# Amount: Rp 500rb–50jt per transaksi setelah aktivasi (tidak konsisten profil)

def inject_dormant(n_accounts: int = 2000) -> pd.DataFrame:
    rows = []
    for i in range(n_accounts):
        acc        = f"DORM-{i:04d}"
        activator  = f"DORM-SRC-{i % 50:03d}"
        exit_acc   = f"DORM-EXIT-{i % 20:02d}"

        # Gap dormant: 180-365 hari (PPATK: >10 tahun, kita model 6-12 bulan)
        dormant_gap  = random.randint(180, 365)
        activation_day = dormant_gap + random.randint(0, 30)

        # Transaksi terakhir sebelum dormant (profil lama, kecil-normal)
        last_active_day = activation_day - dormant_gap
        if last_active_day >= 0:
            rows.append({
                "tx_id": _txid("DRM-OLD"),
                "from_account": f"DORM-HIST-{i % 100:03d}",
                "to_account": acc,
                "amount": round(random.uniform(100_000, 2_000_000)),
                "currency": "IDR", "channel": "teller",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(max(0, last_active_day), random.randint(9, 15)),
                "device_id": f"DEV-DORM-OLD-{i:04d}",
                "institution_id": "BANK_A", "is_laundering": 0, "typology": None,
            })

        # Aktivasi: transfer besar masuk (tidak konsisten profil)
        rows.append({
            "tx_id": _txid("DRM-ACT"),
            "from_account": activator, "to_account": acc,
            "amount": round(random.uniform(5_000_000, 50_000_000)),
            "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(activation_day, random.randint(0, 6)),
            "device_id": f"DEV-DORM-NEW-{i:04d}",  # device baru (beda dari sebelumnya)
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "dormant",
        })

        # Burst setelah aktivasi: banyak transaksi dalam 1-3 hari
        n_burst = random.randint(10, 30)
        for j in range(n_burst):
            burst_hour = random.choices(range(24), weights=[
                5,5,5,5,5,3, 2,2,2,2,2,3,
                3,3,3,4,5,6, 7,8,8,8,7,6], k=1)[0]
            rows.append({
                "tx_id": _txid("DRM-BST"),
                "from_account": acc,
                "to_account": f"DORM-RCP-{random.randint(0, 49):02d}",
                "amount": round(random.uniform(500_000, 20_000_000) / 1000) * 1000,
                "currency": "IDR",
                "channel": random.choice(["mobile", "atm"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(activation_day + j // 10, burst_hour),
                "device_id": f"DEV-DORM-NEW-{i:04d}",
                "institution_id": random.choice(["BANK_A", "BANK_B"]),
                "is_laundering": 1, "typology": "dormant",
            })

        # Cash-out akhir
        rows.append({
            "tx_id": _txid("DRM-EXIT"),
            "from_account": acc, "to_account": exit_acc,
            "amount": round(random.uniform(10_000_000, 100_000_000)),
            "currency": "IDR", "channel": "atm",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(activation_day + 3, random.randint(1, 4)),
            "device_id": f"DEV-DORM-NEW-{i:04d}",
            "institution_id": "BANK_B", "is_laundering": 1, "typology": "dormant",
        })

    return pd.DataFrame(rows)


# ── 4. PEP — Politically Exposed Person Layering ──────────────────────────────
# Parameter: PPATK NRA 2021 (11% STR, korupsi dominan)
# Amount: Rp 100jt – < Rp 500jt per transaksi (di bawah threshold LTKT Rp 500jt)
# Timing: jam kerja 09:00-17:00 WIB (menyamar transaksi bisnis)
# Hop count: 3-5 layer via nominee/PT cangkang

def inject_pep(n_pep: int = 200, n_intermediaries: int = 4) -> pd.DataFrame:
    rows = []
    for p in range(n_pep):
        pep_acc       = f"PEP-MAIN-{p:03d}"
        source        = f"PEP-SRC-{p:03d}"
        intermediaries = [f"PEP-INT-{p:03d}-{j}" for j in range(n_intermediaries)]
        shell_a       = f"PEP-SHELL-A-{p:03d}"
        shell_b       = f"PEP-SHELL-B-{p:03d}"
        day           = random.randint(0, 330)

        # Amount selalu di bawah LTKT threshold (Rp 100jt–499jt)
        amount = round(random.uniform(100_000_000, _SUB_THRESHOLD) / 1_000_000) * 1_000_000

        # Source → intermediary 1 (jam kerja)
        rows.append({
            "tx_id": _txid("PEP-S"),
            "from_account": source, "to_account": intermediaries[0],
            "amount": amount, "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day, random.randint(9, 12)),
            "device_id": f"DEV-PEP-{p:03d}",
            "institution_id": random.choice(_MULTI_BANKS[:4]),
            "is_laundering": 1, "typology": "pep",
        })

        # Intermediary chain (3-5 hop, lintas bank)
        for j in range(len(intermediaries) - 1):
            amt_hop = round(amount * (0.97 ** (j + 1)) / 1_000_000) * 1_000_000
            rows.append({
                "tx_id": _txid("PEP-L"),
                "from_account": intermediaries[j],
                "to_account": intermediaries[j + 1],
                "amount": max(10_000_000, amt_hop),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + j + 1, random.randint(10, 15)),
                "device_id": f"DEV-PEP-INT-{j}",
                "institution_id": random.choice(_MULTI_BANKS),  # lintas bank
                "is_laundering": 1, "typology": "pep",
            })

        # → PT cangkang A
        rows.append({
            "tx_id": _txid("PEP-SHA"),
            "from_account": intermediaries[-1], "to_account": shell_a,
            "amount": round(amount * 0.88 / 1_000_000) * 1_000_000,
            "currency": "IDR", "channel": "internet", "payment_format": "Transfer",
            "tx_timestamp": _ts(day + n_intermediaries + 1, random.randint(14, 16)),
            "device_id": f"DEV-PEP-SH-{p:03d}",
            "institution_id": random.choice(_MULTI_BANKS),
            "is_laundering": 1, "typology": "pep",
        })

        # → PT cangkang B → PEP (placement sebagai "pendapatan bisnis")
        rows.append({
            "tx_id": _txid("PEP-SHB"),
            "from_account": shell_a, "to_account": shell_b,
            "amount": round(amount * 0.85 / 1_000_000) * 1_000_000,
            "currency": "IDR", "channel": "internet", "payment_format": "Transfer",
            "tx_timestamp": _ts(day + n_intermediaries + 3, random.randint(9, 11)),
            "device_id": f"DEV-PEP-SH-{p:03d}",
            "institution_id": random.choice(_MULTI_BANKS),
            "is_laundering": 1, "typology": "pep",
        })
        rows.append({
            "tx_id": _txid("PEP-P"),
            "from_account": shell_b, "to_account": pep_acc,
            "amount": round(amount * 0.82 / 1_000_000) * 1_000_000,
            "currency": "IDR", "channel": "internet", "payment_format": "Transfer",
            "tx_timestamp": _ts(day + n_intermediaries + 5, random.randint(9, 14)),
            "device_id": f"DEV-PEP-{p:03d}",
            "institution_id": random.choice(_MULTI_BANKS),
            "is_laundering": 1, "typology": "pep",
        })

    return pd.DataFrame(rows)


# ── 5. VENDOR CANGKANG — Layering via Perusahaan Fiktif ──────────────────────
# Parameter: kasus PT AST/TDC (Rp530M, 197 rekening, 8 bank)
# Amount: Rp 50jt – Rp 499jt (selalu di bawah LTKT threshold)
# Struktur: sumber → banyak PT cangkang → placement ke aset (obligasi, properti)
# Bank: lintas 8 bank berbeda untuk mempersulit tracing

def inject_vendor(n_vendors: int = 100, n_layers: int = 3) -> pd.DataFrame:
    rows = []
    for v in range(n_vendors):
        corp       = f"CORP-SRC-{v:03d}"
        # 3-4 layer, tiap layer 3-4 PT
        vendors    = [[f"VND-{v:03d}-L{l}-{k:02d}" for k in range(random.randint(3, 4))]
                      for l in range(n_layers)]
        placement  = f"VND-PLACE-{v:03d}"  # aset akhir (obligasi/properti)
        day        = random.randint(0, 300)

        # Amount Rp 50jt–499jt (sub-threshold LTKT)
        total = round(random.uniform(50_000_000, _SUB_THRESHOLD) / 1_000_000) * 1_000_000

        prev_layer = [corp]
        for l, layer in enumerate(vendors):
            for dst in layer:
                src = random.choice(prev_layer)
                amt_layer = round(total / len(layer) * random.uniform(0.8, 1.2)
                                  / 1_000_000) * 1_000_000
                rows.append({
                    "tx_id": _txid(f"VND-{v}-L{l}"),
                    "from_account": src, "to_account": dst,
                    "amount": max(5_000_000, min(amt_layer, _SUB_THRESHOLD)),
                    "currency": "IDR", "channel": "internet",
                    "payment_format": "Transfer",
                    "tx_timestamp": _ts(day + l * 2, random.randint(9, 16)),
                    "device_id": f"DEV-VND-{v:03d}-L{l}",
                    "institution_id": random.choice(_MULTI_BANKS),  # 8 bank
                    "is_laundering": 1, "typology": "vendor",
                })
            prev_layer = layer

        # Placement: PT layer terakhir → aset
        for node in vendors[-1]:
            rows.append({
                "tx_id": _txid("VND-PLC"),
                "from_account": node, "to_account": placement,
                "amount": round(random.uniform(10_000_000, 200_000_000) / 1_000_000) * 1_000_000,
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + n_layers * 2 + 2, random.randint(9, 14)),
                "device_id": f"DEV-VND-PLC-{v:03d}",
                "institution_id": random.choice(_MULTI_BANKS),
                "is_laundering": 1, "typology": "vendor",
            })

    return pd.DataFrame(rows)


# ── 6. BENDAHARA — Korupsi APBD/Dana Desa ────────────────────────────────────
# Parameter: korupsi pemerintah daerah Indonesia (Bareskrim, KPK kasus 2022-2024)
# Amount: Rp 50jt–500jt per transaksi (anggaran proyek APBD)
# Timing: jam kerja (menyamar sebagai pencairan anggaran resmi)

def inject_bendahara(n_bendahara: int = 500) -> pd.DataFrame:
    rows = []
    for b in range(n_bendahara):
        gov_acc    = f"GOV-DINAS-{b:03d}"
        bendahara  = f"BND-{b:03d}"
        recipients = [f"BND-RCP-{b:03d}-{i:02d}" for i in range(random.randint(3, 10))]
        private_acc = f"BND-PRIV-{b:03d}"
        day        = random.randint(0, 330)

        # Pencairan anggaran (tampak resmi, jam kerja)
        budget = round(random.uniform(50_000_000, _SUB_THRESHOLD) / 1_000_000) * 1_000_000
        rows.append({
            "tx_id": _txid("BND-GOV"),
            "from_account": gov_acc, "to_account": bendahara,
            "amount": budget, "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day, random.randint(8, 10)),
            "device_id": f"DEV-BND-{b:03d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "bendahara",
        })

        # Distribusi ke vendor fiktif (sub-threshold per transaksi)
        for rcp in recipients:
            rows.append({
                "tx_id": _txid("BND-OUT"),
                "from_account": bendahara, "to_account": rcp,
                "amount": round(random.uniform(5_000_000, 80_000_000) / 1_000_000) * 1_000_000,
                "currency": "IDR",
                "channel": random.choice(["internet", "teller"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + random.randint(1, 3), random.randint(9, 15)),
                "device_id": f"DEV-BND-{b:03d}",
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "bendahara",
            })

        # Kickback ke rekening pribadi bendahara (malam hari, ATM)
        kickback = round(budget * random.uniform(0.10, 0.25) / 1_000_000) * 1_000_000
        rows.append({
            "tx_id": _txid("BND-KICK"),
            "from_account": bendahara, "to_account": private_acc,
            "amount": kickback, "currency": "IDR", "channel": "atm",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day + 5, random.randint(20, 23)),
            "device_id": f"DEV-BND-PRIV-{b:03d}",
            "institution_id": "BANK_B", "is_laundering": 1, "typology": "bendahara",
        })

    return pd.DataFrame(rows)


# ── 7. QRIS — Merchant Fraud Ring ────────────────────────────────────────────
# Parameter: kasus Bareskrim Jan 2026 (15 PT fiktif, 2 aggregator, 21 situs judol, 11 PJSP)
# Struktur: pemain judol → 15 PT QRIS fiktif → 2 PT penampung → exit
# Amount: Rp 10rb–250jt (QRIS, semua ukuran)

def inject_qris(
    n_pt_fiktif: int = 15,      # sesuai kasus Bareskrim
    n_penampung: int = 2,        # PT penampung akhir
    n_pjsp: int = 11,            # jumlah payment provider (sesuai kasus)
    n_transactions: int = 60000,
) -> pd.DataFrame:
    pt_fiktif  = [f"QRIS-PT-{i:02d}"    for i in range(n_pt_fiktif)]
    penampung  = [f"QRIS-PMP-{i:02d}"   for i in range(n_penampung)]
    pjsp_banks = _MULTI_BANKS[:min(n_pjsp, len(_MULTI_BANKS))]
    exit_acc   = "QRIS-EXIT-OVERSEAS"
    shared_dev = f"DEV-QRIS-CTRL-{random.randint(1000,9999)}"
    rows       = []

    # Layer 1: pemain judol → PT QRIS fiktif (pembayaran kecil)
    for _ in range(n_transactions):
        buyer = f"QRIS-BUY-{random.randint(0, 9999):04d}"
        p = random.random()
        if p < 0.70:    amount = round(random.uniform(10_000, 100_000) / 1000) * 1000
        elif p < 0.95:  amount = round(random.uniform(100_000, 2_000_000) / 1000) * 1000
        else:           amount = round(random.uniform(2_000_000, 50_000_000))

        hour = random.choices(range(24), weights=[
            5,4,4,4,4,3, 2,2,2,2,2,3,
            3,3,3,4,5,6, 7,8,9,9,8,7], k=1)[0]

        rows.append({
            "tx_id": _txid("QRIS-IN"),
            "from_account": buyer,
            "to_account": random.choice(pt_fiktif),
            "amount": amount, "currency": "IDR", "channel": "qris",
            "payment_format": "QRIS",
            "tx_timestamp": _ts(random.randint(0, 364), hour),
            "device_id": f"DEV-BUY-{random.randint(1, 2000):04d}",
            "institution_id": random.choice(pjsp_banks),
            "is_laundering": 1, "typology": "qris",
        })

    # Layer 2: PT fiktif → PT penampung (settlement harian, malam)
    for pt in pt_fiktif:
        for day in range(0, 365, 1):
            if random.random() < 0.65:
                rows.append({
                    "tx_id": _txid("QRIS-SET"),
                    "from_account": pt,
                    "to_account": random.choice(penampung),
                    "amount": round(random.uniform(5_000_000, 50_000_000)),
                    "currency": "IDR", "channel": "internet",
                    "payment_format": "Transfer",
                    "tx_timestamp": _ts(day, 23, random.randint(0, 59)),
                    "device_id": shared_dev,
                    "institution_id": random.choice(pjsp_banks),
                    "is_laundering": 1, "typology": "qris",
                })

    # Layer 3: penampung → exit overseas
    for pmp in penampung:
        for day in range(0, 365, 5):
            rows.append({
                "tx_id": _txid("QRIS-EXIT"),
                "from_account": pmp, "to_account": exit_acc,
                "amount": round(random.uniform(100_000_000, _SUB_THRESHOLD)),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day, random.randint(2, 5)),
                "device_id": shared_dev,
                "institution_id": "BANK_A",
                "is_laundering": 1, "typology": "qris",
            })

    return pd.DataFrame(rows)


# ── AML BASE PATTERNS (replikasi pola AMLWorld) ───────────────────────────────

def inject_aml_fanout(n_clusters: int = 2000) -> pd.DataFrame:
    rows = []
    for c in range(n_clusters):
        source = f"AML-FO-SRC-{c:05d}"
        n_dst  = random.randint(20, 150)
        dsts   = [f"AML-FO-DST-{c:05d}-{i:03d}" for i in range(n_dst)]
        dev    = f"DEV-FO-{c:05d}"
        day    = random.randint(0, 350)
        for dst in dsts:
            rows.append({
                "tx_id": _txid("AML-FO"),
                "from_account": source, "to_account": dst,
                "amount": _aml_amount(),
                "tx_timestamp": _ts(day + random.randint(0, 4), random.randint(0, 23)),
                "device_id": dev if random.random() < 0.4 else f"DEV-FO-R-{random.randint(0,999):03d}",
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
    return pd.DataFrame(rows)


def inject_aml_fanin(n_clusters: int = 1000) -> pd.DataFrame:
    rows = []
    for c in range(n_clusters):
        collector = f"AML-FI-COL-{c:05d}"
        exit_acc  = f"AML-FI-EXIT-{c % 100:03d}"
        n_src     = random.randint(15, 80)
        day       = random.randint(0, 345)
        dev       = f"DEV-FI-{c:05d}"
        for i in range(n_src):
            rows.append({
                "tx_id": _txid("AML-FI"),
                "from_account": f"AML-FI-SRC-{c:05d}-{i:03d}",
                "to_account": collector,
                "amount": _aml_amount(),
                "tx_timestamp": _ts(day + random.randint(0, 6), random.randint(0, 23)),
                "device_id": f"DEV-FI-S-{random.randint(0,999):03d}",
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
        for _ in range(random.randint(2, 8)):
            rows.append({
                "tx_id": _txid("AML-FI-OUT"),
                "from_account": collector, "to_account": exit_acc,
                "amount": _aml_amount(),
                "tx_timestamp": _ts(day + 7 + random.randint(0, 3), random.randint(1, 6)),
                "device_id": dev,
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
    return pd.DataFrame(rows)


def inject_aml_layering(n_chains: int = 5000) -> pd.DataFrame:
    rows = []
    for c in range(n_chains):
        n_hops = random.randint(3, 8)
        accs   = [f"AML-LY-{c:05d}-{h}" for h in range(n_hops + 1)]
        day    = random.randint(0, 355)
        amount = _aml_amount()
        dev    = f"DEV-LY-{c:05d}"
        for h in range(n_hops):
            amount = amount * random.uniform(0.88, 0.99)
            rows.append({
                "tx_id": _txid("AML-LY"),
                "from_account": accs[h], "to_account": accs[h + 1],
                "amount": round(amount, 2),
                "tx_timestamp": _ts(day + h, random.randint(8, 22)),
                "device_id": dev,
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
    return pd.DataFrame(rows)


def inject_aml_scatter_gather(n_clusters: int = 1000) -> pd.DataFrame:
    rows = []
    for c in range(n_clusters):
        source = f"AML-SG-SRC-{c:05d}"
        sink   = f"AML-SG-SNK-{c:05d}"
        n_mid  = random.randint(5, 20)
        mids   = [f"AML-SG-MID-{c:05d}-{i:02d}" for i in range(n_mid)]
        day    = random.randint(0, 345)
        total  = _aml_amount() * random.uniform(10, 100)
        dev    = f"DEV-SG-{c:05d}"
        for mid in mids:
            amt = total / n_mid * random.uniform(0.7, 1.3)
            rows.append({
                "tx_id": _txid("AML-SG-S"),
                "from_account": source, "to_account": mid,
                "amount": round(amt, 2),
                "tx_timestamp": _ts(day, random.randint(8, 18)),
                "device_id": dev,
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
            rows.append({
                "tx_id": _txid("AML-SG-G"),
                "from_account": mid, "to_account": sink,
                "amount": round(amt * random.uniform(0.85, 0.98), 2),
                "tx_timestamp": _ts(day + random.randint(1, 5), random.randint(1, 8)),
                "device_id": f"DEV-SG-M-{random.randint(0,99):02d}",
                "is_laundering": 1, "typology": None,
                **_aml_meta(),
            })
    return pd.DataFrame(rows)


def inject_aml_cycle(n_cycles: int = 500) -> pd.DataFrame:
    rows = []
    for c in range(n_cycles):
        n_nodes  = random.randint(3, 7)
        accs     = [f"AML-CY-{c:04d}-{n}" for n in range(n_nodes)]
        day      = random.randint(0, 350)
        amount   = _aml_amount()
        dev      = f"DEV-CY-{c:04d}"
        n_rounds = random.randint(3, 10)
        for r in range(n_rounds):
            for i in range(n_nodes):
                amount = amount * random.uniform(0.90, 1.05)
                rows.append({
                    "tx_id": _txid("AML-CY"),
                    "from_account": accs[i],
                    "to_account": accs[(i + 1) % n_nodes],
                    "amount": round(amount, 2),
                    "tx_timestamp": _ts(day + r * 2 + i // n_nodes, random.randint(0, 23)),
                    "device_id": dev,
                    "is_laundering": 1, "typology": None,
                    **_aml_meta(),
                })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    final_cols = [
        "tx_id", "from_account", "to_account", "amount", "currency",
        "channel", "payment_format", "tx_timestamp", "device_id",
        "institution_id", "is_laundering", "typology",
    ]

    print("[1/3] Generate typologi + AML base patterns...")
    print("      Parameter dikalibrasi: PPATK Laporan Tipologi 2024 + BI-FAST stats")
    injections = [
        # 7 Typologi Indonesia (parameter PPATK-calibrated)
        ("judol",     inject_judol(n_players=5000, n_pt_qris=15, n_aggregators=2, n_transactions=80000)),
        ("scam",      inject_scam(n_victims=3000, n_chains=300)),
        ("dormant",   inject_dormant(n_accounts=2000)),
        ("pep",       inject_pep(n_pep=450, n_intermediaries=4)),
        ("vendor",    inject_vendor(n_vendors=180, n_layers=3)),
        ("bendahara", inject_bendahara(n_bendahara=500)),
        ("qris",      inject_qris(n_pt_fiktif=15, n_penampung=2, n_pjsp=11, n_transactions=60000)),
        # AML base patterns
        ("aml_fanout",         inject_aml_fanout(n_clusters=10000)),
        ("aml_fanin",          inject_aml_fanin(n_clusters=5000)),
        ("aml_layering",       inject_aml_layering(n_chains=25000)),
        ("aml_scatter_gather", inject_aml_scatter_gather(n_clusters=5000)),
        ("aml_cycle",          inject_aml_cycle(n_cycles=2500)),
    ]
    for name, inj_df in injections:
        print(f"      {name:20s}: {len(inj_df):>10,} transaksi")

    print(f"\n[2/3] Streaming base data → {args.output}...")
    total_base = 0
    for i, chunk in enumerate(pd.read_csv(args.input, chunksize=args.chunk_size, low_memory=False)):
        chunk = chunk.reindex(columns=final_cols)
        chunk.to_csv(args.output, mode="w" if i == 0 else "a", index=False, header=(i == 0))
        total_base += len(chunk)
        if i % 10 == 0:
            print(f"  streamed {total_base:,} rows...", end="\r")
    print(f"  streamed {total_base:,} rows base data")

    print("[3/3] Append injected rows...")
    inj_combined = pd.concat([x[1] for x in injections], ignore_index=True)
    inj_combined = inj_combined.reindex(columns=final_cols)
    inj_combined.to_csv(args.output, mode="a", index=False, header=False)

    total = total_base + len(inj_combined)
    print("\n=== SELESAI ===")
    print(f"Total transaksi      : {total:,}")
    print(f"Typologi injected    : {len(inj_combined):,}")
    print(f"\nBreakdown:")
    for name, inj_df in injections:
        print(f"  {name:20s}: {len(inj_df):>10,}")


if __name__ == "__main__":
    main()
