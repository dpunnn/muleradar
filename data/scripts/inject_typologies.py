"""
Inject 7 typologi fraud Indonesia ke atas AMLWorld base data.

Cara pakai:
  python inject_typologies.py --input ../processed/transactions_li.csv \
                               --output ../processed/transactions_li_injected.csv

Typologi:
  1. judol       - Judi online deposit ring
  2. scam        - Transfer penipuan berantai
  3. dormant     - Rekening tidur tiba-tiba aktif
  4. pep         - PEP layering via perantara
  5. vendor      - Vendor cangkang layering
  6. bendahara   - Bendahara/APBD korupsi
  7. qris        - QRIS merchant fraud ring
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

BASE_TS = datetime(2025, 1, 1)

# Karakteristik AMLWorld base (dari analisis LI-Large)
_AML_CURRENCIES = ["US Dollar","Euro","Yuan","Rupee","Yen","Canadian Dollar",
                   "Ruble","Swiss Franc","UK Pound","Australian Dollar",
                   "Brazil Real","Mexican Peso","Shekel","Saudi Riyal","Bitcoin"]
_AML_CUR_W      = [0.38,0.25,0.055,0.039,0.036,0.035,0.034,0.031,0.030,0.029,
                   0.021,0.020,0.018,0.014,0.013]
_AML_CHANNELS   = ["mobile","internet","atm","qris","teller"]
_AML_CHAN_W     = [0.546,0.149,0.148,0.097,0.060]
_AML_FORMATS    = ["ACH","Cheque","Credit Card","Cash","Bitcoin"]
_AML_FMT_W      = [0.658,0.183,0.095,0.051,0.013]
_AML_BANKS      = ["BANK_A","BANK_B"]


def _aml_amount() -> float:
    """Sample amount sesuai distribusi AMLWorld illicit (median ~4K, heavy tail)."""
    p = random.random()
    if p < 0.10:   return round(random.uniform(0.001, 100), 3)
    elif p < 0.50: return round(random.uniform(100, 10_000), 2)
    elif p < 0.80: return round(random.uniform(10_000, 200_000), 2)
    elif p < 0.95: return round(random.uniform(200_000, 5_000_000), 2)
    else:          return round(random.uniform(5_000_000, 500_000_000), 2)


def _aml_meta() -> dict:
    return {
        "currency":       random.choices(_AML_CURRENCIES, _AML_CUR_W)[0],
        "channel":        random.choices(_AML_CHANNELS,   _AML_CHAN_W)[0],
        "payment_format": random.choices(_AML_FORMATS,    _AML_FMT_W)[0],
        "institution_id": random.choice(_AML_BANKS),
    }


def _ts(day_offset: int, hour: int = None, minute: int = None) -> datetime:
    h = hour if hour is not None else random.randint(0, 23)
    m = minute if minute is not None else random.randint(0, 59)
    return BASE_TS + timedelta(days=day_offset, hours=h, minutes=m)


def _txid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10].upper()}"


# ── 1. JUDOL — Judi Online Deposit Ring ──────────────────────────────────────

def inject_judol(
    n_players: int = 800,
    n_collectors: int = 5,
    n_transactions: int = 12000,
) -> pd.DataFrame:
    players = [f"JUDOL-PLY-{i:04d}" for i in range(n_players)]
    collectors = [f"JUDOL-COL-{i:02d}" for i in range(n_collectors)]
    crypto_out = "JUDOL-CRYPTO-EXIT"
    shared_dev = f"DEV-JUDOL-{random.randint(1000,9999)}"
    rows = []

    for _ in range(n_transactions):
        hour = random.choices(range(24),
            weights=[2,2,2,2,2,1,1,1,2,2,2,2,2,2,2,2,3,4,5,6,7,8,9,6], k=1)[0]
        rows.append({
            "tx_id": _txid("JDL"),
            "from_account": random.choice(players),
            "to_account": random.choice(collectors),
            "amount": round(random.uniform(50_000, 500_000) / 1000) * 1000,
            "currency": "IDR", "channel": "mobile",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(random.randint(0, 89), hour),
            "device_id": shared_dev if random.random() < 0.3 else f"DEV-PLY-{random.randint(1,300):04d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "judol",
        })

    for col in collectors:
        for day in range(0, 90, 2):
            rows.append({
                "tx_id": _txid("JDL-OUT"),
                "from_account": col, "to_account": crypto_out,
                "amount": round(random.uniform(8_000_000, 40_000_000)),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day, random.randint(1, 5)),
                "device_id": shared_dev, "institution_id": "BANK_A",
                "is_laundering": 1, "typology": "judol",
            })

    return pd.DataFrame(rows)


# ── 2. SCAM — Transfer Penipuan Berantai ────────────────────────────────────

def inject_scam(n_victims: int = 200, n_chains: int = 20) -> pd.DataFrame:
    rows = []
    for c in range(n_chains):
        victim = f"SCAM-VIC-{c:03d}"
        mule1 = f"SCAM-M1-{c:03d}"
        mule2 = f"SCAM-M2-{c:03d}"
        mule3 = f"SCAM-M3-{c:03d}"
        exit_acc = f"SCAM-EXIT-{c % 5:02d}"
        day = random.randint(0, 80)
        amount = round(random.uniform(2_000_000, 50_000_000))

        chain = [
            (victim, mule1, amount, day, 9),
            (mule1, mule2, amount * 0.95, day, 9, 30),
            (mule2, mule3, amount * 0.90, day, 10),
            (mule3, exit_acc, amount * 0.85, day, 11),
        ]
        for i, hop in enumerate(chain):
            src, dst, amt, d = hop[0], hop[1], hop[2], hop[3]
            h = hop[4] if len(hop) > 4 else random.randint(9, 22)
            rows.append({
                "tx_id": _txid(f"SCM-{c}"),
                "from_account": src, "to_account": dst,
                "amount": round(amt), "currency": "IDR",
                "channel": random.choice(["mobile", "internet", "atm"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(d, h, i * 5),
                "device_id": f"DEV-SCM-{c:03d}",
                "institution_id": random.choice(["BANK_A", "BANK_B"]),
                "is_laundering": 1, "typology": "scam",
            })

        for v in range(n_victims // n_chains):
            victim_extra = f"SCAM-VIC-{c:03d}-{v:02d}"
            rows.append({
                "tx_id": _txid("SCM-V"),
                "from_account": victim_extra, "to_account": mule1,
                "amount": round(random.uniform(500_000, 5_000_000)),
                "currency": "IDR", "channel": "mobile",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(random.randint(0, day), random.randint(8, 21)),
                "device_id": f"DEV-VIC-{v:03d}",
                "institution_id": "BANK_B", "is_laundering": 1, "typology": "scam",
            })

    return pd.DataFrame(rows)


# ── 3. DORMANT — Rekening Tidur Tiba-Tiba Aktif ─────────────────────────────

def inject_dormant(n_accounts: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n_accounts):
        acc = f"DORM-{i:03d}"
        activator = f"DORM-SRC-{i % 10:02d}"
        exit_acc = f"DORM-EXIT-{i % 5:02d}"
        activation_day = random.randint(10, 70)

        rows.append({
            "tx_id": _txid("DRM-IN"),
            "from_account": activator, "to_account": acc,
            "amount": round(random.uniform(50_000_000, 500_000_000)),
            "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(activation_day, 8),
            "device_id": f"DEV-DORM-{i:03d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "dormant",
        })

        for j in range(random.randint(8, 20)):
            rows.append({
                "tx_id": _txid("DRM-OUT"),
                "from_account": acc,
                "to_account": f"DORM-RCP-{random.randint(0,19):02d}",
                "amount": round(random.uniform(1_000_000, 20_000_000)),
                "currency": "IDR",
                "channel": random.choice(["mobile", "atm", "internet"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(activation_day + j // 3, random.randint(8, 22)),
                "device_id": f"DEV-DORM-{i:03d}",
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "dormant",
            })

        rows.append({
            "tx_id": _txid("DRM-EXIT"),
            "from_account": acc, "to_account": exit_acc,
            "amount": round(random.uniform(30_000_000, 200_000_000)),
            "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(activation_day + 7, 2),
            "device_id": f"DEV-DORM-{i:03d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "dormant",
        })

    return pd.DataFrame(rows)


# ── 4. PEP — Politically Exposed Person Layering ────────────────────────────

def inject_pep(n_pep: int = 10, n_intermediaries: int = 3) -> pd.DataFrame:
    rows = []
    for p in range(n_pep):
        pep_acc = f"PEP-{p:02d}"
        source = f"PEP-SRC-{p:02d}"
        intermediaries = [f"PEP-INT-{p:02d}-{j}" for j in range(n_intermediaries)]
        shell = f"PEP-SHELL-{p:02d}"
        day = random.randint(0, 60)
        amount = round(random.uniform(100_000_000, 5_000_000_000))

        rows.append({
            "tx_id": _txid("PEP-S"),
            "from_account": source, "to_account": intermediaries[0],
            "amount": amount, "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day, 10),
            "device_id": f"DEV-PEP-{p:02d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "pep",
        })

        for j in range(len(intermediaries) - 1):
            rows.append({
                "tx_id": _txid("PEP-L"),
                "from_account": intermediaries[j],
                "to_account": intermediaries[j + 1],
                "amount": round(amount * (0.97 ** (j + 1))),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + j + 1, 14),
                "device_id": f"DEV-PEP-{p:02d}",
                "institution_id": random.choice(["BANK_A", "BANK_B"]),
                "is_laundering": 1, "typology": "pep",
            })

        rows.append({
            "tx_id": _txid("PEP-SH"),
            "from_account": intermediaries[-1], "to_account": shell,
            "amount": round(amount * 0.90), "currency": "IDR",
            "channel": "internet", "payment_format": "Transfer",
            "tx_timestamp": _ts(day + n_intermediaries + 1, 16),
            "device_id": f"DEV-PEP-{p:02d}",
            "institution_id": "BANK_B", "is_laundering": 1, "typology": "pep",
        })

        rows.append({
            "tx_id": _txid("PEP-P"),
            "from_account": shell, "to_account": pep_acc,
            "amount": round(amount * 0.85), "currency": "IDR",
            "channel": "internet", "payment_format": "Transfer",
            "tx_timestamp": _ts(day + n_intermediaries + 3, 9),
            "device_id": f"DEV-PEP-{p:02d}",
            "institution_id": "BANK_B", "is_laundering": 1, "typology": "pep",
        })

    return pd.DataFrame(rows)


# ── 5. VENDOR CANGKANG — Layering via Perusahaan Fiktif ─────────────────────

def inject_vendor(n_vendors: int = 15, n_layers: int = 3) -> pd.DataFrame:
    rows = []
    for v in range(n_vendors):
        corp = f"CORP-{v:02d}"
        vendors = [[f"VND-{v:02d}-L{l}-{k:02d}" for k in range(4)] for l in range(n_layers)]
        final_exit = f"VND-EXIT-{v % 5:02d}"
        day = random.randint(0, 60)
        amount = round(random.uniform(500_000_000, 10_000_000_000))

        prev_layer = [corp]
        for l, layer in enumerate(vendors):
            for dst in layer:
                src = random.choice(prev_layer)
                rows.append({
                    "tx_id": _txid(f"VND-{v}-L{l}"),
                    "from_account": src, "to_account": dst,
                    "amount": round(amount / len(layer) * random.uniform(0.8, 1.2)),
                    "currency": "IDR", "channel": "internet",
                    "payment_format": "Transfer",
                    "tx_timestamp": _ts(day + l * 3, random.randint(9, 17)),
                    "device_id": f"DEV-VND-{v:02d}",
                    "institution_id": random.choice(["BANK_A", "BANK_B"]),
                    "is_laundering": 1, "typology": "vendor",
                })
            prev_layer = layer

        for node in vendors[-1]:
            rows.append({
                "tx_id": _txid("VND-EXIT"),
                "from_account": node, "to_account": final_exit,
                "amount": round(random.uniform(10_000_000, 100_000_000)),
                "currency": "IDR", "channel": "atm",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + n_layers * 3 + 2, random.randint(1, 5)),
                "device_id": f"DEV-VND-{v:02d}",
                "institution_id": "BANK_B", "is_laundering": 1, "typology": "vendor",
            })

    return pd.DataFrame(rows)


# ── 6. BENDAHARA — Korupsi APBD/Dana Desa ───────────────────────────────────

def inject_bendahara(n_bendahara: int = 20) -> pd.DataFrame:
    rows = []
    for b in range(n_bendahara):
        gov_acc = f"GOV-DINAS-{b:02d}"
        bendahara = f"BND-{b:02d}"
        recipients = [f"BND-RCP-{b:02d}-{i:02d}" for i in range(random.randint(5, 15))]
        private_acc = f"BND-PRIV-{b:02d}"
        day = random.randint(0, 75)
        budget = round(random.uniform(200_000_000, 2_000_000_000))

        rows.append({
            "tx_id": _txid("BND-GOV"),
            "from_account": gov_acc, "to_account": bendahara,
            "amount": budget, "currency": "IDR", "channel": "internet",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day, 8),
            "device_id": f"DEV-BND-{b:02d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "bendahara",
        })

        for rcp in recipients:
            rows.append({
                "tx_id": _txid("BND-OUT"),
                "from_account": bendahara, "to_account": rcp,
                "amount": round(random.uniform(5_000_000, 50_000_000)),
                "currency": "IDR",
                "channel": random.choice(["mobile", "teller"]),
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day + random.randint(1, 5), random.randint(9, 16)),
                "device_id": f"DEV-BND-{b:02d}",
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "bendahara",
            })

        rows.append({
            "tx_id": _txid("BND-PRIV"),
            "from_account": bendahara, "to_account": private_acc,
            "amount": round(budget * random.uniform(0.10, 0.30)),
            "currency": "IDR", "channel": "atm",
            "payment_format": "Transfer",
            "tx_timestamp": _ts(day + 7, random.randint(20, 23)),
            "device_id": f"DEV-BND-PRIV-{b:02d}",
            "institution_id": "BANK_B", "is_laundering": 1, "typology": "bendahara",
        })

    return pd.DataFrame(rows)


# ── 7. QRIS — Merchant Fraud Ring ───────────────────────────────────────────

def inject_qris(n_merchants: int = 50, n_transactions: int = 5000) -> pd.DataFrame:
    merchants = [f"QRIS-MRC-{i:03d}" for i in range(n_merchants)]
    settlement = "QRIS-SETTLE-MASTER"
    shared_dev = f"DEV-QRIS-{random.randint(1000,9999)}"
    rows = []

    for _ in range(n_transactions):
        buyer = f"QRIS-BUY-{random.randint(0, 999):03d}"
        merchant = random.choice(merchants)
        rows.append({
            "tx_id": _txid("QRIS"),
            "from_account": buyer, "to_account": merchant,
            "amount": round(random.uniform(10_000, 200_000) / 1000) * 1000,
            "currency": "IDR", "channel": "qris",
            "payment_format": "QRIS",
            "tx_timestamp": _ts(random.randint(0, 89), random.randint(8, 22)),
            "device_id": f"DEV-BUY-{random.randint(1,500):03d}",
            "institution_id": random.choice(["BANK_A", "BANK_B"]),
            "is_laundering": 1, "typology": "qris",
        })

    for merchant in merchants:
        for day in range(0, 90, 3):
            rows.append({
                "tx_id": _txid("QRIS-SET"),
                "from_account": merchant, "to_account": settlement,
                "amount": round(random.uniform(500_000, 5_000_000)),
                "currency": "IDR", "channel": "internet",
                "payment_format": "Transfer",
                "tx_timestamp": _ts(day, 23),
                "device_id": shared_dev,
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "qris",
            })

    return pd.DataFrame(rows)


# ── AML BASE PATTERNS (replikasi pola AMLWorld) ──────────────────────────────

def inject_aml_fanout(n_clusters: int = 2000) -> pd.DataFrame:
    """Satu source kirim ke banyak tujuan — pola money mule distribusi dana."""
    rows = []
    for c in range(n_clusters):
        source  = f"AML-FO-SRC-{c:05d}"
        n_dst   = random.randint(20, 150)
        dsts    = [f"AML-FO-DST-{c:05d}-{i:03d}" for i in range(n_dst)]
        dev     = f"DEV-FO-{c:05d}"
        day     = random.randint(0, 85)
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
    """Banyak source kirim ke satu kolektor — pola pengumpulan dana."""
    rows = []
    for c in range(n_clusters):
        collector = f"AML-FI-COL-{c:05d}"
        exit_acc  = f"AML-FI-EXIT-{c % 100:03d}"
        n_src     = random.randint(15, 80)
        day       = random.randint(0, 80)
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
        # Cash-out dari kolektor
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
    """Chain A→B→C→D→E — pola layering bertahap."""
    rows = []
    for c in range(n_chains):
        n_hops  = random.randint(3, 8)
        accs    = [f"AML-LY-{c:05d}-{h}" for h in range(n_hops + 1)]
        day     = random.randint(0, 82)
        amount  = _aml_amount()
        dev     = f"DEV-LY-{c:05d}"
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
    """Source → banyak intermediary → satu sink — pola scatter-gather."""
    rows = []
    for c in range(n_clusters):
        source  = f"AML-SG-SRC-{c:05d}"
        sink    = f"AML-SG-SNK-{c:05d}"
        n_mid   = random.randint(5, 20)
        mids    = [f"AML-SG-MID-{c:05d}-{i:02d}" for i in range(n_mid)]
        day     = random.randint(0, 80)
        total   = _aml_amount() * random.uniform(10, 100)
        dev     = f"DEV-SG-{c:05d}"
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
    """A→B→C→A — perputaran dana melingkar untuk obscure origin."""
    rows = []
    for c in range(n_cycles):
        n_nodes = random.randint(3, 7)
        accs    = [f"AML-CY-{c:04d}-{n}" for n in range(n_nodes)]
        day     = random.randint(0, 75)
        amount  = _aml_amount()
        dev     = f"DEV-CY-{c:04d}"
        n_rounds = random.randint(3, 10)
        for r in range(n_rounds):
            for i in range(n_nodes):
                amount = amount * random.uniform(0.90, 1.05)
                rows.append({
                    "tx_id": _txid("AML-CY"),
                    "from_account": accs[i],
                    "to_account": accs[(i + 1) % n_nodes],
                    "amount": round(amount, 2),
                    "tx_timestamp": _ts(day + r * 2 + i // n_nodes,
                                        random.randint(0, 23)),
                    "device_id": dev,
                    "is_laundering": 1, "typology": None,
                    **_aml_meta(),
                })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()

    final_cols = [
        "tx_id", "from_account", "to_account", "amount", "currency",
        "channel", "payment_format", "tx_timestamp", "device_id",
        "institution_id", "is_laundering", "typology",
    ]

    print("[1/3] Generate typologi + AML base patterns...")
    injections = [
        # Tipologi Indonesia — 5x volume untuk training HI-Large
        ("judol",     inject_judol(n_players=15000, n_collectors=50, n_transactions=250000)),
        ("scam",      inject_scam(n_victims=10000, n_chains=1000)),
        ("dormant",   inject_dormant(n_accounts=2500)),
        ("pep",       inject_pep(n_pep=500, n_intermediaries=5)),
        ("vendor",    inject_vendor(n_vendors=500, n_layers=5)),
        ("bendahara", inject_bendahara(n_bendahara=1000)),
        ("qris",      inject_qris(n_merchants=1000, n_transactions=150000)),
        # AML base patterns — 5x volume
        ("aml_fanout",        inject_aml_fanout(n_clusters=10000)),
        ("aml_fanin",         inject_aml_fanin(n_clusters=5000)),
        ("aml_layering",      inject_aml_layering(n_chains=25000)),
        ("aml_scatter_gather",inject_aml_scatter_gather(n_clusters=5000)),
        ("aml_cycle",         inject_aml_cycle(n_cycles=2500)),
    ]
    for name, inj_df in injections:
        print(f"      {name:12s}: {len(inj_df):,} transaksi")

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
    print(f"Total transaksi  : {total:,}")
    print(f"\nBreakdown typologi:")
    for name, inj_df in injections:
        print(f"  {name:12s}: {len(inj_df):,}")


if __name__ == "__main__":
    main()
