"""
Generate mini synthetic dataset TANPA SynthAML — untuk test pipeline malam ini.
Output: ../processed/transactions_test.csv (~50K transaksi)

Cara pakai:
  python postprocess_test.py
"""

import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BASE_DATE = datetime(2024, 1, 1)
N_ACCOUNTS = 2000
N_TX = 50_000


def gen_accounts(n):
    return [f"ACC-{i:06d}" for i in range(n)]


def gen_normal_transactions(accounts, n):
    rows = []
    for _ in range(n):
        frm, to = random.sample(accounts, 2)
        day = random.randint(0, 89)
        hour = random.randint(8, 22)
        ts = BASE_DATE + timedelta(days=day, hours=hour, minutes=random.randint(0, 59))
        rows.append({
            "tx_id": f"TX-{uuid.uuid4().hex[:12].upper()}",
            "from_account": frm,
            "to_account": to,
            "amount": round(random.uniform(10_000, 5_000_000)),
            "currency": "IDR",
            "channel": random.choices(["mobile","atm","internet","teller","qris"], weights=[55,15,15,5,10])[0],
            "payment_format": random.choice(["Transfer","Cheque","Credit Card","Cash"]),
            "tx_timestamp": ts,
            "device_id": f"DEV-{random.randint(0, 800):06d}",
            "institution_id": random.choices(["BANK_A","BANK_B"], weights=[70,30])[0],
            "is_laundering": 0,
            "typology": None,
        })
    return pd.DataFrame(rows)


def gen_judol_ring():
    players = [f"JUDOL-PLY-{i:04d}" for i in range(300)]
    collectors = [f"JUDOL-COL-{i:02d}" for i in range(2)]
    crypto_out = "JUDOL-CRYPTO-01"
    dev = "DEV-JUDOL-SHARED"
    rows = []
    for _ in range(5000):
        hour = random.choices(range(24), weights=[2,2,2,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,4,5,6,7,8,6])[0]
        ts = BASE_DATE + timedelta(days=random.randint(0,89), hours=hour, minutes=random.randint(0,59))
        rows.append({
            "tx_id": f"JDL-{uuid.uuid4().hex[:10].upper()}",
            "from_account": random.choice(players),
            "to_account": random.choice(collectors),
            "amount": round(random.uniform(50_000, 500_000) / 1000) * 1000,
            "currency": "IDR", "channel": "mobile", "payment_format": "Transfer",
            "tx_timestamp": ts,
            "device_id": dev if random.random() < 0.3 else f"DEV-PLY-{random.randint(1,200):04d}",
            "institution_id": "BANK_A", "is_laundering": 1, "typology": "judol",
        })
    for col in collectors:
        for d in range(0, 90, 3):
            ts = BASE_DATE + timedelta(days=d, hours=random.randint(1,4))
            rows.append({
                "tx_id": f"JDL-OUT-{uuid.uuid4().hex[:10].upper()}",
                "from_account": col, "to_account": crypto_out,
                "amount": round(random.uniform(5_000_000, 30_000_000)),
                "currency": "IDR", "channel": "internet", "payment_format": "Transfer",
                "tx_timestamp": ts, "device_id": dev,
                "institution_id": "BANK_A", "is_laundering": 1, "typology": "judol",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("[1/3] Generate normal transactions...")
    accounts = gen_accounts(N_ACCOUNTS)
    df = gen_normal_transactions(accounts, N_TX)

    print("[2/3] Inject judol ring...")
    judol = gen_judol_ring()
    df = pd.concat([df, judol], ignore_index=True)
    df = df.sort_values("tx_timestamp").reset_index(drop=True)

    out = "../processed/transactions_test.csv"
    print(f"[3/3] Simpan ke {out}...")
    df.to_csv(out, index=False)

    print(f"\nTotal: {len(df):,} transaksi")
    print(f"Illicit: {df['is_laundering'].sum():,} ({df['is_laundering'].mean()*100:.1f}%)")
    print(f"Judol: {(df['typology']=='judol').sum():,}")
    print("Selesai.")
