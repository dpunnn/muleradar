"""
Jalankan Phase 3 detection pipeline end-to-end:
  1. Typology rules → flagged accounts
  2. Feature extraction → ML feature matrix
  3. XGBoost training → model tersimpan
  4. Predict → risk scores
  5. Generate alerts → INSERT ke PostgreSQL

Cara pakai:
  cd muleradar/backend
  python run_detection.py
  python run_detection.py --skip-train   (pakai model yang sudah ada)
  python run_detection.py --limit 50000  (sample lebih kecil untuk test cepat)
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
)

from detection.rules import check_aml_core_rules, check_typology_rules
from detection.features import extract_features
from detection.model import train_xgboost, predict
from detection.alerts import generate_alerts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, load existing model")
    parser.add_argument("--limit", type=int, default=100_000,
                        help="Max accounts untuk feature extraction (default 100000)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Risk score threshold untuk alert (default 0.5)")
    args = parser.parse_args()

    engine = create_engine(DATABASE_URL)

    # ── Step 1a: AML Core Rules ───────────────────────────────────────
    print("\n" + "="*50)
    print("STEP 1a: AML Core Detection (sistem utama)")
    print("="*50)
    t0 = time.time()
    aml_rules = check_aml_core_rules(engine)
    print(f"Done in {time.time()-t0:.1f}s → {len(aml_rules):,} AML core hits\n")

    # ── Step 1b: Indonesia Typology Rules ────────────────────────────
    print("="*50)
    print("STEP 1b: Indonesia Typology Detection (layer tambahan)")
    print("="*50)
    t0 = time.time()
    typology_rules = check_typology_rules(engine)
    print(f"Done in {time.time()-t0:.1f}s → {len(typology_rules):,} typology hits\n")

    rules = aml_rules + typology_rules
    print(f"TOTAL RULES: {len(rules):,} hits ({len(aml_rules):,} AML core + {len(typology_rules):,} typologi)\n")

    # ── Step 2: Features ─────────────────────────────────────────────
    print("="*50)
    print("STEP 2: Feature Extraction")
    print("="*50)
    t0 = time.time()
    feats = extract_features(engine, limit=args.limit)
    if feats.empty:
        print("ERROR: Feature extraction returned empty DataFrame")
        sys.exit(1)
    print(f"Done in {time.time()-t0:.1f}s → {len(feats)} accounts, "
          f"{feats['is_laundering_label'].sum()} illicit\n")

    # ── Step 3: Train / Load Model ───────────────────────────────────
    print("="*50)
    print("STEP 3: XGBoost Model")
    print("="*50)
    t0 = time.time()
    if args.skip_train:
        print("--skip-train: loading existing model...")
        model = None  # predict() will load from pkl
    else:
        model = train_xgboost(feats)
    print(f"Done in {time.time()-t0:.1f}s\n")

    # ── Step 4: Predict ───────────────────────────────────────────────
    print("="*50)
    print("STEP 4: Predict Risk Scores")
    print("="*50)
    t0 = time.time()
    preds = predict(feats, model=model)
    high = (preds["risk_score"] >= args.threshold).sum()
    print(f"Done in {time.time()-t0:.1f}s → {high} accounts above threshold {args.threshold}")
    print(preds["risk_level"].value_counts().to_string(), "\n")

    # ── Step 5: Generate Alerts ───────────────────────────────────────
    print("="*50)
    print("STEP 5: Generate Alerts")
    print("="*50)

    # Override threshold kalau user set via args
    import detection.alerts as _alerts_mod
    _alerts_mod.RISK_THRESHOLD = args.threshold

    t0 = time.time()
    count = generate_alerts(engine, preds, rules)
    print(f"Done in {time.time()-t0:.1f}s → {count} alerts inserted\n")

    print("="*50)
    print("PIPELINE SELESAI")
    print("="*50)
    print(f"AML core hits  : {len(aml_rules):,}")
    print(f"Typology hits  : {len(typology_rules):,}")
    print(f"Total rule hits: {len(rules):,}")
    print(f"Accounts scored: {len(feats):,}")
    print(f"High risk      : {high:,}")
    print(f"Alerts saved   : {count:,}")


if __name__ == "__main__":
    main()
