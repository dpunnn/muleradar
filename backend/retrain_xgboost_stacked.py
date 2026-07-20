"""
Retrain XGBoost + DyGFormer-score STACKING (12-Jul).

Latar belakang: XGBoost standalone (retrain_xgboost.py) cuma PR-AUC 0.4589 di
test temporal-inductive, jauh di bawah DyGFormer (0.9623) di split yang sama.
Didiagnosis: XGBoost cuma punya fitur tabular flat (agregat riwayat akun
SENDIRI), tak punya mekanisme "pinjam sinyal dari tetangga di graph" — padahal
mule account seringkali terdeteksi justru dari POSISI-nya di jaringan, bukan
perilaku individualnya. Fix: suntikkan skor DyGFormer (dyg_v1.pt, sudah
di-training & FROZEN, TIDAK di-retrain di sini) sbg fitur tambahan ke XGBoost.

KETERBATASAN JUJUR (WAJIB disebut di README/proposal, JANGAN disembunyikan):
dyg_score SEKARANG (13-Jul) dihitung dari graph coverage r=0.65 (naik dari
r=0.31) — TIDAK 100%, cuma dinaikkan bertahap krn keterbatasan RAM laptop
(16.9GB, sempat OOM di percobaan sebelumnya). Coverage naik dari 1,981,734
akun (92.9% dari 2,133,214 total) ke 2,086,611 (97.8%) — masih ~46,603 akun
(2.2%) tanpa dyg_score sama sekali, dibiarkan NaN NATIVE (bukan konstan).
Bias historis (missing condong licit) yang terukur di r=0.31 (illicit rate
7.67% grup missing vs 23.57% covered) BERKURANG signifikan di r=0.65 tapi
belum diverifikasi ulang persis angkanya — kalau mau presisi, ukur ulang.
model dyg_v1.pt TIDAK di-retrain (bobot 100% sama), cuma graph saat SCORING
yang lebih lengkap dari sebelumnya — lihat build_full_coverage_dyg_scores.py.

Jalankan:
    cd backend
    python retrain_xgboost_stacked.py
"""

import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sqlalchemy import create_engine
from xgboost import XGBClassifier

from detection.features import extract_features_bulk
from feature_defs import FEATURE_COLS

TS_CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "retrain_ts_checkpoint.pkl")
# Fix (13-Jul): pakai skor coverage-diperluas (r=0.65, 2,086,611 akun vs
# 1,981,734 sebelumnya) — model dyg_v1.pt TETAP SAMA/frozen, cuma graph
# saat SCORING yang lebih lengkap. dyg_split (label train/val/test ASLI
# DyGFormer) tetap diambil dari cache LAMA krn itu satu-satunya sumber
# otoritatif split — file baru tidak (dan tidak perlu) menyimpan itu.
DYG_SCORES_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "dyg_scores_cache_fullcov_r0.65.pkl"
)
DYG_SPLIT_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "dyg_scores_cache.pkl"
)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "xgboost_stacked_v1.pkl"
)

STACKED_FEATURE_COLS = FEATURE_COLS + ["dyg_score"]


def main():
    t0 = time.time()
    url = os.getenv(
        "DATABASE_URL",
        "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar",
    )
    engine = create_engine(url)

    feats = extract_features_bulk(engine, checkpoint_path=TS_CHECKPOINT_PATH)
    n_illicit = int(feats["is_laundering_label"].sum())
    print(
        f"[STACK] features Postgres siap: shape={feats.shape} illicit={n_illicit:,} "
        f"({n_illicit/len(feats)*100:.2f}%) in {time.time()-t0:.0f}s",
        flush=True,
    )

    # --- merge dyg_score + dyg_split (frozen DyGFormer inference + label
    # train/val/test ASLI DyGFormer sendiri) — dari graph SAMPLED. dyg_split
    # WAJIB dipakai nanti utk filter evaluasi: dyg_score utk akun yg berstatus
    # "train"/"val" bagi DyGFormer itu IN-SAMPLE (DyGFormer sudah "melihat"
    # label akun itu saat training dirinya sendiri) — kalau akun begini
    # kebetulan jatuh di TEST-nya XGBoost, dyg_score-nya bukan skor genuine
    # out-of-sample, melainkan berpotensi "hafalan". Fix: evaluasi akhir HANYA
    # pakai subset akun yg statusnya test-nya XGBoost DAN test-nya DyGFormer
    # SEKALIGUS (double hold-out) — itu satu-satunya cara pastikan nol leak.
    dyg_df = pd.read_pickle(DYG_SCORES_CACHE)  # kolom: account_id, dyg_score (coverage r=0.65)
    feats = feats.merge(dyg_df, on="account_id", how="left")
    dyg_split_df = pd.read_pickle(DYG_SPLIT_CACHE)[["account_id", "dyg_split"]]
    feats = feats.merge(dyg_split_df, on="account_id", how="left")
    # dyg_split NaN (bukan "train"/"val"/"test") = akun TIDAK PERNAH ada di
    # graph training DyGFormer manapun (r=0.31) -> tidak pernah dipakai utk
    # update bobot model sama sekali -> SAMA amannya dgn "test" utk evaluasi
    # (dibuktikan 13-Jul: 104,877 akun begini, ditemukan pas boost coverage
    # ke r=0.65). Lihat clean_mask di bawah.

    n_covered = feats["dyg_score"].notna().sum()
    coverage_pct = n_covered / len(feats) * 100
    illicit_covered = feats.loc[feats["dyg_score"].notna(), "is_laundering_label"].mean()
    illicit_missing = feats.loc[feats["dyg_score"].isna(), "is_laundering_label"].mean()
    print(
        f"[STACK] dyg_score coverage: {n_covered:,}/{len(feats):,} ({coverage_pct:.1f}%). "
        f"Illicit rate COVERED={illicit_covered:.4f} vs MISSING={illicit_missing:.4f} "
        f"(bias terukur, lihat docstring file ini). NaN dibiarkan native, TIDAK diisi konstan.",
        flush=True,
    )
    print(f"[STACK] dyg_split composition (seluruh akun Postgres): "
          f"{feats['dyg_split'].value_counts(dropna=False).to_dict()}", flush=True)

    X = feats[STACKED_FEATURE_COLS].copy()
    for col in FEATURE_COLS:  # dyg_score SENGAJA tidak di-fillna (native missing)
        X[col] = X[col].fillna(0)
    y = feats["is_laundering_label"]

    order = feats["first_seen_ts"].values.argsort()
    n = len(order)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:]

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    print(
        f"[STACK] Split TEMPORAL-INDUCTIVE (70/15/15 by first_seen_ts): "
        f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}",
        flush=True,
    )
    cov_test = feats.iloc[test_idx]["dyg_score"].notna().mean() * 100
    print(f"[STACK] dyg_score coverage DI TEST SET SAJA: {cov_test:.1f}%", flush=True)

    test_dyg_split = feats.iloc[test_idx]["dyg_split"].value_counts(dropna=False)
    print(f"[STACK] Komposisi dyg_split DI TEST SET XGBoost: {test_dyg_split.to_dict()}", flush=True)
    n_double_test = int((feats.iloc[test_idx]["dyg_split"] == "test").sum())
    print(
        f"[STACK] Akun test XGBoost yg JUGA test-nya DyGFormer (double hold-out, "
        f"ZERO LEAK): {n_double_test:,}/{len(test_idx):,} "
        f"({n_double_test/len(test_idx)*100:.1f}%)", flush=True
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        missing=np.nan,  # eksplisit: NaN ditangani mekanisme resmi XGBoost
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 70)
    print("[TEST-STACKED, FULL] — semua akun test XGBoost (termasuk yg dyg_score-nya")
    print("  in-sample utk DyGFormer train/val, atau missing sama sekali). ANGKA INI")
    print("  BISA SEDIKIT TERINFLASI, jangan jadi headline, cuma konteks/perbandingan.")
    print(f"  PR-AUC : {pr_auc:.4f}  (baseline standalone XGBoost: 0.4589)")
    print(f"  F1@0.5 : {f1:.4f}")
    print(f"  Prec   : {prec:.4f}")
    print(f"  Recall : {rec:.4f}")
    print("=" * 70)

    # --- ANGKA UTAMA: double hold-out, ZERO LEAK ---
    # Aman utk evaluasi kalau: (a) test-nya DyGFormer sendiri (dyg_split=="test"),
    # ATAU (b) akun itu TIDAK PERNAH ada di split manapun (dyg_split NaN) ->
    # tidak pernah dipakai update bobot DyGFormer sama sekali (fix 13-Jul,
    # ditemukan pas boost coverage r=0.31->0.65: 104,877 akun begini).
    # Yang DIKECUALIKAN cuma dyg_split=="train"/"val" (in-sample utk DyGFormer).
    test_dyg_split_series = feats.iloc[test_idx]["dyg_split"]
    has_score = feats.iloc[test_idx]["dyg_score"].notna().values
    clean_mask = has_score & ((test_dyg_split_series == "test") | test_dyg_split_series.isna()).values
    if clean_mask.sum() > 10:
        y_test_clean = y_test.values[clean_mask]
        y_prob_clean = y_prob[clean_mask]
        y_pred_clean = (y_prob_clean >= 0.5).astype(int)
        pr_auc_clean = average_precision_score(y_test_clean, y_prob_clean)
        f1_clean = f1_score(y_test_clean, y_pred_clean, zero_division=0)
        prec_clean = precision_score(y_test_clean, y_pred_clean, zero_division=0)
        rec_clean = recall_score(y_test_clean, y_pred_clean, zero_division=0)
        print("\n" + "=" * 70)
        print(f"[TEST-STACKED, CLEAN/DOUBLE-HOLDOUT+NEVERSEEN] — akun test-nya XGBoost")
        print(f"  YANG JUGA (test-nya DyGFormer ATAU tak pernah dipakai training DyGFormer)")
        print(f"  ({clean_mask.sum():,}/{len(test_idx):,} akun, {clean_mask.mean()*100:.1f}% dari")
        print(f"  test set). NOL LEAK, INI ANGKA HEADLINE.")
        print(f"  PR-AUC : {pr_auc_clean:.4f}  (baseline standalone XGBoost: 0.4589, "
              f"stacking r=0.31 sebelumnya: 0.5540)")
        print(f"  F1@0.5 : {f1_clean:.4f}")
        print(f"  Prec   : {prec_clean:.4f}")
        print(f"  Recall : {rec_clean:.4f}")
        print("=" * 70)
    else:
        print("[STACK] WARNING: subset double hold-out terlalu kecil (<=10 akun) — "
              "tak bisa hitung PR-AUC clean yang stabil.")

    imp = sorted(zip(STACKED_FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
    print("[STACK] Feature importance (top 8):")
    for name, v in imp[:8]:
        marker = "  <-- STACKED (DyGFormer)" if name == "dyg_score" else ""
        print(f"    {name:<20} {v:.4f}{marker}")

    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {os.path.abspath(MODEL_PATH)}")
    print(f"[STACK] RETRAIN_STACKED_DONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
