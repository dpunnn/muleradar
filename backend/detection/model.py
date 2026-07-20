"""
MuleRadar Phase 3 — XGBoost Risk Scoring Model.
Train on account-level features, predict laundering probability.
"""

import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_defs import FEATURE_COLS   # definisi kanonik (fix duplikasi 6-Jul)

load_dotenv()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "xgboost_v1.pkl"
)


def train_xgboost(
    features_df: pd.DataFrame,
    labels: pd.Series = None,
    feature_cols: list = None,
    model_path: str = None,
) -> XGBClassifier:
    """
    Train XGBoost classifier on account features.

    feature_cols/model_path (opsional, fix 17-Jul): dipakai utk melatih
    model TERPISAH dgn subset fitur berbeda (mis. model real-time yg cuma
    punya 20 fitur murah dihitung, vs FEATURE_COLS kanonik 24 fitur di
    sini) TANPA menimpa MODEL_PATH default (xgboost_v1.pkl, dipakai
    ml/ensemble.py — jangan sampai angka PR-AUC ensemble yg sudah divalidasi
    ikut berubah). Default None -> perilaku identik sebelum perubahan ini.

    Fix (7-Jul, konsistensi dgn TGN/DyGFormer): SEBELUMNYA train_test_split
    stratified ACAK — test-nya jadi "gampang" (akun test bisa saja sudah
    lama ada, banyak riwayat), tidak sebanding dgn TGN/DyGFormer yg dievaluasi
    temporal-inductive (akun test BENAR-BENAR baru). Sekarang: kalau
    features_df punya kolom 'first_seen_ts' (diisi extract_features_bulk),
    split jadi temporal-inductive 70/15/15 by first-appearance — SAMA rasio &
    metode dgn ml.tgn_dataset.temporal_inductive_split. Fallback ke random
    split HANYA kalau first_seen_ts tak ada (mis. dipanggil manual tanpa
    Postgres) — supaya tidak crash, tapi seharusnya tak pernah kejadian di
    jalur retrain_xgboost.py normal.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain FEATURE_COLS + ideally 'first_seen_ts'. If *labels* is
        None, must also contain 'is_laundering_label'.
    labels : pd.Series, optional
        Binary labels (0/1). Defaults to features_df["is_laundering_label"].

    Returns
    -------
    XGBClassifier
        Trained model (also saved to MODEL_PATH).
    """
    if labels is None:
        labels = features_df["is_laundering_label"]

    cols = feature_cols or FEATURE_COLS
    out_path = model_path or MODEL_PATH

    X = features_df[cols].fillna(0)
    y = labels

    if "first_seen_ts" in features_df.columns:
        order = features_df["first_seen_ts"].values.argsort()
        n = len(order)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)
        train_idx = order[:n_train]
        val_idx = order[n_train:n_train + n_val]
        test_idx = order[n_train + n_val:]

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        print(f"[XGBOOST] Split TEMPORAL-INDUCTIVE (70/15/15 by first_seen_ts): "
              f"train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")
        print(f"[XGBOOST] Illicit rate — train:{y_train.mean():.4f} "
              f"val:{y_val.mean():.4f} test:{y_test.mean():.4f}")
    else:
        print("[XGBOOST] WARNING: 'first_seen_ts' tak ada — fallback ke "
              "train_test_split ACAK (test lebih mudah, tak sebanding TGN/DyG).")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_val, y_val, test_size=0.5, stratify=y_val, random_state=42
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
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # Evaluate on held-out TEST set (bukan val — val cuma monitoring training)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"[TEST] PR-AUC : {pr_auc:.4f}")
    print(f"[TEST] F1@0.5 : {f1:.4f}")
    print(f"[TEST] Prec   : {prec:.4f}")
    print(f"[TEST] Recall : {rec:.4f}")

    # Persist model
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {os.path.abspath(out_path)}")

    return model


def predict(
    features_df: pd.DataFrame,
    model: XGBClassifier = None,
) -> pd.DataFrame:
    """
    Predict risk scores for accounts.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain FEATURE_COLS and 'account_id'.
    model : XGBClassifier, optional
        Pre-loaded model. Loaded from MODEL_PATH if None.

    Returns
    -------
    pd.DataFrame
        Columns: account_id, risk_score, risk_level (LOW/MEDIUM/HIGH).
    """
    if model is None:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

    X = features_df[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    result = pd.DataFrame({
        "account_id": features_df["account_id"].values,
        "risk_score": probs,
    })
    result["risk_level"] = pd.cut(
        result["risk_score"],
        bins=[-0.001, 0.5, 0.8, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"],
    )
    return result
