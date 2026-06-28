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

load_dotenv()

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "xgboost_v1.pkl"
)

FEATURE_COLS = [
    # 13 baseline
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
    # 7 behavioral
    "burst_ratio", "inter_tx_std", "dormancy_days",
    "counterparty_hhi", "channel_entropy",
    "structuring_score", "round_amount_ratio",
]


def train_xgboost(
    features_df: pd.DataFrame,
    labels: pd.Series = None,
) -> XGBClassifier:
    """
    Train XGBoost classifier on account features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Must contain FEATURE_COLS. If *labels* is None, must also contain
        'is_laundering_label'.
    labels : pd.Series, optional
        Binary labels (0/1). Defaults to features_df["is_laundering_label"].

    Returns
    -------
    XGBClassifier
        Trained model (also saved to MODEL_PATH).
    """
    if labels is None:
        labels = features_df["is_laundering_label"]

    X = features_df[FEATURE_COLS].fillna(0)
    y = labels

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
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
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Evaluate on held-out set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"PR-AUC : {pr_auc:.4f}")
    print(f"F1@0.5 : {f1:.4f}")
    print(f"Prec   : {prec:.4f}")
    print(f"Recall : {rec:.4f}")

    # Persist model
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {os.path.abspath(MODEL_PATH)}")

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
