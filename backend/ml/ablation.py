"""
Ablation study: run semua model dan compare metrics.
Save results ke results/ablation_results.csv

Cara pakai:
    cd backend
    python -m ml.ablation
    python -m ml.ablation --max-rows 2000000 --epochs 30
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def run_ablation(
    max_rows: int = 5_000_000,
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
    patience: int = 10,
    csv_path: str = None,
    sample_ratio: float = 0.1,
    seed: int = 42,
    device: str = None,
):
    """Run all models on the same dataset split, compare metrics."""
    import torch
    from ml.tgn_dataset import load_temporal_dataset_fast, CSV_DEFAULT
    from ml.train_tgn import train_graphsage, compute_metrics, save_training_log
    from ml.graphsage_baseline import train_graphsage_baseline

    if csv_path is None:
        csv_path = CSV_DEFAULT
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Load dataset (shared across all models)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[ABLATION] Loading shared dataset ...")
    print("=" * 70)
    t0 = time.time()

    try:
        data = load_temporal_dataset_fast(
            csv_path=csv_path,
            max_rows=max_rows,
            sample_licit_ratio=sample_ratio,
            random_seed=seed,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except MemoryError:
        print("\n[OOM] Not enough memory. Try smaller --max-rows")
        sys.exit(1)

    load_time = time.time() - t0
    print(f"[ABLATION] Dataset loaded in {load_time:.1f}s\n")

    results = []

    # ------------------------------------------------------------------
    # Model 1: TGN / FallbackTemporalGNN (from train_tgn)
    # ------------------------------------------------------------------
    print("=" * 70)
    print("[ABLATION] Model 1: TGN / FallbackTemporalGNN")
    print("=" * 70)
    t0 = time.time()

    try:
        tgn_result = train_graphsage(
            data=data,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            patience=patience,
            device=device,
        )
        tgn_time = time.time() - t0
        results.append({
            "model": "TGN_FallbackGNN",
            "pr_auc": tgn_result["test_metrics"]["pr_auc"],
            "f1": tgn_result["test_metrics"]["f1"],
            "precision": tgn_result["test_metrics"]["precision"],
            "recall": tgn_result["test_metrics"]["recall"],
            "best_val_prauc": tgn_result["best_val_prauc"],
            "epochs_trained": len(tgn_result["training_log"]),
            "train_time_s": tgn_time,
        })
        save_training_log(
            tgn_result["training_log"],
            os.path.join(RESULTS_DIR, "ablation_tgn_log.csv"),
        )
    except Exception as e:
        print(f"[ABLATION] TGN failed: {e}")
        results.append({
            "model": "TGN_FallbackGNN",
            "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0,
            "best_val_prauc": 0, "epochs_trained": 0,
            "train_time_s": time.time() - t0,
            "error": str(e),
        })

    # ------------------------------------------------------------------
    # Model 2: GraphSAGE Baseline
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[ABLATION] Model 2: GraphSAGE Baseline")
    print("=" * 70)
    t0 = time.time()

    # Reset seeds for fair comparison
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        sage_result = train_graphsage_baseline(
            data=data,
            epochs=epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            patience=patience,
            device=device,
        )
        sage_time = time.time() - t0
        results.append({
            "model": "GraphSAGE_Baseline",
            "pr_auc": sage_result["test_metrics"]["pr_auc"],
            "f1": sage_result["test_metrics"]["f1"],
            "precision": sage_result["test_metrics"]["precision"],
            "recall": sage_result["test_metrics"]["recall"],
            "best_val_prauc": sage_result["best_val_prauc"],
            "epochs_trained": len(sage_result["training_log"]),
            "train_time_s": sage_time,
        })
        save_training_log(
            sage_result["training_log"],
            os.path.join(RESULTS_DIR, "ablation_graphsage_log.csv"),
        )
    except Exception as e:
        print(f"[ABLATION] GraphSAGE failed: {e}")
        results.append({
            "model": "GraphSAGE_Baseline",
            "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0,
            "best_val_prauc": 0, "epochs_trained": 0,
            "train_time_s": time.time() - t0,
            "error": str(e),
        })

    # ------------------------------------------------------------------
    # Model 3: XGBoost (node-level, using tabular features)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[ABLATION] Model 3: XGBoost (tabular features)")
    print("=" * 70)
    t0 = time.time()

    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import average_precision_score, f1_score
        from sklearn.metrics import precision_score, recall_score

        # Use node features + node labels for XGBoost
        X = data["node_features"]
        y = data["node_labels"]

        # Split: use same temporal idea — first 60% train, next 20% val, last 20% test
        # For node-level, we split by node index (arbitrary but consistent)
        n = len(y)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        idx = np.arange(n)
        np.random.shuffle(idx)

        train_idx = idx[:n_train]
        val_idx = idx[n_train: n_train + n_val]
        test_idx = idx[n_train + n_val:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)

        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="aucpr",
            random_state=seed, n_jobs=-1,
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

        y_prob = xgb.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        xgb_metrics = {
            "pr_auc": average_precision_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
        }
        xgb_time = time.time() - t0

        print(f"\n[ABLATION XGB] PR-AUC={xgb_metrics['pr_auc']:.4f}, "
              f"F1={xgb_metrics['f1']:.4f}")

        results.append({
            "model": "XGBoost_Tabular",
            "pr_auc": xgb_metrics["pr_auc"],
            "f1": xgb_metrics["f1"],
            "precision": xgb_metrics["precision"],
            "recall": xgb_metrics["recall"],
            "best_val_prauc": xgb_metrics["pr_auc"],  # no val split for xgb
            "epochs_trained": 300,  # n_estimators
            "train_time_s": xgb_time,
        })
    except Exception as e:
        print(f"[ABLATION] XGBoost failed: {e}")
        results.append({
            "model": "XGBoost_Tabular",
            "pr_auc": 0, "f1": 0, "precision": 0, "recall": 0,
            "best_val_prauc": 0, "epochs_trained": 0,
            "train_time_s": time.time() - t0,
            "error": str(e),
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "ablation_results.csv")

    fieldnames = [
        "model", "pr_auc", "f1", "precision", "recall",
        "best_val_prauc", "epochs_trained", "train_time_s", "error",
    ]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 70)
    print("[ABLATION] COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<25} {'PR-AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model']:<25} "
            f"{r['pr_auc']:>8.4f} "
            f"{r['f1']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['train_time_s']:>7.1f}s"
        )
    print("-" * 70)
    print(f"\nResults saved to: {os.path.abspath(results_path)}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: compare all AML detection models"
    )
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_ablation(
        max_rows=args.max_rows,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        csv_path=args.csv,
        sample_ratio=args.sample_ratio,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
