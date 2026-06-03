"""
GraphSAGE baseline untuk ablation study.
Sama feature set, sama train/val/test split sebagai TGN.

Cara pakai:
    cd backend
    python -m ml.graphsage_baseline
    python -m ml.graphsage_baseline --epochs 30 --max-rows 2000000
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml.tgn_dataset import load_temporal_dataset_fast, CSV_DEFAULT
from ml.train_tgn import compute_metrics, save_model, save_training_log

MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)

try:
    from torch_geometric.nn import SAGEConv
    _SAGE_AVAILABLE = True
except ImportError:
    _SAGE_AVAILABLE = False
    print("[WARNING] torch_geometric.nn.SAGEConv not available.")
    print("Install torch-geometric: pip install torch-geometric")


class GraphSAGEBaseline(nn.Module):
    """
    2-layer GraphSAGE for edge-level AML detection.
    Ablation baseline: no temporal memory, just message passing.
    """

    def __init__(
        self,
        node_feat_dim: int = 13,
        edge_feat_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not _SAGE_AVAILABLE:
            raise ImportError("SAGEConv not available")

        self.conv1 = SAGEConv(node_feat_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        # Edge classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """Forward pass: node embeddings + edge classification."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst]], dim=-1)

        if edge_attr is not None:
            edge_repr = torch.cat([edge_repr, edge_attr.float()], dim=-1)
        else:
            pad = torch.zeros(
                edge_repr.size(0), 3, device=edge_repr.device
            )
            edge_repr = torch.cat([edge_repr, pad], dim=-1)

        return self.classifier(edge_repr).squeeze(-1)


def train_graphsage_baseline(
    data: dict,
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 64,
    patience: int = 10,
    device: str = "cpu",
) -> dict:
    """Train GraphSAGE baseline model."""
    x = torch.tensor(data["node_features"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(data["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float32, device=device)
    edge_labels = torch.tensor(data["edge_labels"], dtype=torch.float32, device=device)

    train_idx = data["split"]["train"]
    val_idx = data["split"]["val"]
    test_idx = data["split"]["test"]

    n_pos = int(edge_labels[train_idx].sum().item())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"[BASELINE] Class balance: {n_pos:,} pos / {n_neg:,} neg")

    model = GraphSAGEBaseline(
        node_feat_dim=data["node_features"].shape[1],
        edge_feat_dim=data["edge_attr"].shape[1],
        hidden_dim=hidden_dim,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_prauc = 0.0
    best_model_state = None
    patience_counter = 0
    training_log = []

    print(f"\n[BASELINE] Training GraphSAGE: {epochs} epochs, lr={lr}")
    print(f"[BASELINE] Params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        model.train()
        optimizer.zero_grad()

        try:
            logits = model(x, edge_index, edge_attr)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] Out of memory at epoch {epoch}.")
                print("[OOM] Try smaller --max-rows or --hidden-dim")
                if device != "cpu":
                    torch.cuda.empty_cache()
                raise
            raise

        loss = criterion(logits[train_idx], edge_labels[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = logits[val_idx].cpu().numpy()
            val_labels = edge_labels[val_idx].cpu().numpy()
            val_metrics = compute_metrics(val_labels, val_logits)

        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": loss.item(),
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "time_s": elapsed,
        }
        training_log.append(log_entry)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={loss.item():.4f} | "
                f"PR-AUC={val_metrics['pr_auc']:.4f} | "
                f"F1={val_metrics['f1']:.4f} | "
                f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} | "
                f"{elapsed:.1f}s"
            )

        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[BASELINE] Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        test_logits = logits[test_idx].cpu().numpy()
        test_labels = edge_labels[test_idx].cpu().numpy()
        test_metrics = compute_metrics(test_labels, test_logits)

    print("\n" + "=" * 70)
    print("[BASELINE TEST] GraphSAGE results:")
    print(f"  PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"  F1@0.5   : {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print("=" * 70)

    return {
        "model": model,
        "test_metrics": test_metrics,
        "best_val_prauc": best_val_prauc,
        "training_log": training_log,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE baseline for AML detection (ablation)"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--csv", type=str, default=CSV_DEFAULT)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not _SAGE_AVAILABLE:
        print("[ERROR] torch_geometric not available. Cannot run baseline.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CONFIG] Device: {device}")

    # Load dataset (same as TGN)
    print("\n[PHASE 1] Loading dataset ...")
    t0 = time.time()
    try:
        data = load_temporal_dataset_fast(
            csv_path=args.csv,
            max_rows=args.max_rows,
            sample_licit_ratio=args.sample_ratio,
            random_seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except MemoryError:
        print("\n[OOM] Not enough memory. Try smaller --max-rows")
        sys.exit(1)

    print(f"[PHASE 1] Loaded in {time.time() - t0:.1f}s")

    # Train
    print("\n[PHASE 2] Training GraphSAGE baseline ...")
    result = train_graphsage_baseline(
        data=data,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        patience=args.patience,
        device=device,
    )

    # Save
    model_path = os.path.join(MODEL_DIR, "graphsage_v1.pt")
    save_model(
        model=result["model"],
        path=model_path,
        metadata={
            "test_metrics": result["test_metrics"],
            "best_val_prauc": result["best_val_prauc"],
            "epochs_trained": len(result["training_log"]),
            "model_type": "GraphSAGEBaseline",
        },
    )

    log_path = os.path.join(RESULTS_DIR, "graphsage_training_log.csv")
    save_training_log(result["training_log"], log_path)

    print(f"\n[DONE] Model: {os.path.abspath(model_path)}")
    print(f"[DONE] Log:   {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
