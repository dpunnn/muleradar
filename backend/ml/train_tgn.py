"""
Training loop untuk TGN/GNN model.
Input: tgn_dataset.load_temporal_dataset()
Output: models/tgn_v1.pt

Cara pakai:
    cd backend
    python -m ml.train_tgn
    python -m ml.train_tgn --epochs 30 --max-rows 5000000
    python -m ml.train_tgn --no-tgn   # force GraphSAGE fallback
"""

import argparse
import os
import sys
import time
import csv

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
from sklearn.model_selection import train_test_split

# Ensure backend/ is on path when run as module
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml.tgn_dataset import load_temporal_dataset_fast, CSV_DEFAULT
from ml.tgn_model import get_model, FallbackTemporalGNN, ManualTGN

MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def compute_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5):
    """Compute PR-AUC, F1, Precision, Recall from raw logits."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))  # sigmoid (stable)
    preds = (probs >= threshold).astype(int)

    pr_auc = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return {"pr_auc": pr_auc, "f1": f1, "precision": prec, "recall": rec}


class FocalLoss(nn.Module):
    """Focal Loss untuk class imbalance — fokus pada hard examples."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        return focal.mean()


def train_graphsage(
    data: dict,
    epochs: int = 50,
    lr: float = 0.001,
    hidden_dim: int = 128,
    patience: int = 15,
    device: str = "cpu",
) -> dict:
    """
    Train FallbackTemporalGNN (GraphSAGE-based) on edge classification.

    Returns dict with model, best metrics, training log.
    """
    # Convert to tensors
    x = torch.tensor(data["node_features"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(data["edge_index"], dtype=torch.long, device=device)
    edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float32, device=device)
    edge_labels = torch.tensor(data["edge_labels"], dtype=torch.float32, device=device)

    train_idx = data["split"]["train"]
    val_idx = data["split"]["val"]
    test_idx = data["split"]["test"]

    # Class imbalance: compute pos_weight
    n_pos = int(edge_labels[train_idx].sum().item())
    n_neg = len(train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"[TRAIN] Class balance: {n_pos:,} pos / {n_neg:,} neg, "
          f"pos_weight={pos_weight.item():.2f}")

    # Build model
    model = FallbackTemporalGNN(
        node_feat_dim=data["node_features"].shape[1],
        edge_feat_dim=data["edge_attr"].shape[1],
        hidden_dim=hidden_dim,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.90, gamma=2.0)

    # Training state
    best_val_prauc = 0.0
    best_model_state = None
    patience_counter = 0
    training_log = []

    print(f"\n[TRAIN] Starting training: {epochs} epochs, lr={lr}, "
          f"hidden_dim={hidden_dim}")
    print(f"[TRAIN] Device: {device}")
    print(f"[TRAIN] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        optimizer.zero_grad()

        try:
            logits = model(x, edge_index, edge_attr)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] GPU out of memory at epoch {epoch}.")
                print("[OOM] Try: --max-rows (smaller), --hidden-dim 32, "
                      "or remove --device cuda")
                if device != "cpu":
                    torch.cuda.empty_cache()
                raise
            raise

        train_logits = logits[train_idx]
        train_labels = edge_labels[train_idx]
        loss = criterion(train_logits, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- Validate ---
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

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"loss={loss.item():.4f} | "
                f"val_PR-AUC={val_metrics['pr_auc']:.4f} | "
                f"val_F1={val_metrics['f1']:.4f} | "
                f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} | "
                f"{elapsed:.1f}s"
            )

        # Early stopping
        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[TRAIN] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    # --- Test evaluation ---
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, edge_attr)
        test_logits = logits[test_idx].cpu().numpy()
        test_labels = edge_labels[test_idx].cpu().numpy()
        test_metrics = compute_metrics(test_labels, test_logits)

    print("\n" + "=" * 70)
    print("[TEST] Final evaluation on test set:")
    print(f"  PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"  F1@0.5   : {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  Best val PR-AUC: {best_val_prauc:.4f}")
    print("=" * 70)

    return {
        "model": model,
        "test_metrics": test_metrics,
        "best_val_prauc": best_val_prauc,
        "training_log": training_log,
    }


def train_tgn_manual(
    data: dict,
    epochs: int = 200,
    lr: float = 0.001,
    hidden_dim: int = 128,
    patience: int = 20,
    device: str = "cpu",
    mini_batch_size: int = 2048,
) -> dict:
    """
    Train ManualTGN untuk NODE classification.
    Fase 1: replay semua edge temporal untuk bangun memory state.
    Fase 2: classify nodes dari concat(memory, node_features).
    """
    # Node features + memory kecil → di GPU. Edge tensors BESAR → tetap di CPU,
    # batch-nya saja yang dikirim ke GPU (hindari OOM di full 181M edge).
    x = torch.tensor(data["node_features"], dtype=torch.float32, device=device)
    edge_index = torch.tensor(data["edge_index"], dtype=torch.long)        # CPU
    edge_attr = torch.tensor(data["edge_attr"], dtype=torch.float32)       # CPU
    edge_timestamps = torch.tensor(data["edge_timestamps"], dtype=torch.float32)  # CPU
    node_labels_np = data["node_labels"]
    node_labels_t = torch.tensor(node_labels_np, dtype=torch.float32, device=device)

    num_nodes = data["node_features"].shape[0]
    src_all = edge_index[0]    # CPU
    dst_all = edge_index[1]    # CPU

    # --- Node split (stratified 70/15/15) ---
    all_nodes = np.arange(num_nodes)
    train_nodes_np, temp_nodes_np = train_test_split(
        all_nodes, test_size=0.3, stratify=node_labels_np, random_state=42)
    val_nodes_np, test_nodes_np = train_test_split(
        temp_nodes_np, test_size=0.5, stratify=node_labels_np[temp_nodes_np], random_state=42)

    train_nodes = torch.tensor(train_nodes_np, dtype=torch.long, device=device)
    val_nodes = torch.tensor(val_nodes_np, dtype=torch.long, device=device)
    test_nodes = torch.tensor(test_nodes_np, dtype=torch.long, device=device)

    # Edge temporal order (sort ALL edges by timestamp)
    edge_order = torch.argsort(edge_timestamps).cpu()

    # Class balance
    n_pos = int(node_labels_np[train_nodes_np].sum())
    n_neg = len(train_nodes_np) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"[TGN-NODE] Class balance (nodes): {n_pos:,} pos / {n_neg:,} neg, "
          f"pos_weight={pos_weight.item():.2f}")

    # Model
    model = ManualTGN(
        num_nodes=num_nodes,
        node_feat_dim=x.shape[1],
        edge_feat_dim=edge_attr.shape[1],
        memory_dim=hidden_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_prauc = 0.0
    best_model_state = None
    patience_counter = 0
    training_log = []

    print(f"\n[TGN-NODE] Training node classification: {epochs} epochs, lr={lr}, "
          f"memory_dim={hidden_dim}, batch={mini_batch_size}")
    print(f"[TGN-NODE] Nodes: {num_nodes:,} | Edges: {edge_index.shape[1]:,}")
    print(f"[TGN-NODE] Split: train={len(train_nodes_np):,} | "
          f"val={len(val_nodes_np):,} | test={len(test_nodes_np):,}")
    print(f"[TGN-NODE] Device: {device}")
    print(f"[TGN-NODE] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)

    # Batch besar untuk replay memory (no-grad) — jauh lebih cepat untuk graph besar
    replay_batch = max(mini_batch_size, 65536)

    def _batch_to_device(eidx_cpu):
        """Index edge tensor (CPU) lalu pindah hasil batch ke device."""
        return (src_all[eidx_cpu].to(device),
                dst_all[eidx_cpu].to(device),
                edge_attr[eidx_cpu].to(device),
                edge_timestamps[eidx_cpu].to(device))

    def _replay_all_memory():
        """Replay seluruh edge (temporal order) untuk bangun memory state."""
        for i in range(0, len(edge_order), replay_batch):
            eidx = edge_order[i:i + replay_batch]      # CPU index
            sb, db, eab, tsb = _batch_to_device(eidx)
            model.update_memory_only(x, sb, db, eab, tsb)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        model.reset_memory()

        # FASE 1: bangun memory dengan replay SEMUA edge temporal (no-grad)
        with torch.no_grad():
            _replay_all_memory()

        # FASE 2: node classification + loss pada TRAIN nodes (mini-batch)
        perm = torch.randperm(len(train_nodes), device=device)
        train_nodes_shuf = train_nodes[perm]
        total_loss = 0.0
        nb = 0
        for i in range(0, len(train_nodes_shuf), 8192):
            nb_idx = train_nodes_shuf[i:i + 8192]
            optimizer.zero_grad()
            logits = model.classify_nodes(x, nb_idx)
            loss = criterion(logits, node_labels_t[nb_idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        avg_loss = total_loss / max(nb, 1)

        # VALIDATION — rebuild memory dengan weights final (konsisten dgn test)
        model.eval()
        with torch.no_grad():
            model.reset_memory()
            _replay_all_memory()
            val_logits = model.classify_nodes(x, val_nodes)

        val_metrics = compute_metrics(
            node_labels_np[val_nodes_np],
            val_logits.cpu().numpy(),
        )
        val_prauc = val_metrics["pr_auc"]
        scheduler.step(1.0 - val_prauc)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_PR-AUC={val_prauc:.4f} | "
                  f"val_F1={val_metrics['f1']:.4f} | "
                  f"P={val_metrics['precision']:.4f} "
                  f"R={val_metrics['recall']:.4f} | {elapsed:.1f}s")

        training_log.append({
            "epoch": epoch, "loss": avg_loss, "val_pr_auc": val_prauc,
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "pr_auc"},
        })

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[TGN-NODE] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # --- Test evaluation ---
    model.eval()
    model.reset_memory()
    with torch.no_grad():
        # Replay ALL edges untuk rebuild memory
        _replay_all_memory()
        test_logits = model.classify_nodes(x, test_nodes)

    test_metrics = compute_metrics(
        node_labels_np[test_nodes_np],
        test_logits.cpu().numpy(),
    )

    print("\n" + "=" * 70)
    print("[TEST] Final evaluation on test set (NODE classification):")
    print(f"  PR-AUC   : {test_metrics['pr_auc']:.4f}")
    print(f"  F1@0.5   : {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall   : {test_metrics['recall']:.4f}")
    print(f"  Best val PR-AUC: {best_val_prauc:.4f}")
    print("=" * 70)

    return {
        "model": model,
        "test_metrics": test_metrics,
        "best_val_prauc": best_val_prauc,
        "training_log": training_log,
    }


def save_model(model: nn.Module, path: str, metadata: dict = None):
    """Save model checkpoint with optional metadata."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    if metadata:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, path)
    print(f"[SAVE] Model saved to: {os.path.abspath(path)}")


def save_training_log(log: list, path: str):
    """Save training log as CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not log:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)
    print(f"[SAVE] Training log saved to: {os.path.abspath(path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TGN/GNN for AML detection"
    )
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max CSV rows to load (default: None = seluruh CSV)")
    parser.add_argument("--csv", type=str, default=CSV_DEFAULT,
                        help="Path to transactions CSV")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension (default: 128)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu/cuda (default: auto)")
    parser.add_argument("--no-tgn", action="store_true",
                        help="Force GraphSAGE fallback (skip TGN)")
    parser.add_argument("--sample-licit", type=float, default=0.005,
                        help="Licit sampling ratio dari seluruh CSV (default: 0.005)")
    parser.add_argument("--sample-illicit", type=float, default=0.05,
                        help="Illicit sampling ratio dari seluruh CSV (default: 0.05)")
    parser.add_argument("--mini-batch-size", type=int, default=2048,
                        help="Mini-batch size untuk TGN training (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Skip data cache, paksa reload dari CSV")
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[CONFIG] Device: {device}")
    if device == "cuda":
        print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[CONFIG] GPU Memory: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load dataset
    print("\n" + "=" * 70)
    print("[PHASE 1] Loading dataset ...")
    print("=" * 70)
    t0 = time.time()

    cache_npz = args.csv.replace(".csv", "_traindata.npz")
    if os.path.exists(cache_npz) and not args.no_cache:
        print(f"[CACHE] Loading data dari {os.path.basename(cache_npz)}")
        npz = np.load(cache_npz, allow_pickle=True)
        data = {
            "node_features": npz["node_features"],
            "edge_index": npz["edge_index"],
            "edge_attr": npz["edge_attr"],
            "edge_timestamps": npz["edge_timestamps"],
            "edge_labels": npz["edge_labels"],
            "node_labels": npz["node_labels"],
        }
    else:
        try:
            data = load_temporal_dataset_fast(
                csv_path=args.csv,
                max_rows=args.max_rows,
                sample_licit_ratio=args.sample_licit,
                sample_illicit_ratio=args.sample_illicit,
                random_seed=args.seed,
            )
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            print("Please ensure the training CSV exists at the specified path.")
            sys.exit(1)
        except MemoryError:
            print("\n[OOM] Not enough memory to load dataset.")
            print("Try reducing --max-rows (e.g., --max-rows 1000000)")
            sys.exit(1)

        print(f"[CACHE] Saving data ke {os.path.basename(cache_npz)}")
        np.savez(cache_npz,
                 node_features=data["node_features"],
                 edge_index=data["edge_index"],
                 edge_attr=data["edge_attr"],
                 edge_timestamps=data["edge_timestamps"],
                 edge_labels=data["edge_labels"],
                 node_labels=data["node_labels"],
                 account_to_idx=np.array(data.get("account_to_idx", {}), dtype=object))

    load_time = time.time() - t0
    print(f"[PHASE 1] Dataset loaded in {load_time:.1f}s")

    # Train
    print("\n" + "=" * 70)
    print("[PHASE 2] Training model ...")
    print("=" * 70)

    # Pilih training function berdasarkan model yang tersedia
    test_model = get_model(
        num_nodes=data["node_features"].shape[0],
        node_feat_dim=data["node_features"].shape[1],
        edge_feat_dim=data["edge_attr"].shape[1],
        hidden_dim=args.hidden_dim,
    )

    if isinstance(test_model, ManualTGN):
        result = train_tgn_manual(
            data=data,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            patience=args.patience,
            device=device,
            mini_batch_size=args.mini_batch_size,
        )
    else:
        result = train_graphsage(
            data=data,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            patience=args.patience,
            device=device,
        )

    # Save model
    model_path = os.path.join(MODEL_DIR, "tgn_v1.pt")
    save_model(
        model=result["model"],
        path=model_path,
        metadata={
            "test_metrics": result["test_metrics"],
            "best_val_prauc": result["best_val_prauc"],
            "epochs_trained": len(result["training_log"]),
            "max_rows": args.max_rows,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "model_type": result["model"].__class__.__name__,
        },
    )

    # Save training log
    log_path = os.path.join(RESULTS_DIR, "tgn_training_log.csv")
    save_training_log(result["training_log"], log_path)

    print("\n[DONE] Training complete.")
    print(f"  Model: {os.path.abspath(model_path)}")
    print(f"  Log:   {os.path.abspath(log_path)}")


if __name__ == "__main__":
    main()
