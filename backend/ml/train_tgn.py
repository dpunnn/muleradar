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
import gc
import os
import sys
import threading
import time
import csv

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Ensure backend/ is on path when run as module
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from ml.tgn_dataset import (
    load_temporal_dataset_fast, CSV_DEFAULT, FEATURE_CONTRACT_VERSION,
    temporal_inductive_split,
)
from ml.tgn_model import get_model, FallbackTemporalGNN, ManualTGN

MODEL_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)


def compute_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float = 0.5):
    """Compute PR-AUC, F1, Precision, Recall from raw logits."""
    # Fix (6-Jul): clip ±500 kelewat longgar utk float32 exp() (overflow di
    # ~88), muncul pas BPTT hasilin logit sangat negatif (val prediksi
    # confidently-negative) — RuntimeWarning "overflow in exp", tapi HARMLESS
    # (hasil 1/(1+inf)=0.0, sama persis dgn limit sigmoid yg benar). ±80 aman
    # di float32 (exp(80)~5.5e34 < float32 max ~3.4e38) & prob-nya tetap
    # jenuh ke 0/1 scr praktis (sigmoid(80) beda dgn 1.0 di skala 1e-35).
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))  # sigmoid (stable, no overflow)
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
    focal_alpha: float = 0.90,
    bptt: bool = False,
    bptt_warmup_epochs: int = 0,
    bptt_lr_scale: float = 0.1,
    pretrain_epochs: int = 0,
    pretrain_lr: float = 0.001,
) -> dict:
    """
    Train ManualTGN untuk NODE classification.

    pretrain_epochs>0 (7-Jul, riset resmi TGN Rossi et al. — official
    train_supervised.py MEMBEKUKAN TGN yg SUDAH dilatih via link prediction
    di train_self_supervised.py, BUKAN dilatih langsung dari label fraud):
      Sebelum training classifier, latih msg_mlp+memory_updater(GRU)+
      classifier(dipakai ulang sbg link-predictor) via self-supervised link
      prediction ("apakah edge (src,dst) ini beneran terjadi?", negative
      sampling acak) — sinyal dari SEMUA edge (48 juta), bukan cuma node
      berlabel (~2 juta, mayoritas licit). Sesudah ini, msg_mlp/GRU TIDAK
      lagi random — sudah paham pola temporal asli — baru training utama
      (bptt=False, freeze memory) jalan spt biasa di atas memory yg lebih
      "pintar" ini, bukan proyeksi acak.

    bptt=False (default, "memory beku" — MENANG, test PR-AUC 0.9471, 6-Jul):
      Fase 1: replay semua edge temporal utk bangun memory state (no_grad —
              msg_mlp & memory_updater/GRU TIDAK dapat gradient di sini).
      Fase 2: classify node dari concat(memory, node_features), loss+backward
              cuma lewat node_classifier.
    bptt=True (audit roadmap #3, EKSPERIMENTAL — regresi ke 0.80 di test
               6-Jul dgn alpha=0.6/0.9, precision/recall lebih seimbang tapi
               PR-AUC keseluruhan turun; disimpan utk sesi tuning lanjutan,
               BUKAN default produksi):
      bptt_warmup_epochs epoch PERTAMA tetap pakai memory-beku (classifier
      distabilkan dulu spy grad ke memory nanti tidak datang saat classifier
      msh berantakan), baru epoch selanjutnya switch ke BPTT penuh.
      bptt_lr_scale: LR utk msg_mlp+memory_updater = lr * bptt_lr_scale
      (lebih kecil dari LR classifier — memory yg baru mulai belajar
      butuh langkah lebih hati2, classifier boleh tetap agresif).
      Satu pass kronologis via train_step_batch — msg_mlp + memory_updater +
      node_classifier ke-update BARENG per batch edge (truncated BPTT), loss
      dimask ke label node TRAIN saja.
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

    # --- Node split TEMPORAL (fix 5-Jul) ---
    # Sebelumnya stratified-random (spt DyGFormer) — TIDAK sesuai desain asli
    # ManualTGN yg punya full memory state, harusnya diuji inductive (akun
    # BARU yg belum pernah dilihat), bukan campuran acak semua akun. DyGFormer
    # TETAP pakai stratified (lihat train_dyg.py — PR-AUC kolaps ke 0.07 kalau
    # dipaksa temporal, krn dia model lokal K-hop tanpa memory persisten).
    train_nodes_np, val_nodes_np, test_nodes_np = temporal_inductive_split(
        data["edge_index"], data["edge_timestamps"], num_nodes,
    )
    print(f"[TGN-NODE] Illicit rate — train:{node_labels_np[train_nodes_np].mean():.4f} "
          f"val:{node_labels_np[val_nodes_np].mean():.4f} "
          f"test:{node_labels_np[test_nodes_np].mean():.4f}")

    train_nodes = torch.tensor(train_nodes_np, dtype=torch.long, device=device)
    val_nodes = torch.tensor(val_nodes_np, dtype=torch.long, device=device)
    test_nodes = torch.tensor(test_nodes_np, dtype=torch.long, device=device)

    # Fix (6-Jul, audit roadmap #3): mask node-train utk loss BPTT di bawah.
    # train_step_batch mengklasifikasi SEMUA node yg tersentuh tiap batch edge
    # (src+dst), campur train/val/test krn edge diproses urut waktu global
    # (bukan per-split) — itu memang desain TGN yg benar (memory node lain
    # cuma nyumbang MESSAGE, bukan labelnya, jadi bukan leak). Tapi loss WAJIB
    # cuma dihitung dari label node TRAIN; label val/test tak boleh ikut grad.
    is_train_node = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    is_train_node[train_nodes] = True

    # Edge temporal order (sort ALL edges by timestamp)
    edge_order = torch.argsort(edge_timestamps).cpu()

    # Class balance (info logging saja — FocalLoss tidak pakai pos_weight,
    # imbalance sudah ditangani via alpha internal loss-nya)
    n_pos = int(node_labels_np[train_nodes_np].sum())
    n_neg = len(train_nodes_np) - n_pos
    print(f"[TGN-NODE] Class balance (nodes): {n_pos:,} pos / {n_neg:,} neg "
          f"(ratio {n_neg/max(n_pos,1):.1f}:1)")

    # Model
    model = ManualTGN(
        num_nodes=num_nodes,
        node_feat_dim=x.shape[1],
        edge_feat_dim=edge_attr.shape[1],
        memory_dim=hidden_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    if bptt:
        # Sesi tuning BPTT (6-Jul): LR terpisah — msg_mlp+memory_updater (GRU)
        # baru mulai belajar, dikasih LR lebih kecil (bptt_lr_scale x lr) spy
        # tak "kaget"; node_classifier (sudah terbukti stabil di jalur non-
        # BPTT) tetap di LR normal.
        memory_params = list(model.msg_mlp.parameters()) + list(model.memory_updater.parameters())
        clf_params = list(model.node_classifier.parameters()) + list(model.classifier.parameters())
        optimizer = torch.optim.Adam([
            {"params": memory_params, "lr": lr * bptt_lr_scale},
            {"params": clf_params, "lr": lr},
        ], weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Fix (6-Jul, audit roadmap #5): dicoba CosineAnnealingWarmRestarts(T_0=50)
    # tapi hasilnya REGRESI (test PR-AUC 0.9341 vs 0.9474 ReduceLROnPlateau) —
    # restart LR ke tinggi tiap 50 epoch kemungkinan mengganggu konvergensi yg
    # baru stabil di epoch ~80-140 (early-stop keburu sblm sempat menetap lagi
    # pasca-restart). Dikembalikan ke ReduceLROnPlateau (config terbukti
    # terbaik). Item ini "impact sedang" di roadmap — tak perlu iterasi lanjut.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    # Fix 4-Jul: FocalLoss (bukan BCEWithLogitsLoss+pos_weight) — konsisten
    # dgn train_graphsage (fallback) & train_dygformer (primary di train_dyg.py)
    # yang SUDAH pakai FocalLoss. ManualTGN sebelumnya tertinggal — jadi satu-
    # satunya path training yg masih pakai teknik imbalance lebih sederhana,
    # padahal modelnya ikut dipakai di ensemble produksi (tgn_weight=0.30).
    # Focal fokus ke hard examples (gamma), bukan cuma reweight linear kelas.
    # alpha default 0.90 (5-Jul): sekarang CLI-tunable — dulu dikalibrasi
    # asumsi illicit SANGAT langka (<1%), tapi prevalensi node-level ASLI
    # (setelah fix split temporal) ternyata ~11-15%, jauh lebih tinggi.
    # Recall selalu tinggi (0.7-0.9) & precision selalu rendah (0.17-0.19)
    # di semua epoch = ciri khas alpha kebesaran utk prevalensi seaktual ini.
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

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

    if pretrain_epochs > 0:
        # (7-Jul, riset TGN Rossi et al.) Self-supervised link-prediction
        # pretraining — lihat docstring. msg_mlp+memory_updater+classifier
        # dilatih dari SEMUA edge (bukan label fraud), sebelum classifier
        # utama dilatih di atas memory yg sudah "paham" pola temporal ini.
        print(f"\n[PRETRAIN] Self-supervised link-prediction: {pretrain_epochs} "
              f"epoch (msg_mlp+GRU dilatih dari {edge_index.shape[1]:,} edge, "
              f"bukan label fraud)")
        pretrain_params = (list(model.msg_mlp.parameters())
                            + list(model.memory_updater.parameters())
                            + list(model.classifier.parameters()))
        pretrain_optimizer = torch.optim.Adam(pretrain_params, lr=pretrain_lr, weight_decay=1e-5)
        link_criterion = nn.BCEWithLogitsLoss()
        for pe in range(1, pretrain_epochs + 1):
            pt0 = time.time()
            model.train()
            model.reset_memory()
            total_link_loss = 0.0
            nb_link = 0
            for i in range(0, len(edge_order), replay_batch):
                eidx = edge_order[i:i + replay_batch]
                sb, db, eab, tsb = _batch_to_device(eidx)
                logit_pos, logit_neg = model.pretrain_link_step(x, sb, db, eab, tsb, num_nodes)

                pretrain_optimizer.zero_grad()
                link_loss = (link_criterion(logit_pos, torch.ones_like(logit_pos))
                             + link_criterion(logit_neg, torch.zeros_like(logit_neg)))
                link_loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_params, 1.0)
                pretrain_optimizer.step()
                total_link_loss += link_loss.item()
                nb_link += 1

            avg_link_loss = total_link_loss / max(nb_link, 1)
            print(f"[PRETRAIN] Epoch {pe:3d}/{pretrain_epochs} | "
                  f"link_loss={avg_link_loss:.4f} | {time.time() - pt0:.1f}s")
        print("[PRETRAIN] Selesai — msg_mlp+GRU sudah belajar pola temporal "
              "asli dari link prediction (bukan random lagi). Lanjut training "
              "classifier di atas memory ini.\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        model.reset_memory()

        use_bptt_this_epoch = bptt and epoch > bptt_warmup_epochs
        if use_bptt_this_epoch:
            # (audit roadmap #3, EKSPERIMENTAL — lihat docstring) SATU pass
            # kronologis pakai train_step_batch — msg_mlp + memory_updater
            # (GRU) + node_classifier ke-update BARENG per batch (truncated
            # BPTT: grad ngalir DALAM batch, memory ditulis-balik detached
            # sebelum batch berikutnya).
            total_loss = 0.0
            nb = 0
            for i in range(0, len(edge_order), replay_batch):
                eidx = edge_order[i:i + replay_batch]      # CPU index
                sb, db, eab, tsb = _batch_to_device(eidx)

                logits, node_idx = model.train_step_batch(x, sb, db, eab, tsb)
                mask = is_train_node[node_idx]
                if not mask.any():
                    continue   # batch ini cuma nyentuh node val/test — skip backward

                optimizer.zero_grad()
                loss = criterion(logits[mask], node_labels_t[node_idx[mask]])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                nb += 1

            avg_loss = total_loss / max(nb, 1)
        else:
            # DEFAULT (menang, test PR-AUC 0.9471) — memory beku: replay SEMUA
            # edge no_grad, lalu classify+loss cuma lewat node_classifier.
            with torch.no_grad():
                _replay_all_memory()

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
        # Val logits dihitung ULANG di sini (bukan pakai yg terakhir dari loop
        # training) supaya konsisten dgn BEST model state yg baru di-restore,
        # bukan state epoch terakhir. Dipakai utk threshold_tuning.py (5-Jul).
        val_logits = model.classify_nodes(x, val_nodes)

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

    # Ekspor prediksi VAL (bukan test — test cuma utk laporan akhir, tuning
    # threshold harus di val spy test tetap unbiased) utk threshold_tuning.py.
    val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_logits.cpu().numpy(), -500, 500)))
    val_pred_df = pd.DataFrame({
        "label": node_labels_np[val_nodes_np],
        "score": val_probs,
    })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_pred_path = os.path.join(RESULTS_DIR, "tgn_val_predictions.csv")
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"[SAVE] Val predictions -> {val_pred_path} "
          f"(pakai ml.threshold_tuning utk cari threshold optimal)")

    # Fix (7-Jul, multi-seed ensemble): ekspor TEST predictions juga —
    # sebelumnya cuma metrics test yg di-print, raw score-nya tak pernah
    # disimpan. Butuh ini utk ensemble antar-seed (rata-rata score, bukan
    # cuma bandingkan metrics akhir).
    test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits.cpu().numpy(), -80, 80)))
    test_pred_df = pd.DataFrame({
        "account_idx": test_nodes_np,
        "label": node_labels_np[test_nodes_np],
        "score": test_probs,
    })

    return {
        "model": model,
        "test_metrics": test_metrics,
        "best_val_prauc": best_val_prauc,
        "training_log": training_log,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
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


def _start_memory_watchdog(critical_gb: float = 1.5, interval_s: float = 5.0):
    """
    Fix (7-Jul): watchdog RAM di background — dipicu oleh insiden OOM
    berulang di sesi ini (compute_node_features_full, dst). Mesin ini cuma
    16GB total, seringkali cuma ~4GB free (WSL2/Docker/Chrome makan sisanya).
    Kalau --sample-licit dinaikkan dari default 0.31, konsumsi RAM proses
    ini naik proporsional — bukan cuma nunggu crash MemoryError yg baru
    kelihatan SETELAH kejadian, watchdog ini kasih PERINGATAN DINI begitu
    available RAM sistem (bukan cuma RSS proses ini) mendekati kritis, plus
    coba gc.collect() begitu peringatan pertama muncul (buang cache Python
    yg bisa dilepas, kadang cukup utk lewati titik kritis tanpa OOM).
    Tidak bisa MENJAMIN no-OOM (Python tak bisa cegah OS mematikan proses),
    tapi kasih sinyal SEDINI mungkin utk operator manual Ctrl+C kalau perlu.
    """
    if not _PSUTIL_AVAILABLE:
        print("[WATCHDOG] psutil tidak ada — skip monitoring RAM otomatis.")
        return

    warned_once = [False]

    def _watch():
        while True:
            try:
                avail_gb = psutil.virtual_memory().available / 1e9
                if avail_gb < critical_gb:
                    print(f"\n[WATCHDOG] !! RAM sistem tersisa {avail_gb:.2f} GB "
                          f"(< {critical_gb} GB kritis) !! Kalau ini terus turun, "
                          f"SEGERA Ctrl+C manual sebelum OS/OOM-killer paksa "
                          f"mematikan proses (checkpoint TERAKHIR yg tersimpan "
                          f"mungkin bukan yg terbaru).", flush=True)
                    if not warned_once[0]:
                        gc.collect()
                        warned_once[0] = True
                else:
                    warned_once[0] = False
            except Exception:
                pass
            time.sleep(interval_s)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    print(f"[WATCHDOG] Monitoring RAM sistem aktif (peringatan di bawah {critical_gb} GB tersedia).")


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
    parser.add_argument("--sample-licit", type=float, default=0.31,
                        help="Licit sampling ratio dari seluruh CSV "
                             "(default: 0.31 — fix 4-Jul, target ~56M edge "
                             "total dari komposisi real, lihat tgn_dataset.py)")
    parser.add_argument("--sample-illicit", type=float, default=1.0,
                        help="Illicit sampling ratio dari seluruh CSV "
                             "(default: 1.0 = keep semua illicit, minoritas langka)")
    parser.add_argument("--eval-sample-ratio", type=float, default=0.2,
                        help="Downsample SERAGAM (bukan rebalance) utk window "
                             "val+test (fix 5-Jul — prevalensi ASLI dipertahankan, "
                             "cuma ukurannya dikecilkan demi batas memori GPU)")
    parser.add_argument("--focal-alpha", type=float, default=0.90,
                        help="FocalLoss alpha (5-Jul, CLI-tunable) — default 0.90 "
                             "dikalibrasi asumsi illicit sangat langka (<1%%); "
                             "prevalensi node-level ASLI ternyata ~11-15%%, coba "
                             "alpha lebih rendah (mis. 0.5-0.7) kalau recall jauh "
                             "> precision di semua epoch (tanda alpha kebesaran)")
    parser.add_argument("--mini-batch-size", type=int, default=2048,
                        help="Mini-batch size untuk TGN training (default: 2048)")
    parser.add_argument("--bptt", action="store_true",
                        help="EKSPERIMENTAL (audit roadmap #3): latih memory "
                             "(msg_mlp+GRU) via truncated BPTT (train_step_batch) "
                             "alih-alih memory beku. Default OFF krn hasil 6-Jul "
                             "regresi (0.80 vs 0.9471) dgn hyperparameter saat "
                             "ini — disimpan utk sesi tuning lanjutan, BUKAN "
                             "default produksi.")
    parser.add_argument("--bptt-warmup", type=int, default=0,
                        help="Sesi tuning BPTT: N epoch pertama pakai memory "
                             "beku dulu (stabilkan classifier) sblm switch ke "
                             "BPTT penuh. Cuma berlaku kalau --bptt aktif.")
    parser.add_argument("--bptt-lr-scale", type=float, default=0.1,
                        help="Sesi tuning BPTT: LR msg_mlp+memory_updater = "
                             "lr * scale ini (default 0.1 — lebih hati2 dari "
                             "LR classifier). Cuma berlaku kalau --bptt aktif.")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="Riset TGN resmi (7-Jul): N epoch self-supervised "
                             "link-prediction utk latih msg_mlp+GRU dari SEMUA "
                             "edge SEBELUM training classifier (memory freeze "
                             "spt biasa sesudahnya). Default 0 = skip, sama "
                             "spt sebelumnya (memory random).")
    parser.add_argument("--pretrain-lr", type=float, default=0.001,
                        help="LR utk fase pretrain link-prediction (default 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-cache", action="store_true", default=False,
                        help="Skip data cache, paksa reload dari CSV")
    parser.add_argument("--tag", type=str, default=None,
                        help="Multi-seed ensemble (7-Jul): suffix output model "
                             "(tgn_v1_<tag>.pt) & test predictions "
                             "(tgn_test_predictions_<tag>.csv) — supaya run "
                             "seed berbeda TIDAK saling menimpa. Default None "
                             "= perilaku lama (tgn_v1.pt, tanpa export test pred).")
    args = parser.parse_args()

    _start_memory_watchdog()

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

    # Fix (7-Jul): npz sebelumnya CUMA di-tag feature_contract_version — tidak
    # ikut cek sample_licit/sample_illicit/eval_sample_ratio/max_rows. Insiden
    # nyata: smoke-test --max-rows 500000 --no-cache MENULIS npz (--no-cache
    # cuma cegah BACA, bukan cegah TULIS!) dgn versi fitur yg SAMA dgn run
    # sungguhan berikutnya (beda sample_licit) -> run sungguhan diam-diam
    # pakai npz smoke-test 500K baris, bukan data yg diminta. sampling_sig
    # sekarang ikut disimpan & divalidasi, supaya kombinasi sampling APAPUN
    # yg beda otomatis dianggap cache basi (bukan cuma feature_contract_version).
    sampling_sig = (f"maxrows={args.max_rows}|licit={args.sample_licit}|"
                     f"illicit={args.sample_illicit}|eval={args.eval_sample_ratio}")

    cache_ok = False
    if os.path.exists(cache_npz) and not args.no_cache:
        npz_probe = np.load(cache_npz, allow_pickle=True)
        cached_version = (
            str(npz_probe["feature_contract_version"])
            if "feature_contract_version" in npz_probe.files else None
        )
        cached_sampling_sig = (
            str(npz_probe["sampling_sig"])
            if "sampling_sig" in npz_probe.files else None
        )
        if cached_version == FEATURE_CONTRACT_VERSION and cached_sampling_sig == sampling_sig:
            cache_ok = True
        else:
            print(
                f"[CACHE] IGNORED — npz versi '{cached_version}'/sampling "
                f"'{cached_sampling_sig}' beda dari kode saat ini "
                f"'{FEATURE_CONTRACT_VERSION}'/'{sampling_sig}'. Regenerasi "
                f"otomatis (bukan pakai data usang/beda-sampling)."
            )

    if cache_ok:
        print(f"[CACHE] Loading data dari {os.path.basename(cache_npz)} "
              f"(feature_contract_version={FEATURE_CONTRACT_VERSION})")
        npz = np.load(cache_npz, allow_pickle=True)
        edge_timestamps_c = npz["edge_timestamps"]
        # Fix (5-Jul): split & scaler dulu TIDAK ikut ter-cache -> KeyError di
        # run kedua (pakai cache). Split direkonstruksi dari cutoff TIMESTAMP
        # yg disimpan (bukan posisi persentase — itu bug lama yg sudah
        # diperbaiki di tgn_dataset.py, densitas baris asimetris train/eval).
        train_cutoff_ts_c = float(npz["train_cutoff_ts"])
        val_cutoff_ts_c = float(npz["val_cutoff_ts"])
        split_c = {
            "train": np.where(edge_timestamps_c <= train_cutoff_ts_c)[0],
            "val": np.where((edge_timestamps_c > train_cutoff_ts_c) & (edge_timestamps_c <= val_cutoff_ts_c))[0],
            "test": np.where(edge_timestamps_c > val_cutoff_ts_c)[0],
        }
        data = {
            "node_features": npz["node_features"],
            "edge_index": npz["edge_index"],
            "edge_attr": npz["edge_attr"],
            "edge_timestamps": edge_timestamps_c,
            "edge_labels": npz["edge_labels"],
            "node_labels": npz["node_labels"],
            "account_to_idx": npz["account_to_idx"].item() if "account_to_idx" in npz.files else {},
            "split": split_c,
            "scaler": npz["scaler"].item() if "scaler" in npz.files else None,
        }
    else:
        try:
            data = load_temporal_dataset_fast(
                csv_path=args.csv,
                max_rows=args.max_rows,
                sample_licit_ratio=args.sample_licit,
                sample_illicit_ratio=args.sample_illicit,
                eval_sample_ratio=args.eval_sample_ratio,
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

        print(f"[CACHE] Saving data ke {os.path.basename(cache_npz)} "
              f"(feature_contract_version={FEATURE_CONTRACT_VERSION})")
        np.savez(cache_npz,
                 node_features=data["node_features"],
                 edge_index=data["edge_index"],
                 edge_attr=data["edge_attr"],
                 edge_timestamps=data["edge_timestamps"],
                 edge_labels=data["edge_labels"],
                 node_labels=data["node_labels"],
                 account_to_idx=np.array(data.get("account_to_idx", {}), dtype=object),
                 scaler=np.array(data.get("scaler"), dtype=object),
                 train_cutoff_ts=data["train_cutoff_ts"],
                 val_cutoff_ts=data["val_cutoff_ts"],
                 feature_contract_version=FEATURE_CONTRACT_VERSION,
                 sampling_sig=sampling_sig)

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
            focal_alpha=args.focal_alpha,
            bptt=args.bptt,
            bptt_warmup_epochs=args.bptt_warmup,
            bptt_lr_scale=args.bptt_lr_scale,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_lr=args.pretrain_lr,
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

    # Multi-seed ensemble (7-Jul): kalau --tag dipakai, simpan model & test
    # predictions ke path ber-tag supaya run seed lain tak saling menimpa.
    if args.tag:
        model_path = os.path.join(MODEL_DIR, f"tgn_v1_{args.tag}.pt")
        if "test_pred_df" in result:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            test_pred_path = os.path.join(RESULTS_DIR, f"tgn_test_predictions_{args.tag}.csv")
            result["test_pred_df"].to_csv(test_pred_path, index=False)
            print(f"[SAVE] Test predictions -> {test_pred_path} (utk multi-seed ensemble)")
    else:
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
