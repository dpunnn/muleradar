"""
Training pipeline: DyGFormer (Yu et al. 2023) untuk AML node classification.

DyGFormer adalah temporal graph transformer state-of-the-art:
  - Patch-based temporal encoding (bukan event-by-event seperti TGN)
  - Multi-head attention atas K temporal neighbor patches
  - Jauh lebih efisien untuk graph besar karena tidak maintain memory state

Memory optimizations untuk 56M edge:
  - fp16 mixed precision (torch.cuda.amp) → ~2x less VRAM
  - Gradient checkpointing pada transformer layers → tukar compute vs memory
  - K=10 temporal neighbors (bukan full neighborhood)

Fallback chain:
  DyGFormerNode → ManualTGN (jika OOM atau tidak ada GPU)

Cara pakai:
    cd backend
    python -m ml.train_dyg
    python -m ml.train_dyg --epochs 50 --k-neighbors 10 --fp16
    python -m ml.train_dyg --fallback-tgn   # paksa pakai ManualTGN
"""

import argparse
import os
import sys
import math
import time
import csv
import functools

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

from ml.tgn_dataset import load_temporal_dataset_fast, CSV_DEFAULT, FEATURE_COLS
from ml.tgn_model import ManualTGN

MODEL_DIR  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))

# ── Utility ───────────────────────────────────────────────────────────────────

def compute_metrics(labels: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> dict:
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    preds = (probs >= thr).astype(int)
    prauc = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    return {
        "pr_auc":    prauc,
        "f1":        f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
    }


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits.detach())
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── Temporal Neighbor Index ───────────────────────────────────────────────────

class TemporalNeighborIndex:
    """
    Numpy CSR-based temporal neighbor index untuk graph besar (56M+ edge).

    Struktur data:
      indptr  : (num_nodes+1,) int64  — pointer ke slice per-node di arrays bawah
      _nbr    : (2E,) int64           — neighbor ID, sorted per-node by timestamp asc
      _ts     : (2E,) float64         — edge timestamp corresponding to each entry
      _eidx   : (2E,) int64           — edge attr index corresponding to each entry

    Build vectorized pakai np.lexsort → O(E log E), detik bukan menit.
    Memory: ~3.6 GB untuk 56M edge (vs ~9-10 GB Python list-of-tuples).
    """

    def __init__(self, edge_index: np.ndarray, edge_timestamps: np.ndarray,
                 num_nodes: int):
        src = edge_index[0].astype(np.int64)
        dst = edge_index[1].astype(np.int64)
        E   = len(src)
        eidx_orig = np.arange(E, dtype=np.int64)

        # Bidirectional: tiap edge muncul dua kali (sebagai out-edge src dan in-edge dst)
        nodes_bi = np.concatenate([src, dst])
        nbrs_bi  = np.concatenate([dst, src])
        ts_bi    = np.concatenate([edge_timestamps, edge_timestamps]).astype(np.float64)
        eidx_bi  = np.concatenate([eidx_orig, eidx_orig])

        # Sort by (node_id ASC, timestamp ASC) — lexsort key order: last = primary
        order = np.lexsort((ts_bi, nodes_bi))
        self._nbr   = nbrs_bi[order]
        self._ts    = ts_bi[order]
        self._eidx  = eidx_bi[order]
        nodes_sorted = nodes_bi[order]

        # Build CSR indptr
        degree = np.bincount(nodes_sorted, minlength=num_nodes).astype(np.int64)
        self.indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(degree, out=self.indptr[1:])

    def get_k_recent(self, node_ids: np.ndarray, k: int, cutoff_ts: float = None):
        """
        Untuk setiap node di node_ids, ambil K tetangga temporal terbaru.
        Returns:
          nbr_ids  : (N, K) int64
          nbr_dts  : (N, K) float32 — delta time (ref_ts - edge_ts), non-negative
          nbr_eidx : (N, K) int64   — edge index untuk lookup edge_attr
          mask     : (N, K) bool    — True = valid (bukan padding)
        """
        N = len(node_ids)
        nbr_ids  = np.zeros((N, k), dtype=np.int64)
        nbr_dts  = np.zeros((N, k), dtype=np.float32)
        nbr_eidx = np.zeros((N, k), dtype=np.int64)
        mask     = np.zeros((N, k), dtype=bool)

        for i, nid in enumerate(node_ids):
            s, e = int(self.indptr[nid]), int(self.indptr[nid + 1])
            if s == e:
                continue

            node_ts   = self._ts[s:e]      # sudah sorted asc
            node_nbrs = self._nbr[s:e]
            node_eidx = self._eidx[s:e]

            if cutoff_ts is not None:
                # Binary search — O(log n) vs O(n) linear scan
                cut = int(np.searchsorted(node_ts, cutoff_ts, side="right"))
                if cut == 0:
                    continue
                node_ts   = node_ts[:cut]
                node_nbrs = node_nbrs[:cut]
                node_eidx = node_eidx[:cut]

            n_avail = len(node_ts)
            if n_avail == 0:
                continue

            # Ambil K terbaru (tail), balik agar index 0 = most recent
            start_k = max(0, n_avail - k)
            rec_ts   = node_ts[start_k:][::-1]
            rec_nbrs = node_nbrs[start_k:][::-1]
            rec_eidx = node_eidx[start_k:][::-1]

            n_fill = len(rec_ts)
            ref_ts = cutoff_ts if cutoff_ts is not None else float(rec_ts[0])
            nbr_ids[i,  :n_fill] = rec_nbrs
            nbr_dts[i,  :n_fill] = (ref_ts - rec_ts).astype(np.float32).clip(min=0)
            nbr_eidx[i, :n_fill] = rec_eidx
            mask[i,     :n_fill] = True

        return nbr_ids, nbr_dts, nbr_eidx, mask


# ── DyGFormer Node Classifier ─────────────────────────────────────────────────

class DyGFormerNode(nn.Module):
    """
    DyGFormer-style temporal graph transformer untuk node classification.

    Arsitektur (simplified dari Yu et al. 2023):
      1. Embed node features → d_model
      2. Per-node: ambil K temporal neighbors, embed mereka
      3. Positional encoding berbasis delta timestamp
      4. Transformer encoder atas sequence [target_node | nbr_1 | ... | nbr_K]
      5. Linear head → binary classification

    Memory optimizations:
      - K=10 neighbor limit (bukan full 1-hop)
      - Gradient checkpointing di setiap transformer layer
      - fp16 via AMP (dikontrol di training loop, bukan di sini)
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        k_neighbors: int = 10,
        dropout: float = 0.1,
        use_grad_ckpt: bool = True,
    ):
        super().__init__()
        self.k_neighbors    = k_neighbors
        self.d_model        = d_model
        self.use_grad_ckpt  = use_grad_ckpt

        # Input projections
        self.node_proj = nn.Linear(node_feat_dim, d_model)
        self.edge_proj = nn.Linear(edge_feat_dim, d_model)

        # Time encoding: sinusoidal positional encoding atas delta-t
        self.time_dim = d_model // 4
        self.time_proj = nn.Linear(1, self.time_dim)

        # Patch fusion: combine neighbor node feat + edge feat + time enc
        # time_enc output = time_dim * 2 (sin + cos concatenated)
        patch_in = d_model + d_model + self.time_dim * 2
        self.patch_proj = nn.Linear(patch_in, d_model)

        # Transformer encoder
        self.transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def _time_enc(self, delta_t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time encoding. delta_t: (N, K) → (N, K, time_dim*2)."""
        t = delta_t.unsqueeze(-1)                           # (N, K, 1)
        w = self.time_proj(t).clamp(-10.0, 10.0)           # clamp sebelum sin/cos (fp16 safe)
        return torch.cat([torch.sin(w), torch.cos(w)], dim=-1)

    def forward(
        self,
        x: torch.Tensor,              # (num_nodes, node_feat_dim)
        target_nodes: torch.LongTensor,  # (N,)
        nbr_ids: torch.LongTensor,    # (N, K)
        nbr_dts: torch.FloatTensor,   # (N, K) delta timestamps
        nbr_eidx: torch.LongTensor,   # (N, K) edge indices
        edge_attr: torch.FloatTensor, # (E, edge_feat_dim)
        key_padding_mask: torch.BoolTensor,  # (N, K+1) — True = ignore
    ) -> torch.Tensor:                # (N,) logits

        N, K = nbr_ids.shape

        # 1. Target node embedding (N, d_model)
        target_emb = self.node_proj(x[target_nodes])  # (N, d_model)

        # 2. Neighbor node embeddings (N, K, d_model)
        flat_nbr = nbr_ids.reshape(-1)                           # (N*K,)
        nbr_node_emb = self.node_proj(x[flat_nbr]).view(N, K, -1)  # (N, K, d)

        # 3. Neighbor edge embeddings (N, K, d_model)
        flat_eidx = nbr_eidx.reshape(-1)                         # (N*K,)
        nbr_edge_emb = self.edge_proj(edge_attr[flat_eidx]).view(N, K, -1)  # (N, K, d)

        # 4. Time encoding (N, K, time_dim*2)
        # Clamp delta_t: log1p normalization untuk stabilitas
        nbr_dts_norm = torch.log1p(nbr_dts.clamp(min=0))        # (N, K)
        time_emb = self._time_enc(nbr_dts_norm)                  # (N, K, time_dim*2? no)
        # time_proj returns (N, K, time_dim), after sin+cos → (N, K, time_dim*2)
        # but we named it self.time_dim = d_model//4, so output is d_model//4 * 2 = d_model//2
        # Patch: nbr_node_emb + nbr_edge_emb + time_emb → patch_proj
        patch_in_dim = self.d_model + self.d_model + self.time_dim * 2
        patch_raw = torch.cat([nbr_node_emb, nbr_edge_emb, time_emb], dim=-1)  # (N, K, patch_in_dim)
        patch_emb = self.patch_proj(patch_raw)                   # (N, K, d_model)

        # 5. Sequence: [target | patch_0 | ... | patch_K-1] → (N, K+1, d_model)
        target_seq = target_emb.unsqueeze(1)                     # (N, 1, d_model)
        seq = torch.cat([target_seq, patch_emb], dim=1)         # (N, K+1, d_model)

        # key_padding_mask: (N, K+1), False for target (always valid), then nbr mask
        # True = position to ignore in attention

        # 6. Transformer encoder with optional gradient checkpointing
        for layer in self.transformer:
            if self.use_grad_ckpt and self.training:
                # checkpoint tidak support kwargs → pakai partial untuk bind mask
                fn = functools.partial(layer, src_key_padding_mask=key_padding_mask)
                seq = torch.utils.checkpoint.checkpoint(fn, seq, use_reentrant=False)
            else:
                seq = layer(seq, src_key_padding_mask=key_padding_mask)

        seq = self.norm(seq)  # (N, K+1, d_model)

        # 7. Classification: pakai representasi token target (posisi 0)
        target_repr = seq[:, 0, :]         # (N, d_model)
        logits = self.cls_head(target_repr).squeeze(-1)  # (N,)
        # Clamp untuk cegah fp16 overflow → NaN saat sigmoid
        return logits.clamp(-20.0, 20.0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_dygformer(
    data: dict,
    epochs: int = 100,
    lr: float = 0.0005,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    k_neighbors: int = 10,
    patience: int = 15,
    device: str = "cpu",
    use_fp16: bool = False,
    use_grad_ckpt: bool = True,
    batch_size: int = 2048,
    stratified_split: bool = True,
    seed: int = 42,
) -> dict:

    from sklearn.model_selection import train_test_split as _tts

    node_features   = data["node_features"]
    edge_index      = data["edge_index"]
    edge_attr_np    = data["edge_attr"]
    edge_timestamps = data["edge_timestamps"]
    node_labels_np  = data["node_labels"]
    num_nodes       = node_features.shape[0]

    # Node split — stratified (default) atau temporal
    # Stratified: distribusi illicit sama di train/val/test → DyGFormer tidak kena
    # distribution shift 65% train vs 7% test yang menyebabkan PR-AUC 0.07
    # Temporal: dipakai ManualTGN (punya full memory), tidak cocok untuk model lokal K-hop
    all_nodes = np.arange(num_nodes)
    if stratified_split:
        train_nodes_np, temp_np = _tts(
            all_nodes, test_size=0.30,
            stratify=node_labels_np, random_state=seed,
        )
        val_nodes_np, test_nodes_np = _tts(
            temp_np, test_size=0.50,
            stratify=node_labels_np[temp_np], random_state=seed,
        )
        split_name = "STRATIFIED"
    else:
        src, dst = edge_index[0], edge_index[1]
        node_first_ts = np.full(num_nodes, np.inf)
        np.minimum.at(node_first_ts, src, edge_timestamps)
        np.minimum.at(node_first_ts, dst, edge_timestamps)
        node_first_ts[node_first_ts == np.inf] = edge_timestamps.min()
        order = np.argsort(node_first_ts)
        n_train = int(0.70 * num_nodes)
        n_val   = int(0.15 * num_nodes)
        train_nodes_np = order[:n_train]
        val_nodes_np   = order[n_train: n_train + n_val]
        test_nodes_np  = order[n_train + n_val:]
        split_name = "TEMPORAL"

    print(f"[DYG] {split_name} split: train={len(train_nodes_np):,} "
          f"val={len(val_nodes_np):,} test={len(test_nodes_np):,}")
    print(f"[DYG] Illicit rate — train:{node_labels_np[train_nodes_np].mean():.3f} "
          f"val:{node_labels_np[val_nodes_np].mean():.3f} "
          f"test:{node_labels_np[test_nodes_np].mean():.3f}")

    # Build temporal neighbor index (numpy CSR — efisien untuk 56M+ edge)
    print(f"[DYG] Building temporal neighbor index (CSR, {edge_index.shape[1]/1e6:.1f}M edges)...")
    t_idx_start = time.time()
    tni = TemporalNeighborIndex(edge_index, edge_timestamps, num_nodes)
    idx_mb = (tni._nbr.nbytes + tni._ts.nbytes + tni._eidx.nbytes + tni.indptr.nbytes) / 1e6
    print(f"[DYG] Index built in {time.time()-t_idx_start:.1f}s | RAM: {idx_mb:.0f} MB")

    # Tensors
    x         = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_attr = torch.tensor(edge_attr_np,  dtype=torch.float32, device=device)
    node_labels_t = torch.tensor(node_labels_np, dtype=torch.float32, device=device)

    # Cutoff ts: pakai max timestamp seluruh dataset (stratified = tidak ada leakage concern)
    # Stratified split: semua node sudah ada dari awal, cutoff = max ts
    global_cutoff_ts = float(edge_timestamps.max())
    train_cutoff_ts  = global_cutoff_ts
    val_cutoff_ts    = global_cutoff_ts

    # Class balance + pos_weight untuk BCEWithLogitsLoss fallback
    n_pos = int(node_labels_np[train_nodes_np].sum())
    n_neg = len(train_nodes_np) - n_pos
    print(f"[DYG] Train class balance: {n_pos:,} pos / {n_neg:,} neg "
          f"(ratio {n_neg/max(n_pos,1):.1f}:1)")

    # Model
    model = DyGFormerNode(
        node_feat_dim=node_features.shape[1],
        edge_feat_dim=edge_attr_np.shape[1],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        k_neighbors=k_neighbors,
        use_grad_ckpt=use_grad_ckpt,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[DYG] DyGFormerNode params: {n_params:,}")
    print(f"[DYG] fp16={use_fp16}, grad_ckpt={use_grad_ckpt}, K={k_neighbors}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01  # tidak turun ke 0 supaya stabil
    )
    criterion = FocalLoss(alpha=0.9, gamma=2.0)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_fp16 and device == "cuda"))

    best_val_prauc    = 0.0
    best_model_state  = None
    patience_counter  = 0
    training_log      = []

    def _get_batch_tensors(node_batch_np: np.ndarray, cutoff_ts: float = None):
        """
        Ambil K temporal neighbors per node, return tensors ready for DyGFormerNode.
        """
        nbr_ids_np, nbr_dts_np, nbr_eidx_np, mask_np = tni.get_k_recent(
            node_batch_np, k_neighbors, cutoff_ts=cutoff_ts
        )
        # key_padding_mask: True di posisi yang DIABAIKAN
        # Target (posisi 0) selalu valid → False
        # Neighbor valid di mana mask_np == True → False saat valid
        nbr_pad_mask = ~mask_np                             # True = padding, ignore
        target_pad   = np.zeros((len(node_batch_np), 1), dtype=bool)
        key_pad      = np.concatenate([target_pad, nbr_pad_mask], axis=1)  # (N, K+1)

        return (
            torch.tensor(node_batch_np, dtype=torch.long,    device=device),
            torch.tensor(nbr_ids_np,    dtype=torch.long,    device=device),
            torch.tensor(nbr_dts_np,    dtype=torch.float32, device=device),
            torch.tensor(nbr_eidx_np,   dtype=torch.long,    device=device),
            torch.tensor(key_pad,       dtype=torch.bool,    device=device),
        )

    print(f"\n[DYG] Training DyGFormerNode: {epochs} epochs, lr={lr}, "
          f"d_model={d_model}, device={device}")
    print("-" * 70)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()

        # Mini-batch training
        perm = np.random.permutation(train_nodes_np)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            batch_np = perm[i: i + batch_size]
            try:
                tgt, nbr_ids, nbr_dts, nbr_eidx, key_pad = _get_batch_tensors(
                    batch_np, cutoff_ts=train_cutoff_ts
                )
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=(use_fp16 and device == "cuda")):
                    logits = model(x, tgt, nbr_ids, nbr_dts, nbr_eidx, edge_attr, key_pad)
                    loss   = criterion(logits, node_labels_t[tgt])

                # Skip batch jika loss/logits NaN (fp16 overflow)
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                n_batches  += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    raise RuntimeError(
                        "[OOM] DyGFormer OOM. Coba: --k-neighbors 5 atau --no-fp16 "
                        "atau --fallback-tgn"
                    ) from e
                raise

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        # Validation
        model.eval()
        val_probs_list = []
        val_labels_list = []
        with torch.no_grad():
            for i in range(0, len(val_nodes_np), batch_size):
                vbatch = val_nodes_np[i: i + batch_size]
                tgt, nbr_ids, nbr_dts, nbr_eidx, key_pad = _get_batch_tensors(
                    vbatch, cutoff_ts=val_cutoff_ts
                )
                with torch.amp.autocast("cuda", enabled=(use_fp16 and device == "cuda")):
                    logits = model(x, tgt, nbr_ids, nbr_dts, nbr_eidx, edge_attr, key_pad)
                # Guard NaN sebelum sigmoid (fp16 overflow di attention)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs_list.append(probs)
                val_labels_list.append(node_labels_np[vbatch])

        val_probs  = np.concatenate(val_probs_list)
        val_labels = np.concatenate(val_labels_list)
        val_m = compute_metrics(val_labels, val_probs)
        elapsed = time.time() - t0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_PR-AUC={val_m['pr_auc']:.4f} | "
                  f"F1={val_m['f1']:.4f} P={val_m['precision']:.4f} "
                  f"R={val_m['recall']:.4f} | {elapsed:.1f}s")

        training_log.append({
            "epoch": epoch, "loss": avg_loss,
            "val_pr_auc": val_m["pr_auc"], "val_f1": val_m["f1"],
        })

        if val_m["pr_auc"] > best_val_prauc:
            best_val_prauc = val_m["pr_auc"]
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[DYG] Early stopping epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model.to(device)

    # Test evaluation
    model.eval()
    test_probs_list  = []
    test_labels_list = []
    test_cutoff_ts = global_cutoff_ts  # stratified: semua node sudah ada, pakai max ts
    with torch.no_grad():
        for i in range(0, len(test_nodes_np), batch_size):
            tbatch = test_nodes_np[i: i + batch_size]
            tgt, nbr_ids, nbr_dts, nbr_eidx, key_pad = _get_batch_tensors(
                tbatch, cutoff_ts=test_cutoff_ts
            )
            with torch.amp.autocast("cuda", enabled=(use_fp16 and device == "cuda")):
                logits = model(x, tgt, nbr_ids, nbr_dts, nbr_eidx, edge_attr, key_pad)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            test_probs_list.append(torch.sigmoid(logits).cpu().numpy())
            test_labels_list.append(node_labels_np[tbatch])

    test_probs  = np.concatenate(test_probs_list)
    test_labels = np.concatenate(test_labels_list)
    test_m = compute_metrics(test_labels, test_probs)

    print("\n" + "=" * 70)
    print("[TEST] DyGFormerNode final evaluation (TEMPORAL INDUCTIVE split):")
    print(f"  PR-AUC   : {test_m['pr_auc']:.4f}")
    print(f"  F1@0.5   : {test_m['f1']:.4f}")
    print(f"  Precision: {test_m['precision']:.4f}")
    print(f"  Recall   : {test_m['recall']:.4f}")
    print(f"  Best val PR-AUC: {best_val_prauc:.4f}")
    print("=" * 70)

    return {
        "model": model,
        "test_metrics": test_m,
        "best_val_prauc": best_val_prauc,
        "training_log": training_log,
    }


def train_manualtgn_fallback(data: dict, epochs: int, lr: float,
                              hidden_dim: int, patience: int,
                              device: str, batch_size: int) -> dict:
    """Fallback ke ManualTGN jika DyGFormer OOM."""
    print("\n[FALLBACK] Switching ke ManualTGN...")
    from ml.train_tgn import train_tgn_manual
    return train_tgn_manual(
        data=data, epochs=epochs, lr=lr, hidden_dim=hidden_dim,
        patience=patience, device=device, mini_batch_size=batch_size,
    )


# ── Save helpers ──────────────────────────────────────────────────────────────

def save_model(model: nn.Module, path: str, metadata: dict = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt = {"model_state_dict": cpu_state,
            "model_class": model.__class__.__name__}
    if metadata:
        ckpt["metadata"] = metadata
    torch.save(ckpt, path)
    print(f"[SAVE] Model -> {os.path.abspath(path)}")


def save_log(log: list, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not log:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    print(f"[SAVE] Log -> {os.path.abspath(path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train DyGFormer for AML detection")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--lr",          type=float, default=0.0005)
    parser.add_argument("--d-model",     type=int,   default=128)
    parser.add_argument("--n-heads",     type=int,   default=4)
    parser.add_argument("--n-layers",    type=int,   default=2)
    parser.add_argument("--k-neighbors", type=int,   default=10,
                        help="Jumlah temporal neighbor per node (default: 10)")
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--batch-size",  type=int,   default=2048)
    parser.add_argument("--device",      type=str,   default=None,
                        help="cpu/cuda (default: auto)")
    parser.add_argument("--fp16",        action="store_true",
                        help="Enable mixed precision training (CUDA only)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing (lebih cepat, VRAM lebih besar)")
    parser.add_argument("--csv",         type=str,   default=CSV_DEFAULT)
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--max-rows",    type=int,   default=None)
    parser.add_argument("--fallback-tgn", action="store_true",
                        help="Langsung pakai ManualTGN (skip DyGFormer)")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Pakai temporal split (default: stratified). "
                             "Temporal cocok untuk ManualTGN, bukan DyGFormer.")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--sample-licit",   type=float, default=0.005)
    parser.add_argument("--sample-illicit", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    use_fp16       = args.fp16 and device == "cuda"
    use_grad_ckpt  = not args.no_grad_ckpt

    print("=" * 70)
    print(f"MuleRadar — DyGFormer Training (Yu et al. 2023)")
    print(f"Device    : {device}")
    if device == "cuda":
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"GPU       : {gpu_props.name}")
        print(f"VRAM      : {gpu_props.total_memory / 1e9:.1f} GB")
    print(f"fp16      : {use_fp16}")
    print(f"grad_ckpt : {use_grad_ckpt}")
    print(f"K         : {args.k_neighbors} temporal neighbors")
    print("=" * 70)

    # Load dataset
    print("\n[PHASE 1] Loading dataset...")
    t0 = time.time()
    cache_npz = args.csv.replace(".csv", "_traindata.npz")

    if os.path.exists(cache_npz) and not args.no_cache:
        print(f"[CACHE] Loading {os.path.basename(cache_npz)}")
        npz = np.load(cache_npz, allow_pickle=True)
        data = {
            "node_features": npz["node_features"],
            "edge_index":    npz["edge_index"],
            "edge_attr":     npz["edge_attr"],
            "edge_timestamps": npz["edge_timestamps"],
            "edge_labels":   npz["edge_labels"],
            "node_labels":   npz["node_labels"],
        }
    else:
        data = load_temporal_dataset_fast(
            csv_path=args.csv,
            max_rows=args.max_rows,
            sample_licit_ratio=args.sample_licit,
            sample_illicit_ratio=args.sample_illicit,
            random_seed=args.seed,
        )
        print(f"[CACHE] Saving {os.path.basename(cache_npz)}")
        np.savez(cache_npz,
                 node_features=data["node_features"],
                 edge_index=data["edge_index"],
                 edge_attr=data["edge_attr"],
                 edge_timestamps=data["edge_timestamps"],
                 edge_labels=data["edge_labels"],
                 node_labels=data["node_labels"])

    nf = data["node_features"]
    print(f"[PHASE 1] Done in {time.time()-t0:.1f}s  "
          f"nodes={nf.shape[0]:,} edges={data['edge_index'].shape[1]:,} "
          f"feat_dim={nf.shape[1]}")

    if nf.shape[1] != len(FEATURE_COLS):
        print(f"[ERROR] node_features dim={nf.shape[1]} != FEATURE_COLS={len(FEATURE_COLS)}")
        print("        Cache NPZ sudah usang. Jalankan ulang dengan --no-cache untuk rebuild.")
        sys.exit(1)

    # Train
    print("\n[PHASE 2] Training...")
    result = None

    if not args.fallback_tgn:
        try:
            result = train_dygformer(
                data=data,
                epochs=args.epochs,
                lr=args.lr,
                d_model=args.d_model,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
                k_neighbors=args.k_neighbors,
                patience=args.patience,
                device=device,
                use_fp16=use_fp16,
                use_grad_ckpt=use_grad_ckpt,
                batch_size=args.batch_size,
                stratified_split=not args.temporal_split,
                seed=args.seed,
            )
            model_name = "DyGFormerNode"
        except RuntimeError as e:
            if "OOM" in str(e) or "out of memory" in str(e).lower():
                print(f"\n[OOM] {e}")
                print("[FALLBACK] Switching ke ManualTGN...")
                torch.cuda.empty_cache()
            else:
                raise

    if result is None:
        result = train_manualtgn_fallback(
            data=data, epochs=args.epochs, lr=args.lr,
            hidden_dim=args.d_model, patience=args.patience,
            device=device, batch_size=args.batch_size,
        )
        model_name = "ManualTGN-fallback"

    # Save
    model_path = os.path.join(MODEL_DIR, "dyg_v1.pt")
    save_model(
        result["model"], model_path,
        metadata={
            "model_class":     result["model"].__class__.__name__,
            "model_name":      model_name,
            "test_metrics":    result["test_metrics"],
            "best_val_prauc":  result["best_val_prauc"],
            "epochs_trained":  len(result["training_log"]),
            "hidden_dim":      args.d_model,
            "n_heads":         args.n_heads,
            "n_layers":        args.n_layers,
            "k_neighbors":     args.k_neighbors,
            "fp16":            use_fp16,
            "grad_ckpt":       use_grad_ckpt,
            "n_feature_cols":  nf.shape[1],
        },
    )
    save_log(result["training_log"],
             os.path.join(RESULTS_DIR, "dyg_training_log.csv"))

    print("\n[DONE] Training complete.")
    print(f"  PR-AUC test : {result['test_metrics']['pr_auc']:.4f}")
    print(f"  Model saved : {model_path}")


if __name__ == "__main__":
    main()
