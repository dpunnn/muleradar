"""
TGN (Temporal Graph Network) dan Fallback GNN untuk AML detection.

Exports:
    - MuleRadarTGN: Full TGN dengan temporal memory (requires PyG TGN)
    - FallbackTemporalGNN: GraphSAGE-based edge classifier (always available)
    - get_model(): auto-detect dan return model terbaik yang available
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Check PyG TGN availability
# ------------------------------------------------------------------
_TGN_AVAILABLE = False
try:
    from torch_geometric.nn.models import TGN
    from torch_geometric.nn.models.tgn import (
        IdentityMessage,
        LastAggregator,
    )
    _TGN_AVAILABLE = True
    logger.info("PyG TGN module available")
except ImportError:
    logger.warning(
        "torch_geometric.nn.models.TGN not available. "
        "Will use FallbackTemporalGNN instead."
    )

_SAGE_AVAILABLE = False
try:
    from torch_geometric.nn import SAGEConv
    _SAGE_AVAILABLE = True
except ImportError:
    logger.warning("torch_geometric.nn.SAGEConv not available.")


# ------------------------------------------------------------------
# MuleRadarTGN: Full temporal graph network
# ------------------------------------------------------------------
if _TGN_AVAILABLE:
    class MuleRadarTGN(nn.Module):
        """
        TGN-based AML detector with temporal memory.

        Uses PyG's TGN implementation:
        - Per-node temporal memory updated via message passing
        - IdentityMessage: raw edge features as messages
        - LastAggregator: keep most recent message per node

        Forward pass returns edge-level logits (before sigmoid).
        """

        def __init__(
            self,
            num_nodes: int,
            node_feat_dim: int = 13,
            edge_feat_dim: int = 3,
            memory_dim: int = 64,
            time_dim: int = 16,
            embedding_dim: int = 64,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_nodes = num_nodes
            self.node_feat_dim = node_feat_dim
            self.edge_feat_dim = edge_feat_dim
            self.memory_dim = memory_dim

            # Raw message dim = edge features only (IdentityMessage uses this)
            raw_msg_dim = edge_feat_dim

            self.tgn = TGN(
                num_nodes=num_nodes,
                raw_msg_dim=raw_msg_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
                embedding_dim=embedding_dim,
                message_module=IdentityMessage(
                    raw_msg_dim, memory_dim, time_dim
                ),
                aggregator_module=LastAggregator(),
            )

            # Edge-level classifier: concat src+dst embeddings -> predict
            self.classifier = nn.Sequential(
                nn.Linear(embedding_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, src, dst, t, msg, n_id=None):
            """
            Parameters
            ----------
            src : LongTensor (B,) source node indices
            dst : LongTensor (B,) destination node indices
            t : FloatTensor (B,) timestamps
            msg : FloatTensor (B, edge_feat_dim) edge features
            n_id : LongTensor, optional, neighborhood node ids

            Returns
            -------
            logits : FloatTensor (B,) — raw logits (use BCEWithLogitsLoss)
            """
            # Update memory and get embeddings
            self.tgn.update(src, dst, t, msg)

            # Get node embeddings for src and dst
            z_src = self.tgn.embedding(src)
            z_dst = self.tgn.embedding(dst)

            # Classify edge
            edge_repr = torch.cat([z_src, z_dst], dim=-1)
            return self.classifier(edge_repr).squeeze(-1)

        def reset_memory(self):
            """Reset temporal memory state (call between epochs/splits)."""
            self.tgn.reset_state()

        def detach_memory(self):
            """Detach memory from computation graph (call each batch)."""
            self.tgn.detach_memory()

else:
    # Placeholder when TGN is not available
    class MuleRadarTGN(nn.Module):
        """Placeholder — TGN not available. Use FallbackTemporalGNN."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyG TGN module not available. "
                "Install torch-geometric with TGN support, "
                "or use FallbackTemporalGNN instead."
            )


# ------------------------------------------------------------------
# FallbackTemporalGNN: GraphSAGE + edge classifier
# ------------------------------------------------------------------
class FallbackTemporalGNN(nn.Module):
    """
    Fallback GNN when TGN is not available.
    2-layer GraphSAGE for node embeddings + MLP for edge classification.

    No temporal memory, but includes temporal features (hour) in edge_attr.
    """

    def __init__(
        self,
        node_feat_dim: int = 13,
        edge_feat_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim

        if not _SAGE_AVAILABLE:
            raise ImportError(
                "torch_geometric.nn.SAGEConv not available. "
                "Install torch-geometric."
            )

        self.conv1 = SAGEConv(node_feat_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = dropout

        # Edge classifier: src_emb + dst_emb + edge_attr -> 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Parameters
        ----------
        x : FloatTensor (N, node_feat_dim)
        edge_index : LongTensor (2, E)
        edge_attr : FloatTensor (E, edge_feat_dim), optional

        Returns
        -------
        logits : FloatTensor (E,) — raw logits per edge
        """
        # Node embeddings via GraphSAGE
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Edge-level prediction
        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst]], dim=-1)

        if edge_attr is not None:
            edge_repr = torch.cat([edge_repr, edge_attr.float()], dim=-1)
        else:
            # Pad with zeros if no edge attributes
            pad = torch.zeros(
                edge_repr.size(0), self.edge_feat_dim,
                device=edge_repr.device
            )
            edge_repr = torch.cat([edge_repr, pad], dim=-1)

        return self.edge_mlp(edge_repr).squeeze(-1)

    def get_node_embeddings(self, x, edge_index):
        """Get node embeddings without edge classification (for ensemble)."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h


# ------------------------------------------------------------------
# ManualTGN: pure-PyTorch TGN (no PyG TGN dependency)
# ------------------------------------------------------------------
class ManualTGN(nn.Module):
    """
    Manual TGN implementation (Rossi et al. 2020).
    Node memory update via GRU per temporal interaction.
    Tidak butuh PyG TGN module — pure PyTorch.

    Flow per edge (src->dst, t):
      1. Read memory[src] dan memory[dst]
      2. Compute message: concat(memory[src], memory[dst], node_feat[src], edge_attr)
      3. Update memory[src] via GRU
      4. Predict: MLP(memory[src], memory[dst], edge_attr) -> logit
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int = 13,
        edge_feat_dim: int = 3,
        memory_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.memory_dim = memory_dim

        # Node memory: persistent state per node (not a parameter, updated in-place)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes))

        # Message: concat(mem_src, mem_dst, node_feat_src, edge_attr) -> msg_dim
        msg_dim = memory_dim * 2 + node_feat_dim + edge_feat_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(msg_dim, memory_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Memory updater: GRU cell
        self.memory_updater = nn.GRUCell(memory_dim, memory_dim)

        self.node_feat_dim = node_feat_dim

        # Edge classifier (lama, untuk kompatibilitas edge-level — biarkan ada)
        clf_in_edge = memory_dim * 2 + edge_feat_dim
        self.classifier = nn.Sequential(
            nn.Linear(clf_in_edge, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # NODE classifier: concat(memory, node_features) -> logit per node
        clf_in_node = memory_dim + node_feat_dim
        self.node_classifier = nn.Sequential(
            nn.Linear(clf_in_node, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def reset_memory(self):
        """Reset semua memory ke zero. Dipanggil di awal setiap epoch."""
        self.memory.zero_()
        self.last_update.zero_()

    def forward_batch(
        self,
        node_features: torch.Tensor,   # (N, 13)
        src_batch: torch.Tensor,        # (B,) edge src indices
        dst_batch: torch.Tensor,        # (B,) edge dst indices
        edge_attr_batch: torch.Tensor,  # (B, 3)
        timestamps_batch: torch.Tensor, # (B,) unix timestamps
    ) -> torch.Tensor:
        """
        Process one mini-batch of edges IN TEMPORAL ORDER.
        Update memory untuk setiap edge, return logits.
        """
        B = src_batch.size(0)
        logits = torch.zeros(B, device=self.memory.device)

        for i in range(B):
            s = src_batch[i].item()
            d = dst_batch[i].item()
            ea = edge_attr_batch[i]           # (3,)
            nf_s = node_features[s]           # (13,)

            mem_s = self.memory[s]            # (memory_dim,)
            mem_d = self.memory[d]            # (memory_dim,)

            # Classify BEFORE memory update (predict dari state sebelum transaksi ini)
            clf_input = torch.cat([mem_s, mem_d, ea])
            logits[i] = self.classifier(clf_input.unsqueeze(0)).squeeze()

            # Compute message
            msg_raw = torch.cat([mem_s, mem_d, nf_s, ea])
            msg = self.msg_mlp(msg_raw.unsqueeze(0)).squeeze(0)

            # Update memory[src] dengan GRU
            new_mem_s = self.memory_updater(msg.unsqueeze(0), mem_s.unsqueeze(0)).squeeze(0)
            self.memory[s] = new_mem_s.detach()
            self.last_update[s] = timestamps_batch[i]

        return logits

    def forward_batch_vectorized(
        self,
        node_features: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        edge_attr_batch: torch.Tensor,
        timestamps_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized version: approximate TGN.
        Baca memory state saat ini (tidak sequential), jauh lebih cepat.
        Cocok untuk batch besar, slight accuracy trade-off vs sequential.
        """
        mem_s = self.memory[src_batch]   # (B, memory_dim)
        mem_d = self.memory[dst_batch]   # (B, memory_dim)
        ea = edge_attr_batch             # (B, 3)

        # Classify
        clf_input = torch.cat([mem_s, mem_d, ea], dim=-1)  # (B, 2*mem + 3)
        logits = self.classifier(clf_input).squeeze(-1)    # (B,)

        # Update memory for unique src nodes (last write wins for duplicates)
        nf_s = node_features[src_batch]  # (B, 13)
        msg_raw = torch.cat([mem_s, mem_d, nf_s, ea], dim=-1)  # (B, msg_dim)
        msgs = self.msg_mlp(msg_raw)                            # (B, memory_dim)
        new_mem_s = self.memory_updater(msgs, mem_s)            # (B, memory_dim)

        # Scatter update: untuk setiap unique src, ambil update terakhir
        self.memory[src_batch] = new_mem_s.detach()
        self.last_update[src_batch] = timestamps_batch

        return logits

    def update_memory_only(
        self,
        node_features: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        edge_attr_batch: torch.Tensor,
        timestamps_batch: torch.Tensor,
    ):
        """
        Proses satu batch edge HANYA untuk update memory (tanpa classify).
        Dipakai untuk membangun temporal memory state sebelum node classification.
        Memory di-update via GRU, detach supaya tidak membangun graph antar-batch.
        """
        mem_s = self.memory[src_batch]
        mem_d = self.memory[dst_batch]
        nf_s = node_features[src_batch]
        ea = edge_attr_batch

        msg_raw = torch.cat([mem_s, mem_d, nf_s, ea], dim=-1)
        msgs = self.msg_mlp(msg_raw)
        new_mem_s = self.memory_updater(msgs, mem_s)

        # Update memory in-place (detach supaya tidak membangun graph antar-batch)
        self.memory[src_batch] = new_mem_s.detach()
        # Juga update dst supaya penerima dana ikut punya memory
        mem_d_new = self.memory_updater(msgs, mem_d)
        self.memory[dst_batch] = mem_d_new.detach()
        self.last_update[src_batch] = timestamps_batch
        self.last_update[dst_batch] = timestamps_batch

    def classify_nodes(
        self,
        node_features: torch.Tensor,
        node_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify node dari concat(memory, node_features).
        node_idx: (B,) indeks node yang mau diklasifikasi.
        Return: (B,) logits.
        """
        mem = self.memory[node_idx]           # (B, memory_dim)
        nf = node_features[node_idx]          # (B, node_feat_dim)
        clf_input = torch.cat([mem, nf], dim=-1)
        return self.node_classifier(clf_input).squeeze(-1)

    def train_step_batch(
        self,
        node_features: torch.Tensor,
        src_batch: torch.Tensor,
        dst_batch: torch.Tensor,
        edge_attr_batch: torch.Tensor,
        timestamps_batch: torch.Tensor,
    ):
        """
        BPTT step: update memory src+dst DENGAN gradient, lalu classify
        node-node tersebut dari memory baru + features.
        Gradient mengalir ke msg_mlp + memory_updater + node_classifier.
        Memory ditulis-balik dalam keadaan detached (truncated BPTT antar-batch).

        Return: (logits, node_idx) di mana node_idx = concat(src, dst).
        """
        mem_s = self.memory[src_batch]
        mem_d = self.memory[dst_batch]
        nf_s = node_features[src_batch]
        nf_d = node_features[dst_batch]
        ea = edge_attr_batch

        # Message dari interaksi ini
        msg_raw = torch.cat([mem_s, mem_d, nf_s, ea], dim=-1)
        msgs = self.msg_mlp(msg_raw)

        # Update memory src & dst (WITH grad)
        new_mem_s = self.memory_updater(msgs, mem_s)
        new_mem_d = self.memory_updater(msgs, mem_d)

        # Classify src & dst dari memory baru + features (WITH grad)
        logit_s = self.node_classifier(
            torch.cat([new_mem_s, nf_s], dim=-1)).squeeze(-1)
        logit_d = self.node_classifier(
            torch.cat([new_mem_d, nf_d], dim=-1)).squeeze(-1)

        # Tulis-balik memory dalam keadaan detached
        self.memory[src_batch] = new_mem_s.detach()
        self.memory[dst_batch] = new_mem_d.detach()
        self.last_update[src_batch] = timestamps_batch
        self.last_update[dst_batch] = timestamps_batch

        logits = torch.cat([logit_s, logit_d])
        node_idx = torch.cat([src_batch, dst_batch])
        return logits, node_idx


# ------------------------------------------------------------------
# Auto-detect best available model
# ------------------------------------------------------------------
def get_model(
    num_nodes: int,
    node_feat_dim: int = 13,
    edge_feat_dim: int = 3,
    hidden_dim: int = 64,
    dropout: float = 0.1,
    prefer_tgn: bool = True,
) -> nn.Module:
    """
    Factory: return the best available model.

    Returns MuleRadarTGN if available and prefer_tgn=True,
    otherwise FallbackTemporalGNN.
    """
    if prefer_tgn and _TGN_AVAILABLE:
        try:
            model = MuleRadarTGN(
                num_nodes=num_nodes,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                embedding_dim=hidden_dim,
                memory_dim=hidden_dim,
                dropout=dropout,
            )
            print(f"[MODEL] Using MuleRadarTGN (num_nodes={num_nodes:,})")
            return model
        except Exception as e:
            logger.warning("TGN init failed: %s. Falling back to GraphSAGE.", e)

    # ManualTGN selalu available — pakai jika TGN PyG tidak ada
    if prefer_tgn:
        try:
            model = ManualTGN(
                num_nodes=num_nodes,
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                memory_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            print(f"[MODEL] Using ManualTGN (num_nodes={num_nodes:,}, memory_dim={hidden_dim})")
            return model
        except Exception as e:
            logger.warning("ManualTGN init failed: %s. Falling back to GraphSAGE.", e)

    model = FallbackTemporalGNN(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    print(f"[MODEL] Using FallbackTemporalGNN (GraphSAGE-based)")
    return model
