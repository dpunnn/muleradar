"""
Ensemble XGBoost + ManualTGN + DyGFormerNode untuk production risk scoring.

Arsitektur:
  - XGBoost      : tabular, 20 node features, real-time capable
  - ManualTGN    : temporal graph memory + node classification
  - DyGFormerNode: patch-based temporal transformer (PR-AUC 0.9843, best model)
  - Ensemble     : weighted average (default xgb=0.30, tgn=0.30, dyg=0.40)

Dua path:
  - predict()                : real-time, XGBoost-only (GNN butuh full graph)
  - score_batch_from_cache() : batch, full 3-model ensemble dari npz cache

Cara pakai:
    from ml.ensemble import EnsemblePredictor
    pred = EnsemblePredictor()

    # Real-time (XGBoost only):
    df = pred.predict(features_df)

    # Batch (full ensemble):
    df = pred.score_batch_from_cache()
"""

import os
import sys
import time
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from feature_defs import FEATURE_COLS   # definisi kanonik (fix duplikasi 6-Jul)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models")
)
XGB_MODEL_PATH = os.path.join(_MODELS_DIR, "xgboost_v1.pkl")
TGN_MODEL_PATH = os.path.join(_MODELS_DIR, "tgn_v1.pt")
DYG_MODEL_PATH = os.path.join(_MODELS_DIR, "dyg_v1.pt")

_DATA_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
)
NPZ_DEFAULT = os.path.join(_DATA_DIR, "transactions_hi_injected_traindata.npz")

_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "results")
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _classify_risk(score):
    """Classify risk level from probability score."""
    if score > 0.8:
        return "HIGH"
    elif score > 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def _load_xgb_model(path):
    """Load XGBoost model from pickle file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "XGBoost model not found at {}. "
            "Train it first with: python -m detection.model".format(path)
        )
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# EnsemblePredictor
# ---------------------------------------------------------------------------
class EnsemblePredictor:
    """
    Production ensemble: XGBoost + ManualTGN + DyGFormerNode.

    XGBoost loaded eagerly (wajib).
    TGN dan DyGFormer loaded lazily saat score_batch_from_cache() dipanggil.
    """

    def __init__(
        self,
        xgb_weight=0.30,
        tgn_weight=0.30,
        dyg_weight=0.40,
        xgb_path=None,
        tgn_path=None,
        dyg_path=None,
    ):
        self.xgb_weight = xgb_weight
        self.tgn_weight = tgn_weight
        self.dyg_weight = dyg_weight
        self.xgb_path = xgb_path or XGB_MODEL_PATH
        self.tgn_path = tgn_path or TGN_MODEL_PATH
        self.dyg_path = dyg_path or DYG_MODEL_PATH

        # Load XGBoost (required)
        self.xgb_model = _load_xgb_model(self.xgb_path)
        logger.info("XGBoost model loaded from %s", self.xgb_path)

    # ------------------------------------------------------------------
    # Real-time path: XGBoost only
    # ------------------------------------------------------------------
    def predict_xgb(self, features_df):
        """
        XGBoost predict_proba pada FEATURE_COLS.

        Parameters
        ----------
        features_df : pd.DataFrame
            Harus punya FEATURE_COLS. Kolom account_id opsional.

        Returns
        -------
        np.ndarray shape (N,) — probability scores
        """
        X = features_df[FEATURE_COLS].fillna(0)
        return self.xgb_model.predict_proba(X)[:, 1]

    def predict(self, features_df):
        """
        Real-time prediction: XGBoost only.
        TGN butuh full graph + memory replay, tidak feasible per-request.

        Parameters
        ----------
        features_df : pd.DataFrame
            Harus punya FEATURE_COLS. Kolom account_id opsional.

        Returns
        -------
        pd.DataFrame: account_id (jika ada), xgb_score, ensemble_score, risk_level
        """
        xgb_scores = self.predict_xgb(features_df)

        result = pd.DataFrame({"xgb_score": xgb_scores})

        if "account_id" in features_df.columns:
            result.insert(0, "account_id", features_df["account_id"].values)

        # Real-time: ensemble_score = xgb_score (no TGN available)
        result["ensemble_score"] = xgb_scores
        result["risk_level"] = result["ensemble_score"].apply(_classify_risk)

        return result

    # ------------------------------------------------------------------
    # Batch path: full ensemble (XGBoost + TGN)
    # ------------------------------------------------------------------
    def score_batch_from_cache(self, npz_path=None):
        """
        BATCH scoring: load npz, build TGN memory, classify ALL nodes.

        Steps:
          1. Load npz cache (node_features, edge_index, edge_attr, edge_timestamps)
          2. XGBoost: predict_proba pada node_features
          3. TGN: load ManualTGN, replay edges (temporal order), classify_nodes
          4. Ensemble = xgb_weight * xgb + tgn_weight * tgn
          5. Build DataFrame dengan risk levels

        Parameters
        ----------
        npz_path : str, optional
            Path ke npz cache. Default: NPZ_DEFAULT.

        Returns
        -------
        pd.DataFrame: node_idx, account_id (jika ada), xgb_score, tgn_score,
                       ensemble_score, risk_level
        """
        npz_path = npz_path or NPZ_DEFAULT

        # --- 1. Load npz ---
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(
                "NPZ cache not found at {}. "
                "Run train_tgn first to generate it.".format(npz_path)
            )

        logger.info("Loading npz cache: %s", npz_path)
        npz = np.load(npz_path, allow_pickle=True)

        # Cek feature-contract version (fix 3-Jul) — npz basi (mis. HHI lama,
        # sebelum fix train/serve skew) tetap BISA dipakai (bukan blocking,
        # ensemble cuma skor bukan train), tapi WAJIB diberi warning eksplisit
        # supaya operator tahu skor bisa tidak konsisten dgn xgb_model terkini.
        try:
            from ml.tgn_dataset import FEATURE_CONTRACT_VERSION
            cached_version = (
                str(npz["feature_contract_version"])
                if "feature_contract_version" in npz.files else None
            )
            if cached_version != FEATURE_CONTRACT_VERSION:
                logger.warning(
                    "npz feature_contract_version='%s' BEDA dari kode saat ini "
                    "'%s' — kemungkinan HHI/scaling npz basi (train/serve skew). "
                    "Regenerasi npz: python -m ml.train_tgn --no-cache",
                    cached_version, FEATURE_CONTRACT_VERSION,
                )
        except ImportError:
            pass

        node_features = npz["node_features"]  # (N, 20)
        edge_index = npz["edge_index"]        # (2, E)
        edge_attr = npz["edge_attr"]          # (E, 3)
        edge_timestamps = npz["edge_timestamps"]  # (E,)
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        # account_to_idx mungkin tidak ada di cache lama
        account_to_idx = None
        if "account_to_idx" in npz.files:
            account_to_idx = npz["account_to_idx"].item()

        logger.info(
            "Loaded: %d nodes, %d edges", num_nodes, num_edges
        )

        # --- 2. XGBoost scores ---
        logger.info("Computing XGBoost scores...")
        xgb_input = pd.DataFrame(node_features, columns=FEATURE_COLS)
        try:
            xgb_scores = self.xgb_model.predict_proba(xgb_input)[:, 1]
        except ValueError as e:
            if "feature_names mismatch" in str(e) or "did not have the following fields" in str(e):
                logger.warning(
                    "XGBoost model trained on old features. Retraining on 20 features..."
                )
                xgb_scores = self._retrain_and_score_xgb(
                    node_features, npz, xgb_input
                )
            else:
                raise

        # --- 3. TGN scores ---
        tgn_scores = self._compute_tgn_scores(
            node_features, edge_index, edge_attr, edge_timestamps, num_nodes
        )

        # --- 4. DyGFormer scores ---
        dyg_scores = self._compute_dyg_scores(
            node_features, edge_index, edge_attr, edge_timestamps, num_nodes
        )

        # --- 5. Ensemble ---
        # Normalise weights agar selalu sum ke 1.0 meski ada model yang fallback ke 0
        total_w = self.xgb_weight + self.tgn_weight + self.dyg_weight
        ensemble_scores = (
            (self.xgb_weight * xgb_scores
             + self.tgn_weight * tgn_scores
             + self.dyg_weight * dyg_scores) / total_w
        )

        # --- 6. Build DataFrame ---
        result = pd.DataFrame({
            "node_idx":      np.arange(num_nodes),
            "xgb_score":     xgb_scores,
            "tgn_score":     tgn_scores,
            "dyg_score":     dyg_scores,
            "ensemble_score": ensemble_scores,
        })

        # Map node_idx -> account_id jika tersedia
        if account_to_idx is not None:
            idx_to_account = {v: k for k, v in account_to_idx.items()}
            result["account_id"] = result["node_idx"].map(idx_to_account)
            cols = ["node_idx", "account_id", "xgb_score", "tgn_score",
                    "dyg_score", "ensemble_score"]
            result = result[cols]

        result["risk_level"] = result["ensemble_score"].apply(_classify_risk)

        # --- Ringkasan ---
        counts = result["risk_level"].value_counts()
        n_high = counts.get("HIGH", 0)
        n_med  = counts.get("MEDIUM", 0)
        n_low  = counts.get("LOW", 0)
        logger.info(
            "Ensemble done: %d HIGH, %d MEDIUM, %d LOW (total %d nodes)",
            n_high, n_med, n_low, num_nodes,
        )
        print("=" * 60)
        print("ENSEMBLE SCORING SUMMARY")
        print("=" * 60)
        print("  Nodes scored : {:,}".format(num_nodes))
        print("  HIGH  (>0.8) : {:,}".format(n_high))
        print("  MEDIUM(>0.5) : {:,}".format(n_med))
        print("  LOW   (<=0.5): {:,}".format(n_low))
        print("  Weights      : xgb={:.2f}, tgn={:.2f}, dyg={:.2f}".format(
            self.xgb_weight / total_w,
            self.tgn_weight / total_w,
            self.dyg_weight / total_w,
        ))
        print("=" * 60)

        return result

    def _compute_tgn_scores(
        self, node_features, edge_index, edge_attr, edge_timestamps, num_nodes
    ):
        """
        Load ManualTGN, replay all edges to build memory, classify all nodes.

        Edge tensors stay on CPU; only per-batch slices move to device.
        Node classification is batched (100k per batch) to avoid OOM.

        Falls back to zeros (= xgb-only effective) if tgn checkpoint missing.
        """
        # Check TGN checkpoint
        if not os.path.isfile(self.tgn_path):
            logger.warning(
                "TGN model not found at %s. "
                "Degrading to XGBoost-only (tgn_scores = 0).",
                self.tgn_path,
            )
            return np.zeros(num_nodes, dtype=np.float32)

        # Lazy import torch + ManualTGN
        try:
            import torch
            from ml.tgn_model import ManualTGN
        except ImportError as e:
            logger.warning(
                "Cannot import torch/ManualTGN: %s. "
                "Degrading to XGBoost-only.", e
            )
            return np.zeros(num_nodes, dtype=np.float32)

        # Load checkpoint
        logger.info("Loading TGN checkpoint: %s", self.tgn_path)
        ckpt = torch.load(self.tgn_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        hidden_dim   = meta.get("hidden_dim", 128)
        edge_feat_dim = 3

        # Infer node_feat_dim dari shape checkpoint (backward-compatible dengan model lama)
        # msg_mlp.0.weight shape: [hidden_dim, memory_dim*2 + node_feat_dim + edge_feat_dim + 1]
        # (+1 utk delta_t_log, fix 6-Jul roadmap #4 — WAJIB dihitung, kalau tidak
        # tgn_feat_dim ke-infer 1 lebih besar dari sebenarnya (25 bukan 24) ->
        # ManualTGN dikonstruksi salah dimensi -> load_state_dict gagal size
        # mismatch. Baru ketemu 12-Jul saat coba scoring TGN utk ensemble.)
        msg_w = ckpt["model_state_dict"]["msg_mlp.0.weight"]
        tgn_feat_dim = msg_w.shape[1] - hidden_dim * 2 - edge_feat_dim - 1
        if tgn_feat_dim != len(FEATURE_COLS):
            logger.info(
                "[TGN] Checkpoint trained on %d features (current FEATURE_COLS=%d). "
                "Using first %d features for TGN inference.",
                tgn_feat_dim, len(FEATURE_COLS), tgn_feat_dim,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("TGN device: %s", device)

        tgn = ManualTGN(
            num_nodes=num_nodes,
            node_feat_dim=tgn_feat_dim,
            edge_feat_dim=edge_feat_dim,
            memory_dim=hidden_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        tgn.load_state_dict(ckpt["model_state_dict"])
        tgn.eval()

        # Prepare tensors — slice node_features ke feat_dim yang dipakai TGN
        nf_tgn = node_features[:, :tgn_feat_dim]
        x = torch.tensor(nf_tgn, dtype=torch.float32, device=device)
        src_all = torch.tensor(edge_index[0], dtype=torch.long)   # CPU
        dst_all = torch.tensor(edge_index[1], dtype=torch.long)   # CPU
        ea = torch.tensor(edge_attr, dtype=torch.float32)         # CPU
        ts = torch.tensor(edge_timestamps, dtype=torch.float32)   # CPU

        # Temporal order
        edge_order = torch.argsort(ts).cpu()

        # Replay edges to build memory (batch 8192, per-batch to device)
        logger.info("Building TGN memory: replaying %d edges...", len(edge_order))
        mb = 8192
        with torch.no_grad():
            tgn.reset_memory()
            for i in range(0, len(edge_order), mb):
                eidx = edge_order[i:i + mb]
                tgn.update_memory_only(
                    x,
                    src_all[eidx].to(device),
                    dst_all[eidx].to(device),
                    ea[eidx].to(device),
                    ts[eidx].to(device),
                )

        # Classify ALL nodes (batch 100k to avoid OOM)
        logger.info("Classifying %d nodes...", num_nodes)
        all_scores = np.zeros(num_nodes, dtype=np.float32)
        node_batch_size = 100_000
        with torch.no_grad():
            for start in range(0, num_nodes, node_batch_size):
                end = min(start + node_batch_size, num_nodes)
                node_idx = torch.arange(start, end, dtype=torch.long, device=device)
                logits = tgn.classify_nodes(x, node_idx)
                scores = torch.sigmoid(logits).cpu().numpy()
                all_scores[start:end] = scores

        logger.info("TGN scoring complete.")
        return all_scores

    def _retrain_and_score_xgb(self, node_features, npz, xgb_input):
        """
        Retrain XGBoost dari NPZ cache (dipakai saat feature mismatch dengan pkl lama).
        Menyimpan model baru ke xgb_path supaya run berikutnya tidak perlu retrain.
        """
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split

        node_labels = npz["node_labels"]
        all_nodes   = np.arange(len(node_labels))
        train_idx, _ = train_test_split(
            all_nodes, test_size=0.20, stratify=node_labels, random_state=42
        )
        X_train = node_features[train_idx]
        y_train = node_labels[train_idx]
        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

        xgb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw, eval_metric="aucpr",
            random_state=42, n_jobs=-1,
        )
        logger.info("[XGB] Training on %d nodes with %d features...",
                    len(train_idx), node_features.shape[1])
        xgb.fit(X_train, y_train)

        # Simpan model baru
        with open(self.xgb_path, "wb") as f:
            pickle.dump(xgb, f)
        self.xgb_model = xgb
        logger.info("[XGB] Retrained model saved -> %s", self.xgb_path)

        return xgb.predict_proba(xgb_input)[:, 1]

    def _compute_dyg_scores(
        self, node_features, edge_index, edge_attr, edge_timestamps, num_nodes
    ):
        """
        Load DyGFormerNode, build TemporalNeighborIndex, classify all nodes.

        Falls back to zeros (= xgb+tgn effective) jika dyg checkpoint missing.
        """
        if not os.path.isfile(self.dyg_path):
            logger.warning(
                "DyGFormer model not found at %s. Degrading to 0.",
                self.dyg_path,
            )
            return np.zeros(num_nodes, dtype=np.float32)

        try:
            import torch
            from ml.train_dyg import TemporalNeighborIndex, DyGFormerNode
        except ImportError as e:
            logger.warning(
                "Cannot import DyGFormerNode: %s. Degrading to 0.", e
            )
            return np.zeros(num_nodes, dtype=np.float32)

        logger.info("Loading DyGFormer checkpoint: %s", self.dyg_path)
        ckpt = torch.load(self.dyg_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {})
        d_model       = meta.get("hidden_dim", 128)
        n_heads       = meta.get("n_heads", 4)
        n_layers      = meta.get("n_layers", 2)
        k_neighbors   = meta.get("k_neighbors", 10)
        node_feat_dim = meta.get("n_feature_cols", len(FEATURE_COLS))
        edge_feat_dim = edge_attr.shape[1] if edge_attr.ndim > 1 else 1

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("DyGFormer device: %s", device)

        model = DyGFormerNode(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            k_neighbors=k_neighbors,
            use_grad_ckpt=False,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Build CSR neighbor index
        logger.info("[DYG] Building TemporalNeighborIndex (%d edges)...", edge_index.shape[1])
        t0 = time.time()
        nbr_index = TemporalNeighborIndex(edge_index, edge_timestamps, num_nodes)
        logger.info("[DYG] Index built in %.1fs", time.time() - t0)

        x           = torch.tensor(node_features, dtype=torch.float32, device=device)
        edge_attr_t = torch.tensor(edge_attr,      dtype=torch.float32, device=device)
        cutoff_ts   = float(edge_timestamps.max())

        all_scores = np.zeros(num_nodes, dtype=np.float32)
        batch_size = 2048

        logger.info("[DYG] Classifying %d nodes (batch=%d)...", num_nodes, batch_size)
        with torch.no_grad():
            for start in range(0, num_nodes, batch_size):
                end      = min(start + batch_size, num_nodes)
                batch_np = np.arange(start, end)

                nbr_ids_np, nbr_dts_np, nbr_eidx_np, mask_np = nbr_index.get_k_recent(
                    batch_np, k=k_neighbors, cutoff_ts=cutoff_ts
                )
                # mask_np: (N, K) True=valid → flip ke True=padded, prepend False untuk target
                nbr_pad = ~mask_np
                target_col = np.zeros((len(batch_np), 1), dtype=bool)
                key_pad = np.concatenate([target_col, nbr_pad], axis=1)  # (N, K+1)

                tgt       = torch.tensor(batch_np,    dtype=torch.long,    device=device)
                nbr_ids   = torch.tensor(nbr_ids_np,  dtype=torch.long,    device=device)
                nbr_dts   = torch.tensor(nbr_dts_np,  dtype=torch.float32, device=device)
                nbr_eidx  = torch.tensor(nbr_eidx_np, dtype=torch.long,    device=device)
                key_pad_t = torch.tensor(key_pad,      dtype=torch.bool,    device=device)

                logits = model(x, tgt, nbr_ids, nbr_dts, nbr_eidx, edge_attr_t, key_pad_t)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
                all_scores[start:end] = torch.sigmoid(logits).cpu().numpy()

        logger.info("[DYG] Scoring complete.")
        return all_scores

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------
    @property
    def has_tgn(self):
        """Whether TGN checkpoint file exists."""
        return os.path.isfile(self.tgn_path)

    @property
    def has_dyg(self):
        """Whether DyGFormer checkpoint file exists."""
        return os.path.isfile(self.dyg_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("Initializing EnsemblePredictor...")
    predictor = EnsemblePredictor()

    print("Running batch scoring from cache...")
    df = predictor.score_batch_from_cache()

    # Save head 1000 to CSV
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(_RESULTS_DIR, "ensemble_scores.csv")
    df.head(1000).to_csv(out_path, index=False)
    print("Saved top 1000 rows -> {}".format(out_path))

    # Print sample
    print("\nTop 20 highest-risk nodes:")
    top20 = df.nlargest(20, "ensemble_score")
    print(top20.to_string(index=False))
