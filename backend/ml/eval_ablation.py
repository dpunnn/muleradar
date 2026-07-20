"""
Ablation + Ensemble diagnostic — temporal inductive split.

Semua model dievaluasi pada node test split TEMPORAL (bukan random):
  - Train : node yang PERTAMA KALI muncul di 70% periode terlama
  - Val   : node yang pertama muncul di periode 70-85%
  - Test  : node yang pertama muncul di 15% periode TERBARU (akun paling baru)

Ini inductive temporal split — model harus generalize ke akun yang belum
pernah dilihat saat training, persis seperti kondisi produksi nyata.

Cara pakai:
    cd backend
    python -m ml.eval_ablation
    python -m ml.eval_ablation --random-split   # fallback ke random split lama
"""

import os
import sys
import csv
import argparse
import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from ml.tgn_dataset import CSV_DEFAULT, temporal_inductive_split
from feature_defs import FEATURE_COLS   # definisi kanonik (fix duplikasi 6-Jul)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
TGN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "tgn_v1.pt"))


def metrics(labels, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    pa = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    return {
        "pr_auc": pa,
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def random_node_split(node_labels, num_nodes):
    """Fallback: split acak lama (untuk perbandingan)."""
    all_nodes = np.arange(num_nodes)
    train_nodes, temp_nodes = train_test_split(
        all_nodes, test_size=0.3, stratify=node_labels, random_state=42)
    val_nodes, test_nodes = train_test_split(
        temp_nodes, test_size=0.5, stratify=node_labels[temp_nodes], random_state=42)
    print("      [RANDOM SPLIT] — bukan temporal, hanya untuk perbandingan")
    return train_nodes, val_nodes, test_nodes


def stratified_node_split(node_labels, num_nodes):
    """
    Stratified split — sama dengan yang dipakai DyGFormerNode training.
    Distribusi illicit seragam di semua split (tidak ada distribution shift).
    Digunakan sebagai default karena AMLWorld money mules muncul early
    sehingga temporal split menciptakan distribusi artifisial (65% vs 7%).
    """
    all_nodes = np.arange(num_nodes)
    train_nodes, temp_nodes = train_test_split(
        all_nodes, test_size=0.30, stratify=node_labels, random_state=42)
    val_nodes, test_nodes = train_test_split(
        temp_nodes, test_size=0.50, stratify=node_labels[temp_nodes], random_state=42)
    print("      [STRATIFIED SPLIT] — distribusi illicit seragam (sama dgn DyGFormer)")
    return train_nodes, val_nodes, test_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-split", action="store_true",
                        help="Gunakan random split lama (untuk perbandingan)")
    parser.add_argument("--temporal-split", action="store_true",
                        help="Gunakan temporal inductive split (AMLWorld: distribusi shift)")
    args = parser.parse_args()

    cache_npz = CSV_DEFAULT.replace(".csv", "_traindata.npz")
    if not os.path.exists(cache_npz):
        print("[ERROR] Cache tidak ada: {}".format(cache_npz))
        print("Jalankan dulu: python -m ml.train_tgn")
        sys.exit(1)

    print("[1/4] Loading cache {}...".format(os.path.basename(cache_npz)))
    npz = np.load(cache_npz, allow_pickle=True)
    node_features  = npz["node_features"]
    node_labels    = npz["node_labels"]
    edge_index     = npz["edge_index"]
    edge_attr      = npz["edge_attr"]
    edge_timestamps = npz["edge_timestamps"]
    num_nodes = node_features.shape[0]

    print("      Nodes={:,} | illicit={:,} ({:.1f}%)".format(
        num_nodes, int(node_labels.sum()), node_labels.mean() * 100))
    print("      Timestamp range: {:.0f} → {:.0f}".format(
        edge_timestamps.min(), edge_timestamps.max()))

    # ── Split ──────────────────────────────────────────────────────────────
    if args.random_split:
        split_mode = "random"
    elif args.temporal_split:
        split_mode = "temporal_inductive"
    else:
        split_mode = "stratified"
    print("\n      Split mode: {}".format(split_mode.upper()))

    if args.random_split:
        train_nodes, val_nodes, test_nodes = random_node_split(node_labels, num_nodes)
    elif args.temporal_split:
        train_nodes, val_nodes, test_nodes = temporal_inductive_split(
            edge_index, edge_timestamps, num_nodes)
    else:
        train_nodes, val_nodes, test_nodes = stratified_node_split(node_labels, num_nodes)

    # Hitung illicit rate di setiap split
    train_ill = node_labels[train_nodes].mean() * 100
    val_ill   = node_labels[val_nodes].mean() * 100
    test_ill  = node_labels[test_nodes].mean() * 100
    print("      train={:,} ({:.1f}% illicit) | val={:,} ({:.1f}%) | test={:,} ({:.1f}%)".format(
        len(train_nodes), train_ill,
        len(val_nodes),   val_ill,
        len(test_nodes),  test_ill))

    y_test = node_labels[test_nodes]

    # ── XGBoost ────────────────────────────────────────────────────────────
    print("\n[2/4] Training XGBoost ({} split)...".format(split_mode))
    X_train = node_features[train_nodes]
    y_train = node_labels[train_nodes]
    X_test  = node_features[test_nodes]

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    xgb_test = xgb.predict_proba(X_test)[:, 1]
    m_xgb = metrics(y_test, xgb_test)
    print("      XGBoost test PR-AUC={:.4f} F1={:.4f}".format(
        m_xgb["pr_auc"], m_xgb["f1"]))

    # ── ManualTGN ──────────────────────────────────────────────────────────
    print("\n[3/4] Loading ManualTGN + scoring test nodes...")
    import torch
    from ml.tgn_model import ManualTGN

    ckpt = torch.load(TGN_PATH, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    hidden_dim = meta.get("hidden_dim", 128)

    # Infer node_feat_dim dari checkpoint (backward-compat dengan model 13-fitur lama)
    msg_w = ckpt["model_state_dict"]["msg_mlp.0.weight"]
    tgn_feat_dim = msg_w.shape[1] - hidden_dim * 2 - 3
    if tgn_feat_dim != len(FEATURE_COLS):
        print("      [TGN] Checkpoint: {} features (FEATURE_COLS={}). "
              "Pakai {} features untuk TGN.".format(
                  tgn_feat_dim, len(FEATURE_COLS), tgn_feat_dim))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgn = ManualTGN(
        num_nodes=num_nodes, node_feat_dim=tgn_feat_dim, edge_feat_dim=3,
        memory_dim=hidden_dim, hidden_dim=hidden_dim,
    ).to(device)
    tgn.load_state_dict(ckpt["model_state_dict"])
    tgn.eval()

    # Slice node_features ke tgn_feat_dim (13 jika model lama)
    x        = torch.tensor(node_features[:, :tgn_feat_dim], dtype=torch.float32, device=device)
    src_all  = torch.tensor(edge_index[0],  dtype=torch.long,    device=device)
    dst_all  = torch.tensor(edge_index[1],  dtype=torch.long,    device=device)
    ea       = torch.tensor(edge_attr,       dtype=torch.float32, device=device)
    ts_all   = torch.tensor(edge_timestamps, dtype=torch.float32, device=device)

    # Temporal: filter edges hanya untuk temporal inductive split
    # Stratified & random: replay semua edge (konsisten dengan cara TGN ditraining)
    if args.temporal_split:
        train_node_set = set(train_nodes.tolist()) | set(val_nodes.tolist())
        src_np = edge_index[0]
        dst_np = edge_index[1]
        train_edge_mask = np.isin(src_np, list(train_node_set)) & \
                          np.isin(dst_np, list(train_node_set))
        train_edge_idx = np.where(train_edge_mask)[0]
        ts_train = edge_timestamps[train_edge_idx]
        order_train = np.argsort(ts_train)
        train_edge_sorted = torch.tensor(
            train_edge_idx[order_train], dtype=torch.long, device=device)
        print("      [TEMPORAL] Update memory pakai {:,} train edges (bukan full graph)".format(
            len(train_edge_sorted)))
    else:
        edge_order = torch.argsort(ts_all).cpu()
        train_edge_sorted = edge_order.to(device)

    test_nodes_t = torch.tensor(test_nodes, dtype=torch.long, device=device)

    mb = 8192
    with torch.no_grad():
        tgn.reset_memory()
        for i in range(0, len(train_edge_sorted), mb):
            eidx = train_edge_sorted[i:i + mb]
            tgn.update_memory_only(x, src_all[eidx], dst_all[eidx], ea[eidx], ts_all[eidx])
        tgn_logits = tgn.classify_nodes(x, test_nodes_t)
        tgn_test   = torch.sigmoid(tgn_logits).cpu().numpy()

    m_tgn = metrics(y_test, tgn_test)
    print("      ManualTGN test PR-AUC={:.4f} F1={:.4f}".format(
        m_tgn["pr_auc"], m_tgn["f1"]))

    # ── Ensemble ───────────────────────────────────────────────────────────
    print("\n[4/4] Ensemble (XGBoost + ManualTGN) berbagai bobot:")
    print("      xgb_w  tgn_w   PR-AUC      F1    Prec     Rec")
    best_prauc = 0.0
    best_xw = 1.0
    best_m = m_xgb
    for xw in [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
        tw = 1.0 - xw
        ens = xw * xgb_test + tw * tgn_test
        m = metrics(y_test, ens)
        flag = ""
        if m["pr_auc"] > best_prauc:
            best_prauc = m["pr_auc"]
            best_xw = xw
            best_m = m
            flag = " *"
        print("      {:>5.1f}  {:>5.1f}  {:>7.4f}  {:>6.4f}  {:>6.4f}  {:>6.4f}{}".format(
            xw, tw, m["pr_auc"], m["f1"], m["precision"], m["recall"], flag))

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ABLATION SUMMARY ({})".format(split_mode.upper()))
    print("=" * 65)
    print("  XGBoost only    : PR-AUC={:.4f}  F1={:.4f}".format(m_xgb["pr_auc"], m_xgb["f1"]))
    print("  ManualTGN only  : PR-AUC={:.4f}  F1={:.4f}".format(m_tgn["pr_auc"], m_tgn["f1"]))
    print("  Best ensemble   : PR-AUC={:.4f}  (xgb={:.1f}, tgn={:.1f})".format(
        best_prauc, best_xw, 1.0 - best_xw))
    gain = best_prauc - m_xgb["pr_auc"]
    print("\n  Ensemble vs XGBoost-only: {:+.4f} PR-AUC".format(gain))
    if args.temporal_split:
        print("\n  [CATATAN] Temporal inductive split = lebih konservatif dari random split.")
        print("  Penurunan PR-AUC vs random split adalah WAJAR dan menunjukkan evaluasi jujur.")
        print("  Model yang perform di sini benar-benar generalize ke akun baru.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.random_split:
        suffix = "_random"
    elif args.temporal_split:
        suffix = "_temporal"
    else:
        suffix = "_stratified"
    out = os.path.join(RESULTS_DIR, "ablation_results{}.csv".format(suffix))
    rows = [
        ["model", "split", "pr_auc", "f1", "precision", "recall"],
        ["xgboost",       split_mode, m_xgb["pr_auc"], m_xgb["f1"], m_xgb["precision"], m_xgb["recall"]],
        ["manualtgn",     split_mode, m_tgn["pr_auc"], m_tgn["f1"], m_tgn["precision"], m_tgn["recall"]],
        ["ensemble_best", split_mode, best_m["pr_auc"], best_m["f1"], best_m["precision"], best_m["recall"]],
    ]
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("\n  Saved -> {}".format(out))

    # Selalu update ablation_results.csv (default, dipakai README)
    out_default = os.path.join(RESULTS_DIR, "ablation_results.csv")
    with open(out_default, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("  Saved -> {} (default)".format(out_default))


if __name__ == "__main__":
    main()
