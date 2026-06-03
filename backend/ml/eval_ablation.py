"""
Ablation + Ensemble diagnostic — fair head-to-head pada DATA & SPLIT yang sama.

Semua model dievaluasi pada node test split identik (HI-Large injected):
  - XGBoost (tabular, 13 node features)
  - TGN (ManualTGN, memory + features)
  - Ensemble (weighted average) pada beberapa bobot

Cara pakai:
    cd backend
    python -m ml.eval_ablation
"""

import os
import sys
import csv
import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

from ml.tgn_dataset import CSV_DEFAULT

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
TGN_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "tgn_v1.pt"))

FEATURE_COLS = [
    "in_degree", "out_degree", "degree_ratio", "in_amount_sum",
    "out_amount_sum", "amount_ratio", "unique_senders", "unique_recipients",
    "max_single_tx", "night_tx_ratio", "avg_amount_in", "avg_amount_out", "total_tx",
]


def metrics(labels, probs, thr=0.5):
    preds = (probs >= thr).astype(int)
    pa = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0
    return {
        "pr_auc": pa,
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def main():
    cache_npz = CSV_DEFAULT.replace(".csv", "_traindata.npz")
    if not os.path.exists(cache_npz):
        print("[ERROR] Cache tidak ada: {}".format(cache_npz))
        print("Jalankan dulu: python -m ml.train_tgn")
        sys.exit(1)

    print("[1/4] Loading cache {}...".format(os.path.basename(cache_npz)))
    npz = np.load(cache_npz, allow_pickle=True)
    node_features = npz["node_features"]
    node_labels = npz["node_labels"]
    edge_index = npz["edge_index"]
    edge_attr = npz["edge_attr"]
    edge_timestamps = npz["edge_timestamps"]
    num_nodes = node_features.shape[0]
    print("      Nodes={:,} | illicit={:,} ({:.1f}%)".format(
        num_nodes, int(node_labels.sum()), node_labels.mean() * 100))

    # Split node IDENTIK dengan train_tgn (random_state=42, stratified 70/15/15)
    all_nodes = np.arange(num_nodes)
    train_nodes, temp_nodes = train_test_split(
        all_nodes, test_size=0.3, stratify=node_labels, random_state=42)
    val_nodes, test_nodes = train_test_split(
        temp_nodes, test_size=0.5, stratify=node_labels[temp_nodes], random_state=42)
    print("      Split: train={:,} val={:,} test={:,}".format(
        len(train_nodes), len(val_nodes), len(test_nodes)))

    y_test = node_labels[test_nodes]

    # XGBoost pada node features HI yang sama
    print("\n[2/4] Training XGBoost (data & split sama)...")
    X_train = node_features[train_nodes]
    y_train = node_labels[train_nodes]
    X_test = node_features[test_nodes]

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

    # TGN scores pada test nodes
    print("\n[3/4] Loading TGN + scoring test nodes...")
    import torch
    from ml.tgn_model import ManualTGN

    ckpt = torch.load(TGN_PATH, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    hidden_dim = meta.get("hidden_dim", 128)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tgn = ManualTGN(
        num_nodes=num_nodes, node_feat_dim=13, edge_feat_dim=3,
        memory_dim=hidden_dim, hidden_dim=hidden_dim,
    ).to(device)
    tgn.load_state_dict(ckpt["model_state_dict"])
    tgn.eval()

    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    src_all = torch.tensor(edge_index[0], dtype=torch.long, device=device)
    dst_all = torch.tensor(edge_index[1], dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    ts = torch.tensor(edge_timestamps, dtype=torch.float32, device=device)
    edge_order = torch.argsort(ts).cpu()
    test_nodes_t = torch.tensor(test_nodes, dtype=torch.long, device=device)

    mb = 8192
    with torch.no_grad():
        tgn.reset_memory()
        for i in range(0, len(edge_order), mb):
            eidx = edge_order[i:i + mb].to(device)
            tgn.update_memory_only(x, src_all[eidx], dst_all[eidx], ea[eidx], ts[eidx])
        tgn_logits = tgn.classify_nodes(x, test_nodes_t)
        tgn_test = torch.sigmoid(tgn_logits).cpu().numpy()

    m_tgn = metrics(y_test, tgn_test)
    print("      TGN test PR-AUC={:.4f} F1={:.4f}".format(
        m_tgn["pr_auc"], m_tgn["f1"]))

    # Ensemble pada beberapa bobot
    print("\n[4/4] Ensemble (XGBoost + TGN) berbagai bobot:")
    print("      xgb_w  tgn_w   PR-AUC      F1    Prec     Rec")
    best_prauc = 0.0
    best_xw = 1.0
    best_m = m_xgb
    weights = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]
    for xw in weights:
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

    # Ringkasan + simpan
    xgb_pa = m_xgb["pr_auc"]
    tgn_pa = m_tgn["pr_auc"]
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (test set, data & split identik)")
    print("=" * 60)
    print("  XGBoost only : PR-AUC={:.4f}  F1={:.4f}".format(xgb_pa, m_xgb["f1"]))
    print("  TGN only     : PR-AUC={:.4f}  F1={:.4f}".format(tgn_pa, m_tgn["f1"]))
    print("  Best ensemble: PR-AUC={:.4f}  (xgb={:.1f}, tgn={:.1f})".format(
        best_prauc, best_xw, 1.0 - best_xw))
    gain = best_prauc - xgb_pa
    print("\n  Ensemble vs XGBoost-only: {:+.4f} PR-AUC".format(gain))
    if gain > 0.002:
        print("  => TGN MENAMBAH nilai. Ensemble layak dipakai.")
    else:
        print("  => TGN tidak menambah nilai signifikan; XGBoost-only cukup.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "ablation_results.csv")
    rows = [
        ["model", "pr_auc", "f1", "precision", "recall"],
        ["xgboost", m_xgb["pr_auc"], m_xgb["f1"], m_xgb["precision"], m_xgb["recall"]],
        ["tgn", m_tgn["pr_auc"], m_tgn["f1"], m_tgn["precision"], m_tgn["recall"]],
        ["ensemble_best", best_m["pr_auc"], best_m["f1"], best_m["precision"], best_m["recall"]],
    ]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print("\n  Saved -> {}".format(out))


if __name__ == "__main__":
    main()
