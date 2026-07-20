"""
Bangun graph coverage LEBIH TINGGI (bukan 31% licit warisan training) khusus
utk SCORING DyGFormer (12-13 Jul) — model dyg_v1.pt TIDAK dilatih ulang di
sini, cuma di-load & dipakai forward-pass/inference thd graph yang lebih
lengkap, supaya lebih banyak akun (idealnya 100%, minimal jauh di atas 92.9%
sekarang) dapat dyg_score asli, bukan NaN karena tak pernah muncul di graph
training yang di-sample.

Kenapa aman TANPA retrain: DyGFormer attention-based, terima graph SEBAGAI
INPUT saat forward-pass (bukan hafalan permanen di bobot per-node) — jadi
model yg SAMA persis (sudah PR-AUC 0.9623 terverifikasi) bisa dikasih graph
yg lebih lengkap tanpa perlu re-training.

Kenapa dijalankan BERTAHAP (bukan langsung coverage_ratio=1.0): RAM laptop
16.9GB, cuma 2.6GB tersedia (sebelum tutup app lain). Struktur graph 100%
edge diperkirakan ~12-14GB. Coba coverage_ratio lebih kecil dulu (default
0.65), kalau sukses & RAM masih ada sisa, naikkan bertahap.

PENTING: TIDAK overwrite npz training asli (transactions_hi_injected_
traindata.npz, dipakai train_tgn.py/train_dyg.py) — simpan ke file TERPISAH
supaya model training reference tetap utuh.

Jalankan (dari folder backend/):
    python build_full_coverage_dyg_scores.py --coverage-ratio 0.65
    # kalau sukses & RAM sisa banyak, coba lagi lebih tinggi:
    python build_full_coverage_dyg_scores.py --coverage-ratio 0.85
"""

import argparse
import gc
import os
import time

import numpy as np
import pandas as pd
import psutil
import torch

from ml.tgn_dataset import load_temporal_dataset_fast, CSV_DEFAULT, FEATURE_CONTRACT_VERSION
from ml.train_dyg import TemporalNeighborIndex, DyGFormerNode  # arsitektur sama persis dgn training

DYG_CKPT = os.path.join(os.path.dirname(__file__), "..", "models", "dyg_v1.pt")


def _mem_report(label):
    vm = psutil.virtual_memory()
    print(f"[MEM] {label}: available={vm.available/1e9:.2f}GB / total={vm.total/1e9:.2f}GB", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coverage-ratio", type=float, default=0.65,
                    help="Dipakai utk sample_licit_ratio DAN eval_sample_ratio sekaligus "
                         "(uniform utk semua window, bukan direbalance) — 1.0 = 100% data.")
    p.add_argument("--csv", default=CSV_DEFAULT)
    p.add_argument("--out-npz", default=None,
                    help="Default: <csv>_fullcov_r<ratio>.npz (TIDAK overwrite npz training)")
    p.add_argument("--out-scores", default=None,
                    help="Default: data/processed/dyg_scores_cache_fullcov_r<ratio>.pkl")
    args = p.parse_args()

    r = args.coverage_ratio
    out_npz = args.out_npz or args.csv.replace(".csv", f"_fullcov_r{r}.npz")
    out_scores = args.out_scores or os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", f"dyg_scores_cache_fullcov_r{r}.pkl"
    )

    _mem_report("sebelum load")
    t0 = time.time()

    if os.path.exists(out_npz):
        print(f"[BUILD] npz coverage r={r} sudah ada, load dari cache: {out_npz}", flush=True)
        npz = np.load(out_npz, allow_pickle=True)
        node_features = npz["node_features"]
        edge_index = npz["edge_index"]
        edge_attr = npz["edge_attr"]
        edge_timestamps = npz["edge_timestamps"]
        node_labels = npz["node_labels"]
        account_to_idx = npz["account_to_idx"].item()
    else:
        print(f"[BUILD] Membangun graph coverage r={r} (sample_licit=eval_sample_ratio={r}) "
              f"dari CSV penuh — TIDAK overwrite npz training asli...", flush=True)
        data = load_temporal_dataset_fast(
            csv_path=args.csv,
            sample_licit_ratio=r,
            sample_illicit_ratio=1.0,
            eval_sample_ratio=r,
            random_seed=42,
        )
        _mem_report("setelah load_temporal_dataset_fast")
        node_features = data["node_features"]
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        edge_timestamps = data["edge_timestamps"]
        node_labels = data["node_labels"]
        account_to_idx = data["account_to_idx"]

        print(f"[BUILD] Nodes={len(account_to_idx):,} Edges={edge_index.shape[1]:,} "
              f"({time.time()-t0:.0f}s) -> saving npz...", flush=True)
        np.savez(
            out_npz,
            node_features=node_features, edge_index=edge_index, edge_attr=edge_attr,
            edge_timestamps=edge_timestamps, node_labels=node_labels,
            account_to_idx=account_to_idx,
            feature_contract_version=FEATURE_CONTRACT_VERSION,
            coverage_ratio=r,
        )
        del data
        gc.collect()
        _mem_report("setelah save npz")

    num_nodes = node_features.shape[0]
    print(f"[BUILD] Total akun ter-cover: {num_nodes:,}", flush=True)

    # --- Inference pakai dyg_v1.pt yang SUDAH dilatih (TIDAK diubah/retrain) ---
    print("[SCORE] Load checkpoint dyg_v1.pt (frozen, no retrain)...", flush=True)
    ckpt = torch.load(DYG_CKPT, map_location="cpu", weights_only=False)
    meta = ckpt.get("metadata", {})
    d_model       = meta.get("hidden_dim", 128)
    n_heads       = meta.get("n_heads", 4)
    n_layers      = meta.get("n_layers", 2)
    k_neighbors   = meta.get("k_neighbors", 10)
    node_feat_dim = node_features.shape[1]
    edge_feat_dim = edge_attr.shape[1] if edge_attr.ndim > 1 else 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SCORE] Device: {device}", flush=True)

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
    print(f"[SCORE] Checkpoint test_metrics asli (referensi, TIDAK berubah): "
          f"{meta.get('test_metrics')}", flush=True)

    print(f"[SCORE] Building TemporalNeighborIndex ({edge_index.shape[1]:,} edges)...", flush=True)
    t_idx = time.time()
    nbr_index = TemporalNeighborIndex(edge_index, edge_timestamps, num_nodes)
    print(f"[SCORE] Index built in {time.time()-t_idx:.1f}s", flush=True)
    _mem_report("setelah build neighbor index")

    x = torch.tensor(node_features, dtype=torch.float32, device=device)
    edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    cutoff_ts = float(edge_timestamps.max())  # scoring: pakai semua histori yg ada (bukan train-only)

    batch_size = 2048  # sama persis dgn ensemble.py::_compute_dyg_scores (proven stabil)
    all_scores = np.zeros(num_nodes, dtype=np.float32)
    t_score = time.time()
    print(f"[SCORE] Classifying {num_nodes:,} nodes (batch={batch_size})...", flush=True)
    with torch.no_grad():
        for start in range(0, num_nodes, batch_size):
            end = min(start + batch_size, num_nodes)
            batch_np = np.arange(start, end)
            nbr_ids_np, nbr_dts_np, nbr_eidx_np, mask_np = nbr_index.get_k_recent(
                batch_np, k=k_neighbors, cutoff_ts=cutoff_ts
            )
            # mask_np: (N, K) True=valid -> flip ke True=padded, prepend False utk target
            nbr_pad = ~mask_np
            target_col = np.zeros((len(batch_np), 1), dtype=bool)
            key_pad = np.concatenate([target_col, nbr_pad], axis=1)  # (N, K+1)

            tgt       = torch.tensor(batch_np,    dtype=torch.long,    device=device)
            nbr_ids   = torch.tensor(nbr_ids_np,  dtype=torch.long,    device=device)
            nbr_dts   = torch.tensor(nbr_dts_np,  dtype=torch.float32, device=device)
            nbr_eidx  = torch.tensor(nbr_eidx_np, dtype=torch.long,    device=device)
            key_pad_t = torch.tensor(key_pad,     dtype=torch.bool,    device=device)

            logits = model(x, tgt, nbr_ids, nbr_dts, nbr_eidx, edge_attr_t, key_pad_t)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
            all_scores[start:end] = torch.sigmoid(logits).cpu().numpy()

            if (start // batch_size) % 50 == 0:
                elapsed = time.time() - t_score
                print(f"[SCORE] {end:,}/{num_nodes:,} ({elapsed:.0f}s)", flush=True)

    idx_to_account = {v: k for k, v in account_to_idx.items()}
    accounts = [idx_to_account[i] for i in range(num_nodes)]
    df_out = pd.DataFrame({"account_id": accounts, "dyg_score": all_scores})
    df_out.to_pickle(out_scores)
    print(f"[DONE] {num_nodes:,} akun ter-skor -> {out_scores} "
          f"(total {time.time()-t0:.0f}s)", flush=True)
    print(f"[DONE] Bandingkan coverage: sebelumnya 1,981,734 akun (31% licit) -> "
          f"sekarang {num_nodes:,} akun (coverage_ratio={r})", flush=True)


if __name__ == "__main__":
    main()
