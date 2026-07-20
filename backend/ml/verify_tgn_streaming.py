"""
Verifikasi P4.8 (20-Jul): apakah math TGN-streaming (tgn_streaming_scorer)
FAITHFUL thd jalur batch reference yg sudah proven (ensemble._compute_tgn_scores,
PR-AUC test 0,9930 di cache)?

Strategi: replay SAMPEL edge dari npz training (fitur SUDAH ter-scale exact,
edge_attr exact) lewat DUA jalur, bandingkan skor node:
  (A) REFERENCE: ManualTGN.update_memory_only (batch=1) + classify_nodes
      — persis yg dipakai _compute_tgn_scores yg hasilkan cache 0,9930.
  (B) MINE: replikasi math score_tx (msg_mlp -> GRU src&dst -> node_classifier)
      pakai modul model yg SAMA, memory dikelola manual (dict, meniru Redis).

Kalau (A) == (B) (korelasi ~1, max-diff ~0) -> transkripsi streaming saya
faithful. Digabung fakta cache batch = 0,9930 -> jalur streaming reproduksi
akurasi TGN (modulo: fitur live feature_store vs training, & serialisasi
float32 Redis — residual terdokumentasi).

Cara pakai: cd backend && python -m ml.verify_tgn_streaming
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.tgn_model import ManualTGN

_DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
_MODELS = os.path.join(os.path.dirname(__file__), "..", "..", "models")
NPZ = os.path.join(_DATA, "transactions_hi_injected_traindata.npz")
CKPT = os.path.join(_MODELS, "tgn_v1.pt")
N_EDGES_SAMPLE = int(os.getenv("VERIFY_EDGES", "8000"))
MEM_DIM = 256


def _load_model(num_nodes):
    """Load ManualTGN dgn cara SAMA dgn scorer saya (strict=False minus memory)."""
    m = ManualTGN(num_nodes=num_nodes, node_feat_dim=24, edge_feat_dim=3,
                  memory_dim=MEM_DIM, hidden_dim=MEM_DIM, dropout=0.0)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd_no_mem = {k: v for k, v in sd.items() if k not in ("memory", "last_update")}
    m.load_state_dict(sd_no_mem, strict=False)
    m.eval()
    return m


def main():
    print(f"[1/4] Load npz + sampel {N_EDGES_SAMPLE:,} edge temporal-awal...")
    npz = np.load(NPZ, allow_pickle=True)
    nf = npz["node_features"][:, :24].astype(np.float32)   # exact-scaled
    ei = npz["edge_index"]; ea = npz["edge_attr"].astype(np.float32)
    ts = npz["edge_timestamps"].astype(np.float64)

    order = np.argsort(ts)[:N_EDGES_SAMPLE]
    src = ei[0][order]; dst = ei[1][order]
    eaa = ea[order]; tss = ts[order]

    # remap node yg terlibat -> index kecil kontigu (biar buffer reference kecil)
    involved = np.unique(np.concatenate([src, dst]))
    remap = {int(o): i for i, o in enumerate(involved)}
    n_small = len(involved)
    nf_small = nf[involved]                          # (n_small, 24)
    src_s = np.array([remap[int(s)] for s in src], dtype=np.int64)
    dst_s = np.array([remap[int(d)] for d in dst], dtype=np.int64)
    print(f"      {n_small:,} node unik terlibat")

    # ── (A) REFERENCE: ManualTGN update_memory_only batch=1 + classify ──
    print("[2/4] REFERENCE replay (ManualTGN, batch=1)...")
    ref = _load_model(n_small)
    x = torch.tensor(nf_small, dtype=torch.float32)
    with torch.no_grad():
        ref.reset_memory()
        for i in range(len(src_s)):
            ref.update_memory_only(
                x,
                torch.tensor([src_s[i]]), torch.tensor([dst_s[i]]),
                torch.tensor(eaa[i:i+1], dtype=torch.float32),
                torch.tensor(tss[i:i+1], dtype=torch.float32),
            )
        ref_logits = ref.classify_nodes(x, torch.arange(n_small))
        ref_scores = torch.sigmoid(ref_logits).numpy()

    # ── (B) MINE: replikasi math score_tx, memory di dict (meniru Redis) ──
    print("[3/4] MINE replay (transkripsi score_tx, memory dict)...")
    m = _load_model(2)  # num_nodes tak relevan (buffer tak dipakai)
    mem = {}    # node -> np.float32(256)
    last = {}   # node -> float ts
    def gmem(n): return mem.get(n, np.zeros(MEM_DIM, dtype=np.float32))
    with torch.no_grad():
        for i in range(len(src_s)):
            s = int(src_s[i]); d = int(dst_s[i])
            mem_s = torch.tensor(gmem(s)).unsqueeze(0)
            mem_d = torch.tensor(gmem(d)).unsqueeze(0)
            nf_s = torch.tensor(nf_small[s]).unsqueeze(0)
            ea_t = torch.tensor(eaa[i]).unsqueeze(0)
            dt = np.float32(np.log1p(max(float(tss[i]) - last.get(s, 0.0), 0.0)) / 20.0)
            dt_t = torch.tensor([[dt]], dtype=torch.float32)
            msg = m.msg_mlp(torch.cat([mem_s, mem_d, nf_s, ea_t, dt_t], dim=-1))
            new_s = m.memory_updater(msg, mem_s)
            new_d = m.memory_updater(msg, mem_d)
            # serialisasi float32 via bytes (meniru Redis round-trip)
            mem[s] = np.frombuffer(new_s.squeeze(0).numpy().astype(np.float32).tobytes(), dtype=np.float32).copy()
            mem[d] = np.frombuffer(new_d.squeeze(0).numpy().astype(np.float32).tobytes(), dtype=np.float32).copy()
            last[s] = float(tss[i]); last[d] = float(tss[i])
        # classify semua node dari memory final
        my_scores = np.zeros(n_small, dtype=np.float32)
        for n in range(n_small):
            mm = torch.tensor(gmem(n)).unsqueeze(0)
            nf_n = torch.tensor(nf_small[n]).unsqueeze(0)
            logit = m.node_classifier(torch.cat([mm, nf_n], dim=-1))
            my_scores[n] = torch.sigmoid(logit).item()

    # ── (4) Bandingkan ──
    print("[4/4] Bandingkan REFERENCE vs MINE...")
    diff = np.abs(ref_scores - my_scores)
    corr = np.corrcoef(ref_scores, my_scores)[0, 1]
    print(f"      node dibandingkan : {n_small:,}")
    print(f"      korelasi          : {corr:.6f}  (target ~1.0)")
    print(f"      max abs diff      : {diff.max():.2e}")
    print(f"      mean abs diff     : {diff.mean():.2e}")
    ok = corr > 0.9999 and diff.max() < 1e-3
    print(f"\n      {'LULUS' if ok else 'PERLU DICEK'} — transkripsi streaming "
          f"{'FAITHFUL thd batch reference' if ok else 'ADA DIVERGENSI'}")


if __name__ == "__main__":
    main()
