"""
Validasi P4.8 (20-Jul): apakah TGN-streaming FLAG kolektor illicit nyata di
JALUR PRODUKSI ASLI (feature_store fitur LIVE, bukan npz training)?

Ini menguji sekaligus:
  - Feature train/serve skew: pakai fitur feature_store real-time (sebagian
    approx) — bukan fitur npz. Kalau TGN tetap flag -> skew tak fatal.
  - End-to-end scoring produksi: replay incoming (bangun fan_in + memory TGN
    kolektor sbg dst) lalu outgoing (kolektor jadi src -> DISKOR).
  - Bandingkan TGN vs XGBoost di akun SAMA.

Cara pakai: cd backend && python -m ml.validate_tgn_production_path
"""

import os
import sys

import numpy as np
from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streaming"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_scorer import RealtimeScorer
from feature_store import FEATURE_COLS

DATABASE_URL = os.getenv("DATABASE_URL",
                         "postgresql://muleradar:muleradar_secret@localhost:5432/muleradar")
COLLECTOR = os.getenv("VALIDATE_COLLECTOR", "QRIS-PT-14")
MAX_IN = int(os.getenv("VALIDATE_MAX_IN", "500"))
MAX_OUT = int(os.getenv("VALIDATE_MAX_OUT", "20"))


def _fetch(engine, where, params, limit):
    q = text(f"""
        SELECT tx_id, from_account, to_account, amount, channel, tx_timestamp,
               device_id, is_laundering, typology
        FROM transactions WHERE {where}
        ORDER BY tx_timestamp LIMIT :lim
    """)
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, {**params, "lim": limit}).mappings().all()]


def _clear_state(store, accounts):
    for a in accounts:
        for suf in ("", ":tgnmem", ":tgnlast", ":dev", ":in_cp", ":out_cp",
                    ":txwin", ":inwin", ":chan"):
            store.r.delete(f"acct:{a}{suf}")


def run_mode(mode, engine, incoming, outgoing):
    sc = RealtimeScorer(mode=mode)
    # bersihkan state semua akun terlibat
    accts = {COLLECTOR}
    for t in incoming + outgoing:
        accts.add(str(t["from_account"])); accts.add(str(t["to_account"]))
    _clear_state(sc.store, accts)

    # replay incoming (bangun fan_in + memory kolektor sbg dst)
    for t in incoming:
        sc.score(_norm(t), apply_update=True)
    # replay outgoing (kolektor jadi src -> DISKOR) — ambil skor tertinggi
    best = None
    for t in outgoing:
        r = sc.score(_norm(t), apply_update=True)
        if best is None or r["risk_score"] > best["risk_score"]:
            best = r
    # cek ketersediaan 4 fitur graph (butuh cache refresh_graph_cache)
    feat = sc.store.get_model_features(COLLECTOR)
    gi = {c: feat[FEATURE_COLS.index(c)] for c in
          ("pagerank", "kcore_number", "device_sharing_count", "n_institutions")
          if c in FEATURE_COLS}
    return best, gi


def _norm(t):
    return {"from_account": str(t["from_account"]), "to_account": str(t["to_account"]),
            "amount": float(t["amount"] or 0), "channel": t.get("channel") or "internet",
            "tx_timestamp": str(t["tx_timestamp"]), "device_id": t.get("device_id") or "",
            "typology": t.get("typology"), "tx_id": t.get("tx_id")}


def main():
    engine = create_engine(DATABASE_URL)
    print(f"[1/3] Ambil transaksi kolektor {COLLECTOR} (in<= {MAX_IN}, out<= {MAX_OUT})...")
    incoming = _fetch(engine, "to_account = :acc", {"acc": COLLECTOR}, MAX_IN)
    outgoing = _fetch(engine, "from_account = :acc", {"acc": COLLECTOR}, MAX_OUT)
    print(f"      incoming={len(incoming)} outgoing={len(outgoing)}")
    if not outgoing:
        print("      kolektor tak punya transaksi keluar -> tak bisa diskor. Ganti VALIDATE_COLLECTOR.")
        return

    print("[2/3] Skor jalur produksi mode TGN...")
    tgn_best, tgn_graph = run_mode("tgn", engine, incoming, outgoing)
    print("[3/3] Skor jalur produksi mode XGBoost (baseline)...")
    xgb_best, xgb_graph = run_mode("xgb", engine, incoming, outgoing)

    print("\n=== HASIL (kolektor illicit nyata, fitur feature_store LIVE) ===")
    print(f"TGN  : base_score={tgn_best['base_score']:.4f} risk={tgn_best['risk_score']:.4f} "
          f"level={tgn_best['risk_level']} decision={tgn_best['decision']}")
    print(f"XGB  : base_score={xgb_best['base_score']:.4f} risk={xgb_best['risk_score']:.4f} "
          f"level={xgb_best['risk_level']} decision={xgb_best['decision']}")
    print(f"\nFitur graph (butuh cache refresh_graph_cache) di feature_store live:")
    for k, v in tgn_graph.items():
        print(f"  {k}: {v}  {'(0 -> cache belum populate akun ini = feature skew)' if v==0 else ''}")
    print(f"\nsinyal TGN: {tgn_best['reasons']}")
    flagged_tgn = tgn_best["decision"] != "NONE"
    print(f"\n-> TGN {'FLAG' if flagged_tgn else 'TIDAK flag'} kolektor illicit ini "
          f"({'base model tinggi' if tgn_best['base_score']>0.5 else 'base model rendah, tergantung sinyal'})")


if __name__ == "__main__":
    main()
