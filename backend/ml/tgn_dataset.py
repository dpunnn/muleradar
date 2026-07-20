"""
TGN Dataset: load transaksi dari CSV -> PyG-compatible dict.
Support chunked loading untuk file besar (179M+ rows).

Cara pakai:
    from ml.tgn_dataset import load_temporal_dataset
    data = load_temporal_dataset(max_rows=5_000_000)
"""

import os
import sys
import math
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Backend dir on path agar bisa import feature_defs (kanonik) meski dipanggil
# dari konteks berbeda (mis. ml.ensemble).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_defs import counterparty_hhi as _canon_hhi   # fix HHI 3-Jul
from feature_defs import FEATURE_COLS   # definisi kanonik (fix duplikasi 6-Jul)

_STRUCTURING_BANDS = [
    (9_500, 10_000), (95_000, 100_000), (950_000, 1_000_000),
    (9_500_000, 10_000_000), (95_000_000, 100_000_000), (450_000_000, 500_000_000),
]

logger = logging.getLogger(__name__)


def temporal_inductive_split(edge_index, edge_timestamps, num_nodes,
                              train_ratio=0.70, val_ratio=0.15):
    """
    Temporal inductive node split berbasis first-appearance timestamp.

    Untuk setiap node, cari timestamp PERTAMA kali node tersebut muncul
    (sebagai sender atau receiver). Lalu urutkan semua node berdasarkan
    waktu pertama muncul.

    - Train : node paling awal muncul (70% pertama)
    - Val   : node muncul di rentang 70-85%
    - Test  : node paling baru muncul (15% terakhir) — truly unseen at train time

    Tidak ada edge yang menyeberang train↔test karena node test SELURUHNYA
    baru muncul setelah cutoff waktu train selesai.

    Dipindah ke sini (5-Jul) dari ml/eval_ablation.py supaya jadi SATU sumber
    kebenaran, dipakai konsisten oleh eval_ablation.py DAN train_tgn.py
    (ManualTGN) — lihat catatan design intent di train_dyg.py: ManualTGN
    (full memory state) cocok pakai temporal split, DyGFormer (lokal K-hop,
    tanpa memory persisten) TETAP pakai stratified (PR-AUC kolaps ke 0.07
    kalau dipaksa temporal — JANGAN diubah).
    """
    src, dst = edge_index[0], edge_index[1]

    # Vectorized: cari first timestamp per node (numpy minimum.at)
    node_first_ts = np.full(num_nodes, np.inf, dtype=np.float64)
    np.minimum.at(node_first_ts, src, edge_timestamps)
    np.minimum.at(node_first_ts, dst, edge_timestamps)

    # Node tanpa edge (isolated) → assign ke training (timestamp max → masuk train)
    no_edge_mask = node_first_ts == np.inf
    node_first_ts[no_edge_mask] = edge_timestamps.min()

    # Urutkan node berdasarkan first appearance
    node_order = np.argsort(node_first_ts)  # ascending: paling lama → paling baru
    n = len(node_order)

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_nodes = node_order[:n_train]
    val_nodes   = node_order[n_train: n_train + n_val]
    test_nodes  = node_order[n_train + n_val:]

    t_train_end = node_first_ts[node_order[n_train - 1]]
    t_val_end   = node_first_ts[node_order[n_train + n_val - 1]]
    t_test_end  = node_first_ts[node_order[-1]]

    print("      Temporal cutoff:")
    print("        Train ends : {:.0f} (Unix ts)".format(t_train_end))
    print("        Val ends   : {:.0f} (Unix ts)".format(t_val_end))
    print("        Test ends  : {:.0f} (Unix ts)".format(t_test_end))
    print("      [INDUCTIVE] Test nodes = akun yang BARU muncul setelah train period selesai")

    return train_nodes, val_nodes, test_nodes

CSV_DEFAULT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                 "transactions_hi_injected.csv")
)

CHANNEL_MAP = {"mobile": 0, "atm": 1, "internet": 2, "teller": 3, "qris": 4}

# --- Penyelarasan mata uang: AMLWorld (IBM) multi-currency -> semua ke IDR ---
# Kurs referensi tetap (proxy) agar nominal sebanding lintas-rekening & konsisten
# dengan konteks Rupiah. Typologi Indonesia hasil injeksi sudah "IDR" (faktor 1.0).
CURRENCY_TO_IDR = {
    "IDR": 1.0,
    "US Dollar": 16000.0, "Euro": 17500.0, "UK Pound": 20500.0,
    "Swiss Franc": 18000.0, "Canadian Dollar": 11800.0, "Australian Dollar": 10700.0,
    "Yuan": 2250.0, "Yen": 105.0, "Rupee": 190.0, "Ruble": 175.0,
    "Brazil Real": 2900.0, "Saudi Riyal": 4270.0, "Mexican Peso": 800.0,
    "Shekel": 4400.0, "Bitcoin": 1_000_000_000.0,
}


def harmonize_amount_to_idr(chunk: pd.DataFrame) -> pd.DataFrame:
    """Konversi kolom amount ke IDR berdasarkan currency (in-place, mengembalikan chunk).
    Mata uang tak dikenal / kosong dianggap sudah IDR (faktor 1.0)."""
    if "currency" in chunk.columns:
        rate = chunk["currency"].map(CURRENCY_TO_IDR).fillna(1.0)
        chunk["amount"] = chunk["amount"].astype(float) * rate.to_numpy()
    return chunk


def _merge_running(running, batch_dfs, sum_cols, max_cols):
    """Merge batch of chunk-level agg DataFrames into running totals."""
    batch = pd.concat(batch_dfs).groupby(level=0).agg(
        {c: "sum" for c in sum_cols} | {c: "max" for c in max_cols}
    )
    if running is None:
        return batch
    # sum cols: add with fill_value=0 untuk akun yang baru muncul
    result = running.reindex(running.index.union(batch.index), fill_value=0)
    batch_r = batch.reindex(result.index, fill_value=0)
    for c in sum_cols:
        result[c] += batch_r[c]
    for c in max_cols:
        result[c] = np.maximum(result[c].values, batch_r[c].values)
    return result


def _merge_channel(running, batch_dfs):
    """Merge channel-count batch (account_id, channel, cnt) — sum-mergeable
    (fix 6-Jul, roadmap #1: dasar utk channel_entropy sungguhan)."""
    batch = (pd.concat(batch_dfs, ignore_index=True)
             .groupby(["account_id", "channel"], sort=False)["cnt"].sum().reset_index())
    if running is None:
        return batch
    combined = pd.concat([running, batch], ignore_index=True)
    return combined.groupby(["account_id", "channel"], sort=False)["cnt"].sum().reset_index()


def _merge_pairs(running, batch_dfs):
    """Merge batch pasangan (kolA, kolB) unik — dedup union, dasar counterparty
    ASLI (fix 6-Jul, roadmap #1: ganti approksimasi unique_senders/recipients=degree).

    Fix (7-Jul): batch_dfs bisa KOSONG kalau dipakai utk accumulator yg
    difilter cutoff (device/inst/cpout_causal) — begitu scan lewat cutoff,
    tiap chunk selanjutnya nyumbang 0 baris causal, tapi merge tetap ke-
    trigger by len(in_agg) yg TIDAK difilter. pd.concat([]) crash "No objects
    to concatenate" kalau dibiarkan — guard di sini, bukan larang pemanggil.
    """
    if not batch_dfs:
        return running
    batch = pd.concat(batch_dfs, ignore_index=True).drop_duplicates()
    if running is None:
        return batch
    return pd.concat([running, batch], ignore_index=True).drop_duplicates()


def _merge_topk_ts(running, batch_dfs, k=20):
    """Merge batch (account_id, tx_timestamp), simpan cuma K TERBARU per akun.
    Fix 6-Jul, roadmap #1: dasar burst_ratio/inter_tx_std/dormancy_days
    sungguhan. K=20 (bukan K=500 spt versi Postgres LATERAL) krn di sini
    SEMUA ditahan di RAM (bukan bounded index scan sisi DB) — trade-off
    disengaja demi keamanan memori utk skala 181 juta baris CSV, mengingat
    insiden OOM akun hub 6,3 juta transaksi sebelumnya."""
    parts = list(batch_dfs)
    if running is not None:
        parts.append(running)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values("tx_timestamp", ascending=False)
    topk = combined.groupby("account_id", sort=False).head(k)
    return topk.reset_index(drop=True)


def compute_node_features_full(csv_path: str, chunk_size: int = 1_000_000,
                                topk_ts: int = 20,
                                network_feature_cutoff_ts: float = None) -> pd.DataFrame:
    """
    Full scan CSV untuk compute per-node features yang akurat dari semua 181M rows.
    Return DataFrame dengan index=account_id dan 24 kolom features (13 baseline +
    7 behavioral + 2 network: device_sharing_count, n_institutions — fix 6-Jul roadmap #2
    + 2 graph struktural: pagerank, kcore_number — fix 7-Jul).

    network_feature_cutoff_ts (fix 7-Jul, KRITIS — insiden leak): device_sharing_
    count/n_institutions/pagerank/kcore_number itu fitur LINTAS-AKUN (bergantung
    pada akun LAIN, bukan cuma riwayat akun itu sendiri). Kalau dihitung dari
    SELURUH dataset (default, cutoff=None), akun TEST (baru muncul setelah
    training) diam-diam "mengintip" struktur graf yg baru terbentuk BELAKANGAN
    — termasuk koneksi ring fraud yg sama yg baru kelihatan di periode test.
    Diverifikasi empiris: kcore_number AUC causal cuma 0.439 (di BAWAH acak!)
    vs 0.772 kalau "bocor" dari full graph — SEMUA sinyalnya ternyata palsu.
    Kalau parameter ini diisi (val_cutoff_ts dari split), device/institution/
    pagerank/kcore CUMA dihitung dari edge tx_timestamp <= cutoff (exclude
    SELURUH periode test) — akun test yg genuinely baru akan dapat nilai
    default/lemah utk fitur INI SAJA, itu JUJUR (bukan bug), bukan diam-diam
    dilonggarkan. unique_recipients/unique_senders/counterparty_hhi TETAP dari
    seluruh riwayat AKUN ITU SENDIRI (self-aggregate, bukan lintas-akun — beda
    kategori leak, lebih bisa diterima, tak perlu restriksi sama).
    Hasil di-cache ke .pkl (ber-versi, lihat FEATURE_CONTRACT_VERSION) supaya run
    berikutnya langsung load (~5 detik).

    Fix (6-Jul, audit roadmap #1): sebelumnya burst_ratio/inter_tx_std/
    dormancy_days/channel_entropy di-hardcode 0.0 ("tidak bisa dihitung dari
    aggregated scan") — terverifikasi AUC single-feature-nya persis 0.500
    (mati total, 20% fitur terbuang). unique_senders/unique_recipients juga
    cuma approksimasi via degree (bukan nunique asli), merusak counterparty_hhi.
    Sekarang keempatnya dihitung SUNGGUHAN:
      - channel_entropy   : channel count per akun (sum-mergeable, sama pola
                             dgn in/out_agg) + reuse _compute_channel_entropy
                             dari detection/features.py (definisi kanonik).
      - burst/inter_tx_std/dormancy : top-K=20 timestamp TERBARU per akun
                             (bounded, di-merge inkremental spy tak OOM di
                             akun hub — ingat insiden 6,3 juta transaksi/akun)
                             + reuse _compute_temporal_features (kanonik).
      - unique_senders/recipients   : true nunique via dedup pair tracking,
                             bukan degree lagi.
    K=20 (bukan K=500 spt versi Postgres LATERAL) krn di sini semua ditahan
    di RAM (bukan bounded index scan sisi DB) — trade-off disengaja demi
    keamanan memori. Precision lebih rendah dari XGBoost/Postgres-path utk
    akun berdegree sangat tinggi, tapi tak lagi NOL seperti sebelumnya.
    """
    from detection.features import (
        _compute_temporal_features, _compute_channel_entropy,
        _compute_device_sharing, _compute_institution_diversity,
        _compute_graph_structural,
    )

    # Fix (7-Jul, KRITIS): cache SEBELUMNYA cuma cek FEATURE_CONTRACT_VERSION —
    # TIDAK cek network_feature_cutoff_ts. Insiden nyata: pkl sempat kesimpan
    # dari test PARSIAL (cutoff dari max_rows=60 juta, bukan dataset penuh) —
    # kalau tak divalidasi, run TRAINING SUNGGUHAN (cutoff beda, dataset
    # penuh) bisa diam-diam pakai cache yg SALAH SCOPE ini (versi cocok,
    # cutoff-nya yg beda). Sama pola dgn insiden sampling_sig train_tgn.py.
    cache_path = csv_path.replace(".csv", "_node_features.pkl")
    if os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        cached_cutoff = cached.get("network_feature_cutoff_ts") if isinstance(cached, dict) else None
        if (isinstance(cached, dict) and cached.get("version") == FEATURE_CONTRACT_VERSION
                and cached_cutoff == network_feature_cutoff_ts):
            print(f"  [node-feat] Loading from cache: {os.path.basename(cache_path)} "
                  f"(version={FEATURE_CONTRACT_VERSION}, cutoff={network_feature_cutoff_ts})")
            return cached["df"]
        print(f"  [node-feat] Cache IGNORED — versi/cutoff beda "
              f"(cached_cutoff={cached_cutoff} vs {network_feature_cutoff_ts}), regenerasi otomatis.")

    _IN_SUM  = ["in_degree", "in_amount_sum", "night_in", "illicit_in"]
    _IN_MAX  = ["in_max"]
    _OUT_SUM = ["out_degree", "out_amount_sum", "night_out", "illicit_out",
                "structuring_count", "round_tx_count"]
    _OUT_MAX = ["out_max"]
    MERGE_EVERY = 20     # merge sum/max setiap 20 chunk (murah)
    TS_MERGE_EVERY = 3   # merge top-K timestamp lebih sering (lebih berat per-merge);
                         # K=20 + merge tiap 3 chunk dipilih konservatif setelah cek
                         # RAM tersedia cuma ~5.8GB saat fix ini dibuat (6-Jul)

    in_agg = []
    out_agg = []
    in_running = None
    out_running = None

    chan_agg = []
    chan_running = None

    cpout_agg = []    # (from_account, to_account) unik — dasar unique_recipients
    cpout_running = None
    cpin_agg = []     # (to_account, from_account) unik — dasar unique_senders
    cpin_running = None

    ts_agg = []       # (account_id, tx_timestamp) long-format — dasar temporal features
    ts_running = None

    device_agg = []   # (account_id=from_account, device_id) unik — dasar device_sharing_count
    device_running = None
    inst_agg = []     # (account_id, institution_id) unik, in+out — dasar n_institutions
    inst_running = None

    # Fix (7-Jul, KRITIS): pasangan (from,to) KHUSUS utk pagerank/kcore, di-
    # filter cutoff (BEDA dari cpout_agg/cpout_running di atas yg TETAP penuh
    # riwayat — itu basis unique_recipients, self-aggregate, bukan lintas-akun,
    # tak butuh restriksi). device_agg/inst_agg di atas JUGA akan difilter
    # cutoff yg sama (lihat blok if di bawah).
    cpout_causal_agg = []
    cpout_causal_running = None

    rows_read = 0

    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype={"from_account": str, "to_account": str,
               "amount": float, "is_laundering": float,
               "typology": str, "channel": str,
               "device_id": str, "institution_id": str},
        parse_dates=["tx_timestamp"],
        low_memory=False,
    )

    _network_cutoff_dt = (pd.Timestamp(network_feature_cutoff_ts, unit="s")
                           if network_feature_cutoff_ts is not None else None)

    for ci, chunk in enumerate(reader, start=1):
        chunk = harmonize_amount_to_idr(chunk)
        chunk["is_laundering"] = chunk["is_laundering"].fillna(0).astype(int)
        chunk["tx_timestamp"] = pd.to_datetime(chunk["tx_timestamp"], errors="coerce")
        chunk["is_night"] = chunk["tx_timestamp"].dt.hour.isin([22, 23, 0, 1, 2, 3]).astype(int)
        chunk["is_round"] = (
            (chunk["amount"] > 0) &
            (chunk["amount"].apply(lambda x: abs(x - round(x / 10000) * 10000) < 1.0))
        ).astype(int)
        chunk["is_struct"] = chunk["amount"].apply(
            lambda a: 1 if any(lo <= a < hi for lo, hi in _STRUCTURING_BANDS) else 0
        )

        ib = chunk.groupby("to_account", sort=False).agg(
            in_degree=("amount", "count"),
            in_amount_sum=("amount", "sum"),
            in_max=("amount", "max"),
            night_in=("is_night", "sum"),
            illicit_in=("is_laundering", "sum"),
        )
        in_agg.append(ib)

        ob = chunk.groupby("from_account", sort=False).agg(
            out_degree=("amount", "count"),
            out_amount_sum=("amount", "sum"),
            out_max=("amount", "max"),
            night_out=("is_night", "sum"),
            illicit_out=("is_laundering", "sum"),
            structuring_count=("is_struct", "sum"),
            round_tx_count=("is_round", "sum"),
        )
        out_agg.append(ob)

        # --- Channel count per akun (fix 6-Jul: dasar channel_entropy sungguhan) ---
        cb = pd.concat([
            chunk[["from_account", "channel"]].rename(columns={"from_account": "account_id"}),
            chunk[["to_account", "channel"]].rename(columns={"to_account": "account_id"}),
        ], ignore_index=True).groupby(["account_id", "channel"], sort=False).size()
        chan_agg.append(cb.rename("cnt").reset_index())

        # --- Pasangan counterparty unik (fix 6-Jul: true nunique, bukan degree) ---
        cpout_agg.append(chunk[["from_account", "to_account"]].drop_duplicates())
        cpin_agg.append(chunk[["to_account", "from_account"]].drop_duplicates())

        # --- Timestamp per akun (fix 6-Jul: dasar burst/inter_tx_std/dormancy) ---
        ts_agg.append(pd.concat([
            chunk[["from_account", "tx_timestamp"]].rename(columns={"from_account": "account_id"}),
            chunk[["to_account", "tx_timestamp"]].rename(columns={"to_account": "account_id"}),
        ], ignore_index=True))

        # --- Device & institution per akun (fix 6-Jul, audit roadmap #2) ---
        # Fix (7-Jul, KRITIS): fitur INI lintas-akun (bocor kalau lihat masa
        # depan akun LAIN) — kalau cutoff diisi, filter dulu ke edge SEBELUM
        # cutoff (exclude periode test) SEBELUM diakumulasi. device_id cuma
        # bermakna sbg sender (lihat postprocess.py assign_device — di-derive
        # dari from_account saja, to_account tak pernah "pakai" device di sini).
        chunk_causal = (chunk[chunk["tx_timestamp"] <= _network_cutoff_dt]
                        if _network_cutoff_dt is not None else chunk)
        if "device_id" in chunk_causal.columns and len(chunk_causal):
            device_agg.append(
                chunk_causal[["from_account", "device_id"]]
                .rename(columns={"from_account": "account_id"})
                .drop_duplicates()
            )
        if "institution_id" in chunk_causal.columns and len(chunk_causal):
            inst_agg.append(pd.concat([
                chunk_causal[["from_account", "institution_id"]].rename(columns={"from_account": "account_id"}),
                chunk_causal[["to_account", "institution_id"]].rename(columns={"to_account": "account_id"}),
            ], ignore_index=True).drop_duplicates())
        # cpout_causal: basis pagerank/kcore — SAMA pasangan dgn cpout_agg
        # (dasar unique_recipients) TAPI difilter cutoff (beda tujuan: graph-
        # position lintas-akun butuh restriksi, unique_recipients self-
        # aggregate tidak).
        if len(chunk_causal):
            cpout_causal_agg.append(chunk_causal[["from_account", "to_account"]].drop_duplicates())

        rows_read += len(chunk)
        if rows_read % 10_000_000 == 0:
            print(f"  [node-feat] scanned {rows_read:,} rows...", end="\r")

        # Merge setiap MERGE_EVERY chunk supaya intermediate concat tetap kecil
        if len(in_agg) >= MERGE_EVERY:
            in_running  = _merge_running(in_running,  in_agg,  _IN_SUM,  _IN_MAX)
            out_running = _merge_running(out_running, out_agg, _OUT_SUM, _OUT_MAX)
            in_agg  = []
            out_agg = []
            chan_running  = _merge_channel(chan_running, chan_agg)
            chan_agg = []
            cpout_running = _merge_pairs(cpout_running, cpout_agg)
            cpout_agg = []
            cpin_running  = _merge_pairs(cpin_running, cpin_agg)
            cpin_agg = []
            device_running = _merge_pairs(device_running, device_agg)
            device_agg = []
            inst_running  = _merge_pairs(inst_running, inst_agg)
            inst_agg = []
            cpout_causal_running = _merge_pairs(cpout_causal_running, cpout_causal_agg)
            cpout_causal_agg = []

        # Top-K timestamp di-merge LEBIH SERING (operasi sort+head lebih berat
        # kalau dibiarkan menumpuk terlalu banyak chunk sekaligus)
        if ci % TS_MERGE_EVERY == 0:
            ts_running = _merge_topk_ts(ts_running, ts_agg, k=topk_ts)
            ts_agg = []

    print(f"  [node-feat] scanned {rows_read:,} rows — final merge...")

    # Merge sisa chunk yang belum di-merge
    if in_agg:
        in_running  = _merge_running(in_running,  in_agg,  _IN_SUM,  _IN_MAX)
        out_running = _merge_running(out_running, out_agg, _OUT_SUM, _OUT_MAX)
    if chan_agg:
        chan_running = _merge_channel(chan_running, chan_agg)
    if cpout_agg:
        cpout_running = _merge_pairs(cpout_running, cpout_agg)
    if cpin_agg:
        cpin_running = _merge_pairs(cpin_running, cpin_agg)
    if device_agg:
        device_running = _merge_pairs(device_running, device_agg)
    if inst_agg:
        inst_running = _merge_pairs(inst_running, inst_agg)
    if cpout_causal_agg:
        cpout_causal_running = _merge_pairs(cpout_causal_running, cpout_causal_agg)
    if ts_agg:
        ts_running = _merge_topk_ts(ts_running, ts_agg, k=topk_ts)

    in_df  = in_running
    out_df = out_running

    df = in_df.join(out_df, how="outer").fillna(0)

    df["max_single_tx"]    = df[["in_max", "out_max"]].max(axis=1)
    df["total_tx"]         = df["in_degree"] + df["out_degree"]
    df["night_tx_count"]   = df["night_in"] + df["night_out"]
    df["illicit_count"]    = df["illicit_in"] + df["illicit_out"]

    df["degree_ratio"]     = df["out_degree"] / (df["in_degree"] + 1)
    df["amount_ratio"]     = df["out_amount_sum"] / (df["in_amount_sum"] + 1)
    df["night_tx_ratio"]   = df["night_tx_count"] / df["total_tx"].clip(lower=1)
    df["avg_amount_in"]    = df["in_amount_sum"] / (df["in_degree"] + 1)
    df["avg_amount_out"]   = df["out_amount_sum"] / (df["out_degree"] + 1)
    df["is_laundering_label"] = (df["illicit_count"] > 0).astype(int)

    # Fix (6-Jul): true unique counterparty (nunique dari pasangan dedup),
    # BUKAN approksimasi degree lagi.
    n_recip = (cpout_running.groupby("from_account").size()
               if cpout_running is not None else pd.Series(dtype="int64"))
    n_send  = (cpin_running.groupby("to_account").size()
               if cpin_running is not None else pd.Series(dtype="int64"))
    df["unique_recipients"] = n_recip.reindex(df.index).fillna(0)
    df["unique_senders"]    = n_send.reindex(df.index).fillna(0)

    # Behavioral features yang bisa dihitung dari vectorized scan
    _out_deg1 = df["out_degree"].clip(lower=1)
    df["structuring_score"]  = df["structuring_count"] / _out_deg1
    df["round_amount_ratio"] = df["round_tx_count"]    / _out_deg1
    # counterparty_hhi — definisi KANONIK (feature_defs, fix 3-Jul finding #4),
    # sekarang dari unique_recipients ASLI (fix 6-Jul), bukan proxy degree lagi.
    df["counterparty_hhi"]   = _canon_hhi(df["unique_recipients"].clip(lower=1))

    # Fix (6-Jul): channel_entropy sungguhan (reuse fungsi kanonik, satu
    # definisi dgn detection/features.py — hindari train/serve skew lagi).
    if chan_running is not None and len(chan_running):
        df_ent = _compute_channel_entropy(chan_running).set_index("account_id")
        df = df.join(df_ent, how="left")
    if "channel_entropy" not in df.columns:
        df["channel_entropy"] = 0.0
    df["channel_entropy"] = df["channel_entropy"].fillna(0.0)

    # Fix (6-Jul): burst_ratio/inter_tx_std/dormancy_days sungguhan dari
    # top-K=20 timestamp terbaru/akun (reuse fungsi kanonik).
    if ts_running is not None and len(ts_running):
        df_tmp = _compute_temporal_features(ts_running).set_index("account_id")
        df = df.join(df_tmp, how="left")
    for col in ["burst_ratio", "inter_tx_std", "dormancy_days"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Fix (6-Jul, audit roadmap #2): device_sharing_count & n_institutions —
    # dua fitur network baru (lihat _compute_device_sharing/_compute_institution_diversity
    # utk justifikasi kenapa masing2 bersih dari noise data-generation).
    if device_running is not None and len(device_running):
        df_dev = _compute_device_sharing(device_running).set_index("account_id")
        df = df.join(df_dev, how="left")
    if "device_sharing_count" not in df.columns:
        df["device_sharing_count"] = 0.0
    df["device_sharing_count"] = df["device_sharing_count"].fillna(0.0)

    if inst_running is not None and len(inst_running):
        df_inst = _compute_institution_diversity(inst_running).set_index("account_id")
        df = df.join(df_inst, how="left")
    if "n_institutions" not in df.columns:
        df["n_institutions"] = 0.0
    df["n_institutions"] = df["n_institutions"].fillna(0.0)

    # Fix (7-Jul, dorongan AUC; DIPERBAIKI 7-Jul sore — insiden leak): pagerank
    # + kcore_number, fitur graph struktural lintas-akun. PAKAI cpout_causal_
    # running (difilter cutoff), BUKAN cpout_running (itu full-lifetime, basis
    # unique_recipients — leak kalau dipakai di sini, lihat docstring fungsi).
    if cpout_causal_running is not None and len(cpout_causal_running):
        df_graph = _compute_graph_structural(cpout_causal_running).set_index("account_id")
        df = df.join(df_graph, how="left")
    for col in ["pagerank", "kcore_number"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    print(f"  [node-feat] {len(df):,} accounts, {df['is_laundering_label'].sum():,} illicit")
    pd.to_pickle({"version": FEATURE_CONTRACT_VERSION, "df": df,
                  "network_feature_cutoff_ts": network_feature_cutoff_ts}, cache_path)
    print(f"  [node-feat] Cache saved: {os.path.basename(cache_path)} "
          f"(version={FEATURE_CONTRACT_VERSION})")
    return df


# Naikkan versi ini SETIAP KALI logic pembangunan fitur/label di modul ini
# berubah (kolom baru, definisi HHI berubah, cara scaling berubah, dst).
# train_tgn.py & train_dyg.py men-stamp versi ini ke npz cache saat generate,
# dan MENOLAK memakai cache yang versinya beda — supaya training TIDAK PERNAH
# diam-diam pakai data lama walau lupa pasang --no-cache secara manual.
# History:
#   v1                    : awal (HHI = 1/unique_recipients tanpa acuan modul
#                            lain; StandardScaler fit di seluruh data — bocor test)
#   v2-canonical-hhi-2jul  : fix 3 Jul 2026 — HHI dari feature_defs.counterparty_hhi
#                            (satu definisi dgn detection/features & feature_store);
#                            StandardScaler fit HANYA di train split (fix leak)
#   v3-live-temporal-6jul  : fix 6 Jul 2026 (audit roadmap #1) — burst_ratio/
#                            inter_tx_std/dormancy_days/channel_entropy dulu
#                            hardcode 0.0 (AUC=0.500, mati total), sekarang
#                            dihitung sungguhan (top-K=20 timestamp/akun +
#                            reuse detection/features._compute_temporal_features
#                            & _compute_channel_entropy). unique_senders/
#                            unique_recipients dulu approx=degree, sekarang
#                            true nunique counterparty (dedup pair tracking).
#   v4-device-inst-6jul    : fix 6 Jul 2026 (audit roadmap #2) — tambah 2 fitur
#                            network: device_sharing_count (cluster device
#                            <=20 akun, threshold empiris motong noise
#                            hash-pool organik ~3300 akun/device — lihat
#                            _compute_device_sharing) & n_institutions
#                            (nunique institution_id, >2 mustahil organik —
#                            lihat _compute_institution_diversity). FEATURE_COLS
#                            dipindah ke feature_defs.py (single source of
#                            truth, sebelumnya ter-duplikasi 6x file berbeda).
#                            Juga bawa fix NaT-overflow di
#                            _compute_temporal_features (detection/features.py)
#                            — subtract int64 NaT sentinel overflow senyap,
#                            ketahuan dari RuntimeWarning pas full-run v3.
#   v5-graph-struct-7jul   : fix 7 Jul 2026 (dorongan kejar AUC lebih tinggi
#                            pasca-roadmap) — tambah 2 fitur GRAPH STRUKTURAL
#                            pertama (semua fitur sebelumnya 1-hop/langsung):
#                            pagerank (posisi "hub" di graf transaksi
#                            keseluruhan) & kcore_number (kepadatan koneksi
#                            lokal). Dihitung dari cpout_running (pasangan
#                            unik, reuse basis unique_recipients) via
#                            scipy sparse power-iteration + peeling
#                            divektorisasi (lihat _compute_graph_structural).
#   v6-causal-network-7jul : fix 7 Jul 2026 SORE (KRITIS — insiden leak,
#                            ditemukan user bertanya "ini nyontek ga?").
#                            device_sharing_count/n_institutions/pagerank/
#                            kcore_number SEMUA lintas-akun (bergantung akun
#                            LAIN) — dihitung dari SELURUH dataset (v5) berarti
#                            akun test diam-diam intip struktur graf yg baru
#                            terbentuk BELAKANGAN. Diverifikasi empiris:
#                            kcore_number AUC causal cuma 0.439 (DI BAWAH
#                            ACAK!) vs 0.772 "bocor" — HAMPIR SEMUA sinyalnya
#                            palsu. pagerank 0.406 vs 0.571 (sama parah).
#                            device_sharing 0.530 vs 0.631 (~23% asli).
#                            n_institutions 0.347 vs 0.556 (parah, malah
#                            terbalik). Sekarang ke-4 fitur ini dihitung CUMA
#                            dari edge tx_timestamp <= val_cutoff_ts (exclude
#                            SELURUH periode test) via network_feature_
#                            cutoff_ts param baru — akun test genuinely baru
#                            akan dapat nilai lemah/default utk fitur INI SAJA
#                            (JUJUR, bukan bug). unique_recipients/senders/
#                            counterparty_hhi TETAP full-lifetime (self-
#                            aggregate akun itu sendiri, kategori leak beda &
#                            lebih bisa diterima — TIDAK direstriksi sama).
FEATURE_CONTRACT_VERSION = "v6-causal-network-7jul"


def load_temporal_dataset(
    csv_path: str = CSV_DEFAULT,
    chunk_size: int = 500_000,
    max_rows: int = None,
    sample_licit_ratio: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Load CSV chunked -> aggregate per-node features + temporal edges.

    Parameters
    ----------
    csv_path : str
        Path to the transactions CSV file.
    chunk_size : int
        Number of rows per chunk for streaming read.
    max_rows : int, optional
        Maximum total rows to load (None = all rows).
    sample_licit_ratio : float
        Fraction of licit rows to sample (0.1 = 10%). All illicit rows kept.
    random_seed : int
        Random seed for reproducibility.

    Catatan (4-Jul): fungsi ini TIDAK dipakai train_tgn.py/train_dyg.py (mereka
    pakai load_temporal_dataset_fast di bawah) — tapi filosofinya "keep semua
    illicit" di sini SUDAH BENAR sejak awal, jadi jadi acuan saat fast-path
    default-nya diperbaiki (lihat FEATURE_CONTRACT_VERSION & docstring
    load_temporal_dataset_fast untuk detail perhitungan real 4-Jul).

    Returns
    -------
    dict with keys:
        - node_features: np.ndarray (N, 20)  [13 baseline + 7 behavioral]
        - edge_index: np.ndarray (2, E)
        - edge_attr: np.ndarray (E, 3)  [amount_log_norm, channel_enc, hour]
        - edge_timestamps: np.ndarray (E,) unix timestamp
        - edge_labels: np.ndarray (E,) is_laundering per edge
        - node_labels: np.ndarray (N,) 1 if any edge illicit
        - account_to_idx: dict {account_id: int}
        - split: dict {train/val/test: np.ndarray of edge indices}
        - scaler: StandardScaler fitted on node features
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected training data at this path. "
            f"Use --csv to specify a different location."
        )

    rng = np.random.RandomState(random_seed)

    # ------------------------------------------------------------------
    # PASS 1: Stream CSV, accumulate per-account stats + sampled edges
    # ------------------------------------------------------------------
    logger.info("Pass 1: scanning CSV (chunked, chunk_size=%d) ...", chunk_size)
    print(f"[DATASET] Loading from: {csv_path}")
    print(f"[DATASET] chunk_size={chunk_size}, max_rows={max_rows}, "
          f"sample_licit_ratio={sample_licit_ratio}")

    # Per-account accumulators
    acc_stats = defaultdict(lambda: {
        "in_degree": 0, "out_degree": 0,
        "in_amount_sum": 0.0, "out_amount_sum": 0.0,
        "night_tx_in": 0, "night_tx_out": 0,
        "total_tx_in": 0, "total_tx_out": 0,
        "max_tx": 0.0,
        "senders": set(), "recipients": set(),
        "illicit_count": 0,
        # behavioral counters
        "round_tx_count": 0, "structuring_count": 0,
        "channel_counts": defaultdict(int),
    })

    # Sampled edges for graph construction
    edge_rows = []  # list of (from_acc, to_acc, amount, channel, timestamp, is_laund)

    rows_read = 0
    illicit_kept = 0
    licit_sampled = 0

    try:
        reader = pd.read_csv(
            csv_path,
            chunksize=chunk_size,
            dtype={
                "from_account": str, "to_account": str,
                "amount": float, "is_laundering": int,
            },
            parse_dates=["tx_timestamp"],
            low_memory=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to open CSV: {e}") from e

    for chunk_idx, chunk in enumerate(reader):
        if max_rows is not None and rows_read >= max_rows:
            break

        if max_rows is not None:
            remaining = max_rows - rows_read
            chunk = chunk.iloc[:remaining]

        rows_read += len(chunk)

        chunk = harmonize_amount_to_idr(chunk)   # selaraskan amount -> IDR

        # Separate illicit vs licit
        illicit_mask = chunk["is_laundering"] == 1
        illicit_rows = chunk[illicit_mask]
        licit_rows = chunk[~illicit_mask]

        # Sample licit rows
        n_licit_sample = max(1, int(len(licit_rows) * sample_licit_ratio))
        if n_licit_sample < len(licit_rows):
            licit_sample = licit_rows.sample(
                n=n_licit_sample, random_state=rng.randint(0, 2**31)
            )
        else:
            licit_sample = licit_rows

        # Combine sampled rows
        sampled = pd.concat([illicit_rows, licit_sample], ignore_index=True)
        illicit_kept += len(illicit_rows)
        licit_sampled += len(licit_sample)

        # Update per-account stats from ALL rows (not just sampled)
        # to get accurate node features
        for _, row in chunk.iterrows():
            from_acc = str(row["from_account"])
            to_acc = str(row["to_account"])
            amount = float(row["amount"])
            is_laund = int(row["is_laundering"])

            # Determine night hour
            ts = row["tx_timestamp"]
            if pd.notna(ts):
                hour = ts.hour if hasattr(ts, "hour") else 0
            else:
                hour = 0
            is_night = 1 if hour in (22, 23, 0, 1, 2, 3) else 0

            # Behavioral flags
            is_round = 1 if (amount > 0 and abs(amount - round(amount / 10000) * 10000) < 1.0) else 0
            _STRUCT_BANDS = [(9_500,10_000),(95_000,100_000),(950_000,1_000_000),
                             (9_500_000,10_000_000),(95_000_000,100_000_000),(450_000_000,500_000_000)]
            is_struct = 1 if any(lo <= amount < hi for lo, hi in _STRUCT_BANDS) else 0
            channel_str = str(row.get("channel", "internet") or "internet").lower()

            # From account (outbound)
            s_from = acc_stats[from_acc]
            s_from["out_degree"] += 1
            s_from["out_amount_sum"] += amount
            s_from["total_tx_out"] += 1
            s_from["night_tx_out"] += is_night
            s_from["recipients"].add(to_acc)
            s_from["max_tx"] = max(s_from["max_tx"], amount)
            s_from["illicit_count"] += is_laund
            s_from["round_tx_count"] += is_round
            s_from["structuring_count"] += is_struct
            s_from["channel_counts"][channel_str] += 1

            # To account (inbound)
            s_to = acc_stats[to_acc]
            s_to["in_degree"] += 1
            s_to["in_amount_sum"] += amount
            s_to["total_tx_in"] += 1
            s_to["night_tx_in"] += is_night
            s_to["senders"].add(from_acc)
            s_to["max_tx"] = max(s_to["max_tx"], amount)
            s_to["illicit_count"] += is_laund
            s_to["channel_counts"][channel_str] += 1

        # Collect sampled edges
        for _, row in sampled.iterrows():
            ts = row["tx_timestamp"]
            if pd.notna(ts):
                unix_ts = ts.timestamp() if hasattr(ts, "timestamp") else 0.0
                hour = ts.hour if hasattr(ts, "hour") else 0
            else:
                unix_ts = 0.0
                hour = 0

            channel_str = str(row.get("channel", "internet")).lower()
            channel_enc = CHANNEL_MAP.get(channel_str, 2)

            edge_rows.append((
                str(row["from_account"]),
                str(row["to_account"]),
                float(row["amount"]),
                channel_enc,
                unix_ts,
                hour,
                int(row["is_laundering"]),
            ))

        if (chunk_idx + 1) % 10 == 0:
            print(f"  ... processed {rows_read:,} rows "
                  f"({illicit_kept:,} illicit, {licit_sampled:,} licit sampled)")

    print(f"[DATASET] Scan complete: {rows_read:,} rows total")
    print(f"[DATASET] Edges kept: {len(edge_rows):,} "
          f"({illicit_kept:,} illicit + {licit_sampled:,} licit)")
    print(f"[DATASET] Unique accounts: {len(acc_stats):,}")

    if len(edge_rows) == 0:
        raise ValueError("No edges found in dataset. Check CSV file content.")

    # ------------------------------------------------------------------
    # PASS 1 was iterrow-heavy; for large datasets, use vectorized version
    # This is a tradeoff: iterrows is memory-safe but slow
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Build account index
    # ------------------------------------------------------------------
    all_accounts = sorted(acc_stats.keys())
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts)}
    num_nodes = len(all_accounts)

    # ------------------------------------------------------------------
    # Compute temporal behavioral features dari sampled edge_rows
    # (burst_ratio, inter_tx_std, dormancy_days via two-pointer per account)
    # ------------------------------------------------------------------
    logger.info("Computing temporal behavioral features from %d sampled edges...", len(edge_rows))
    _acc_ts = defaultdict(list)
    for (frm, to, _amt, _ch, unix_ts, _hr, _il) in edge_rows:
        if unix_ts > 0:
            _acc_ts[frm].append(unix_ts)
            _acc_ts[to].append(unix_ts)

    _acc_temporal = {}  # account_id → (burst_ratio, inter_tx_std, dormancy_days)
    for _acc, _tss in _acc_ts.items():
        _tss.sort()
        _n = len(_tss)
        if _n < 2:
            _acc_temporal[_acc] = (float(_n > 0), 0.0, 0.0)
            continue
        _gaps = [_tss[i+1] - _tss[i] for i in range(_n - 1)]
        _mean_gap = sum(_gaps) / len(_gaps)
        _inter_tx_std = (sum((_g - _mean_gap) ** 2 for _g in _gaps) / len(_gaps)) ** 0.5
        _dormancy_days = max(_gaps) / 86400.0
        # burst_ratio: two-pointer, 1-hour window
        _left = 0
        _max_burst = 1
        for _r in range(_n):
            while _tss[_r] - _tss[_left] > 3600:
                _left += 1
            _max_burst = max(_max_burst, _r - _left + 1)
        _acc_temporal[_acc] = (float(_max_burst) / _n, _inter_tx_std, _dormancy_days)
    del _acc_ts

    # ------------------------------------------------------------------
    # Build node features (N, 20) matching FEATURE_COLS
    # ------------------------------------------------------------------
    node_features_raw = np.zeros((num_nodes, len(FEATURE_COLS)), dtype=np.float32)
    node_labels = np.zeros(num_nodes, dtype=np.int64)

    for acc, idx in account_to_idx.items():
        s = acc_stats[acc]
        in_deg = s["in_degree"]
        out_deg = s["out_degree"]
        in_amt = s["in_amount_sum"]
        out_amt = s["out_amount_sum"]
        total_tx = s["total_tx_in"] + s["total_tx_out"]
        night_tx = s["night_tx_in"] + s["night_tx_out"]

        # Behavioral features
        _burst, _std, _dorm = _acc_temporal.get(acc, (0.0, 0.0, 0.0))
        _u_recv = max(len(s["recipients"]), 1)
        _counterparty_hhi = _canon_hhi(_u_recv)   # kanonik (fix HHI 3-Jul)
        _chan = s["channel_counts"]
        _chan_total = sum(_chan.values()) or 1
        _channel_entropy = -sum(
            (c / _chan_total) * math.log2(c / _chan_total + 1e-12)
            for c in _chan.values()
        ) if _chan else 0.0
        _out_deg1 = max(out_deg, 1)
        _structuring_score  = s["structuring_count"] / _out_deg1
        _round_amount_ratio = s["round_tx_count"] / _out_deg1

        node_features_raw[idx] = [
            in_deg,                                         # in_degree
            out_deg,                                        # out_degree
            out_deg / (in_deg + 1),                         # degree_ratio
            in_amt,                                         # in_amount_sum
            out_amt,                                        # out_amount_sum
            out_amt / (in_amt + 1),                         # amount_ratio
            len(s["senders"]),                              # unique_senders
            len(s["recipients"]),                           # unique_recipients
            s["max_tx"],                                    # max_single_tx
            night_tx / max(total_tx, 1),                    # night_tx_ratio
            in_amt / (in_deg + 1),                          # avg_amount_in
            out_amt / (out_deg + 1),                        # avg_amount_out
            total_tx,                                       # total_tx
            _burst,                                         # burst_ratio
            _std,                                           # inter_tx_std
            _dorm,                                          # dormancy_days
            _counterparty_hhi,                              # counterparty_hhi
            _channel_entropy,                               # channel_entropy
            _structuring_score,                             # structuring_score
            _round_amount_ratio,                            # round_amount_ratio
        ]

        if s["illicit_count"] > 0:
            node_labels[idx] = 1

    # Free memory from sets in acc_stats
    del acc_stats

    # CATATAN (fix StandardScaler-leak 3-Jul): node_features_raw SENGAJA belum
    # di-scale di sini. Scaler di-fit HANYA pada node train (setelah split) supaya
    # statistik normalisasi (mean/std) tidak bocor dari val/test.

    # ------------------------------------------------------------------
    # Build edge arrays (amount kolom 0 = log1p mentah, di-scale setelah split)
    # ------------------------------------------------------------------
    num_edges = len(edge_rows)
    edge_index = np.zeros((2, num_edges), dtype=np.int64)
    edge_attr = np.zeros((num_edges, 3), dtype=np.float32)
    edge_timestamps = np.zeros(num_edges, dtype=np.float64)
    edge_labels = np.zeros(num_edges, dtype=np.int64)

    for i, (from_acc, to_acc, amount, channel, unix_ts, hour, is_laund) in enumerate(edge_rows):
        edge_index[0, i] = account_to_idx[from_acc]
        edge_index[1, i] = account_to_idx[to_acc]
        edge_attr[i, 0] = np.log1p(amount)  # log amount (belum di-scale)
        edge_attr[i, 1] = channel            # channel encoding
        edge_attr[i, 2] = hour               # hour of day
        edge_timestamps[i] = unix_ts
        edge_labels[i] = is_laund

    del edge_rows  # free memory

    # ------------------------------------------------------------------
    # Temporal split by timestamp (60/20/20)
    # ------------------------------------------------------------------
    sorted_idx = np.argsort(edge_timestamps)
    n_train = int(0.6 * num_edges)
    n_val = int(0.2 * num_edges)

    split = {
        "train": sorted_idx[:n_train],
        "val": sorted_idx[n_train: n_train + n_val],
        "test": sorted_idx[n_train + n_val:],
    }

    # ------------------------------------------------------------------
    # Normalisasi — fit HANYA di train (fix StandardScaler-leak 3-Jul)
    # ------------------------------------------------------------------
    # Node features: fit scaler di node yang muncul sebagai src/dst pada TRAIN
    # edges saja, lalu transform semua node.
    train_edge_idx = split["train"]
    train_nodes = np.unique(np.concatenate([
        edge_index[0, train_edge_idx], edge_index[1, train_edge_idx]
    ]))
    scaler = StandardScaler()
    if len(train_nodes) > 0:
        scaler.fit(node_features_raw[train_nodes])
    else:  # fallback (train kosong, tak seharusnya terjadi)
        scaler.fit(node_features_raw)
    node_features = scaler.transform(node_features_raw).astype(np.float32)
    del node_features_raw

    # Edge amount (kolom 0): fit di train edges saja, transform semua.
    amount_scaler = StandardScaler()
    amount_scaler.fit(edge_attr[train_edge_idx, 0].reshape(-1, 1))
    edge_attr[:, 0] = amount_scaler.transform(edge_attr[:, 0].reshape(-1, 1)).ravel()

    train_illicit = edge_labels[split["train"]].sum()
    val_illicit = edge_labels[split["val"]].sum()
    test_illicit = edge_labels[split["test"]].sum()

    print(f"[DATASET] Split sizes: "
          f"train={len(split['train']):,} ({train_illicit:,} illicit), "
          f"val={len(split['val']):,} ({val_illicit:,} illicit), "
          f"test={len(split['test']):,} ({test_illicit:,} illicit)")
    print(f"[DATASET] Node features shape: {node_features.shape}")
    print(f"[DATASET] Edge index shape: {edge_index.shape}")

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_timestamps": edge_timestamps,
        "edge_labels": edge_labels,
        "node_labels": node_labels,
        "account_to_idx": account_to_idx,
        "split": split,
        "scaler": scaler,
    }


def load_temporal_dataset_fast(
    csv_path: str = CSV_DEFAULT,
    chunk_size: int = 500_000,
    max_rows: int = None,
    sample_licit_ratio: float = 0.31,
    sample_illicit_ratio: float = 1.0,
    eval_sample_ratio: float = 0.2,
    random_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> dict:
    """
    Vectorized version: scan seluruh CSV dengan chunked sampling.

    Default DIPERBAIKI 4-Jul (sebelumnya 0.005/0.05 — komentar lama salah
    asumsi 41M illicit yang TIDAK PERNAH diverifikasi terhadap data asli).
    Komposisi REAL transactions_hi_injected (diverifikasi via Postgres exact
    COUNT, 4 Juli 2026): illicit = 525.774, licit ≈ 176,47 juta (total ≈177 juta).

    Fix KEDUA (5-Jul): sampling `sample_licit_ratio`/`sample_illicit_ratio`
    SEKARANG cuma diterapkan ke window TRAIN (temporal). Versi sebelumnya
    nyampling SEMUA data (train+val+test sekaligus) SEBELUM split — artinya
    val/test ikut direbalance ke prevalensi ~41% illicit, bukan distribusi
    ASLI. Itu bikin PR-AUC yg dilaporkan tidak sebanding dgn XGBoost yg
    dievaluasi di populasi natural (17,77%, lihat detection/features.py
    extract_features_bulk). Sekarang: cutoff waktu train/val/test dihitung
    DULU dari timestamp ASLI (pass 1, sebelum sampling apapun), baru window
    train yg di-rebalance (utk efisiensi training + sinyal gradient dari
    kelas minoritas langka — ini praktik standar & TETAP dipertahankan).
    Window eval (val+test) di-downsample SERAGAM (`eval_sample_ratio`, sama
    utk kedua kelas) HANYA demi batas memori GPU, prevalensi relatifnya
    tetap ASLI/jujur — bukan direbalance seperti train.

    Default (dihitung dari angka real di atas):
      sample_illicit_ratio=1.0  → keep SEMUA illicit di window TRAIN
      sample_licit_ratio=0.31   → downsample licit di window TRAIN saja
      eval_sample_ratio=0.2     → window eval (val+test) di-downsample rata
                                   utk kedua kelas (bukan direbalance),
                                   demi muat di batas memori RTX 4050.
    Composisi ASLI (bukan sampled) akan DICETAK saat runtime terpisah utk
    window train vs eval — selalu cek output terminal sbg ground-truth.

    Same return format as load_temporal_dataset().
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rng = np.random.RandomState(random_seed)
    print(f"[DATASET-FAST] Loading from: {csv_path}")

    # --- PASS 1: scan HANYA kolom tx_timestamp (murah) utk cutoff temporal ---
    # Cutoff HARUS dihitung dari distribusi waktu ASLI (sebelum sampling apapun),
    # supaya window train/val/test tidak bias oleh rebalancing.
    print("[DATASET-FAST] Pass 1/2: scan tx_timestamp (kolom saja) utk cutoff "
          "temporal dari data ASLI ...")
    # Fix (5-Jul): JANGAN pakai parse_dates= di read_csv — kalau ada 1 baris
    # saja yg formatnya beda (mis. presisi sub-detik tak konsisten), pandas
    # 3.x diam-diam gagal parse SATU CHUNK PENUH jadi dtype=str, lalu
    # .astype("datetime64[s]") dari string tsb crash "Cannot losslessly
    # convert units". Parse manual pakai format="mixed" (tahan format
    # campuran) + errors="coerce" (baris rusak jadi NaT, di-drop, BUKAN
    # crash seluruh pipeline).
    ts_chunks = []
    rows_read_p1 = 0
    n_dropped_p1 = 0
    reader1 = pd.read_csv(csv_path, chunksize=chunk_size, usecols=["tx_timestamp"],
                           low_memory=False)
    for chunk in reader1:
        if max_rows is not None and rows_read_p1 >= max_rows:
            break
        rows_read_p1 += len(chunk)
        ts = pd.to_datetime(chunk["tx_timestamp"], format="mixed", errors="coerce")
        n_dropped_p1 += int(ts.isna().sum())
        ts = ts.dropna()
        ts_chunks.append(ts.values.astype("datetime64[s]").astype(np.int64))
    if n_dropped_p1 > 0:
        print(f"[DATASET-FAST] WARNING: {n_dropped_p1:,} baris dgn tx_timestamp "
              f"tak valid di-skip (pass 1, dari {rows_read_p1:,} baris di-scan).")
    all_ts_sec = np.concatenate(ts_chunks)
    del ts_chunks
    sorted_ts = np.sort(all_ts_sec)
    n_total_ts = len(sorted_ts)
    n_train_cut = max(1, int(train_ratio * n_total_ts))
    n_val_cut = max(n_train_cut + 1, int((train_ratio + val_ratio) * n_total_ts))
    train_cutoff_ts = float(sorted_ts[n_train_cut - 1])
    val_cutoff_ts = float(sorted_ts[min(n_val_cut, n_total_ts) - 1])
    del sorted_ts, all_ts_sec
    print(f"[DATASET-FAST] Cutoff dari {n_total_ts:,} baris asli: "
          f"train<={train_cutoff_ts:.0f}, val<={val_cutoff_ts:.0f}")

    # --- PASS 2: scan penuh; sample HANYA window train, eval didownsample seragam ---
    # Fix OOM (7-Jul): sebelumnya SEMUA chunk (362x, dari 181 juta baris)
    # ditumpuk di list Python, baru pd.concat() SEKALIGUS di akhir — insiden
    # nyata: RAM sistem sampai 0,50GB tersisa (nyaris OOM keras) pas concat
    # raksasa itu, meski sample_licit cuma 0.5. Sekarang merge berkala tiap
    # MERGE_EVERY chunk (pola sama dgn compute_node_features_full yg sudah
    # terbukti aman di 181 juta baris) — batasi ukuran list tak-tergabung,
    # tak pernah nunggu 362 chunk lalu concat sekaligus.
    MERGE_EVERY = 20
    chunks = []
    running_df = None
    rows_read = 0
    total_illicit_seen = 0
    total_licit_seen = 0
    train_illicit_seen = 0
    train_licit_seen = 0
    eval_illicit_seen = 0
    eval_licit_seen = 0
    # Fix OOM (5-Jul): batasi kolom yg dibaca HANYA yg benar2 dipakai fungsi
    # ini (from/to_account, amount, channel, tx_timestamp, is_laundering).
    # Kolom currency/payment_format/device_id/institution_id/typology TAK
    # PERNAH dipakai di sini tapi ikut kebawa di memori utk 49,7 juta baris —
    # kolom string itu paling boros (overhead object per-baris di pandas).
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        usecols=["from_account", "to_account", "amount", "is_laundering",
                 "channel", "tx_timestamp"],
        dtype={"from_account": str, "to_account": str,
               "amount": float, "is_laundering": float,
               "channel": "category"},  # hemat memori: cuma ~5 jenis channel
        # (7-Jul: category utk from/to_account SEMPAT dicoba, tapi union
        # kategori antar-chunk pas concat malah lebih lambat & nyaris tak
        # hemat memori sama sekali — direvert. Akar OOM tetap di pola
        # akumulasi MERGE_EVERY di atas, bukan di sini.)
        low_memory=False,
    )

    n_dropped_p2 = 0
    for chunk in reader:
        if max_rows is not None and rows_read >= max_rows:
            break
        rows_read += len(chunk)

        chunk["is_laundering"] = chunk["is_laundering"].fillna(0).astype(int)
        # Fix (5-Jul): sama spt pass 1 — parse manual (format="mixed",
        # errors="coerce") supaya 1 baris rusak tak bikin seluruh chunk
        # gagal parse & crash.
        chunk["tx_timestamp"] = pd.to_datetime(chunk["tx_timestamp"], format="mixed", errors="coerce")
        valid_ts = chunk["tx_timestamp"].notna()
        n_dropped_p2 += int((~valid_ts).sum())
        chunk = chunk[valid_ts]

        chunk_ts_sec = chunk["tx_timestamp"].values.astype("datetime64[s]").astype(np.int64).astype(np.float64)
        is_train_window = chunk_ts_sec <= train_cutoff_ts

        train_part = chunk[is_train_window]
        eval_part = chunk[~is_train_window]   # val+test — TIDAK direbalance

        illicit_t = train_part[train_part["is_laundering"] == 1]
        licit_t = train_part[train_part["is_laundering"] == 0]
        illicit_e = eval_part[eval_part["is_laundering"] == 1]
        licit_e = eval_part[eval_part["is_laundering"] == 0]

        train_illicit_seen += len(illicit_t); train_licit_seen += len(licit_t)
        eval_illicit_seen += len(illicit_e); eval_licit_seen += len(licit_e)
        total_illicit_seen = train_illicit_seen + eval_illicit_seen
        total_licit_seen = train_licit_seen + eval_licit_seen

        # Sampling REBALANCE — HANYA window train
        if sample_illicit_ratio < 1.0 and len(illicit_t) > 0:
            n_ill = max(1, int(len(illicit_t) * sample_illicit_ratio))
            if n_ill < len(illicit_t):
                illicit_t = illicit_t.sample(n=n_ill, random_state=rng.randint(0, 2**31))
        if len(licit_t) > 0:
            n_lic = max(1, int(len(licit_t) * sample_licit_ratio))
            if n_lic < len(licit_t):
                licit_t = licit_t.sample(n=n_lic, random_state=rng.randint(0, 2**31))

        # Downsample SERAGAM (bukan rebalance) — HANYA window eval, demi memori
        if eval_sample_ratio < 1.0 and len(illicit_e) > 0:
            n_ill_e = max(1, int(len(illicit_e) * eval_sample_ratio))
            if n_ill_e < len(illicit_e):
                illicit_e = illicit_e.sample(n=n_ill_e, random_state=rng.randint(0, 2**31))
        if eval_sample_ratio < 1.0 and len(licit_e) > 0:
            n_lic_e = max(1, int(len(licit_e) * eval_sample_ratio))
            if n_lic_e < len(licit_e):
                licit_e = licit_e.sample(n=n_lic_e, random_state=rng.randint(0, 2**31))

        chunks.append(pd.concat([illicit_t, licit_t, illicit_e, licit_e], ignore_index=True))

        if len(chunks) >= MERGE_EVERY:
            batch_df = pd.concat(chunks, ignore_index=True)
            running_df = batch_df if running_df is None else pd.concat(
                [running_df, batch_df], ignore_index=True)
            chunks = []

        if rows_read % 5_000_000 == 0:
            print(f"  scanned {rows_read:,} rows...", end="\r")

    if chunks:
        batch_df = pd.concat(chunks, ignore_index=True)
        running_df = batch_df if running_df is None else pd.concat(
            [running_df, batch_df], ignore_index=True)
    df = running_df
    del chunks, running_df

    if n_dropped_p2 > 0:
        print(f"[DATASET-FAST] WARNING: {n_dropped_p2:,} baris dgn tx_timestamp "
              f"tak valid di-skip (pass 2, dari {rows_read:,} baris di-scan).")

    illicit_count = df["is_laundering"].sum()
    print(f"[DATASET-FAST] Komposisi ASLI (sebelum sampling) — window TRAIN: "
          f"{train_illicit_seen:,} illicit + {train_licit_seen:,} licit | "
          f"window EVAL: {eval_illicit_seen:,} illicit + {eval_licit_seen:,} licit "
          f"— GROUND TRUTH aktual, bukan asumsi docstring.")
    print(f"[DATASET-FAST] Loaded {len(df):,} edges from {rows_read:,} rows "
          f"({illicit_count:,} illicit, {illicit_count/len(df)*100:.1f}%) "
          f"[train: sample_licit_ratio={sample_licit_ratio}, "
          f"sample_illicit_ratio={sample_illicit_ratio} | "
          f"eval: eval_sample_ratio={eval_sample_ratio}, prevalensi ASLI dipertahankan]")

    # Build account index dari sampled edges
    all_accounts_sampled = sorted(pd.unique(
        pd.concat([df["from_account"], df["to_account"]])
    ).tolist())
    account_to_idx = {acc: idx for idx, acc in enumerate(all_accounts_sampled)}
    num_nodes = len(all_accounts_sampled)

    # Map edges
    src = df["from_account"].map(account_to_idx).values.astype(np.int64)
    dst = df["to_account"].map(account_to_idx).values.astype(np.int64)
    edge_index = np.stack([src, dst])

    # Pastikan tx_timestamp bertipe datetime
    df["tx_timestamp"] = pd.to_datetime(df["tx_timestamp"], errors="coerce")

    # Edge attributes
    amounts_log = np.log1p(df["amount"].values).astype(np.float32)
    channels = df["channel"].map(
        lambda x: CHANNEL_MAP.get(str(x).lower(), 2)
    ).values.astype(np.float32) if "channel" in df.columns else np.full(len(df), 2, dtype=np.float32)
    hours = df["tx_timestamp"].dt.hour.fillna(0).values.astype(np.float32)

    # amount kolom 0 = log1p MENTAH; di-scale setelah split (fix leak 3-Jul)
    edge_attr = np.stack([amounts_log, channels, hours], axis=1).astype(np.float32)
    # Fix (13-Jul, insiden OOM r=0.65 118 juta baris): .apply(lambda x: x.timestamp())
    # row-by-row itu boros memori PARAH utk skala >100 juta baris — pandas
    # bikin intermediate array object (boxing tiap Timestamp jadi Python
    # float satu-satu via lib.map_infer) sebelum akhirnya di-cast float64,
    # jauh lebih besar dari array final 900MB-nya sendiri -> ArrayMemoryError.
    # Fix: vectorized via Timedelta (resolution-agnostic, sama pola dgn fix
    # bug epoch di detection/features.py — HINDARI .astype("int64")//10**9
    # yg asumsi ns, bisa salah 1000x kalau dtype internalnya [us]).
    edge_timestamps = (
        (df["tx_timestamp"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    ).fillna(0.0).values.astype(np.float64)
    edge_labels = df["is_laundering"].values.astype(np.int64)

    # Node features: full scan CSV untuk akurasi (bukan dari sampled edges)
    # Fix (7-Jul, KRITIS): network_feature_cutoff_ts=val_cutoff_ts supaya
    # device_sharing_count/n_institutions/pagerank/kcore_number (fitur lintas-
    # akun) TIDAK mengintip periode test — lihat docstring compute_node_
    # features_full utk bukti empiris leak-nya (kcore_number AUC causal 0.439
    # vs 0.772 kalau bocor).
    print("[DATASET-FAST] Computing node features (full CSV scan, network "
          "features causal sampai val_cutoff_ts)...")
    node_feat_df = compute_node_features_full(csv_path, network_feature_cutoff_ts=val_cutoff_ts)

    # Align node features ke urutan all_accounts_sampled (20 fitur = FEATURE_COLS global)
    feat_aligned = node_feat_df.reindex(all_accounts_sampled)[FEATURE_COLS].fillna(0).astype(np.float32)
    node_features_raw = feat_aligned.values

    # Node labels dari full scan
    node_labels = np.zeros(num_nodes, dtype=np.int64)
    if "is_laundering_label" in node_feat_df.columns:
        lbl = node_feat_df.reindex(all_accounts_sampled)["is_laundering_label"].fillna(0).astype(int)
        node_labels = lbl.values

    illicit_nodes = int(node_labels.sum())

    # Temporal split (fix 5-Jul): pakai cutoff TIMESTAMP dari pass 1 (bukan
    # posisi persentase array) — krn densitas baris sudah asimetris setelah
    # sampling window train (direbalance) vs eval (downsample seragam).
    # Split by posisi lama SALAH di sini: 60% posisi array != 60% waktu asli.
    split = {
        "train": np.where(edge_timestamps <= train_cutoff_ts)[0],
        "val": np.where((edge_timestamps > train_cutoff_ts) & (edge_timestamps <= val_cutoff_ts))[0],
        "test": np.where(edge_timestamps > val_cutoff_ts)[0],
    }

    # Normalisasi — fit HANYA di train (fix StandardScaler-leak 3-Jul)
    train_edge_idx = split["train"]
    train_nodes = np.unique(np.concatenate([
        edge_index[0, train_edge_idx], edge_index[1, train_edge_idx]
    ]))
    scaler = StandardScaler()
    if len(train_nodes) > 0:
        scaler.fit(node_features_raw[train_nodes])
    else:
        scaler.fit(node_features_raw)
    node_features = scaler.transform(node_features_raw).astype(np.float32)

    amount_scaler = StandardScaler()
    amount_scaler.fit(edge_attr[train_edge_idx, 0].reshape(-1, 1))
    edge_attr[:, 0] = amount_scaler.transform(edge_attr[:, 0].reshape(-1, 1)).ravel()

    print(f"[DATASET-FAST] Nodes: {num_nodes:,}, Edges: {len(df):,}")
    print(f"[DATASET-FAST] Node labels: {node_labels.sum():,} illicit / {num_nodes:,}")
    print(f"[DATASET-FAST] Split: train={len(split['train']):,}, "
          f"val={len(split['val']):,}, test={len(split['test']):,}")

    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_timestamps": edge_timestamps,
        "edge_labels": edge_labels,
        "node_labels": node_labels,
        "account_to_idx": account_to_idx,
        "split": split,
        "scaler": scaler,
        # Disimpan (fix 5-Jul) supaya cache npz bisa rekonstruksi split yg
        # SAMA persis tanpa re-scan CSV — split TIDAK BOLEH dihitung ulang
        # dari posisi persentase edge_timestamps (itu bug lama yg baru
        # diperbaiki: densitas baris asimetris antara window train/eval).
        "train_cutoff_ts": train_cutoff_ts,
        "val_cutoff_ts": val_cutoff_ts,
    }
