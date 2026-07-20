"""
Phase 3.5.9-companion — Cost-based threshold tuning (fix 4-Jul, redesign #4).

MASALAH: ALERT_THRESHOLD=0.5 & FREEZE_THRESHOLD=0.85 di streaming/realtime_scorer.py
dipilih ARBITRER (angka bulat, bukan hasil analisis). Threshold ini menentukan
keputusan HUKUM (freeze rekening nasabah) — harus defensible ke regulator,
bukan tebakan.

SOLUSI: threshold OPTIMAL dihitung dari precision-recall curve model pada data
validasi + rasio biaya nyata:
    - cost_false_negative : biaya SATU fraud yang lolos tak terdeteksi
    - cost_false_positive : biaya SATU alert palsu (waktu investigasi analis)
Threshold yang MEMINIMALKAN total biaya harapan = threshold yang dipakai.

PENTING — biaya TIDAK di-hardcode di modul ini. Rasio cost_fn/cost_fp adalah
keputusan RISK APPETITE compliance/risk team institusi (bank kecil vs bank
besar punya rasio beda), bukan sesuatu yang bisa ditebak dari kode. Yang
disediakan: mesin hitungnya + sensitivity_analysis() supaya operator lihat
bagaimana threshold berubah seiring asumsi biaya — bukan percaya 1 angka.

Referensi biaya yang SUDAH diriset (BUSINESS_PLAN.txt) untuk mulai estimasi:
    - Biaya false positive per alert : ~Rp 34.091 (30 menit x gaji analis
      Rp 12jt/bulan / (22 hari x 8 jam)) — lihat BUSINESS_PLAN.txt section B.1
    - Biaya false negative per fraud  : TIDAK ADA angka pasti/universal.
      Institusi harus tetapkan sendiri berdasarkan risk appetite (mis. porsi
      dari denda OJK maks Rp 15 miliar per pelanggaran, ditimbang probabilitas
      terdeteksi regulator, ditambah kerugian nasabah + reputasi). JANGAN
      pakai angka default modul ini sebagai keputusan final tanpa review
      compliance team.

Cara pakai:
    python -m ml.threshold_tuning --csv path/to/val_predictions.csv \\
        --cost-fp 34091 --cost-fn 5000000
    (CSV wajib punya kolom: label (0/1), score (0-1))

    Atau sebagai library:
        from ml.threshold_tuning import cost_based_threshold, sensitivity_analysis
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def cost_based_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_false_negative: float,
    cost_false_positive: float,
) -> dict:
    """
    Cari threshold yang MEMINIMALKAN total biaya harapan pada data validasi.

    Return dict:
        best              : {threshold, precision, recall, tp, fp, fn, expected_cost}
        sensitivity_table : list semua kandidat threshold + biayanya (untuk plot/tabel)
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0:
        raise ValueError("y_true tidak punya contoh positif (fraud) — tak bisa dihitung.")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # precision_recall_curve return array precisions/recalls (len N+1),
    # thresholds (len N) — index sinkron via i (thresholds[i] pasangan
    # precisions[i]/recalls[i], BUKAN elemen terakhir precisions/recalls
    # yang tak punya threshold, sesuai dokumentasi sklearn).
    results = []
    for i, thr in enumerate(thresholds):
        recall = float(recalls[i])
        precision = float(precisions[i])
        tp = recall * n_pos
        fn = n_pos - tp
        # precision = tp / (tp + fp)  =>  fp = tp * (1 - precision) / precision
        fp = tp * (1 - precision) / precision if precision > 0 else float(n_neg)
        expected_cost = fn * cost_false_negative + fp * cost_false_positive
        results.append({
            "threshold": float(thr),
            "precision": precision,
            "recall": recall,
            "tp": round(tp, 1),
            "fp": round(fp, 1),
            "fn": round(fn, 1),
            "expected_cost": round(expected_cost, 2),
        })

    if not results:
        raise ValueError("Tidak ada kandidat threshold — cek y_true/y_scores.")

    best = min(results, key=lambda r: r["expected_cost"])
    return {"best": best, "sensitivity_table": results}


def sensitivity_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    cost_fp: float,
    cost_fn_candidates: list,
) -> list:
    """
    Untuk beberapa asumsi cost_fn berbeda, tunjukkan threshold optimal
    masing-masing. Membantu compliance team LIHAT trade-off sebelum
    commit ke satu angka — bukan percaya satu hasil tunggal.
    """
    out = []
    for cost_fn in cost_fn_candidates:
        res = cost_based_threshold(y_true, y_scores, cost_fn, cost_fp)
        out.append({"cost_fn": cost_fn, "cost_fp": cost_fp, **res["best"]})
    return out


def _print_report(y_true, y_scores, cost_fn, cost_fp):
    result = cost_based_threshold(y_true, y_scores, cost_fn, cost_fp)
    best = result["best"]
    print("=" * 70)
    print("COST-BASED THRESHOLD — HASIL OPTIMAL")
    print("=" * 70)
    print(f"  cost_false_negative (1 fraud lolos) : {cost_fn:,.0f}")
    print(f"  cost_false_positive (1 alert palsu)  : {cost_fp:,.0f}")
    print(f"  Rasio cost_fn/cost_fp                : {cost_fn/max(cost_fp,1e-9):.1f}x")
    print("-" * 70)
    print(f"  THRESHOLD OPTIMAL     : {best['threshold']:.4f}")
    print(f"  Precision @ threshold : {best['precision']:.4f}")
    print(f"  Recall @ threshold    : {best['recall']:.4f}")
    print(f"  TP={best['tp']:.0f}  FP={best['fp']:.0f}  FN={best['fn']:.0f}")
    print(f"  Expected total cost   : {best['expected_cost']:,.0f}")
    print("=" * 70)

    print("\nSENSITIVITY — threshold optimal kalau asumsi cost_fn berbeda:")
    print(f"  {'cost_fn':>15} {'threshold':>10} {'precision':>10} {'recall':>10}")
    for mult in (0.2, 0.5, 1.0, 2.0, 5.0):
        cfn = cost_fn * mult
        r = cost_based_threshold(y_true, y_scores, cfn, cost_fp)["best"]
        print(f"  {cfn:>15,.0f} {r['threshold']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f}")
    print("\n[PENTING] Threshold di atas adalah HASIL HITUNGAN dari asumsi biaya "
          "yang kamu masukkan --cost-fn. WAJIB direview compliance/risk team "
          "sebelum dipakai produksi — bukan otomatis benar.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cost-based threshold tuning dari CSV prediksi validasi."
    )
    parser.add_argument("--csv", required=True,
                         help="CSV dgn kolom 'label' (0/1) dan 'score' (0-1)")
    parser.add_argument("--cost-fp", type=float, default=34091,
                         help="Biaya 1 false positive (default: estimasi "
                              "30 menit investigasi analis, lihat BUSINESS_PLAN.txt)")
    parser.add_argument("--cost-fn", type=float, required=True,
                         help="Biaya 1 false negative (WAJIB diisi eksplisit oleh "
                              "compliance/risk team — tidak ada default yang aman)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "label" not in df.columns or "score" not in df.columns:
        print("[ERROR] CSV wajib punya kolom 'label' dan 'score'.", file=sys.stderr)
        sys.exit(1)

    _print_report(df["label"].values, df["score"].values, args.cost_fn, args.cost_fp)
