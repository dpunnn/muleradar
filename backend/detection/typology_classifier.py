"""
detection/typology_classifier.py — Klasifikasi TIPOLOGI money-laundering (Opsi B).

Menjawab kebutuhan ML-first: label tipologi (judol/qris/dll) datang dari MODEL,
bukan rule (detection/rules.py). Model deteksi (TGN/DyG) menjawab "illicit atau
bukan"; modul INI menjawab "kalau illicit, TIPE apa".

============================ KONTRAK (STABIL) ============================
WAJIB dijaga sama antara Opsi B (sekarang) dan Opsi A (final) supaya swap MULUS
— caller (alert-generation Phase 4.9/4.10, UI dropdown "Tipologi") TIDAK berubah:

    classify_batch(features: np.ndarray[N,24]) -> list[TypologyPrediction]
    classify_one(features: np.ndarray[24])     -> TypologyPrediction

    TypologyPrediction:
      .typology   : str   — salah satu 7 tipologi ID, atau "UNKNOWN"
      .confidence : float — probabilitas tipologi terpilih (0..1)
      .probs      : dict  — {tipologi: prob} lengkap
      .is_unknown : bool
=========================================================================

IMPLEMENTASI SEKARANG (Opsi B): MLP frozen (models/typology_mlp_v1.pkl) di atas
24 fitur node behavioral+graph (feature_defs.FEATURE_COLS). Deteksi TGN/DyG
TIDAK disentuh (aman). Eval test temporal: ROC-AUC 0.95 / PR-AUC 0.76 macro.

PIVOT KE OPSI A (final): ganti _load_model() + _predict_proba() supaya pakai
"kepala tipologi" multi-task di TGN/DyG. Kontrak di atas TETAP → caller aman.

UNKNOWN: ~92% akun illicit tak punya tipologi berlabel (kasus AMLWorld asli).
Kalau confidence < threshold, tipologi = "UNKNOWN" ("laundering, tipe belum
jelas") — JUJUR, bukan maksa satu dari 7. Threshold via env TYPOLOGY_UNKNOWN_THRESHOLD.
"""
import os
import numpy as np

_MODEL_PATH = os.getenv(
    "TYPOLOGY_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "typology_mlp_v2.pkl"),
)
_UNKNOWN_THRESHOLD = float(os.getenv("TYPOLOGY_UNKNOWN_THRESHOLD", "0.40"))


class TypologyPrediction:
    """Hasil klasifikasi tipologi satu akun (lihat KONTRAK di modul)."""
    __slots__ = ("typology", "confidence", "probs", "is_unknown")

    def __init__(self, typology: str, confidence: float, probs: dict, is_unknown: bool):
        self.typology = typology
        self.confidence = confidence
        self.probs = probs
        self.is_unknown = is_unknown

    def to_dict(self) -> dict:
        return {"typology": self.typology, "confidence": round(self.confidence, 4),
                "is_unknown": self.is_unknown,
                "probs": {k: round(v, 4) for k, v in self.probs.items()}}

    def __repr__(self):
        return f"<Typology {self.typology} conf={self.confidence:.2f}>"


class TypologyClassifier:
    """Klasifikasi tipologi per-akun. Singleton via get_classifier()."""

    def __init__(self, model_path: str = None, unknown_threshold: float = None):
        import joblib
        d = joblib.load(model_path or _MODEL_PATH)
        self._model = d["model"]
        self._scaler = d["scaler"]
        self.typos = d["typos"]                       # 7 tipologi (urutan tetap)
        self.threshold = (_UNKNOWN_THRESHOLD if unknown_threshold is None
                          else unknown_threshold)

    # --- titik yg DIGANTI saat pivot ke Opsi A (multi-task TGN/DyG) ---
    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        """(N,24) fitur -> (N,7) probabilitas tipologi. Opsi B: MLP frozen."""
        Xs = self._scaler.transform(np.asarray(features, dtype=np.float32))
        return self._model.predict_proba(Xs)

    def classify_batch(self, features: np.ndarray) -> list:
        features = np.atleast_2d(np.asarray(features, dtype=np.float32))
        proba = self._predict_proba(features)
        out = []
        for row in proba:
            top = int(np.argmax(row))
            conf = float(row[top])
            typ = self.typos[top]
            probs = {self.typos[i]: float(row[i]) for i in range(len(self.typos))}
            # UNKNOWN kalau: (v2) kelas eksplisit "UNKNOWN" menang, ATAU confidence
            # terlalu rendah (safety utk v1 tanpa kelas UNKNOWN).
            if typ == "UNKNOWN" or conf < self.threshold:
                out.append(TypologyPrediction("UNKNOWN", conf, probs, True))
            else:
                out.append(TypologyPrediction(typ, conf, probs, False))
        return out

    def classify_one(self, features: np.ndarray) -> TypologyPrediction:
        return self.classify_batch(features)[0]


_INSTANCE = None


def get_classifier() -> TypologyClassifier:
    """Singleton — load model sekali."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TypologyClassifier()
    return _INSTANCE


if __name__ == "__main__":
    # Smoke test: ambil beberapa akun test berlabel, bandingkan prediksi vs asli.
    import pandas as pd
    root = os.path.join(os.path.dirname(__file__), "..", "..")
    z = np.load(os.path.join(root, "data/processed/transactions_hi_injected_traindata.npz"),
                allow_pickle=True)
    Xall = z["node_features"]; a2i = z["account_to_idx"].item()
    lab = pd.read_parquet(os.path.join(root, "data/processed/account_typology_labels.parquet"))
    TYPOS = ["judol", "qris", "dormant", "scam", "bendahara", "pep", "vendor"]
    lab["y"] = lab[TYPOS].values.argmax(1)
    lab["idx"] = lab["account_id"].map(lambda a: a2i.get(a, -1))
    lab = lab[lab["idx"] >= 0].sample(12, random_state=1)

    clf = get_classifier()
    feats = Xall[lab["idx"].values]
    preds = clf.classify_batch(feats)
    print(f"KONTRAK OK — {len(clf.typos)} tipologi, threshold UNKNOWN={clf.threshold}\n")
    print(f"{'akun':<12}{'ASLI':<12}{'PREDIKSI':<12}{'conf':>6}  match")
    for (_, r), p in zip(lab.iterrows(), preds):
        asli = TYPOS[r["y"]]
        mark = "OK" if p.typology == asli else ("(unknown)" if p.is_unknown else "X")
        print(f"{r['account_id']:<12}{asli:<12}{p.typology:<12}{p.confidence:>6.2f}  {mark}")
