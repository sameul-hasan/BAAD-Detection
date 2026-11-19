"""Classical MFCC + LogisticRegression baseline on isolated BAAD words."""
from __future__ import annotations

import json
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from baad import paths
from baad.logging_utils import get_logger, log_event


def extract_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    for rel in df["proc_path"]:
        wav, sr = sf.read(str(paths.DATA_DIR / rel), dtype="float32")
        if len(wav) < 512:
            wav = np.pad(wav, (0, 512 - len(wav)))
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=20, n_fft=320, hop_length=160)
        d1 = librosa.feature.delta(mfcc, width=min(9, max(3, (mfcc.shape[1] // 2) * 2 - 1))) if mfcc.shape[1] >= 3 else np.zeros_like(mfcc)
        vec = np.concatenate([mfcc.mean(1), mfcc.std(1), d1.mean(1), d1.std(1)])
        feats.append(vec.astype(np.float32))
    return np.vstack(feats)


def main():
    log = get_logger("baseline_mfcc")
    t0 = time.time()
    tr = pd.read_csv(paths.SPLITS_DIR / "train.csv")
    va = pd.read_csv(paths.SPLITS_DIR / "val.csv")
    te = pd.read_csv(paths.SPLITS_DIR / "test.csv")

    X_tr, y_tr = extract_features(tr), tr["label_id"].to_numpy()
    X_va, y_va = extract_features(va), va["label_id"].to_numpy()
    X_te, y_te = extract_features(te), te["label_id"].to_numpy()

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1)
    clf.fit(X_tr_s, y_tr)

    labels = json.loads(paths.LABELS_JSON.read_text(encoding="utf-8"))
    num_abusive = sum(1 for x in labels if x["is_abusive"])

    results = {"fit_sec": round(time.time() - t0, 2)}
    for name, X_s, y in [("val", X_va_s, y_va), ("test", X_te_s, y_te)]:
        probs = clf.predict_proba(X_s)
        pred = clf.classes_[probs.argmax(axis=1)]
        acc = accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average="macro", zero_division=0)
        y_ab = (y < num_abusive).astype(int)
        p_ab = (pred < num_abusive).astype(int)
        bp, br, bf1, _ = precision_recall_fscore_support(y_ab, p_ab, average="binary", zero_division=0)
        results.update({
            f"{name}_word_acc": round(float(acc), 4),
            f"{name}_macro_f1": round(float(mf1), 4),
            f"{name}_abusive_prec": round(float(bp), 4),
            f"{name}_abusive_rec": round(float(br), 4),
            f"{name}_abusive_f1": round(float(bf1), 4),
        })
    log.info("MFCC Baseline B1 results: %s", results)
    log_event(
        "baseline_mfcc",
        "Evaluated isolated-word MFCC + LogisticRegression baseline (unseen speakers)",
        metrics=results,
    )


if __name__ == "__main__":
    main()
