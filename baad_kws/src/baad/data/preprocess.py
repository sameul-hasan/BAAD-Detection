"""Resample every manifest clip to 16 kHz mono float, trim edge silence, peak-normalise.

Writes data/processed/<label_id>/<speaker>_<n>.wav (16-bit PCM) and adds a
'proc_path' + 'proc_duration' column to the manifest.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from baad import paths
from baad.logging_utils import get_logger, log_event

TRIM_TOP_DB = 35.0
PAD_S = 0.05
PEAK = 0.9


def load_mono_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != paths.SAMPLE_RATE:
        g = gcd(sr, paths.SAMPLE_RATE)
        x = resample_poly(x, paths.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return x


def trim(x: np.ndarray, top_db: float = TRIM_TOP_DB, pad_s: float = PAD_S) -> np.ndarray:
    """Energy-based edge trim (25 ms frames, 10 ms hop) keeping a small pad."""
    sr = paths.SAMPLE_RATE
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(x) < win:
        return x
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt((x[idx] ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms / (rms.max() + 1e-12))
    voiced = np.where(db > -top_db)[0]
    if len(voiced) == 0:
        return x
    pad = int(pad_s * sr)
    s = max(0, voiced[0] * hop - pad)
    e = min(len(x), voiced[-1] * hop + win + pad)
    return x[s:e]


def process_one(args):
    src, dst = args
    x = trim(load_mono_16k(src))
    peak = float(np.abs(x).max())
    if peak > 1e-4:
        x = x * (PEAK / peak)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), x, paths.SAMPLE_RATE, subtype="PCM_16")
    return len(x) / paths.SAMPLE_RATE


def main():
    log = get_logger("preprocess")
    df = pd.read_csv(paths.MANIFEST)
    counters: dict = {}
    proc = []
    for r in df.itertuples():
        k = (r.label_id, r.speaker)
        counters[k] = counters.get(k, 0) + 1
        proc.append(paths.PROCESSED_DIR / f"{r.label_id:03d}" / f"{r.speaker}_{counters[k]}.wav")
    jobs = [(str(paths.DATASET_ROOT / p), d) for p, d in zip(df.path, proc)]
    with ProcessPoolExecutor() as ex:
        durs = list(ex.map(process_one, jobs, chunksize=32))
    df["proc_path"] = [str(p.relative_to(paths.DATA_DIR)) for p in proc]
    df["proc_duration"] = np.round(durs, 4)
    df.to_csv(paths.MANIFEST, index=False)

    before_min = float(df.duration.sum() / 60)
    after_min = float(df.proc_duration.sum() / 60)
    gt_10 = float((df.proc_duration > 1.0).mean())
    gt_125 = float((df.proc_duration > paths.WINDOW_SEC).mean())
    metrics = {
        "clips_processed": int(len(df)),
        "target_sr": paths.SAMPLE_RATE,
        "before_trim_min": round(before_min, 2),
        "after_trim_min": round(after_min, 2),
        "median_dur_s": round(float(df.proc_duration.median()), 3),
        "max_dur_s": round(float(df.proc_duration.max()), 3),
        "pct_gt_1_0s": round(100.0 * gt_10, 2),
        "pct_gt_1_25s": round(100.0 * gt_125, 2),
    }
    log.info("Preprocessed %d clips: %s", len(df), metrics)
    log_event("preprocess", "Resampled to 16 kHz mono, trimmed silence, peak-normalised", metrics=metrics)


if __name__ == "__main__":
    main()
