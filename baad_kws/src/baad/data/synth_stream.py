"""Deterministic synthetic continuous ('sentence') speech stream generator.

Strictly pairs BAAD clips from `data/splits/<split>.csv` with FLEURS-bn carrier
speech from `data/external/fleurs_bn/<split>/` so there is zero speaker leakage.

Generates:
  - Positive/mixed streams (8-12 s): FLEURS carrier + 1-3 BAAD clips spliced in
    via 20 ms raised-cosine crossfade, RMS-matched (+-4 dB), optional light noise.
  - Clean negative-only streams (8-12 s): pure FLEURS speech + optional close-word
    insertions, used to measure False Alarms per Hour (FA/h).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from baad import paths
from baad.data.augment import add_noise_snr, rms_match
from baad.logging_utils import get_logger, log_event

SR = paths.SAMPLE_RATE
FADE_SAMPLES = int(0.020 * SR)  # 20 ms raised-cosine crossfade


def load_carrier_stream(fleurs_files: list[Path], target_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Concatenate random FLEURS clips from the same split until target_samples is reached."""
    chunks = []
    total = 0
    while total < target_samples:
        p = fleurs_files[int(rng.integers(0, len(fleurs_files)))]
        wav, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if len(wav) < int(0.4 * SR):
            continue
        # Normalize carrier segment gently
        rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-8)
        wav = wav * (0.06 / rms)
        chunks.append(wav)
        chunks.append(np.zeros(int(rng.uniform(0.05, 0.25) * SR), dtype=np.float32))
        total += len(chunks[-2]) + len(chunks[-1])
    buf = np.concatenate(chunks)
    if len(buf) > target_samples:
        start = int(rng.integers(0, len(buf) - target_samples + 1))
        buf = buf[start:start + target_samples]
    return buf[:target_samples].astype(np.float32)


def splice_clip(carrier: np.ndarray, clip: np.ndarray, start_idx: int, rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    """Splice `clip` into `carrier` at `start_idx` using RMS matching and 20 ms crossfade."""
    end_idx = min(len(carrier), start_idx + len(clip))
    clip = clip[:end_idx - start_idx]
    n = len(clip)
    local = carrier[max(0, start_idx - SR):min(len(carrier), end_idx + SR)]
    target_rms = float(np.sqrt(np.mean(local ** 2)) + 1e-8)
    target_rms = max(0.03, min(0.18, target_rms))
    matched = rms_match(clip, target_rms, jitter_db=4.0, rng=rng)

    fade = min(FADE_SAMPLES, n // 4)
    if fade > 1:
        ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, dtype=np.float32) / fade))
        mask = np.ones(n, dtype=np.float32)
        mask[:fade] = ramp
        mask[-fade:] = ramp[::-1]
    else:
        mask = np.ones(n, dtype=np.float32)

    out = carrier.copy()
    # Blend smoothly across edges while replacing the speech inside the word slot
    out[start_idx:end_idx] = (1.0 - mask) * out[start_idx:end_idx] + mask * matched
    return out, start_idx, end_idx


def _pick_slots(n_events: int, total_samples: int, clip_lens: list[int], rng: np.random.Generator) -> list[int]:
    """Pick non-overlapping start sample indices with >= 1.5 s spacing and >= 0.4 s margins."""
    margin = int(0.40 * SR)
    min_gap = int(1.50 * SR)
    placed: list[tuple[int, int]] = []
    starts: list[int] = []
    for clen in clip_lens:
        hi = total_samples - margin - clen
        if hi <= margin:
            return []
        for _ in range(50):
            s = int(rng.integers(margin, hi + 1))
            e = s + clen
            if all((e + min_gap <= ps) or (s >= pe + min_gap) for ps, pe in placed):
                placed.append((s, e))
                starts.append(s)
                break
    return starts


def generate_split(split: str, n_pos: int = 150, n_neg: int = 100, seed: int = 2026) -> list[dict]:
    rng = np.random.default_rng(seed + (0 if split == "val" else 1000))
    df = pd.read_csv(paths.SPLITS_DIR / f"{split}.csv")
    abusive_df = df[df["is_abusive"] == 1].reset_index(drop=True)
    close_df = df[df["is_abusive"] == 0].reset_index(drop=True)

    fleurs_dir = paths.EXTERNAL_DIR / "fleurs_bn" / split
    fleurs_files = sorted(fleurs_dir.glob("*.wav"))
    if not fleurs_files:
        raise FileNotFoundError(f"No FLEURS wav files in {fleurs_dir}")

    # Pre-load split BAAD clips into memory for speed
    clip_cache: dict[str, np.ndarray] = {}
    for rel in df["proc_path"]:
        wav, _ = sf.read(str(paths.DATA_DIR / rel), dtype="float32")
        clip_cache[rel] = wav

    out_dir = paths.SYNTH_DIR / split
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records = []

    def make_stream(idx: int, kind: str) -> dict:
        dur_s = float(rng.uniform(8.0, 12.0))
        total_samples = int(round(dur_s * SR))
        wav = load_carrier_stream(fleurs_files, total_samples, rng)

        chosen_rows = []
        if kind == "pos":
            n_ev = int(rng.choice([1, 2, 3], p=[0.40, 0.40, 0.20]))
            # Guarantee at least 1 abusive word; remaining slots can be abusive (65%) or close (35%)
            chosen_rows.append(abusive_df.iloc[int(rng.integers(0, len(abusive_df)))])
            for _ in range(n_ev - 1):
                pool = abusive_df if rng.random() < 0.65 else close_df
                chosen_rows.append(pool.iloc[int(rng.integers(0, len(pool)))])
            rng.shuffle(chosen_rows)
        elif kind == "neg_close":
            n_ev = int(rng.choice([1, 2], p=[0.65, 0.35]))
            for _ in range(n_ev):
                chosen_rows.append(close_df.iloc[int(rng.integers(0, len(close_df)))])

        clips = [clip_cache[r["proc_path"]] for r in chosen_rows]
        starts = _pick_slots(len(clips), total_samples, [len(c) for c in clips], rng)
        events = []
        if starts:
            for r, clip, s in sorted(zip(chosen_rows, clips, starts), key=lambda t: t[2]):
                wav, s_idx, e_idx = splice_clip(wav, clip, s, rng)
                events.append({
                    "word": str(r["label"]),
                    "label_id": int(r["label_id"]),
                    "is_abusive": bool(r["is_abusive"]),
                    "speaker": str(r["speaker"]),
                    "start": round(s_idx / SR, 4),
                    "end": round(e_idx / SR, 4),
                })

        if rng.random() < 0.6:
            wav = add_noise_snr(wav, rng, snr_db_range=(12.0, 28.0))
        peak = float(np.max(np.abs(wav)) + 1e-8)
        if peak > 0.95:
            wav = wav * (0.90 / peak)

        fname = f"{split}_{kind}_{idx:04d}.wav"
        rel_audio = f"synth/{split}/audio/{fname}"
        sf.write(str(audio_dir / fname), wav, SR, subtype="PCM_16")
        return {
            "audio": rel_audio,
            "duration": round(len(wav) / SR, 4),
            "split": split,
            "stream_kind": kind,
            "has_abusive": any(e["is_abusive"] for e in events),
            "events": events,
        }

    for i in range(n_pos):
        records.append(make_stream(i, "pos"))
    n_neg_close = int(round(0.30 * n_neg))
    for i in range(n_neg):
        k = "neg_close" if i < n_neg_close else "neg_pure"
        records.append(make_stream(i, k))

    jsonl_path = out_dir / "streams.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return records


def main():
    log = get_logger("synth_stream")
    summary = {}
    for split in ("val", "test"):
        recs = generate_split(split, n_pos=150, n_neg=100, seed=2026)
        tot_h = sum(r["duration"] for r in recs) / 3600
        neg_h = sum(r["duration"] for r in recs if not r["has_abusive"]) / 3600
        n_ab = sum(1 for r in recs for e in r["events"] if e["is_abusive"])
        n_cl = sum(1 for r in recs for e in r["events"] if not e["is_abusive"])
        summary[split] = {
            "streams": len(recs),
            "total_hours": round(tot_h, 3),
            "clean_neg_hours": round(neg_h, 3),
            "abusive_events": n_ab,
            "close_events": n_cl,
        }
        log.info("Generated %s synthetic streams: %s", split, summary[split])
    log_event("synth_stream", "Generated deterministic val and test synthetic continuous streams", metrics=summary)


if __name__ == "__main__":
    main()
