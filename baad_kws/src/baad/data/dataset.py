"""RAM-cached PyTorch datasets for isolated evaluation and continuous KWS training."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from baad import paths
from baad.data.augment import (
    add_noise_snr,
    random_gain,
    rms_match,
    simple_reverb,
    speed_perturb,
    synthetic_noise,
)


def load_labels() -> list[dict]:
    return json.loads(paths.LABELS_JSON.read_text(encoding="utf-8"))


def pad_or_center_crop(wav: np.ndarray, target_len: int = paths.WINDOW_SAMPLES) -> np.ndarray:
    """Fit `wav` into exact `target_len` samples (center-crop if longer, center-pad if shorter)."""
    n = len(wav)
    if n == target_len:
        return wav.astype(np.float32)
    if n > target_len:
        start = (n - target_len) // 2
        return wav[start:start + target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    offset = (target_len - n) // 2
    out[offset:offset + n] = wav
    return out


class KWSWindowDataset(Dataset):
    """Dataset supporting three modes:
      - 'isolated': deterministic center-padded/cropped BAAD clips (for isolated word eval).
      - 'kws_train': stochastic on-the-fly continuous windows mixing BAAD keywords
        (abusive + close hard negatives) with FLEURS carrier speech, partial-edge
        unknowns, and synthetic noise/silence.
      - 'kws_val': deterministic mix of BAAD clips (half embedded in FLEURS speech,
        half clean) + FLEURS unknown windows + silence windows for stable epoch metrics.
    """

    def __init__(
        self,
        split: str,
        mode: str = "kws_train",
        samples_per_epoch: int = 9600,
        window_samples: int = paths.WINDOW_SAMPLES,
        seed: int = 1337,
        max_fleurs_cache: int = 1800,
    ):
        assert mode in ("isolated", "kws_train", "kws_val")
        self.split = split
        self.mode = mode
        self.window_samples = window_samples
        self.seed = seed

        labels = load_labels()
        self.num_classes = len(labels)
        self.unknown_id = next(x["id"] for x in labels if x["word"] == paths.UNKNOWN_LABEL)
        self.silence_id = next(x["id"] for x in labels if x["word"] == paths.SILENCE_LABEL)
        self.num_abusive = sum(1 for x in labels if x["is_abusive"])

        self.df = pd.read_csv(paths.SPLITS_DIR / f"{split}.csv").reset_index(drop=True)
        # Preload all BAAD split waveforms into RAM (~40-100 MB per split)
        self.baad_wavs: list[np.ndarray] = []
        for rel in self.df["proc_path"]:
            w, _ = sf.read(str(paths.DATA_DIR / rel), dtype="float32")
            self.baad_wavs.append(w)

        self.baad_labels = self.df["label_id"].to_numpy(dtype=np.int64)
        self.baad_is_abusive = self.df["is_abusive"].to_numpy(dtype=np.int64)
        self.abusive_indices = np.where(self.baad_is_abusive == 1)[0]
        self.close_indices = np.where(self.baad_is_abusive == 0)[0]

        # Class-balanced index groups so rare words (31 clips) get equal training exposure
        self.abusive_by_class: dict[int, np.ndarray] = {
            int(c): np.where(self.baad_labels == c)[0]
            for c in np.unique(self.baad_labels[self.abusive_indices])
        }
        self.abusive_classes = sorted(self.abusive_by_class.keys())
        self.close_by_class: dict[int, np.ndarray] = {
            int(c): np.where(self.baad_labels == c)[0]
            for c in np.unique(self.baad_labels[self.close_indices])
        }
        self.close_classes = sorted(self.close_by_class.keys())

        # Index & pre-cache FLEURS waveforms for this split
        fleurs_dir = paths.EXTERNAL_DIR / "fleurs_bn" / split
        self.fleurs_files = sorted(fleurs_dir.glob("*.wav"))
        self.fleurs_cache: list[np.ndarray] = []
        if mode in ("kws_train", "kws_val") and self.fleurs_files:
            rng_init = np.random.default_rng(seed)
            perm = rng_init.permutation(len(self.fleurs_files))[:max_fleurs_cache]
            for idx in perm:
                w, _ = sf.read(str(self.fleurs_files[int(idx)]), dtype="float32")
                if len(w) >= int(0.5 * paths.SAMPLE_RATE):
                    rms = float(np.sqrt(np.mean(w ** 2)) + 1e-8)
                    self.fleurs_cache.append((w * (0.07 / rms)).astype(np.float32))

        if mode == "isolated":
            self._len = len(self.df)
        elif mode == "kws_train":
            self._len = samples_per_epoch
        else:
            # All val BAAD clips + 250 FLEURS unknown + 50 silence = ~1299 fixed val items
            self._n_unk = 250 if self.fleurs_cache else 0
            self._n_sil = 50
            self._len = len(self.df) + self._n_unk + self._n_sil
            self._val_items = self._build_deterministic_val()

    def _sample_fleurs_window(self, rng: np.random.Generator, length: int) -> np.ndarray:
        if not self.fleurs_cache:
            return synthetic_noise(length, rng) * 0.01
        chunks = []
        tot = 0
        while tot < length:
            w = self.fleurs_cache[int(rng.integers(0, len(self.fleurs_cache)))]
            chunks.append(w)
            tot += len(w)
        buf = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if len(buf) == length:
            return buf.copy()
        s = int(rng.integers(0, len(buf) - length + 1))
        return buf[s:s + length].copy()

    def _embed_word(
        self,
        word: np.ndarray,
        rng: np.random.Generator,
        use_speech_bg: bool,
        partial_edge: bool = False,
    ) -> np.ndarray:
        win = self.window_samples
        fade = min(int(0.020 * paths.SAMPLE_RATE), max(2, len(word) // 4))
        if use_speech_bg and self.fleurs_cache:
            bg = self._sample_fleurs_window(rng, win)
            bg_rms = float(np.sqrt(np.mean(bg ** 2)) + 1e-8)
            word_scaled = rms_match(word, max(0.04, min(0.18, bg_rms)), jitter_db=4.5, rng=rng)
        else:
            bg = synthetic_noise(win, rng) * float(rng.uniform(0.001, 0.012))
            word_scaled = rms_match(word, float(rng.uniform(0.05, 0.20)))

        if partial_edge:
            # Place < 25% of keyword at very left or right edge so detector learns not to fire prematurely
            keep = max(int(0.05 * paths.SAMPLE_RATE), int(len(word_scaled) * rng.uniform(0.10, 0.24)))
            out = bg.copy()
            if rng.random() < 0.5:
                seg = word_scaled[-keep:]
                out[:keep] = seg
            else:
                seg = word_scaled[:keep]
                out[-keep:] = seg
            return out.astype(np.float32)

        if len(word_scaled) >= win:
            start = int(rng.integers(0, len(word_scaled) - win + 1)) if len(word_scaled) > win else 0
            return word_scaled[start:start + win].astype(np.float32)

        max_off = win - len(word_scaled)
        offset = int(rng.integers(0, max_off + 1))
        out = bg.copy()
        if fade > 1:
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade, dtype=np.float32) / fade))
            mask = np.ones(len(word_scaled), dtype=np.float32)
            mask[:fade] = ramp
            mask[-fade:] = ramp[::-1]
        else:
            mask = np.ones(len(word_scaled), dtype=np.float32)
        out[offset:offset + len(word_scaled)] = (
            (1.0 - mask) * out[offset:offset + len(word_scaled)] + mask * word_scaled
        )
        return out.astype(np.float32)

    def _build_deterministic_val(self) -> list[tuple[np.ndarray, int]]:
        rng = np.random.default_rng(self.seed + 999)
        items: list[tuple[np.ndarray, int]] = []
        for idx in range(len(self.df)):
            w = self.baad_wavs[idx]
            y = int(self.baad_labels[idx])
            use_bg = (idx % 2 == 1) and bool(self.fleurs_cache)
            if use_bg:
                wav = self._embed_word(w, rng, use_speech_bg=True)
            else:
                wav = pad_or_center_crop(w, self.window_samples)
            items.append((wav.astype(np.float32), y))
        for _ in range(self._n_unk):
            items.append((self._sample_fleurs_window(rng, self.window_samples), self.unknown_id))
        for _ in range(self._n_sil):
            sil = synthetic_noise(self.window_samples, rng) * float(rng.uniform(0.0005, 0.015))
            items.append((sil.astype(np.float32), self.silence_id))
        return items

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.mode == "isolated":
            wav = pad_or_center_crop(self.baad_wavs[idx], self.window_samples)
            return torch.from_numpy(wav), int(self.baad_labels[idx])

        if self.mode == "kws_val":
            wav, y = self._val_items[idx]
            return torch.from_numpy(wav.copy()), y

        # kws_train: stochastic per-sample RNG seeded from numpy global/worker state
        rng = np.random.default_rng()
        r = float(rng.random())
        if r < 0.44:
            # Abusive keyword (class-balanced across 112 abusive words)
            c = self.abusive_classes[int(rng.integers(0, len(self.abusive_classes)))]
            pool = self.abusive_by_class[c]
            i = int(pool[int(rng.integers(0, len(pool)))])
            w = speed_perturb(self.baad_wavs[i], rng, 0.92, 1.08)
            wav = self._embed_word(w, rng, use_speech_bg=(rng.random() < 0.70))
            label = c
        elif r < 0.62:
            # Close non-abusive hard-negative keyword (class-balanced across 40 close words)
            c = self.close_classes[int(rng.integers(0, len(self.close_classes)))]
            pool = self.close_by_class[c]
            i = int(pool[int(rng.integers(0, len(pool)))])
            w = speed_perturb(self.baad_wavs[i], rng, 0.92, 1.08)
            wav = self._embed_word(w, rng, use_speech_bg=(rng.random() < 0.70))
            label = c
        elif r < 0.92:
            # _unknown_: normal Bangla speech (85%) or partial keyword edge (15%)
            if rng.random() < 0.15:
                i = int(rng.integers(0, len(self.baad_wavs)))
                wav = self._embed_word(self.baad_wavs[i], rng, use_speech_bg=True, partial_edge=True)
            else:
                wav = self._sample_fleurs_window(rng, self.window_samples)
            label = self.unknown_id
        else:
            # _silence_ / ambient room noise
            wav = synthetic_noise(self.window_samples, rng) * float(rng.uniform(0.0002, 0.015))
            label = self.silence_id

        if label != self.silence_id:
            if rng.random() < 0.25:
                wav = simple_reverb(wav, rng)
            if rng.random() < 0.60:
                wav = add_noise_snr(wav, rng, snr_db_range=(6.0, 26.0))
            wav = random_gain(wav, rng, db_range=(-10.0, 5.0))
            wav = np.clip(wav, -1.0, 1.0).astype(np.float32)

        return torch.from_numpy(wav), int(label)
