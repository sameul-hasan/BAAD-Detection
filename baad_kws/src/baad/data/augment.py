"""Waveform and Mel-spectrogram augmentation utilities for low-resource KWS."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import fftconvolve, lfilter
from torch import nn

from baad import paths


def speed_perturb(wav: np.ndarray, rng: np.random.Generator, low: float = 0.90, high: float = 1.10) -> np.ndarray:
    """Fast time-stretch via linear interpolation (factor > 1.0 = faster/shorter)."""
    rate = float(rng.uniform(low, high))
    if abs(rate - 1.0) < 0.01 or len(wav) < 16:
        return wav
    n_out = max(16, int(round(len(wav) / rate)))
    x_old = np.linspace(0.0, 1.0, len(wav), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def random_gain(wav: np.ndarray, rng: np.random.Generator, db_range: tuple[float, float] = (-12.0, 6.0)) -> np.ndarray:
    db = float(rng.uniform(*db_range))
    return (wav * (10.0 ** (db / 20.0))).astype(np.float32)


def synthetic_noise(length: int, rng: np.random.Generator, sr: int = paths.SAMPLE_RATE) -> np.ndarray:
    """Generate realistic ambient background noise (white + pink/brown + faint AC hum)."""
    white = rng.standard_normal(length).astype(np.float32)
    # One-pole low-pass for brown/pink-like room rumble
    b = np.array([0.08], dtype=np.float32)
    a = np.array([1.0, -0.92], dtype=np.float32)
    low = lfilter(b, a, white).astype(np.float32)
    w_mix = float(rng.uniform(0.15, 0.55))
    noise = w_mix * white + (1.0 - w_mix) * low

    if rng.random() < 0.5:
        t = np.arange(length, dtype=np.float32) / sr
        f0 = float(rng.choice([50.0, 60.0, 100.0]))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        hum = float(rng.uniform(0.02, 0.12)) * np.sin(2.0 * np.pi * f0 * t + phase)
        noise = noise + hum.astype(np.float32)

    std = float(np.sqrt(np.mean(noise ** 2)) + 1e-8)
    return (noise / std).astype(np.float32)


def add_noise_snr(
    wav: np.ndarray,
    rng: np.random.Generator,
    snr_db_range: tuple[float, float] = (6.0, 25.0),
) -> np.ndarray:
    noise = synthetic_noise(len(wav), rng)
    sig_rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-8)
    snr_db = float(rng.uniform(*snr_db_range))
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    out = wav + noise * noise_rms
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def simple_reverb(wav: np.ndarray, rng: np.random.Generator, sr: int = paths.SAMPLE_RATE) -> np.ndarray:
    """Lightweight synthetic exponential-decay room impulse response."""
    rt60 = float(rng.uniform(0.08, 0.35))
    n_rir = int(rt60 * sr)
    if n_rir < 32:
        return wav
    t = np.arange(n_rir, dtype=np.float32) / sr
    decay = np.exp(-6.91 * t / rt60)
    rir = rng.standard_normal(n_rir).astype(np.float32) * decay * float(rng.uniform(0.15, 0.40))
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2) + 1e-8)
    wet = fftconvolve(wav, rir, mode="full")[:len(wav)].astype(np.float32)
    peak = float(np.max(np.abs(wet)) + 1e-8)
    if peak > 0.99:
        wet *= 0.95 / peak
    return wet


def rms_match(src: np.ndarray, target_rms: float, jitter_db: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    src_rms = float(np.sqrt(np.mean(src ** 2)) + 1e-8)
    gain = target_rms / src_rms
    if jitter_db > 0.0 and rng is not None:
        gain *= 10.0 ** (float(rng.uniform(-jitter_db, jitter_db)) / 20.0)
    return (src * gain).astype(np.float32)


class SpecAugment(nn.Module):
    """Train-only SpecAugment on [B, 1, F, T] log-Mel spectrograms."""

    def __init__(self, time_masks: int = 2, max_time: int = 15, freq_masks: int = 2, max_freq: int = 6):
        super().__init__()
        self.time_masks = time_masks
        self.max_time = max_time
        self.freq_masks = freq_masks
        self.max_freq = max_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        b, _, n_freq, n_time = x.shape
        fill = x.amin(dim=(-2, -1), keepdim=True)
        for i in range(b):
            for _ in range(self.freq_masks):
                w = int(torch.randint(0, min(self.max_freq, n_freq) + 1, (1,), device=x.device).item())
                if w > 0:
                    f0 = int(torch.randint(0, n_freq - w + 1, (1,), device=x.device).item())
                    x[i, :, f0:f0 + w, :] = fill[i]
            for _ in range(self.time_masks):
                w = int(torch.randint(0, min(self.max_time, n_time) + 1, (1,), device=x.device).item())
                if w > 0:
                    t0 = int(torch.randint(0, n_time - w + 1, (1,), device=x.device).item())
                    x[i, :, :, t0:t0 + w] = fill[i]
        return x
