"""Export-friendly log-Mel front-end using only Conv1d (no Unfold/STFT).

Uses win=320 (20 ms), n_fft=320 (161 bins), hop=160 (10 ms), n_mels=40 so the
DFT/Mel weights are tiny (~103 KB in INT8) and export cleanly to ONNX opset 17
and INT8 quantization.
Input:  [B, samples] float32 waveform at 16 kHz.
Output: [B, 1, n_mels, frames] float32 log-Mel spectrogram.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def mel_filterbank(n_fft: int, n_mels: int, sr: int, fmin: float, fmax: float) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mpts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hpts = mel2hz(mpts)
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = hpts[i], hpts[i + 1], hpts[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
    fb *= (2.0 / (hpts[2:] - hpts[:-2]))[:, None]
    return fb


class LogMel(nn.Module):
    def __init__(
        self,
        sr: int = 16000,
        n_fft: int = 320,
        win_ms: float = 20.0,
        hop_ms: float = 10.0,
        n_mels: int = 40,
        fmin: float = 20.0,
        fmax: float = 7600.0,
    ):
        super().__init__()
        self.win = int(round(sr * win_ms / 1000))
        self.hop = int(round(sr * hop_ms / 1000))
        self.n_fft = n_fft
        self.n_mels = n_mels
        n_bins = n_fft // 2 + 1

        window = torch.hann_window(self.win, periodic=True, dtype=torch.float64)
        k = torch.arange(n_bins, dtype=torch.float64)[:, None]
        n = torch.arange(self.win, dtype=torch.float64)[None, :]
        ang = 2.0 * math.pi * k * n / n_fft

        # [n_bins, 1, win] Conv1d kernels with Hann window folded in
        cos_k = (torch.cos(ang) * window[None, :]).float().unsqueeze(1).contiguous()
        sin_k = (-torch.sin(ang) * window[None, :]).float().unsqueeze(1).contiguous()
        mel_k = torch.from_numpy(mel_filterbank(n_fft, n_mels, sr, fmin, fmax)).unsqueeze(-1).contiguous()

        self.register_buffer("cos_k", cos_k)
        self.register_buffer("sin_k", sin_k)
        self.register_buffer("mel_k", mel_k)

    def num_frames(self, n_samples: int) -> int:
        return 1 + (n_samples - self.win) // self.hop

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # Per-window DC removal keeps inference invariant to microphone DC offset
        wav = wav - wav.mean(dim=-1, keepdim=True)
        x = wav.unsqueeze(1)                                    # [B, 1, N]
        re = F.conv1d(x, self.cos_k, stride=self.hop)           # [B, n_bins, T]
        im = F.conv1d(x, self.sin_k, stride=self.hop)           # [B, n_bins, T]
        power = re * re + im * im
        mel = F.conv1d(power, self.mel_k, stride=1)             # [B, n_mels, T]
        logmel = torch.log(mel + 1e-6)
        return logmel.unsqueeze(1)                              # [B, 1, n_mels, T]
