"""Streaming keyword-spotting detector (VAD + sliding window + smoothing + peak picking)."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from baad import paths
from baad.data.preprocess import load_mono_16k


class ModelBackend(Protocol):
    num_classes: int

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        ...


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return (e / (np.sum(e) + 1e-12)).astype(np.float32)


class PyTorchBackend:
    def __init__(self, ckpt_path: Path):
        import torch
        from baad.features import LogMel
        from baad.models.bcresnet import BCResNetBackbone, EndToEndKWS

        self._torch = torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        self.num_classes = int(ckpt["num_classes"])
        logmel = LogMel(sr=paths.SAMPLE_RATE, n_fft=320, win_ms=20.0, hop_ms=10.0, n_mels=int(cfg["n_mels"]))
        backbone = BCResNetBackbone(
            num_classes=self.num_classes,
            scale=int(cfg["scale"]),
            dropout=float(cfg["dropout"]),
        )
        self.model = EndToEndKWS(logmel=logmel, backbone=backbone, spec_augment=None)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model.eval()

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            x = self._torch.from_numpy(window.astype(np.float32))[None, :]
            logits = self.model(x)[0].cpu().numpy()
        return _softmax(logits)


class ONNXBackend:
    def __init__(self, onnx_path: Path, num_threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        out_shape = self.sess.get_outputs()[0].shape
        self.num_classes = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 0

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        logits = self.sess.run(None, {self.in_name: window.astype(np.float32)[None, :]})[0][0]
        if self.num_classes == 0:
            self.num_classes = len(logits)
        return _softmax(logits)


def load_backend(model_path: Path, num_threads: int = 2) -> ModelBackend:
    if model_path.suffix == ".onnx":
        return ONNXBackend(model_path, num_threads=num_threads)
    return PyTorchBackend(model_path)


class EnergyVAD:
    """Fast RMS + hangover gate so pure silence skips neural inference."""

    def __init__(self, rms_thresh: float = 0.005, hangover_hops: int = 8):
        self.rms_thresh = rms_thresh
        self.hangover_hops = hangover_hops
        self._hang = 0

    def reset(self) -> None:
        self._hang = 0

    def is_speech(self, chunk: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(chunk ** 2)) + 1e-12)
        if rms >= self.rms_thresh:
            self._hang = self.hangover_hops
            return True
        if self._hang > 0:
            self._hang -= 1
            return True
        return False


class StreamingDetector:
    """Sliding-window streaming detector emitting Option-A abusive-word events."""

    def __init__(
        self,
        backend: ModelBackend,
        labels: list[dict],
        window_sec: float = paths.WINDOW_SEC,
        hop_sec: float = paths.HOP_SEC,
        smooth_k: int = 3,
        threshold: float = 0.45,
        binary_threshold: float = 0.60,
        refractory_sec: float = 0.70,
        vad_rms_thresh: float = 0.005,
        show_close: bool = False,
    ):
        self.backend = backend
        self.labels = labels
        self.num_classes = len(labels)
        self.num_abusive = sum(1 for x in labels if x["is_abusive"])
        self.unknown_id = next(x["id"] for x in labels if x["word"] == paths.UNKNOWN_LABEL)
        self.silence_id = next(x["id"] for x in labels if x["word"] == paths.SILENCE_LABEL)

        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.window_samples = int(round(window_sec * paths.SAMPLE_RATE))
        self.hop_samples = int(round(hop_sec * paths.SAMPLE_RATE))

        self.smooth_k = max(1, int(smooth_k))
        self.threshold = float(threshold)
        self.binary_threshold = float(binary_threshold)
        self.refractory_sec = float(refractory_sec)
        self.show_close = show_close

        self.vad = EnergyVAD(rms_thresh=vad_rms_thresh, hangover_hops=8)
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(self.window_samples, dtype=np.float32)
        self.hist: deque[np.ndarray] = deque(maxlen=self.smooth_k)
        self.vad.reset()
        self.last_emit_end = -1e9
        self._pending: dict | None = None
        self._pending_hops = 0
        self.n_hops = 0
        self.n_vad_active = 0

    def _finalize_pending(self) -> list[dict]:
        if self._pending is None:
            return []
        ev = self._pending
        self._pending = None
        self._pending_hops = 0
        if ev["t_peak"] - self.last_emit_end < self.refractory_sec:
            return []
        self.last_emit_end = ev["t_peak"]
        # Center inside window minus moving-average smoothing lag
        lag = 0.5 * (self.smooth_k - 1) * self.hop_sec
        center = ev["t_peak"] - 0.5 * self.window_sec - lag
        start = max(0.0, center - 0.375)
        end = max(start + 0.25, center + 0.375)
        return [{
            "word": self.labels[ev["class_id"]]["word"],
            "label_id": int(ev["class_id"]),
            "is_abusive": bool(self.labels[ev["class_id"]]["is_abusive"]),
            "kind": self.labels[ev["class_id"]]["kind"],
            "start": round(start, 3),
            "end": round(end, 3),
            "score": round(float(ev["score"]), 4),
            "p_abusive": round(float(ev["p_abusive"]), 4),
        }]

    def push(self, chunk_16k: np.ndarray, t_end: float) -> tuple[list[dict], dict]:
        """Process one hop chunk (<= hop_samples). Returns (emitted_events, hop_status)."""
        n = len(chunk_16k)
        if n > 0:
            if n >= self.window_samples:
                self.buf[:] = chunk_16k[-self.window_samples:]
            else:
                self.buf[:-n] = self.buf[n:]
                self.buf[-n:] = chunk_16k

        self.n_hops += 1
        speech = self.vad.is_speech(chunk_16k)
        if not speech:
            p = np.zeros(self.num_classes, dtype=np.float32)
            p[self.silence_id] = 1.0
            self.hist.append(p)
            events = self._finalize_pending()
            return events, {"speech": False, "p_abusive": 0.0, "top_word": paths.SILENCE_LABEL, "top_score": 1.0}

        self.n_vad_active += 1
        p = self.backend.predict_probs(self.buf)
        self.hist.append(p)
        p_s = np.mean(self.hist, axis=0)

        p_abusive = float(p_s[:self.num_abusive].sum())
        best_ab_c = int(np.argmax(p_s[:self.num_abusive]))
        best_ab_score = float(p_s[best_ab_c])

        # Peak-confirm abusive trigger (Option A: only emit abusive words unless show_close=True)
        # Skip first ~0.95s while the ring buffer is mostly zero-padded.
        triggered = (t_end >= 0.95) and (
            (best_ab_score >= self.threshold)
            or (p_abusive >= self.binary_threshold and best_ab_score >= 0.65 * self.threshold)
        )
        # Strength score combines specific word confidence and overall abusive mass
        strength = max(best_ab_score, 0.85 * p_abusive if p_abusive >= self.binary_threshold else 0.0)

        emitted: list[dict] = []
        if triggered and (t_end - self.last_emit_end >= self.refractory_sec):
            if self._pending is None:
                self._pending = {
                    "t_peak": t_end,
                    "class_id": best_ab_c,
                    "score": best_ab_score,
                    "p_abusive": p_abusive,
                    "strength": strength,
                }
                self._pending_hops = 1
            elif strength >= self._pending["strength"] and self._pending_hops < 4:
                self._pending = {
                    "t_peak": t_end,
                    "class_id": best_ab_c,
                    "score": best_ab_score,
                    "p_abusive": p_abusive,
                    "strength": strength,
                }
                self._pending_hops += 1
            else:
                emitted.extend(self._finalize_pending())
        elif self._pending is not None:
            emitted.extend(self._finalize_pending())

        if self.show_close and not triggered:
            best_cl_rel = int(np.argmax(p_s[self.num_abusive:self.unknown_id]))
            cl_c = self.num_abusive + best_cl_rel
            cl_score = float(p_s[cl_c])
            if cl_score >= max(self.threshold, 0.50) and (t_end - self.last_emit_end >= self.refractory_sec):
                self.last_emit_end = t_end
                start = max(0.0, t_end - 1.00)
                end = max(start + 0.20, t_end - 0.25)
                emitted.append({
                    "word": self.labels[cl_c]["word"],
                    "label_id": int(cl_c),
                    "is_abusive": False,
                    "kind": "close",
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "score": round(cl_score, 4),
                    "p_abusive": round(p_abusive, 4),
                })

        top_c = int(np.argmax(p_s))
        return emitted, {
            "speech": True,
            "p_abusive": p_abusive,
            "top_word": self.labels[top_c]["word"],
            "top_score": float(p_s[top_c]),
        }

    def flush(self) -> list[dict]:
        return self._finalize_pending()

    def detect_array(self, wav_16k: np.ndarray) -> dict:
        self.reset()
        events: list[dict] = []
        n = len(wav_16k)
        # Pad 0.35 s trailing silence so words near end of file pass window center & flush
        padded = np.concatenate([wav_16k, np.zeros(int(0.35 * paths.SAMPLE_RATE), dtype=np.float32)])
        for pos in range(0, len(padded), self.hop_samples):
            chunk = padded[pos:pos + self.hop_samples]
            if len(chunk) < self.hop_samples:
                chunk = np.pad(chunk, (0, self.hop_samples - len(chunk)))
            t_end = (pos + self.hop_samples) / paths.SAMPLE_RATE
            evs, _ = self.push(chunk, t_end=t_end)
            events.extend(evs)
        events.extend(self.flush())
        abusive_events = [e for e in events if e["is_abusive"]]
        return {
            "duration": round(n / paths.SAMPLE_RATE, 3),
            "sentence_is_offensive": len(abusive_events) > 0,
            "num_abusive_events": len(abusive_events),
            "vad_active_ratio": round(self.n_vad_active / max(1, self.n_hops), 3),
            "events": events,
        }

    def detect_file(self, wav_path: str | Path) -> dict:
        wav = load_mono_16k(str(wav_path))
        res = self.detect_array(wav)
        res["file"] = str(wav_path)
        return res


def load_labels_file(labels_path: Path = paths.LABELS_JSON) -> list[dict]:
    return json.loads(labels_path.read_text(encoding="utf-8"))
