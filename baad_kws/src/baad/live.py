"""Desktop live-microphone & audio-file marking demo using the exported INT8 ONNX model."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from baad import paths
from baad.data.preprocess import load_mono_16k
from baad.detector import StreamingDetector, load_backend, load_labels_file
from baad.logging_utils import configure_matplotlib, ensure_logs_dir, get_logger, log_event


def _append_detection_log(record: dict) -> None:
    ensure_logs_dir()
    rec = {"ts": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), **record}
    with open(paths.LOGS_DIR / "detections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class MicStream:
    """16 kHz mono live mic reader: tries sounddevice, falls back to pw-record/parecord/arecord."""

    def __init__(self, sr: int = paths.SAMPLE_RATE, hop_samples: int = paths.HOP_SAMPLES):
        self.sr = sr
        self.hop_samples = hop_samples
        self._sd_stream = None
        self._proc = None
        self.backend_name = ""

    def __enter__(self):
        try:
            import sounddevice as sd
            self._sd_stream = sd.InputStream(
                samplerate=self.sr, channels=1, dtype="float32", blocksize=self.hop_samples
            )
            self._sd_stream.start()
            self.backend_name = "sounddevice"
            return self
        except (ImportError, OSError):
            self._sd_stream = None

        cmd = None
        if shutil.which("pw-record"):
            cmd = ["pw-record", "--format", "s16", "--rate", str(self.sr), "--channels", "1", "-"]
            self.backend_name = "pw-record"
        elif shutil.which("parecord"):
            cmd = ["parecord", "--raw", "--format=s16le", f"--rate={self.sr}", "--channels=1"]
            self.backend_name = "parecord"
        elif shutil.which("arecord"):
            cmd = ["arecord", "-q", "-f", "S16_LE", "-r", str(self.sr), "-c", "1", "-t", "raw"]
            self.backend_name = "arecord"
        else:
            raise RuntimeError("No audio capture backend available (PortAudio/pw-record/parecord/arecord missing)")

        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return self

    def read_hop(self) -> np.ndarray | None:
        if self._sd_stream is not None:
            data, _ = self._sd_stream.read(self.hop_samples)
            return data[:, 0].astype(np.float32)
        assert self._proc is not None and self._proc.stdout is not None
        need = self.hop_samples * 2  # int16 = 2 bytes
        buf = bytearray()
        while len(buf) < need:
            chunk = self._proc.stdout.read(need - len(buf))
            if not chunk:
                break
            buf.extend(chunk)
        if not buf:
            return None
        arr = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        return arr

    def __exit__(self, *args):
        if self._sd_stream is not None:
            self._sd_stream.stop()
            self._sd_stream.close()
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait(timeout=2)


def _bengali_font():
    from matplotlib.font_manager import FontProperties
    for fp in (
        "/usr/share/fonts/truetype/lohit-bengali/Lohit-Bengali.ttf",
        "/usr/share/fonts/truetype/fonts-beng-extra/Mukti.ttf",
    ):
        if Path(fp).exists():
            return FontProperties(fname=fp)
    return None


def save_marked_plot(wav: np.ndarray, events: list[dict], out_path: Path, title: str = "") -> None:
    configure_matplotlib(headless=True)
    import matplotlib.pyplot as plt

    bn_font = _bengali_font()
    t = np.arange(len(wav), dtype=np.float32) / paths.SAMPLE_RATE
    fig, ax = plt.subplots(figsize=(13, 4.0))
    ax.plot(t, wav, color="#3b5b75", linewidth=0.6, alpha=0.85)
    ymax = max(0.2, float(np.max(np.abs(wav))) * 1.15)
    ax.set_ylim(-ymax, ymax)

    for i, ev in enumerate(events):
        color = "#d62728" if ev["is_abusive"] else "#e6ab02"
        tag = f"{ev['word']} ({ev['score']:.2f})"
        ax.axvspan(ev["start"], ev["end"], color=color, alpha=0.25)
        y_pos = ymax * (0.88 if i % 2 == 0 else 0.70)
        ax.text(
            0.5 * (ev["start"] + ev["end"]),
            y_pos,
            tag,
            color="#900000" if ev["is_abusive"] else "#7a5900",
            fontsize=10,
            fontweight="bold",
            fontproperties=bn_font,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or f"Detected Abusive Events: {sum(1 for e in events if e['is_abusive'])}")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _fmt_time(s: float) -> str:
    m, sec = divmod(s, 60.0)
    return f"{int(m):02d}:{sec:04.1f}"


def build_detector(
    export_dir: Path,
    model_override: Path | None = None,
    threshold_override: float | None = None,
    show_close: bool = False,
) -> StreamingDetector:
    model_path = model_override or (export_dir / "baad_kws_int8.onnx")
    if not model_path.exists():
        model_path = export_dir / "baad_kws_fp32.onnx"
    labels = load_labels_file(export_dir / "labels.json")
    cfg_path = export_dir / "detect_tuned.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    backend = load_backend(model_path, num_threads=2)
    return StreamingDetector(
        backend=backend,
        labels=labels,
        window_sec=float(cfg.get("window_sec", paths.WINDOW_SEC)),
        hop_sec=float(cfg.get("hop_sec", paths.HOP_SEC)),
        smooth_k=int(cfg.get("smooth_k", 3)),
        threshold=float(threshold_override if threshold_override is not None else cfg.get("threshold", 0.45)),
        binary_threshold=float(cfg.get("binary_threshold", 0.65)),
        refractory_sec=float(cfg.get("refractory_sec", 0.70)),
        vad_rms_thresh=float(cfg.get("vad_rms_thresh", 0.005)),
        show_close=show_close,
    )


def run_file_mode(detector: StreamingDetector, wav_path: Path, json_out: Path | None, plot_out: Path | None, log):
    wav = load_mono_16k(str(wav_path))
    res = detector.detect_array(wav)
    res["file"] = str(wav_path)
    log.info(
        "File %s (%.2fs): offensive=%s events=%d",
        wav_path.name, res["duration"], res["sentence_is_offensive"], res["num_abusive_events"],
    )
    for ev in res["events"]:
        tag = "abusive" if ev["is_abusive"] else "close"
        print(f"  [{_fmt_time(ev['start'])} - {_fmt_time(ev['end'])}] {ev['word']} – {tag} (score={ev['score']:.3f})")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    if plot_out is not None:
        save_marked_plot(
            wav, res["events"], plot_out,
            title=f"{wav_path.name} — offensive={res['sentence_is_offensive']} ({res['num_abusive_events']} abusive marks)",
        )
    _append_detection_log({"mode": "file", **res})
    log_event(
        "live_demo_file",
        f"Marked {wav_path.name}: offensive={res['sentence_is_offensive']} ({res['num_abusive_events']} events)",
        metrics={"duration_s": res["duration"], "num_abusive_events": res["num_abusive_events"], "vad_active_ratio": res["vad_active_ratio"]},
    )


def run_live_mode(detector: StreamingDetector, gui: bool, log):
    log.info("Starting live microphone detection (Ctrl+C to stop)...")
    session_events: list[dict] = []
    plt_ctx = None
    if gui:
        configure_matplotlib(headless=False)
        import matplotlib.pyplot as plt
        plt.ion()
        fig, ax = plt.subplots(figsize=(11, 3.6))
        plt_ctx = (plt, fig, ax, deque(maxlen=int(10 * paths.SAMPLE_RATE)))

    bn_font = _bengali_font()
    t0 = time.monotonic()
    try:
        with MicStream() as mic:
            log.info("Microphone capture backend: %s (16 kHz mono)", mic.backend_name)
            t_cur = 0.0
            while True:
                chunk = mic.read_hop()
                if chunk is None:
                    break
                t_cur += len(chunk) / paths.SAMPLE_RATE
                evs, status = detector.push(chunk, t_end=t_cur)
                if plt_ctx is not None:
                    plt_ctx[3].extend(chunk.tolist())

                bar = "#" * min(12, int(np.sqrt(np.mean(chunk ** 2)) * 120))
                sys.stdout.write(
                    f"\r[{_fmt_time(t_cur)}] VAD={'SPEECH' if status['speech'] else 'quiet ':6s} "
                    f"|{bar:<12}| p_ab={status['p_abusive']:.2f}   "
                )
                sys.stdout.flush()

                for ev in evs:
                    session_events.append(ev)
                    tag = "abusive" if ev["is_abusive"] else "close"
                    sys.stdout.write(
                        f"\n\033[1;31m[MARK {_fmt_time(ev['start'])} - {_fmt_time(ev['end'])}] "
                        f"{ev['word']} – {tag} (score={ev['score']:.3f})\033[0m\n"
                    )
                    sys.stdout.flush()

                if plt_ctx is not None and (detector.n_hops % 3 == 0):
                    plt, fig, ax, ring = plt_ctx
                    arr = np.array(ring, dtype=np.float32)
                    ax.clear()
                    t_end = t_cur
                    t_start = max(0.0, t_end - len(arr) / paths.SAMPLE_RATE)
                    xs = np.linspace(t_start, t_end, len(arr))
                    ax.plot(xs, arr, color="#3b5b75", linewidth=0.6)
                    for ev in session_events[-10:]:
                        if ev["end"] >= t_start:
                            ax.axvspan(ev["start"], ev["end"], color="#d62728", alpha=0.28)
                            ax.text(ev["start"], 0.75, ev["word"], color="#900000", fontweight="bold", fontproperties=bn_font)
                    ax.set_ylim(-0.9, 0.9)
                    ax.set_xlim(t_start, max(10.0, t_end))
                    plt.pause(0.001)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        rem = detector.flush()
        for ev in rem:
            session_events.append(ev)
            print(f"[MARK {_fmt_time(ev['start'])} - {_fmt_time(ev['end'])}] {ev['word']} – abusive (score={ev['score']:.3f})")
        _append_detection_log({
            "mode": "live_mic",
            "elapsed_s": round(time.monotonic() - t0, 2),
            "events": session_events,
        })
        log.info("Live session ended: %d total detections logged.", len(session_events))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--file", type=Path, default=None, help="Run streaming detector on WAV file instead of mic")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--save-plot", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--show-close", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    log = get_logger("live")
    det = build_detector(args.export_dir, args.model, args.threshold, args.show_close)
    if args.file is not None:
        run_file_mode(det, args.file, args.json, args.save_plot, log)
    else:
        run_live_mode(det, args.gui, log)


if __name__ == "__main__":
    main()
