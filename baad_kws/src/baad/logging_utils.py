"""Central structured logging for every stage of the BAAD KWS pipeline.

Maintains:
  - logs/master.log       (all stages, chronological text log)
  - logs/<stage>.log      (per-stage detailed log)
  - logs/events.jsonl     (machine-readable JSONL event stream)
  - logs/EXPERIMENT_LOG.md (human-readable Markdown experiment notebook)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from baad import paths

_LOGGERS: dict[str, logging.Logger] = {}
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_logs_dir() -> Path:
    paths.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return paths.LOGS_DIR


def get_logger(stage: str, log_file: Path | None = None) -> logging.Logger:
    """Return a UTF-8 logger writing to stdout, logs/master.log, and logs/<stage>.log."""
    key = f"{stage}:{log_file}"
    if key in _LOGGERS:
        return _LOGGERS[key]

    ensure_logs_dir()
    logger = logging.getLogger(f"baad.{stage}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    master_fh = logging.FileHandler(paths.LOGS_DIR / "master.log", encoding="utf-8")
    master_fh.setFormatter(formatter)
    logger.addHandler(master_fh)

    stage_path = log_file if log_file is not None else (paths.LOGS_DIR / f"{stage}.log")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_fh = logging.FileHandler(stage_path, encoding="utf-8")
    stage_fh.setFormatter(formatter)
    logger.addHandler(stage_fh)

    _LOGGERS[key] = logger
    return logger


def init_experiment_log(force: bool = False) -> Path:
    ensure_logs_dir()
    md_path = paths.LOGS_DIR / "EXPERIMENT_LOG.md"
    if force or not md_path.exists() or md_path.stat().st_size == 0:
        header = (
            "# BAAD Keyword Spotting — Experiment & Pipeline Log\n\n"
            "Chronological record of environment checks, data preprocessing, "
            "speaker splits, synthetic continuous streams, baselines, student training, "
            "threshold tuning, evaluation, ONNX/INT8 export, and live/file verification.\n\n"
            "---\n\n"
        )
        md_path.write_text(header, encoding="utf-8")
    return md_path


def _format_val(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.2f}"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def log_event(
    stage: str,
    summary: str,
    metrics: dict[str, Any] | None = None,
    details: str | None = None,
) -> None:
    """Append an event to logs/events.jsonl and logs/EXPERIMENT_LOG.md."""
    ensure_logs_dir()
    init_experiment_log(force=False)
    ts = _now_iso()

    record = {
        "ts": ts,
        "stage": stage,
        "summary": summary,
        "metrics": metrics or {},
    }
    if details:
        record["details"] = details

    jsonl_path = paths.LOGS_DIR / "events.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [f"## [{ts}] `{stage}` — {summary}\n\n"]
    if metrics:
        lines.append("| Metric | Value |\n|---|---|\n")
        for k, v in metrics.items():
            lines.append(f"| `{k}` | `{_format_val(v)}` |\n")
        lines.append("\n")
    if details:
        lines.append(f"{details.rstrip()}\n\n")
    lines.append("---\n\n")

    md_path = paths.LOGS_DIR / "EXPERIMENT_LOG.md"
    with open(md_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def configure_matplotlib(headless: bool = True) -> None:
    """Configure Matplotlib temp cache, Agg backend (if headless), and Bangla fonts."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    if headless:
        matplotlib.use("Agg", force=True)
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Lohit Bengali", "Mukti"]
    matplotlib.rcParams["axes.unicode_minus"] = False
