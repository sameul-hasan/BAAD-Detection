"""Download Google FLEURS Bengali (bn_in) as background 'normal Bangla speech'.

Files come straight from the Hugging Face dataset repo (no loading script needed):
  data/bn_in/{train,dev,test}.tsv  and  data/bn_in/audio/{split}.tar.gz
Then they are extracted and resampled to 16 kHz mono into data/external/fleurs_bn/<split>/.
FLEURS 'dev' -> our 'val'. Run:  python -m baad.data.download_fleurs
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf

from baad import paths
from baad.data.preprocess import load_mono_16k
from baad.logging_utils import get_logger, log_event

BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/bn_in"
SPLITS = {"train": "train", "dev": "val", "test": "test"}
OUT = paths.EXTERNAL_DIR / "fleurs_bn"
CACHE = paths.EXTERNAL_DIR / "_downloads"


def fetch(url: str, dst: Path, log=None):
    if dst.exists() and dst.stat().st_size > 0:
        if log:
            log.info("Cached: %s (%.1f MB)", dst.name, dst.stat().st_size / 1e6)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    msg = f"downloading {url}"
    log.info(msg) if log else print(msg, flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "baad-kws/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:8.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dst)
    if log:
        log.info("Downloaded %s (%.2f MB)", dst.name, dst.stat().st_size / 1e6)


def convert(args):
    src, dst = args
    x = load_mono_16k(src)
    sf.write(dst, x, paths.SAMPLE_RATE, subtype="PCM_16")
    return len(x) / paths.SAMPLE_RATE


def download_only():
    log = get_logger("download_fleurs")
    sizes_mb = {}
    for fs in SPLITS:
        tsv = CACHE / f"{fs}.tsv"
        tar = CACHE / f"{fs}.tar.gz"
        fetch(f"{BASE}/{fs}.tsv", tsv, log=log)
        fetch(f"{BASE}/audio/{fs}.tar.gz", tar, log=log)
        sizes_mb[f"{fs}_tsv_mb"] = round(tsv.stat().st_size / 1e6, 3)
        sizes_mb[f"{fs}_tar_mb"] = round(tar.stat().st_size / 1e6, 2)
    log_event("download_fleurs_fetch", "Downloaded FLEURS Bengali (bn_in) archives", metrics=sizes_mb)


def extract_and_convert():
    log = get_logger("download_fleurs")
    stats = {}
    for fs, ours in SPLITS.items():
        out_dir = OUT / ours
        existing = sorted(out_dir.glob("*.wav")) if out_dir.exists() else []
        if existing:
            durs = [sf.info(str(p)).frames / sf.info(str(p)).samplerate for p in existing]
            log.info("Exists %s: %d files, %.2f h", out_dir, len(durs), sum(durs) / 3600)
            stats[f"{ours}_files"] = len(durs)
            stats[f"{ours}_hours"] = round(sum(durs) / 3600, 3)
            continue
        tmp = CACHE / f"_x_{fs}"
        if tmp.exists():
            shutil.rmtree(tmp)
        log.info("Extracting %s...", CACHE / f"{fs}.tar.gz")
        with tarfile.open(CACHE / f"{fs}.tar.gz") as t:
            t.extractall(tmp)
        out_dir.mkdir(parents=True, exist_ok=True)
        srcs = sorted(tmp.rglob("*.wav"))
        jobs = [(str(s), str(out_dir / s.name)) for s in srcs]
        with ProcessPoolExecutor() as ex:
            durs = list(ex.map(convert, jobs, chunksize=16))
        shutil.rmtree(tmp)
        shutil.copy(CACHE / f"{fs}.tsv", out_dir / "transcripts.tsv")
        stats[f"{ours}_files"] = len(durs)
        stats[f"{ours}_hours"] = round(sum(durs) / 3600, 3)
        log.info("%s: %d files, %.2f h", ours, len(durs), sum(durs) / 3600)
    log_event("download_fleurs_convert", "Extracted and resampled FLEURS bn_in to 16 kHz mono", metrics=stats)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("download", "all"):
        download_only()
    if step in ("convert", "all"):
        extract_and_convert()
