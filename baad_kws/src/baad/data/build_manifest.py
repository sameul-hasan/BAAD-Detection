"""Scan the BAAD folders and write data/manifest.csv + data/labels.json.

Columns: path, label, label_id, word_index, is_abusive, subset, speaker, sr, channels, duration
- speaker = 'a<N>' (abusive pool) or 'c<N>' (close-word pool), taken from the
  filename prefix '<N>_chunkK.wav' (the metadata 'spaker_id' is unique per clip).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf

from baad import paths
from baad.logging_utils import get_logger, log_event

_NUM_PREFIX = re.compile(r"^\s*(\d+)\s*\.\s*")


def normalize_label(folder_name: str) -> tuple[int, str]:
    """'10. চুতমারানি' -> (10, 'চুতমারানি')."""
    name = unicodedata.normalize("NFC", folder_name)
    m = _NUM_PREFIX.match(name)
    if not m:
        raise ValueError(f"Folder without numeric prefix: {folder_name!r}")
    label = re.sub(r"\s+", " ", name[m.end():]).strip()
    return int(m.group(1)), label


def speaker_of(path: Path, subset: str) -> str:
    stem = path.stem
    if "_chunk" not in stem:
        raise ValueError(f"Unexpected clip name: {path}")
    return ("a" if subset == "abusive" else "c") + stem.split("_chunk")[0].strip()


def scan(root: Path, subset: str) -> list[dict]:
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        idx, label = normalize_label(d.name)
        for f in sorted(d.glob("*.wav")):
            info = sf.info(str(f))
            rows.append(dict(
                path=str(f.relative_to(paths.DATASET_ROOT)),
                source_folder=d.name,
                label=label, word_index=idx, subset=subset,
                speaker_pool=subset,
                is_abusive=int(subset == "abusive"),
                speaker=speaker_of(f, subset),
                sr=info.samplerate, channels=info.channels,
                duration=round(info.frames / info.samplerate, 4),
            ))
    return rows


def build() -> pd.DataFrame:
    rows = scan(paths.ABUSIVE_DIR, "abusive") + scan(paths.CLOSE_DIR, "close")
    df = pd.DataFrame(rows)

    # Harmonize words that exist in both BAAD/ and Abusive Close Words/ (e.g. 'বলদ')
    # so the model never sees contradictory abusive=1 vs abusive=0 labels for the same word.
    abusive_words = set(df.loc[df["subset"] == "abusive", "label"])
    cross_mask = (df["subset"] == "close") & (df["label"].isin(abusive_words))
    harmonized_words = sorted(df.loc[cross_mask, "label"].unique())
    harmonized_clips = int(cross_mask.sum())
    if harmonized_clips > 0:
        a_word_idx = df.loc[df["subset"] == "abusive"].groupby("label")["word_index"].min().to_dict()
        df.loc[cross_mask, "subset"] = "abusive"
        df.loc[cross_mask, "is_abusive"] = 1
        df.loc[cross_mask, "word_index"] = df.loc[cross_mask, "label"].map(a_word_idx)

    # Detect merged duplicate folders within each subset (e.g. '21.বাল পাকনা' & '90.বাল পাকনা')
    dup_folders = (
        df.groupby(["subset", "label"])["source_folder"]
        .unique()
        .loc[lambda s: s.map(len) > 1]
        .to_dict()
    )

    df["label_key"] = df["subset"].str[0] + ":" + df["label"]
    order = (
        df.groupby(["label_key", "subset", "label"], as_index=False)["word_index"]
        .min()
        .sort_values(["subset", "word_index"])
    )
    keys = list(order["label_key"]) + [paths.UNKNOWN_LABEL, paths.SILENCE_LABEL]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate label keys")
    key_to_id = {k: i for i, k in enumerate(keys)}
    df["label_id"] = df["label_key"].map(key_to_id)

    labels = []
    for k in keys:
        if k.startswith(("a:", "c:")):
            labels.append(dict(id=key_to_id[k], key=k, word=k[2:], is_abusive=k.startswith("a:"),
                               kind="abusive" if k.startswith("a:") else "close"))
        else:
            labels.append(dict(id=key_to_id[k], key=k, word=k, is_abusive=False, kind="special"))
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["path", "source_folder", "label", "label_key", "label_id", "word_index",
            "is_abusive", "subset", "speaker_pool", "speaker", "sr", "channels", "duration"]
    df[cols].to_csv(paths.MANIFEST, index=False)
    paths.LABELS_JSON.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    df.attrs["dup_folders"] = {f"{k[0]}:{k[1]}": list(v) for k, v in dup_folders.items()}
    df.attrs["harmonized_words"] = harmonized_words
    df.attrs["harmonized_clips"] = harmonized_clips
    return df


if __name__ == "__main__":
    log = get_logger("build_manifest")
    df = build()
    summary_df = df.groupby("subset").agg(
        clips=("path", "size"),
        words=("label", "nunique"),
        speakers=("speaker", "nunique"),
        minutes=("duration", lambda s: round(s.sum() / 60, 2)),
    )
    sr_counts = df.sr.value_counts().to_dict()
    log.info("Built manifest: clips=%d, word_labels=%d, speakers=%d", len(df), df.label_id.nunique(), df.speaker.nunique())
    log.info("Subset breakdown:\n%s", summary_df.to_string())
    log.info("Sample rates: %s", sr_counts)
    log.info("Merged duplicate folders: %s | Harmonized cross-subset words: %s (%d clips)",
             df.attrs.get("dup_folders"), df.attrs.get("harmonized_words"), df.attrs.get("harmonized_clips", 0))
    log_event(
        "build_manifest",
        "Scanned BAAD dataset, harmonized duplicate/overlapping labels, and wrote manifest.csv + labels.json",
        metrics={
            "total_clips": int(len(df)),
            "abusive_clips": int((df.subset == "abusive").sum()),
            "close_clips": int((df.subset == "close").sum()),
            "abusive_words": int(df[df.subset == "abusive"].label.nunique()),
            "close_words": int(df[df.subset == "close"].label.nunique()),
            "total_classes_with_special": int(df.label_id.nunique() + 2),
            "speakers_total": int(df.speaker.nunique()),
            "harmonized_cross_subset_clips": df.attrs.get("harmonized_clips", 0),
            "raw_sample_rates": sr_counts,
        },
        details=(
            f"- Merged duplicate folders: `{df.attrs.get('dup_folders')}`\n"
            f"- Harmonized close->abusive overlapping words: `{df.attrs.get('harmonized_words')}` ({df.attrs.get('harmonized_clips', 0)} clips)\n\n"
            f"```text\n{summary_df.to_string()}\n```"
        ),
    )
