"""Speaker-independent train/val/test split (separately for abusive and close pools).

Writes data/splits/{train,val,test}.csv and adds a 'split' + 'gender' column to the manifest.
"""
from __future__ import annotations

import random
from pathlib import PurePosixPath

import openpyxl
import pandas as pd

from baad import paths
from baad.logging_utils import get_logger, log_event

SEED = 1337
FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
FORCE_TRAIN_MIN_CLIPS = 20   # speakers with fewer clips always go to train


def speaker_gender() -> dict[str, str]:
    out = {}
    for prefix, xlsx in [("a", paths.METADATA_DIR / "BAAD_Speaker_Information_WordWise.xlsx"),
                         ("c", paths.METADATA_DIR / "Abusive Close Words" / "BAAD_Close-Word_Spaker_info.xlsx")]:
        ws = openpyxl.load_workbook(xlsx, read_only=True).active
        rows = ws.iter_rows(values_only=True)
        header = [str(h).strip().lower() for h in next(rows)]
        ip, ig = header.index("path"), header.index("gender")
        for r in rows:
            if not r[ip]:
                continue
            spk = prefix + PurePosixPath(str(r[ip])).name.split("_chunk")[0].strip()
            g_val = str(r[ig]).strip().capitalize() if r[ig] is not None else "Unknown"
            out.setdefault(spk, g_val)
    return out


def assign(speakers: pd.DataFrame, rng: random.Random) -> dict[str, str]:
    """speakers: columns speaker, gender, n. Returns speaker -> split."""
    res = {}
    small = speakers[speakers.n < FORCE_TRAIN_MIN_CLIPS]
    res.update({s: "train" for s in small.speaker})
    big = speakers[speakers.n >= FORCE_TRAIN_MIN_CLIPS]
    for _, g in big.groupby("gender"):
        spk = list(g.speaker)
        rng.shuffle(spk)
        n = len(spk)
        n_test = max(1, round(n * FRACS["test"]))
        n_val = max(1, round(n * FRACS["val"]))
        for i, s in enumerate(spk):
            res[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return res


def main():
    log = get_logger("splits")
    df = pd.read_csv(paths.MANIFEST)
    gender = speaker_gender()
    df["gender"] = df.speaker.map(gender).fillna("Unknown")
    rng = random.Random(SEED)
    split = {}
    for _, pool in df.groupby("speaker_pool"):
        spk = pool.groupby("speaker").agg(gender=("gender", "first"), n=("path", "size")).reset_index()
        split.update(assign(spk, rng))
    df["split"] = df.speaker.map(split)
    df.to_csv(paths.MANIFEST, index=False)
    paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for s in FRACS:
        df[df.split == s].to_csv(paths.SPLITS_DIR / f"{s}.csv", index=False)
    summary_df = df.groupby(["speaker_pool", "split"]).agg(
        clips=("path", "size"),
        speakers=("speaker", "nunique"),
        female_ratio=("gender", lambda g: round(float((g == "Female").mean()), 3)),
    )
    missing_val = sorted(set(df.label_id) - set(df[df.split == "val"].label_id))
    missing_test = sorted(set(df.label_id) - set(df[df.split == "test"].label_id))
    missing_train = sorted(set(df.label_id) - set(df[df.split == "train"].label_id))
    metrics = {
        "seed": SEED,
        "train_clips": int((df.split == "train").sum()),
        "val_clips": int((df.split == "val").sum()),
        "test_clips": int((df.split == "test").sum()),
        "train_speakers": int(df[df.split == "train"].speaker.nunique()),
        "val_speakers": int(df[df.split == "val"].speaker.nunique()),
        "test_speakers": int(df[df.split == "test"].speaker.nunique()),
        "missing_labels_train": len(missing_train),
        "missing_labels_val": len(missing_val),
        "missing_labels_test": len(missing_test),
    }
    log.info("Split breakdown:\n%s", summary_df.to_string())
    log.info("Missing labels -> train:%s val:%s test:%s", missing_train, missing_val, missing_test)
    log_event(
        "splits",
        "Created speaker-independent train/val/test splits",
        metrics=metrics,
        details=f"```text\n{summary_df.to_string()}\n```",
    )


if __name__ == "__main__":
    main()
