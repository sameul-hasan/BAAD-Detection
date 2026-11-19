import json
import pandas as pd
import soundfile as sf

from baad import paths


def test_synthetic_streams_and_no_speaker_leak():
    for split in ("val", "test"):
        jsonl = paths.SYNTH_DIR / split / "streams.jsonl"
        assert jsonl.exists(), f"Missing {jsonl}"
        records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) == 250

        split_speakers = set(pd.read_csv(paths.SPLITS_DIR / f"{split}.csv")["speaker"])
        other_splits = {"train", "val", "test"} - {split}
        forbidden_speakers = set()
        for osplit in other_splits:
            forbidden_speakers.update(pd.read_csv(paths.SPLITS_DIR / f"{osplit}.csv")["speaker"])

        pos_count = 0
        neg_count = 0
        for rec in records:
            wav_path = paths.DATA_DIR / rec["audio"]
            assert wav_path.exists()
            info = sf.info(str(wav_path))
            assert info.samplerate == paths.SAMPLE_RATE
            assert info.channels == 1
            dur = info.frames / info.samplerate
            assert abs(dur - rec["duration"]) < 0.01

            if rec["has_abusive"]:
                pos_count += 1
            else:
                neg_count += 1

            prev_end = -1.0
            for ev in rec["events"]:
                assert 0.0 <= ev["start"] < ev["end"] <= rec["duration"] + 1e-4
                assert ev["start"] >= prev_end
                prev_end = ev["end"]
                assert ev["speaker"] in split_speakers
                assert ev["speaker"] not in forbidden_speakers

        assert pos_count == 150
        assert neg_count == 100
