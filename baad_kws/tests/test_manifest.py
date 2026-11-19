import json
import pandas as pd
import soundfile as sf

from baad import paths


def test_manifest_and_processed_files():
    assert paths.MANIFEST.exists()
    assert paths.LABELS_JSON.exists()

    df = pd.read_csv(paths.MANIFEST)
    assert len(df) == 5262 + 824 == 6086
    assert (df["subset"] == "abusive").sum() == 5285
    assert (df["subset"] == "close").sum() == 801
    assert (df["speaker_pool"] == "abusive").sum() == 5262
    assert (df["speaker_pool"] == "close").sum() == 824

    labels = json.loads(paths.LABELS_JSON.read_text(encoding="utf-8"))
    assert len(labels) == 154
    abusive_labels = [x for x in labels if x["kind"] == "abusive"]
    close_labels = [x for x in labels if x["kind"] == "close"]
    special_labels = [x for x in labels if x["kind"] == "special"]
    assert len(abusive_labels) == 112
    assert len(close_labels) == 40
    assert len(special_labels) == 2
    assert [x["id"] for x in labels] == list(range(len(labels)))
    assert [x["id"] for x in abusive_labels] == list(range(112))
    assert set(x["word"] for x in abusive_labels).isdisjoint(x["word"] for x in close_labels)
    assert labels[-2]["word"] == paths.UNKNOWN_LABEL
    assert labels[-1]["word"] == paths.SILENCE_LABEL

    assert "proc_path" in df.columns
    for rel in df["proc_path"]:
        p = paths.DATA_DIR / rel
        assert p.exists(), f"Missing processed file: {p}"
    # Spot check format on sample files
    for rel in list(df["proc_path"].iloc[:25]) + list(df["proc_path"].iloc[-25:]):
        info = sf.info(str(paths.DATA_DIR / rel))
        assert info.samplerate == paths.SAMPLE_RATE
        assert info.channels == 1
        assert info.frames > 0
