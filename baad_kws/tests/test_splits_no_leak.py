import pandas as pd

from baad import paths


def test_speaker_independent_splits():
    tr = pd.read_csv(paths.SPLITS_DIR / "train.csv")
    va = pd.read_csv(paths.SPLITS_DIR / "val.csv")
    te = pd.read_csv(paths.SPLITS_DIR / "test.csv")
    df = pd.read_csv(paths.MANIFEST)

    assert len(tr) + len(va) + len(te) == len(df)

    s_tr, s_va, s_te = set(tr["speaker"]), set(va["speaker"]), set(te["speaker"])
    assert len(s_tr & s_va) == 0, f"train/val speaker leak: {s_tr & s_va}"
    assert len(s_tr & s_te) == 0, f"train/test speaker leak: {s_tr & s_te}"
    assert len(s_va & s_te) == 0, f"val/test speaker leak: {s_va & s_te}"

    # Every word class must be represented in training
    assert set(tr["label_id"]) == set(df["label_id"])
    # Both abusive and close pools must appear in val and test
    assert set(va["speaker_pool"]) == {"abusive", "close"}
    assert set(te["speaker_pool"]) == {"abusive", "close"}
    assert set(va["subset"]) == {"abusive", "close"}
    assert set(te["subset"]) == {"abusive", "close"}
