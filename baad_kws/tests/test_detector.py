import json
import numpy as np

from baad import paths
from baad.detector import StreamingDetector


class MockBackend:
    """Mock model returning high probability for a chosen word ID when signal energy is centered."""

    def __init__(self, num_classes: int, target_id: int, fire_at_sec: float, win_samples: int):
        self.num_classes = num_classes
        self.target_id = target_id
        self.unknown_id = num_classes - 2
        self.fire_hop = int(round(fire_at_sec / paths.HOP_SEC))
        self.calls = 0

    def predict_probs(self, window: np.ndarray) -> np.ndarray:
        self.calls += 1
        p = np.full(self.num_classes, 1e-4, dtype=np.float32)
        if abs(self.calls - self.fire_hop) <= 2:
            p[self.target_id] = 0.88
        else:
            p[self.unknown_id] = 0.90
        return p / p.sum()


def test_streaming_detector_marks_target_and_ignores_clean():
    labels = json.loads(paths.LABELS_JSON.read_text(encoding="utf-8"))
    target_id = 1  # কুত্তা
    # Build 6 s speech-level noise signal, expecting mock detection around t=3.2 s
    wav = (0.05 * np.sin(2 * np.pi * 220.0 * np.arange(6 * paths.SAMPLE_RATE) / paths.SAMPLE_RATE)).astype(np.float32)

    backend = MockBackend(len(labels), target_id=target_id, fire_at_sec=3.2, win_samples=paths.WINDOW_SAMPLES)
    det = StreamingDetector(backend, labels, smooth_k=2, threshold=0.45, binary_threshold=0.60, refractory_sec=0.70)
    res = det.detect_array(wav)

    assert res["sentence_is_offensive"] is True
    assert res["num_abusive_events"] == 1
    ev = res["events"][0]
    assert ev["label_id"] == target_id
    assert ev["word"] == labels[target_id]["word"]
    assert ev["is_abusive"] is True
    assert 2.0 <= ev["start"] < ev["end"] <= 4.0

    # Clean stream with target_id never triggered
    clean_backend = MockBackend(len(labels), target_id=target_id, fire_at_sec=99.0, win_samples=paths.WINDOW_SAMPLES)
    det_clean = StreamingDetector(clean_backend, labels, smooth_k=2, threshold=0.45, binary_threshold=0.60)
    res_clean = det_clean.detect_array(wav)
    assert res_clean["sentence_is_offensive"] is False
    assert res_clean["num_abusive_events"] == 0
