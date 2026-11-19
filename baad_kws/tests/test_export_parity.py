import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from baad import paths
from baad.features import LogMel
from baad.models.bcresnet import BCResNetBackbone, EndToEndKWS


def _find_export_dir() -> Path | None:
    if not paths.RUNS_DIR.exists():
        return None
    for d in sorted(paths.RUNS_DIR.iterdir()):
        ed = d / "export"
        if (ed / "baad_kws_fp32.onnx").exists() and (ed / "baad_kws_int8.onnx").exists():
            return ed
    return None


def test_export_parity_and_size():
    ed = _find_export_dir()
    if ed is None:
        pytest.skip("No exported run found yet")

    fp32 = ed / "baad_kws_fp32.onnx"
    int8 = ed / "baad_kws_int8.onnx"
    run_dir = ed.parent

    assert int8.stat().st_size < 1024 * 1024, f"INT8 model exceeds 1 MB: {int8.stat().st_size} bytes"

    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    logmel = LogMel(sr=paths.SAMPLE_RATE, n_fft=320, win_ms=20.0, hop_ms=10.0, n_mels=int(cfg["n_mels"]))
    backbone = BCResNetBackbone(num_classes=int(ckpt["num_classes"]), scale=int(cfg["scale"]), dropout=float(cfg["dropout"]))
    model = EndToEndKWS(logmel=logmel, backbone=backbone, spec_augment=None)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 0.1, size=(4, paths.WINDOW_SAMPLES)).astype(np.float32)
    with torch.no_grad():
        pt_out = model(torch.from_numpy(x)).numpy()

    s_fp32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    in_name = s_fp32.get_inputs()[0].name
    fp32_out = np.vstack([s_fp32.run(None, {in_name: x[i:i + 1]})[0] for i in range(len(x))])
    assert np.max(np.abs(pt_out - fp32_out)) < 1e-3

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    s_int8 = ort.InferenceSession(str(int8), sess_options=opts, providers=["CPUExecutionProvider"])
    in_i = s_int8.get_inputs()[0].name
    int8_out = np.vstack([s_int8.run(None, {in_i: x[i:i + 1]})[0] for i in range(len(x))])
    assert int8_out.shape == pt_out.shape
    assert np.isfinite(int8_out).all()

    # Check single-hop inference latency
    single = x[:1]
    for _ in range(10):
        s_int8.run(None, {in_i: single})
    t0 = time.perf_counter()
    for _ in range(50):
        s_int8.run(None, {in_i: single})
    avg_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    assert avg_ms < 50.0, f"Unexpectedly slow INT8 CPU hop latency: {avg_ms:.2f} ms"
