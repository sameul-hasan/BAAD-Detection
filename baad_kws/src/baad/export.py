"""Export trained BC-ResNet KWS model to ONNX FP32 + INT8 (< 1 MB) with parity & latency checks."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from baad import paths
from baad.data.dataset import KWSWindowDataset, load_labels
from baad.features import LogMel
from baad.logging_utils import get_logger, log_event
from baad.models.bcresnet import BCResNetBackbone, EndToEndKWS


class WindowCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, samples: np.ndarray):
        self.input_name = input_name
        self.samples = samples
        self.idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.idx >= len(self.samples):
            return None
        batch = self.samples[self.idx:self.idx + 1]
        self.idx += 1
        return {self.input_name: batch}


def _benchmark_ort(onnx_path: Path, sample: np.ndarray, threads: int, n_runs: int = 200) -> dict[str, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    feed = {in_name: sample}
    for _ in range(15):
        sess.run(None, feed)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        f"t{threads}_mean_ms": round(float(arr.mean()), 3),
        f"t{threads}_median_ms": round(float(np.median(arr)), 3),
        f"t{threads}_p95_ms": round(float(np.percentile(arr, 95)), 3),
    }


def _eval_parity(pt_model: torch.nn.Module, fp32_path: Path, cand_path: Path, wavs: np.ndarray) -> dict[str, float]:
    with torch.no_grad():
        pt_logits = pt_model(torch.from_numpy(wavs)).numpy()

    s_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    in_name = s_fp32.get_inputs()[0].name
    fp32_logits = np.vstack([s_fp32.run(None, {in_name: wavs[i:i + 1]})[0] for i in range(len(wavs))])

    s_cand = ort.InferenceSession(str(cand_path), providers=["CPUExecutionProvider"])
    in_c = s_cand.get_inputs()[0].name
    cand_logits = np.vstack([s_cand.run(None, {in_c: wavs[i:i + 1]})[0] for i in range(len(wavs))])

    fp32_diff = float(np.max(np.abs(pt_logits - fp32_logits)))
    top1_agree = float((fp32_logits.argmax(axis=1) == cand_logits.argmax(axis=1)).mean())

    def cos_sim(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(np.sum(an * bn, axis=1)))

    return {
        "pt_vs_fp32_max_abs_diff": round(fp32_diff, 6),
        "fp32_vs_int8_top1_agree": round(top1_agree, 4),
        "fp32_vs_int8_cos_sim": round(cos_sim(fp32_logits, cand_logits), 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    export_dir = args.run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("export", log_file=export_dir / "export.log")

    ckpt_path = args.ckpt or (args.run_dir / "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    num_classes = int(ckpt["num_classes"])
    win_samples = int(round(float(cfg["window_sec"]) * paths.SAMPLE_RATE))

    logmel = LogMel(sr=paths.SAMPLE_RATE, n_fft=320, win_ms=20.0, hop_ms=10.0, n_mels=int(cfg["n_mels"]))
    backbone = BCResNetBackbone(num_classes=num_classes, scale=int(cfg["scale"]), dropout=float(cfg["dropout"]))
    model = EndToEndKWS(logmel=logmel, backbone=backbone, spec_augment=None)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    fp32_path = export_dir / "baad_kws_fp32.onnx"
    dummy = torch.randn(1, win_samples, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        opset_version=17,
        input_names=["audio"],
        output_names=["logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(fp32_path)))
    fp32_kb = fp32_path.stat().st_size / 1024.0
    log.info("Exported FP32 ONNX: %s (%.1f KB)", fp32_path, fp32_kb)

    # Collect calibration & parity samples from val set
    val_ds = KWSWindowDataset("val", mode="kws_val", window_samples=win_samples, max_fleurs_cache=200)
    rng = np.random.default_rng(1337)
    idxs = rng.choice(len(val_ds), size=min(200, len(val_ds)), replace=False)
    cal_wavs = np.stack([val_ds[int(i)][0].numpy() for i in idxs]).astype(np.float32)

    # Identify LogMel nodes so we can compare backbone-only INT8 vs full INT8
    m_onnx = onnx.load(str(fp32_path))
    logmel_nodes = [n.name for n in m_onnx.graph.node if "logmel" in n.name.lower()]
    log.info("Found %d logmel nodes in ONNX graph", len(logmel_nodes))

    candidates = {}
    for variant, excl in [("backbone_int8", logmel_nodes), ("full_int8", [])]:
        cand_path = export_dir / f"_tmp_{variant}.onnx"
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(cand_path),
            calibration_data_reader=WindowCalibrationReader("audio", cal_wavs[:120]),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=excl,
        )
        size_kb = cand_path.stat().st_size / 1024.0
        par = _eval_parity(model, fp32_path, cand_path, cal_wavs)
        candidates[variant] = {"path": cand_path, "size_kb": round(size_kb, 1), **par}
        log.info("Quantization candidate %s: size=%.1f KB parity=%s", variant, size_kb, par)

    # Prefer candidate < 1024 KB with highest top1 agreement (using backbone_int8 when under 1 MB!)
    valid = [v for v in candidates.values() if v["size_kb"] < 1024.0]
    chosen = max(valid or list(candidates.values()), key=lambda v: (v["fp32_vs_int8_top1_agree"], v["fp32_vs_int8_cos_sim"]))
    int8_path = export_dir / "baad_kws_int8.onnx"
    shutil.move(str(chosen["path"]), str(int8_path))
    for v in candidates.values():
        if v["path"].exists():
            v["path"].unlink()

    # Copy labels + tuned config
    shutil.copy(paths.LABELS_JSON, export_dir / "labels.json")
    tuned_src = args.run_dir / "eval" / "detect_tuned.yaml"
    if not tuned_src.exists():
        tuned_src = paths.CONFIGS_DIR / "detect_tuned.yaml"
    if tuned_src.exists():
        shutil.copy(tuned_src, export_dir / "detect_tuned.yaml")

    sample = cal_wavs[:1]
    bench = {
        **_benchmark_ort(int8_path, sample, threads=1),
        **_benchmark_ort(int8_path, sample, threads=2),
    }

    metrics = {
        "fp32_kb": round(fp32_kb, 1),
        "int8_kb": chosen["size_kb"],
        "int8_mb": round(chosen["size_kb"] / 1024.0, 3),
        "pt_vs_fp32_max_abs_diff": chosen["pt_vs_fp32_max_abs_diff"],
        "fp32_vs_int8_top1_agree": chosen["fp32_vs_int8_top1_agree"],
        "fp32_vs_int8_cos_sim": chosen["fp32_vs_int8_cos_sim"],
        **bench,
    }
    log.info("Final export metrics: %s", metrics)
    log_event(
        f"export_{args.run_dir.name}",
        f"Exported FP32 ({fp32_kb:.1f} KB) and INT8 ({chosen['size_kb']:.1f} KB) ONNX models",
        metrics=metrics,
        details=f"- Export bundle: `{export_dir}` (`baad_kws_int8.onnx`, `baad_kws_fp32.onnx`, `labels.json`, `detect_tuned.yaml`)",
    )


if __name__ == "__main__":
    main()
