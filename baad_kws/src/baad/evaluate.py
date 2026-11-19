"""Comprehensive evaluation, validation threshold tuning, DET curve, and error analysis."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

from baad import paths
from baad.data.dataset import KWSWindowDataset, load_labels
from baad.data.preprocess import load_mono_16k
from baad.detector import EnergyVAD, StreamingDetector, load_backend
from baad.logging_utils import configure_matplotlib, get_logger, log_event


def _eval_isolated(model_path: Path, labels: list[dict], eval_dir: Path, log) -> dict:
    from baad.detector import PyTorchBackend

    num_abusive = sum(1 for x in labels if x["is_abusive"])
    backend = PyTorchBackend(model_path) if model_path.suffix == ".pt" else None
    results = {}

    for split in ("val", "test"):
        ds = KWSWindowDataset(split, mode="isolated", window_samples=paths.WINDOW_SAMPLES)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        preds_all, probs_all, targets_all = [], [], []
        if backend is not None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = backend.model.to(dev)
            with torch.no_grad():
                for wav, y in loader:
                    logits = model(wav.to(dev))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    probs_all.append(probs)
                    preds_all.append(probs.argmax(axis=1))
                    targets_all.append(y.numpy())
            model.cpu()
        else:
            ob = load_backend(model_path)
            for wav, y in loader:
                for row in wav.numpy():
                    p = ob.predict_probs(row)
                    probs_all.append(p[None, :])
                    preds_all.append(np.array([int(p.argmax())]))
                targets_all.append(y.numpy())

        probs = np.concatenate(probs_all, axis=0)
        preds = np.concatenate(preds_all, axis=0)
        targets = np.concatenate(targets_all, axis=0)

        top1 = float(accuracy_score(targets, preds))
        top5_hits = np.argsort(-probs, axis=1)[:, :5]
        top5 = float(np.mean([t in top5_hits[i] for i, t in enumerate(targets)]))
        mf1 = float(f1_score(targets, preds, average="macro", zero_division=0))

        y_ab = (targets < num_abusive).astype(int)
        p_ab = (preds < num_abusive).astype(int)
        bp, br, bf1, _ = precision_recall_fscore_support(y_ab, p_ab, average="binary", zero_division=0)

        results[f"{split}_iso_top1"] = round(top1, 4)
        results[f"{split}_iso_top5"] = round(top5, 4)
        results[f"{split}_iso_macro_f1"] = round(mf1, 4)
        results[f"{split}_iso_ab_prec"] = round(float(bp), 4)
        results[f"{split}_iso_ab_rec"] = round(float(br), 4)
        results[f"{split}_iso_ab_f1"] = round(float(bf1), 4)

        if split == "test":
            # Abusive vs Close confusion pairs
            pair_counts: Counter[tuple[str, str, str]] = Counter()
            for t, p in zip(targets, preds):
                t_kind, p_kind = labels[t]["kind"], labels[p]["kind"]
                if (t_kind == "close" and p_kind == "abusive") or (t_kind == "abusive" and p_kind == "close"):
                    pair_counts[(f"{t_kind}->{p_kind}", labels[t]["word"], labels[p]["word"])] += 1
            with open(eval_dir / "close_confusion.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["direction", "true_word", "pred_word", "count"])
                for (direction, tw, pw), cnt in pair_counts.most_common():
                    w.writerow([direction, tw, pw, cnt])

            # Gender breakdown on test
            genders = ds.df["gender"].to_numpy()
            g_rows = []
            for g in sorted(set(genders)):
                m = genders == g
                gm_f1 = f1_score(targets[m], preds[m], average="macro", zero_division=0)
                g_acc = accuracy_score(targets[m], preds[m])
                gp, gr, gf1, _ = precision_recall_fscore_support(y_ab[m], p_ab[m], average="binary", zero_division=0)
                g_rows.append({"gender": g, "clips": int(m.sum()), "top1": round(g_acc, 4),
                               "macro_f1": round(gm_f1, 4), "abusive_prec": round(gp, 4),
                               "abusive_rec": round(gr, 4), "abusive_f1": round(gf1, 4)})
            pd.DataFrame(g_rows).to_csv(eval_dir / "gender_breakdown.csv", index=False)

            # Per-word F1
            w_rows = []
            for lbl in labels[:len(labels) - 2]:
                cid = lbl["id"]
                m_true = targets == cid
                m_pred = preds == cid
                tp = int((m_true & m_pred).sum())
                fp = int((~m_true & m_pred).sum())
                fn = int((m_true & ~m_pred).sum())
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-9, prec + rec) if (prec + rec) > 0 else 0.0
                w_rows.append({"id": cid, "word": lbl["word"], "kind": lbl["kind"],
                               "support": int(m_true.sum()), "tp": tp, "fp": fp, "fn": fn,
                               "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
            pd.DataFrame(w_rows).to_csv(eval_dir / "per_word_f1.csv", index=False)

    log.info("Isolated evaluation: %s", results)
    return results


def _cache_stream_posteriors(records: list[dict], backend, labels: list[dict]) -> list[dict]:
    """Run backend ONCE per synthetic stream so threshold sweeps replay in milliseconds."""
    from baad.detector import PyTorchBackend

    num_classes = len(labels)
    silence_id = next(x["id"] for x in labels if x["word"] == paths.SILENCE_LABEL)
    hop_samples = paths.HOP_SAMPLES
    win_samples = paths.WINDOW_SAMPLES

    use_gpu_batch = isinstance(backend, PyTorchBackend) and torch.cuda.is_available()
    dev = torch.device("cuda" if use_gpu_batch else "cpu")
    if use_gpu_batch:
        backend.model.to(dev)

    cached = []
    for rec in records:
        wav, _ = sf.read(str(paths.DATA_DIR / rec["audio"]), dtype="float32")
        padded = np.concatenate([wav, np.zeros(int(0.35 * paths.SAMPLE_RATE), dtype=np.float32)])
        buf = np.zeros(win_samples, dtype=np.float32)
        vad = EnergyVAD(rms_thresh=0.005, hangover_hops=8)

        active_windows = []
        active_indices = []
        t_ends = []
        probs_seq = np.zeros((1 + (len(padded) - 1) // hop_samples, num_classes), dtype=np.float32)
        probs_seq[:, silence_id] = 1.0

        for step_idx, pos in enumerate(range(0, len(padded), hop_samples)):
            chunk = padded[pos:pos + hop_samples]
            if len(chunk) < hop_samples:
                chunk = np.pad(chunk, (0, hop_samples - len(chunk)))
            buf[:-hop_samples] = buf[hop_samples:]
            buf[-hop_samples:] = chunk
            t_end = (pos + hop_samples) / paths.SAMPLE_RATE
            t_ends.append(t_end)
            if vad.is_speech(chunk):
                if use_gpu_batch:
                    active_windows.append(buf.copy())
                    active_indices.append(step_idx)
                else:
                    probs_seq[step_idx] = backend.predict_probs(buf)

        if use_gpu_batch and active_windows:
            with torch.no_grad():
                x = torch.from_numpy(np.stack(active_windows)).to(dev)
                probs = torch.softmax(backend.model(x), dim=1).cpu().numpy()
            for idx, p in zip(active_indices, probs):
                probs_seq[idx] = p

        cached.append({
            "rec": rec,
            "probs": probs_seq,
            "t_ends": t_ends,
        })

    if use_gpu_batch:
        backend.model.cpu()
    return cached


def _replay_detector(
    probs: np.ndarray,
    t_ends: list[float],
    labels: list[dict],
    num_abusive: int,
    smooth_k: int,
    threshold: float,
    binary_threshold: float,
    refractory_sec: float = 0.70,
) -> tuple[list[dict], float]:
    """Replay smoothing + peak picking on precomputed hop posteriors."""
    hist: deque[np.ndarray] = deque(maxlen=max(1, smooth_k))
    last_emit_end = -1e9
    pending: dict | None = None
    pending_hops = 0
    events: list[dict] = []
    max_p_ab = 0.0

    def finalize():
        nonlocal pending, pending_hops, last_emit_end
        if pending is None:
            return
        ev = pending
        pending = None
        pending_hops = 0
        if ev["t_peak"] - last_emit_end < refractory_sec:
            return
        last_emit_end = ev["t_peak"]
        lag = 0.5 * (max(1, smooth_k) - 1) * paths.HOP_SEC
        center = ev["t_peak"] - 0.5 * paths.WINDOW_SEC - lag
        start = max(0.0, center - 0.375)
        end = max(start + 0.25, center + 0.375)
        events.append({
            "word": labels[ev["class_id"]]["word"],
            "label_id": int(ev["class_id"]),
            "is_abusive": True,
            "start": round(start, 3),
            "end": round(end, 3),
            "score": round(float(ev["score"]), 4),
            "p_abusive": round(float(ev["p_abusive"]), 4),
        })

    for p, t_end in zip(probs, t_ends):
        hist.append(p)
        p_s = np.mean(hist, axis=0)
        p_ab = float(p_s[:num_abusive].sum())
        if p_ab > max_p_ab:
            max_p_ab = p_ab
        best_c = int(np.argmax(p_s[:num_abusive]))
        best_score = float(p_s[best_c])
        triggered = (t_end >= 0.95) and (
            (best_score >= threshold)
            or (p_ab >= binary_threshold and best_score >= 0.65 * threshold)
        )
        strength = max(best_score, 0.85 * p_ab if p_ab >= binary_threshold else 0.0)

        if triggered and (t_end - last_emit_end >= refractory_sec):
            if pending is None:
                pending = {"t_peak": t_end, "class_id": best_c, "score": best_score, "p_abusive": p_ab, "strength": strength}
                pending_hops = 1
            elif strength >= pending["strength"] and pending_hops < 4:
                pending = {"t_peak": t_end, "class_id": best_c, "score": best_score, "p_abusive": p_ab, "strength": strength}
                pending_hops += 1
            else:
                finalize()
        elif pending is not None:
            finalize()

    finalize()
    return events, max_p_ab


def _match_events(gt_events: list[dict], pred_events: list[dict], tol_s: float = 0.50, exact_word: bool = False) -> tuple[int, int, int, list[dict], list[dict]]:
    matched_gt = set()
    matched_pred = set()
    for pi, pe in enumerate(pred_events):
        p_mid = 0.5 * (pe["start"] + pe["end"])
        best_gi, best_dist = -1, 1e9
        for gi, ge in enumerate(gt_events):
            if gi in matched_gt:
                continue
            if exact_word and pe["label_id"] != ge["label_id"]:
                continue
            g_mid = 0.5 * (ge["start"] + ge["end"])
            overlap = min(pe["end"], ge["end"]) - max(pe["start"], ge["start"])
            dist = abs(p_mid - g_mid)
            if dist <= tol_s or overlap >= -0.35:
                if dist < best_dist:
                    best_dist = dist
                    best_gi = gi
        if best_gi >= 0:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    fps = [pe for pi, pe in enumerate(pred_events) if pi not in matched_pred]
    fns = [ge for gi, ge in enumerate(gt_events) if gi not in matched_gt]
    return len(matched_pred), len(fps), len(fns), fps, fns


def _score_cached(cached: list[dict], labels: list[dict], num_abusive: int, smooth_k: int, threshold: float, binary_threshold: float) -> tuple[dict, list[dict]]:
    tp_any = fp_any = fn_any = 0
    tp_ex = fp_ex = fn_ex = 0
    neg_hours = 0.0
    neg_fa = 0
    neg_close_streams = 0
    neg_close_fa_streams = 0
    sent_y, sent_p, sent_score = [], [], []
    errors = []

    for item in cached:
        rec = item["rec"]
        preds, max_p_ab = _replay_detector(item["probs"], item["t_ends"], labels, num_abusive, smooth_k, threshold, binary_threshold)
        gt_ab = [e for e in rec["events"] if e["is_abusive"]]

        tpa, fpa, fna, fps, fns = _match_events(gt_ab, preds, tol_s=0.50, exact_word=False)
        tpe, fpe, fne, _, _ = _match_events(gt_ab, preds, tol_s=0.50, exact_word=True)
        tp_any += tpa; fp_any += fpa; fn_any += fna
        tp_ex += tpe; fp_ex += fpe; fn_ex += fne

        if not rec["has_abusive"]:
            neg_hours += rec["duration"] / 3600.0
            neg_fa += len(preds)
            if rec["stream_kind"] == "neg_close":
                neg_close_streams += 1
                if preds:
                    neg_close_fa_streams += 1

        sent_y.append(int(rec["has_abusive"]))
        sent_p.append(int(len(preds) > 0))
        sent_score.append(max_p_ab)

        for e in fps:
            errors.append({"audio": rec["audio"], "kind": rec["stream_kind"], "error": "FP",
                           "start": e["start"], "end": e["end"], "pred_word": e["word"],
                           "true_word": ",".join(x["word"] for x in rec["events"]) or "_clean_", "score": e["score"]})
        for e in fns:
            errors.append({"audio": rec["audio"], "kind": rec["stream_kind"], "error": "FN",
                           "start": e["start"], "end": e["end"], "pred_word": "_missed_",
                           "true_word": e["word"], "score": 0.0})

    def prf(tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r) if (p + r) > 0 else 0.0
        return round(p, 4), round(r, 4), round(f1, 4)

    ap, ar, af1 = prf(tp_any, fp_any, fn_any)
    ep, er, ef1 = prf(tp_ex, fp_ex, fn_ex)
    sp, sr_, sf1, _ = precision_recall_fscore_support(sent_y, sent_p, average="binary", zero_division=0)
    auc = float(roc_auc_score(sent_y, sent_score)) if len(set(sent_y)) > 1 else 0.0
    fa_h = neg_fa / max(1e-6, neg_hours)

    metrics = {
        "smooth_k": smooth_k,
        "threshold": round(threshold, 3),
        "binary_threshold": round(binary_threshold, 3),
        "event_any_prec": ap, "event_any_rec": ar, "event_any_f1": af1,
        "event_exact_prec": ep, "event_exact_rec": er, "event_exact_f1": ef1,
        "sent_prec": round(float(sp), 4), "sent_rec": round(float(sr_), 4),
        "sent_f1": round(float(sf1), 4), "sent_auc": round(auc, 4),
        "clean_neg_fa": int(neg_fa), "clean_neg_hours": round(neg_hours, 4),
        "fa_per_hour": round(fa_h, 2),
        "close_neg_fa_rate": round(neg_close_fa_streams / max(1, neg_close_streams), 4),
    }
    return metrics, errors


def _save_det_plot(sweep_rows: list[dict], best: dict, out_path: Path) -> None:
    configure_matplotlib(headless=True)
    import matplotlib.pyplot as plt

    df = pd.DataFrame(sweep_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, grp in df.groupby("smooth_k"):
        # For each threshold pick the point with best event_any_f1 across binary_threshold
        g = grp.sort_values("event_any_f1", ascending=False).drop_duplicates("threshold").sort_values("fa_per_hour")
        axes[0].plot(g["fa_per_hour"], g["event_any_f1"], marker="o", markersize=3, label=f"smooth_k={k}")
        axes[1].plot(g["fa_per_hour"], 1.0 - g["event_any_rec"], marker="o", markersize=3, label=f"smooth_k={k}")

    axes[0].scatter([best["fa_per_hour"]], [best["event_any_f1"]], color="red", s=80, zorder=5, label="chosen")
    axes[0].axvline(5.0, color="gray", linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Clean Speech False Alarms / Hour")
    axes[0].set_ylabel("Event F1 (±0.5 s)")
    axes[0].set_title("Validation Event F1 vs FA/h")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter([best["fa_per_hour"]], [1.0 - best["event_any_rec"]], color="red", s=80, zorder=5, label="chosen")
    axes[1].axvline(5.0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Clean Speech False Alarms / Hour")
    axes[1].set_ylabel("Miss Rate (1 - Event Recall)")
    axes[1].set_title("DET Curve (Lower-Left is Better)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _eval_raw_test(detector: StreamingDetector, eval_dir: Path, log) -> dict:
    te_df = pd.read_csv(paths.SPLITS_DIR / "test.csv")
    test_ab_speakers = {s[1:] for s in te_df["speaker"].unique() if s.startswith("a")}
    rows = []
    for wav_p in sorted(paths.RAW_DIR.glob("*.wav")):
        m = re.match(r"^(\d+)[_-]", wav_p.name)
        if not m or m.group(1) not in test_ab_speakers:
            continue
        if wav_p.stat().st_size > 150_000_000:
            log.warning("Skipping oversized raw file: %s", wav_p.name)
            continue
        try:
            info = sf.info(str(wav_p))
            dur = info.frames / info.samplerate
            if dur <= 0 or dur > 1800:
                log.warning("Skipping raw file with abnormal duration %.1fs: %s", dur, wav_p.name)
                continue
            res = detector.detect_file(wav_p)
            rows.append({
                "file": wav_p.name,
                "speaker": f"a{m.group(1)}",
                "duration_s": round(dur, 2),
                "detected_events": res["num_abusive_events"],
                "vad_active_ratio": res["vad_active_ratio"],
            })
        except Exception as e:
            log.warning("Skipping unreadable raw file %s: %s", wav_p.name, e)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(eval_dir / "raw_test_summary.csv", index=False)
    total_events = int(out_df["detected_events"].sum()) if len(out_df) else 0
    return {"raw_test_files": len(out_df), "raw_total_detections": total_events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None, help="Override .pt or .onnx model path")
    ap.add_argument("--skip-raw", action="store_true")
    args = ap.parse_args()

    eval_dir = args.run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger("evaluate", log_file=eval_dir / "eval.log")

    model_path = args.model or (args.run_dir / "best.pt")
    labels = load_labels()
    num_abusive = sum(1 for x in labels if x["is_abusive"])

    # Part A: Isolated-word evaluation
    iso_metrics = _eval_isolated(model_path, labels, eval_dir, log)

    # Part B: Validation threshold sweep on synthetic continuous streams
    backend = load_backend(model_path)
    val_recs = [json.loads(l) for l in (paths.SYNTH_DIR / "val" / "streams.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    val_cached = _cache_stream_posteriors(val_recs, backend, labels)

    sweep_rows = []
    for smooth_k in (2, 3, 4):
        for thr in np.round(np.arange(0.35, 0.96, 0.05), 2):
            for bin_thr in (0.75, 0.85, 0.95, 1.01):
                m, _ = _score_cached(val_cached, labels, num_abusive, smooth_k, float(thr), float(bin_thr))
                sweep_rows.append(m)

    pd.DataFrame(sweep_rows).to_csv(eval_dir / "val_threshold_sweep.csv", index=False)

    # Prefer operating points with FA/h <= 5.0; fallback to lowest penalty if none achieves <= 5.0
    low_fa = [r for r in sweep_rows if r["fa_per_hour"] <= 5.0 and r["event_any_rec"] >= 0.40]
    if low_fa:
        best_val = max(low_fa, key=lambda r: (r["event_any_f1"], r["event_exact_f1"], -r["fa_per_hour"]))
    else:
        best_val = max(sweep_rows, key=lambda r: r["event_any_f1"] - 0.025 * max(0.0, r["fa_per_hour"] - 5.0))

    _save_det_plot(sweep_rows, best_val, eval_dir / "det_curve.png")

    tuned_cfg = {
        "window_sec": paths.WINDOW_SEC,
        "hop_sec": paths.HOP_SEC,
        "smooth_k": int(best_val["smooth_k"]),
        "threshold": float(best_val["threshold"]),
        "binary_threshold": float(best_val["binary_threshold"]),
        "refractory_sec": 0.70,
        "vad_rms_thresh": 0.005,
        "show_close": False,
    }
    paths.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    (eval_dir / "detect_tuned.yaml").write_text(yaml.safe_dump(tuned_cfg, sort_keys=False), encoding="utf-8")
    (paths.CONFIGS_DIR / "detect_tuned.yaml").write_text(yaml.safe_dump(tuned_cfg, sort_keys=False), encoding="utf-8")
    log.info("Tuned val threshold: %s", best_val)

    # Part C: Held-out Test Synthetic Streams
    test_recs = [json.loads(l) for l in (paths.SYNTH_DIR / "test" / "streams.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    test_cached = _cache_stream_posteriors(test_recs, backend, labels)
    test_m, test_errors = _score_cached(
        test_cached, labels, num_abusive,
        tuned_cfg["smooth_k"], tuned_cfg["threshold"], tuned_cfg["binary_threshold"],
    )
    pd.DataFrame(test_errors).to_csv(eval_dir / "test_errors.csv", index=False)
    log.info("Held-out Test continuous metrics: %s", test_m)

    raw_m = {}
    if not args.skip_raw:
        det = StreamingDetector(
            backend, labels,
            smooth_k=tuned_cfg["smooth_k"],
            threshold=tuned_cfg["threshold"],
            binary_threshold=tuned_cfg["binary_threshold"],
            refractory_sec=tuned_cfg["refractory_sec"],
            vad_rms_thresh=tuned_cfg["vad_rms_thresh"],
        )
        raw_m = _eval_raw_test(det, eval_dir, log)
        log.info("Raw Data test evaluation: %s", raw_m)

    all_metrics = {
        **iso_metrics,
        "tuned_smooth_k": tuned_cfg["smooth_k"],
        "tuned_threshold": tuned_cfg["threshold"],
        "tuned_binary_threshold": tuned_cfg["binary_threshold"],
        "val_event_any_f1": best_val["event_any_f1"],
        "val_fa_per_hour": best_val["fa_per_hour"],
        **{f"test_{k}": v for k, v in test_m.items() if k not in ("smooth_k", "threshold", "binary_threshold")},
        **raw_m,
    }
    log_event(
        f"evaluate_{args.run_dir.name}",
        f"Evaluated {model_path.name} on isolated test, val tuning, test continuous streams, and Raw Data",
        metrics=all_metrics,
        details=f"- Tuned config: `{eval_dir / 'detect_tuned.yaml'}`\n- DET plot: `{eval_dir / 'det_curve.png'}`\n- Errors CSV: `{eval_dir / 'test_errors.csv'}`",
    )


if __name__ == "__main__":
    main()
