"""Training loop for lightweight streaming BC-ResNet KWS model (CPU or CUDA)."""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from baad import paths
from baad.data.augment import SpecAugment
from baad.data.dataset import KWSWindowDataset, load_labels
from baad.features import LogMel
from baad.logging_utils import configure_matplotlib, get_logger, log_event
from baad.models.bcresnet import BCResNetBackbone, EndToEndKWS


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_abusive: int,
    ce_fn: nn.Module,
    bce_fn: nn.Module,
    aux_weight: float,
) -> torch.Tensor:
    ce = ce_fn(logits, targets)
    if aux_weight <= 0:
        return ce
    ab_lse = torch.logsumexp(logits[:, :num_abusive], dim=1)
    non_lse = torch.logsumexp(logits[:, num_abusive:], dim=1)
    ab_logit = ab_lse - non_lse
    is_ab = (targets < num_abusive).float()
    return ce + aux_weight * bce_fn(ab_logit, is_ab)


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_abusive: int,
    unknown_id: int,
    ce_fn: nn.Module,
    bce_fn: nn.Module,
    aux_weight: float,
) -> dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    tot_loss, n_items = 0.0, 0
    use_amp = device.type == "cuda"
    for wav, y in loader:
        wav, y = wav.to(device, non_blocking=use_amp), y.to(device, non_blocking=use_amp)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(wav)
            loss = compute_loss(logits, y, num_abusive, ce_fn, bce_fn, aux_weight)
        tot_loss += float(loss.item()) * len(y)
        n_items += len(y)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))

    y_ab = (targets < num_abusive).astype(int)
    p_ab = (preds < num_abusive).astype(int)
    ab_p, ab_r, ab_f1, _ = precision_recall_fscore_support(y_ab, p_ab, average="binary", zero_division=0)

    # Specific hard-negative & unknown false-positive rates into abusive classes
    close_mask = (targets >= num_abusive) & (targets < unknown_id)
    unk_mask = targets == unknown_id
    close_fp = float((preds[close_mask] < num_abusive).mean()) if close_mask.any() else 0.0
    unk_fp = float((preds[unk_mask] < num_abusive).mean()) if unk_mask.any() else 0.0

    return {
        "loss": tot_loss / max(1, n_items),
        "acc": acc,
        "macro_f1": macro_f1,
        "ab_prec": float(ab_p),
        "ab_rec": float(ab_r),
        "ab_f1": float(ab_f1),
        "close_fp": close_fp,
        "unk_fp": unk_fp,
    }


def save_curves(history: list[dict], out_path: Path) -> None:
    configure_matplotlib(headless=True)
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_macro_f1"] for h in history], label="kws_val_macro_f1")
    axes[1].plot(epochs, [h["iso_macro_f1"] for h in history], label="iso_val_macro_f1")
    axes[1].plot(epochs, [h["val_ab_f1"] for h in history], label="kws_val_ab_f1")
    axes[1].set_title("F1 Metrics")
    axes[1].legend()

    axes[2].plot(epochs, [h["val_close_fp"] for h in history], label="close_fp_rate")
    axes[2].plot(epochs, [h["val_unk_fp"] for h in history], label="unknown_fp_rate")
    axes[2].set_title("False Positive Rates")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true", help="Quick 2-epoch smoke test on tiny subset")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs in config")
    ap.add_argument("--run-name", type=str, default=None, help="Override run name")
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda",
                    help="Target compute device (default: cuda — fails if GPU unavailable)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    run_name = args.run_name or (f"{cfg['run_name']}_smoke" if args.smoke else cfg["run_name"])
    if args.smoke:
        cfg["epochs"] = 2
        cfg["samples_per_epoch"] = 256
        cfg["num_workers"] = 0

    run_dir = paths.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"train_{run_name}", log_file=run_dir / "train.log")

    set_seed(int(cfg["seed"]))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            msg = (
                "CUDA GPU requested (--device cuda) but torch.cuda.is_available() is False. "
                "Install the NVIDIA driver (e.g. nvidia-driver-570) and reboot, then verify with `nvidia-smi`."
            )
            log.error(msg)
            raise RuntimeError(msg)
        device = torch.device("cuda")
    elif args.device == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            device = torch.device("cpu")

    labels = load_labels()
    num_classes = len(labels)
    num_abusive = sum(1 for x in labels if x["is_abusive"])
    unknown_id = next(x["id"] for x in labels if x["word"] == paths.UNKNOWN_LABEL)
    assert [x["id"] for x in labels if x["is_abusive"]] == list(range(num_abusive))

    win_samples = int(round(float(cfg["window_sec"]) * paths.SAMPLE_RATE))
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    log.info("Starting run=%s device=%s gpu=%s classes=%d (abusive=%d) smoke=%s",
             run_name, device, gpu_name, num_classes, num_abusive, args.smoke)

    train_ds = KWSWindowDataset(
        "train", mode="kws_train",
        samples_per_epoch=int(cfg["samples_per_epoch"]),
        window_samples=win_samples, seed=int(cfg["seed"]),
        max_fleurs_cache=200 if args.smoke else 2000,
    )
    kws_val_ds = KWSWindowDataset(
        "val", mode="kws_val",
        window_samples=win_samples, seed=int(cfg["seed"]),
        max_fleurs_cache=150 if args.smoke else 600,
    )
    iso_val_ds = KWSWindowDataset("val", mode="isolated", window_samples=win_samples, seed=int(cfg["seed"]))

    nw = int(cfg.get("num_workers", 0)) if device.type == "cuda" else 0
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True,
                              num_workers=nw, persistent_workers=(nw > 0), pin_memory=pin)
    kws_val_loader = DataLoader(kws_val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=pin)
    iso_val_loader = DataLoader(iso_val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=0, pin_memory=pin)

    logmel = LogMel(sr=paths.SAMPLE_RATE, n_fft=320, win_ms=20.0, hop_ms=10.0, n_mels=int(cfg["n_mels"]))
    backbone = BCResNetBackbone(num_classes=num_classes, scale=int(cfg["scale"]), dropout=float(cfg["dropout"]))
    model = EndToEndKWS(logmel=logmel, backbone=backbone, spec_augment=SpecAugment()).to(device)

    n_params = sum(p.numel() for p in backbone.parameters())
    log.info("Model BCResNet(scale=%d) trainable params=%d", cfg["scale"], n_params)

    ce_fn = nn.CrossEntropyLoss(label_smoothing=float(cfg.get("label_smoothing", 0.05)))
    bce_fn = nn.BCEWithLogitsLoss()
    aux_w = float(cfg.get("aux_binary_weight", 0.35))

    epochs = int(cfg["epochs"])
    base_lr = float(cfg["lr"])
    min_lr = float(cfg["min_lr"])
    warmup = min(int(cfg.get("warmup_epochs", 2)), max(0, epochs - 1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(cfg["weight_decay"]))

    def lr_lambda(ep: int) -> float:
        if warmup > 0 and ep < warmup:
            return float(ep + 1) / float(warmup)
        progress = (ep - warmup) / max(1, epochs - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1.0 - min_lr / base_lr) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metrics_csv = run_dir / "metrics.csv"
    fieldnames = [
        "epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
        "val_ab_prec", "val_ab_rec", "val_ab_f1", "val_close_fp", "val_unk_fp",
        "iso_acc", "iso_macro_f1", "iso_ab_f1", "score", "lr", "epoch_sec",
    ]
    history: list[dict] = []
    best_score = -1.0
    best_row: dict | None = None
    t_start = time.time()

    with open(metrics_csv, "w", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            model.train()
            tot_loss, n_seen = 0.0, 0
            for wav, y in train_loader:
                wav, y = wav.to(device, non_blocking=use_amp), y.to(device, non_blocking=use_amp)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = model(wav)
                    loss = compute_loss(logits, y, num_abusive, ce_fn, bce_fn, aux_w)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                tot_loss += float(loss.item()) * len(y)
                n_seen += len(y)

            cur_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()

            val_m = evaluate_loader(model, kws_val_loader, device, num_abusive, unknown_id, ce_fn, bce_fn, aux_w)
            iso_m = evaluate_loader(model, iso_val_loader, device, num_abusive, unknown_id, ce_fn, bce_fn, aux_w)
            score = 0.5 * val_m["ab_f1"] + 0.5 * val_m["macro_f1"]
            ep_sec = time.time() - t_ep

            row = {
                "epoch": epoch,
                "train_loss": round(tot_loss / max(1, n_seen), 4),
                "val_loss": round(val_m["loss"], 4),
                "val_acc": round(val_m["acc"], 4),
                "val_macro_f1": round(val_m["macro_f1"], 4),
                "val_ab_prec": round(val_m["ab_prec"], 4),
                "val_ab_rec": round(val_m["ab_rec"], 4),
                "val_ab_f1": round(val_m["ab_f1"], 4),
                "val_close_fp": round(val_m["close_fp"], 4),
                "val_unk_fp": round(val_m["unk_fp"], 4),
                "iso_acc": round(iso_m["acc"], 4),
                "iso_macro_f1": round(iso_m["macro_f1"], 4),
                "iso_ab_f1": round(iso_m["ab_f1"], 4),
                "score": round(score, 4),
                "lr": round(cur_lr, 6),
                "epoch_sec": round(ep_sec, 2),
            }
            writer.writerow(row)
            cf.flush()
            history.append(row)

            log.info(
                "ep %02d/%02d | tr_loss=%.3f va_loss=%.3f | kws_f1=%.3f ab_f1=%.3f (P=%.2f R=%.2f cFP=%.2f uFP=%.2f) | iso_acc=%.3f iso_f1=%.3f | %.1fs",
                epoch, epochs, row["train_loss"], row["val_loss"],
                row["val_macro_f1"], row["val_ab_f1"], row["val_ab_prec"], row["val_ab_rec"],
                row["val_close_fp"], row["val_unk_fp"], row["iso_acc"], row["iso_macro_f1"], ep_sec,
            )

            ckpt = {
                "config": cfg,
                "num_classes": num_classes,
                "num_abusive": num_abusive,
                "epoch": epoch,
                "metrics": row,
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, run_dir / "last.pt")
            if score > best_score:
                best_score = score
                best_row = row
                torch.save(ckpt, run_dir / "best.pt")

    save_curves(history, run_dir / "training_curves.png")
    total_min = (time.time() - t_start) / 60.0
    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.type == "cuda" else 0.0
    summary = {
        "run_name": run_name,
        "smoke": args.smoke,
        "device": str(device),
        "gpu_name": gpu_name or "cpu",
        "peak_vram_mb": peak_vram_mb,
        "params": n_params,
        "epochs": epochs,
        "total_minutes": round(total_min, 2),
        **(best_row or {}),
    }
    log.info("Training finished in %.2f min. Best: %s", total_min, best_row)
    log_event(
        f"train_{run_name}",
        f"{'Smoke' if args.smoke else 'Full'} BC-ResNet-{cfg['scale']} training completed",
        metrics=summary,
        details=f"- Checkpoint: `{run_dir / 'best.pt'}`\n- Per-epoch metrics: `{metrics_csv}`\n- Curves: `{run_dir / 'training_curves.png'}`",
    )


if __name__ == "__main__":
    main()
