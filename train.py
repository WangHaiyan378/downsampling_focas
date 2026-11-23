"""
Unified training script for both Task A and Task B.
Supports config-driven task switching, mixed precision, gradient clipping,
learning rate scheduling, and TensorBoard logging.
"""

import os
import sys
import time
import math
import yaml
import argparse
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from losses import UnifiedLoss
from utils.grid_config import sync_model_grids_from_data


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str=None):
    """推論／評估用：可退回 CPU。訓練請用 resolve_training_device。"""
    if device_str and device_str != "auto":
        if str(device_str).startswith("cuda") and (not torch.cuda.is_available()):
            print("[Device] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_training_device(device_str=None):
    """
    訓練強制使用 NVIDIA GPU：不再在無 CUDA 時靜默退回 CPU。
    預設 device_str 為 None 或 'auto' 時皆視為使用 cuda:0。
    僅允許 cuda / cuda:n；不支援 --device cpu（避免誤跑 CPU 極慢訓練）。
    """
    raw = (device_str or "cuda").strip()
    low = raw.lower()
    if low == "cpu":
        raise RuntimeError(
            "訓練已設定為必須使用 GPU，不支援 --device cpu。\n"
            "若環境無顯卡，請在裝有 CUDA 的機器上訓練；"
            "僅除錯請改程式或暫時使用 demo 以外的腳本。"
        )
    if low == "auto":
        raw = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "訓練需要 GPU，但 torch.cuda.is_available() 為 False。\n"
            "請安裝帶 CUDA 的 PyTorch（https://pytorch.org 選對 CUDA 版本）、"
            "安裝 NVIDIA 驅動，並檢查 CUDA_VISIBLE_DEVICES 是否誤設為空。"
        )
    dev = torch.device(raw)
    if dev.type != "cuda":
        raise RuntimeError(f"訓練僅支援 CUDA 裝置，收到: {raw}")
    return dev


def get_scheduler(optimizer, cfg, steps_per_epoch):
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"].get("warmup_epochs", 5)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def apply_demo_overrides(cfg):
    """Shrink model for fast CPU/GPU demo runs."""
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    dcfg["history_len"] = min(int(dcfg.get("history_len", 45)), 7)
    nl = min(int(dcfg.get("num_leads", 45)), 6)
    dcfg["num_leads"] = nl
    dcfg["lead_times"] = list(range(1, nl + 1))
    mcfg["history_len"] = dcfg["history_len"]
    mcfg["num_leads"] = nl

    mcfg["d_model"] = min(mcfg.get("d_model", 128), 64)
    mcfg["d_hidden"] = min(mcfg.get("d_hidden", 256), 128)
    mcfg["n_heads"] = min(mcfg.get("n_heads", 8), 4)
    mcfg["n_encoder_layers"] = min(mcfg.get("n_encoder_layers", 3), 1)
    mcfg["n_fusion_layers"] = min(mcfg.get("n_fusion_layers", 2), 1)
    mcfg["n_tsdu_layers"] = min(mcfg.get("n_tsdu_layers", 3), 1)
    mcfg["n_correction_blocks"] = min(mcfg.get("n_correction_blocks", 4), 2)


def _pick_amp_dtype(device):
    """bf16 on Ampere/Ada/Hopper or newer (sm >= 80), else fp16. bf16 避免 1e-8 等 eps underflow。"""
    if device.type != "cuda":
        return torch.float32, False
    try:
        major, _ = torch.cuda.get_device_capability(device)
        if major >= 8:
            return torch.bfloat16, True
    except Exception:
        pass
    return torch.float16, True


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device,
                    task_type, epoch, amp_dtype=torch.bfloat16):
    model.train()
    total_loss = 0.0
    comp_sums = {}
    n_batches = 0
    n_skipped = 0
    use_cuda = device.type == "cuda"
    use_scaler = scaler is not None and amp_dtype == torch.float16

    for batch_idx, batch in enumerate(loader):
        x_static = batch["x_static"].to(device, non_blocking=True)
        lead_ids = batch["lead_ids"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)

        if task_type == "history2era5":
            x_input = batch["x_history"].to(device, non_blocking=True)
        else:
            x_input = batch["x_forecast"].to(device, non_blocking=True)

        if not torch.isfinite(x_input).all() or not torch.isfinite(target_uv).all():
            n_skipped += 1
            continue

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            pred = model(x_input, x_static, lead_ids)
            loss, components = criterion(pred, target_uv)

        if not torch.isfinite(loss):
            n_skipped += 1
            if epoch == 1 and batch_idx < 3:
                print(f"    [skip] non-finite loss at batch {batch_idx}: {loss.item()}")
            continue

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item()
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
        n_batches += 1

    if n_skipped and epoch == 1:
        print(f"    [train] skipped {n_skipped} non-finite batches in epoch 1")

    avg_loss = total_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in comp_sums.items()}
    return avg_loss, avg_comps


@torch.no_grad()
def validate(model, loader, criterion, device, task_type, amp_dtype=torch.bfloat16):
    model.eval()
    total_loss = 0.0
    comp_sums = {}
    n_batches = 0
    use_cuda = device.type == "cuda"

    for batch in loader:
        x_static = batch["x_static"].to(device, non_blocking=True)
        lead_ids = batch["lead_ids"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].to(device, non_blocking=True)

        if task_type == "history2era5":
            x_input = batch["x_history"].to(device, non_blocking=True)
        else:
            x_input = batch["x_forecast"].to(device, non_blocking=True)

        if not torch.isfinite(x_input).all() or not torch.isfinite(target_uv).all():
            continue

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            pred = model(x_input, x_static, lead_ids)
            loss, components = criterion(pred, target_uv)

        if not torch.isfinite(loss):
            continue

        total_loss += loss.item()
        for k, v in components.items():
            comp_sums[k] = comp_sums.get(k, 0.0) + v.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in comp_sums.items()}
    return avg_loss, avg_comps


def train(cfg, log_callback=None, device_str=None):
    task_type = cfg["task_type"]
    tcfg = cfg["training"]
    device = resolve_training_device(device_str)

    print(f"[Train] Task: {task_type}")
    print(f"[Train] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    demo = cfg["data"].get("demo_mode", False)
    if demo:
        apply_demo_overrides(cfg)
    sync_model_grids_from_data(cfg, demo=demo)

    train_ds = build_dataset(cfg, split="train")
    val_ds = build_dataset(cfg, split="val")

    use_cuda = device.type == "cuda"
    nw = int(tcfg.get("num_workers", 4 if use_cuda else 0))
    persistent = bool(nw > 0)
    train_loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=use_cuda, drop_last=True,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=use_cuda,
        persistent_workers=persistent,
    )

    model = build_model(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Model parameters: {param_count:,}")

    criterion = UnifiedLoss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    amp_dtype, amp_enabled = _pick_amp_dtype(device)
    print(f"[Train] AMP dtype: {str(amp_dtype).replace('torch.', '')}  enabled={amp_enabled}")
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))

    save_dir = tcfg.get("save_dir", f"checkpoints/{task_type}")
    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_comps = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            task_type, epoch, amp_dtype=amp_dtype,
        )
        val_loss, val_comps = validate(
            model, val_loader, criterion, device, task_type, amp_dtype=amp_dtype,
        )
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_components": {k: round(v, 6) for k, v in train_comps.items()},
            "val_components": {k: round(v, 6) for k, v in val_comps.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time": round(elapsed, 1),
        }
        history.append(log_entry)

        print(
            f"  Epoch {epoch:03d}/{tcfg['epochs']} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s"
        )

        if log_callback:
            log_callback(log_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, ckpt_path)
            print(f"  -> Saved best model (val_loss={val_loss:.5f})")

        if epoch % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    log_path = os.path.join(save_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Train] Done. Best val_loss: {best_val_loss:.5f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train S2S downscaling model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="訓練僅 GPU：cuda（預設）/ cuda:0 / cuda:1 / auto（等同 cuda）。不支援 cpu。",
    )
    parser.add_argument("--demo", action="store_true", help="Force demo mode (small model + dummy data)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.demo:
        cfg["data"]["demo_mode"] = True

    train(cfg, device_str=args.device)


if __name__ == "__main__":
    main()
