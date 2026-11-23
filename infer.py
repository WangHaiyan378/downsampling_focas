"""
Inference script for both Task A and Task B.
Loads a trained checkpoint and runs prediction on a batch.
"""

import os
import argparse
import yaml
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from train import resolve_device, apply_demo_overrides
from utils.grid_config import sync_model_grids_from_data


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def infer(cfg, checkpoint_path, output_dir="outputs", max_samples=10, device_str=None):
    task_type = cfg["task_type"]
    device = resolve_device(device_str)

    print(f"[Infer] Task: {task_type}")
    print(f"[Infer] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    demo = cfg["data"].get("demo_mode", False)
    if demo:
        apply_demo_overrides(cfg)
    sync_model_grids_from_data(cfg, demo=demo)

    model = build_model(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Infer] Loaded checkpoint from {checkpoint_path}")
    else:
        print("[Infer] No checkpoint provided, using random weights (demo)")

    model.eval()
    use_amp = device.type == "cuda"

    dataset = build_dataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, batch in enumerate(loader):
        if i >= max_samples:
            break

        x_static = batch["x_static"].to(device, non_blocking=True)
        lead_ids = batch["lead_ids"].to(device, non_blocking=True)

        if task_type == "history2era5":
            x_input = batch["x_history"].to(device, non_blocking=True)
        else:
            x_input = batch["x_forecast"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x_input, x_static, lead_ids)

        pred_uv = pred["uv"].float().cpu().numpy()
        has_conv = pred.get("direct_conv") is not None
        if has_conv:
            pred_direct_conv = pred["direct_conv"].float().cpu().numpy()
            pred_diag_conv = pred["diag_conv"].float().cpu().numpy()
        else:
            pred_direct_conv = None
            pred_diag_conv = None

        sample_result = {
            "sample_idx": i,
            "pred_uv_shape": list(pred_uv.shape),
            "pred_uv_mean": float(pred_uv.mean()),
            "pred_uv_std": float(pred_uv.std()),
        }
        if has_conv:
            sample_result["pred_conv_mean"] = float(pred_direct_conv.mean())
        results.append(sample_result)

        save_kwargs = {
            "uv": pred_uv[0],
            "lead_ids": batch["lead_ids"].numpy()[0],
        }
        if has_conv:
            save_kwargs["direct_conv"] = pred_direct_conv[0]
            save_kwargs["diag_conv"] = pred_diag_conv[0]
        np.savez(os.path.join(output_dir, f"pred_{i:04d}.npz"), **save_kwargs)
        print(f"  Sample {i}: uv_shape={pred_uv.shape}, mean={pred_uv.mean():.4f}"
              + (", conv=on" if has_conv else ", conv=off"))

    summary_path = os.path.join(output_dir, "inference_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Infer] Done. Results saved to {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Inference for S2S downscaling")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", help="Device: auto / cuda / cpu")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.demo:
        cfg["data"]["demo_mode"] = True

    infer(cfg, args.checkpoint, args.output_dir, args.max_samples, device_str=args.device)


if __name__ == "__main__":
    main()
