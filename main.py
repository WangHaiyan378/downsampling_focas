"""
Main entry point: unified CLI for training, inference, and frontend server.

Usage:
    python main.py train --config configs/task_a.yaml
    python main.py train --config configs/task_b.yaml --demo --epochs 5
    python main.py infer --config configs/task_a.yaml --checkpoint checkpoints/task_a/best_model.pt
    python main.py serve --port 5000
"""

import argparse
import os
import sys
import yaml

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(args):
    from train import train, load_config as lc
    cfg = lc(args.config)
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.demo:
        cfg["data"]["demo_mode"] = True
    train(cfg, device_str=args.device)


def cmd_infer(args):
    from infer import infer, load_config as lc
    cfg = lc(args.config)
    if args.demo:
        cfg["data"]["demo_mode"] = True
    infer(cfg, args.checkpoint, args.output_dir, args.max_samples, device_str=args.device)


def cmd_eval(args):
    from evaluate import evaluate
    from evaluation.rmse_spatial import parse_lead_groups_arg
    cfg = load_config(args.config)
    if args.demo:
        cfg["data"]["demo_mode"] = True
    lead_groups = parse_lead_groups_arg(args.lead_groups) if not args.skip_rmse_maps else None
    evaluate_kwargs = dict(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        device_str=args.device,
        lead_min=args.lead_min,
        lead_max=args.lead_max,
        lead_groups=lead_groups,
        save_rmse_spatial=(not args.skip_rmse_maps),
        rmse_maps_dir=args.rmse_maps_dir,
    )
    # 只有 CLI 顯式給的旗標才覆寫 cfg
    if args.include_taiwan is not None:
        evaluate_kwargs["include_taiwan"] = args.include_taiwan
    if args.rebuild_china_mask is not None:
        evaluate_kwargs["rebuild_china_mask"] = args.rebuild_china_mask
    if args.taiwan_shapefile_path:
        evaluate_kwargs["taiwan_shapefile_path"] = args.taiwan_shapefile_path
    evaluate(**evaluate_kwargs)


def cmd_serve(args):
    from frontend.app import create_app
    app = create_app()
    print(f"[Server] Starting web UI on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def main():
    parser = argparse.ArgumentParser(description="S2S-ERA5 Downscaling Unified System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--batch_size", type=int, default=None)
    train_parser.add_argument("--lr", type=float, default=None)
    train_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="訓練僅 GPU：cuda（預設）/ cuda:0 / auto。無 CUDA 時會報錯而非改用 CPU。",
    )
    train_parser.add_argument("--demo", action="store_true",
                              help="Demo mode: small model + dummy data for quick validation")

    # ---- infer ----
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", type=str, required=True)
    infer_parser.add_argument("--checkpoint", type=str, default=None)
    infer_parser.add_argument("--output_dir", type=str, default="outputs")
    infer_parser.add_argument("--max_samples", type=int, default=10)
    infer_parser.add_argument("--device", type=str, default="auto")
    infer_parser.add_argument("--demo", action="store_true")

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="Evaluate model with comprehensive metrics")
    eval_parser.add_argument("--config", type=str, required=True)
    eval_parser.add_argument("--checkpoint", type=str, default=None)
    eval_parser.add_argument("--output_dir", type=str, default="eval_results")
    eval_parser.add_argument("--max_samples", type=int, default=50)
    eval_parser.add_argument("--device", type=str, default="auto")
    eval_parser.add_argument("--demo", action="store_true")
    eval_parser.add_argument("--lead_min", type=int, default=10,
                             help="評估 lead 天數下界（含），預設 10")
    eval_parser.add_argument("--lead_max", type=int, default=45,
                             help="評估 lead 天數上界（含），預設 45")
    eval_parser.add_argument(
        "--lead_groups", type=str, default="01-10,11-20,21-30,31-40",
        help="逐格點 RMSE 空間圖的時效段，預設 '01-10,11-20,21-30,31-40'",
    )
    eval_parser.add_argument(
        "--skip_rmse_maps", action="store_true",
        help="關閉逐格點 RMSE 空間熱力圖輸出",
    )
    eval_parser.add_argument(
        "--rmse_maps_dir", type=str, default=None,
        help="逐格點 RMSE 圖的輸出資料夾；預設 <output_dir>/rmse_spatial_maps/<task>_<var>",
    )
    eval_parser.add_argument(
        "--include_taiwan", dest="include_taiwan", action="store_true", default=None,
        help="把台灣納入中國陸地 mask 與背景邊界（預設啟用，由 cfg.data.include_taiwan 控制，預設 true）。"
             "首次使用且本地無 TWN shapefile 時會自動透過 cartopy Natural Earth 下載。",
    )
    eval_parser.add_argument(
        "--no_include_taiwan", dest="include_taiwan", action="store_false",
        help="不把台灣納入 mask / 邊界（沿用舊版 China-only 行為）。",
    )
    eval_parser.add_argument(
        "--taiwan_shapefile_path", type=str, default=None,
        help="可選的台灣 shapefile 路徑（例如 gadm41_TWN_0.shp）。"
             "未指定時會在 china_shapefile_dir 內搜尋 TWN/taiwan，再 fallback 到 Natural Earth。",
    )
    eval_parser.add_argument(
        "--rebuild_china_mask", action="store_true", default=None,
        help="即使已有 mask .npy 也根據 shapefile 重新生成（切換 include_taiwan 時建議使用一次）。"
             "未指定時優先讀 cfg.data.rebuild_china_mask，最後 fallback 為 False。",
    )

    # ---- serve ----
    serve_parser = subparsers.add_parser("serve", help="Start web UI")
    serve_parser.add_argument("--port", type=int, default=5000)
    serve_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
