"""
Flask backend for the China-focused S2S-ERA5 downscaling workbench.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import yaml
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
EVAL_DIR = PROJECT_ROOT / "eval_results"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_CACHE_ROOT = PROJECT_ROOT / "sample_cache"
BUILD_ID = "frontend-debug-20260426-1115"

CHINA_REGION = {
    "enabled": True,
    "lat_min": 15,
    "lat_max": 55,
    "lon_min": 70,
    "lon_max": 140,
}

TRAINING_STATE = {
    "running": False,
    "task_type": None,
    "config": None,
    "epoch": 0,
    "total_epochs": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "history": [],
    "components": {},
    "start_time": None,
    "phase": "idle",
    "error": None,
    "traceback": None,
    "error_file": None,
}

EVALUATION_STATE = {
    "running": False,
    "task_type": None,
    "config": None,
    "report_path": None,
    "report": None,
    "start_time": None,
    "error": None,
}

INFERENCE_STATE = {
    "running": False,
    "results": [],
    "task_type": None,
    "config": None,
    "output_dir": None,
    "start_time": None,
    "error": None,
}


def create_app():
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
    )
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    CORS(app)

    @app.after_request
    def no_cache_static(resp):
        resp.headers["Cache-Control"] = "no-store, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
        return resp

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def status():
        return jsonify(_clean_json({
            "build_id": BUILD_ID,
            "training": TRAINING_STATE,
            "evaluation": EVALUATION_STATE,
            "inference": INFERENCE_STATE,
            "gpu_available": _check_gpu(),
            "metadata": _metadata(),
        }))

    @app.route("/api/metadata")
    def metadata():
        return jsonify(_clean_json(_metadata()))

    @app.route("/api/version")
    def version():
        return jsonify({
            "build_id": BUILD_ID,
            "project_root": str(PROJECT_ROOT),
            "app_file": __file__,
            "python": sys.executable,
            "dont_write_bytecode": sys.dont_write_bytecode,
        })

    @app.route("/api/debug/permissions")
    def debug_permissions():
        targets = [
            PROJECT_ROOT,
            CHECKPOINT_DIR,
            OUTPUT_DIR,
            EVAL_DIR,
            Path(tempfile.gettempdir()),
        ]
        results = []
        for target in targets:
            results.append(_write_probe(target))
        return jsonify(_clean_json({"build_id": BUILD_ID, "results": results}))

    @app.route("/api/errors/clear", methods=["POST"])
    def clear_errors():
        TRAINING_STATE["error"] = None
        TRAINING_STATE["traceback"] = None
        TRAINING_STATE["error_file"] = None
        EVALUATION_STATE["error"] = None
        EVALUATION_STATE["traceback"] = None
        INFERENCE_STATE["error"] = None
        INFERENCE_STATE["traceback"] = None
        return jsonify({"status": "cleared"})

    @app.route("/api/config/<name>")
    def get_config(name):
        cfg = _load_config(name)
        return jsonify(_clean_json(cfg))

    @app.route("/api/train", methods=["POST"])
    def start_training():
        if TRAINING_STATE["running"]:
            return jsonify({"error": "训练正在运行"}), 400
        try:
            data = request.json or {}
            cfg_name = data.get("config")
            cfg = _request_config(data)

            tcfg = cfg.setdefault("training", {})
            tcfg["epochs"] = int(data.get("epochs", tcfg.get("epochs", 5)))
            tcfg["batch_size"] = int(data.get("batch_size", tcfg.get("batch_size", 1)))
            tcfg["lr"] = float(data.get("lr", tcfg.get("lr", 1e-4)))
            if data.get("save_dir"):
                tcfg["save_dir"] = data["save_dir"]
            tcfg["num_workers"] = int(data.get("num_workers", 0))

            TRAINING_STATE.update({
                "running": True,
                "task_type": cfg.get("task_type"),
                "config": cfg_name,
                "epoch": 0,
                "total_epochs": tcfg["epochs"],
                "train_loss": 0.0,
                "val_loss": 0.0,
                "history": [],
                "components": {},
                "start_time": time.time(),
                "phase": "queued",
                "error": None,
                "traceback": None,
                "error_file": None,
            })

            def run_training():
                try:
                    TRAINING_STATE["phase"] = "importing training modules"
                    from train import train

                    def log_cb(entry):
                        TRAINING_STATE["phase"] = "training"
                        TRAINING_STATE["epoch"] = entry.get("epoch", 0)
                        TRAINING_STATE["train_loss"] = entry.get("train_loss", 0.0)
                        TRAINING_STATE["val_loss"] = entry.get("val_loss", 0.0)
                        TRAINING_STATE["components"] = entry.get("train_components", {})
                        TRAINING_STATE["history"].append(entry)

                    TRAINING_STATE["phase"] = "initializing dataset/model"
                    train(cfg, log_callback=log_cb, device_str=data.get("device", "cuda"))
                    TRAINING_STATE["phase"] = "finished"
                except Exception as e:
                    _record_training_error(e, phase="failed in training thread")
                    traceback.print_exc()
                finally:
                    TRAINING_STATE["running"] = False

            threading.Thread(target=run_training, daemon=True).start()
            return jsonify({"status": "started", "task_type": cfg.get("task_type"), "build_id": BUILD_ID})
        except Exception as e:
            _record_training_error(e, phase="failed before training thread")
            TRAINING_STATE["running"] = False
            return jsonify(_clean_json({
                "error": str(e),
                "traceback": TRAINING_STATE.get("traceback"),
                "error_file": TRAINING_STATE.get("error_file"),
                "build_id": BUILD_ID,
            })), 500

    @app.route("/api/train/stop", methods=["POST"])
    def stop_training():
        TRAINING_STATE["running"] = False
        return jsonify({"status": "stop_requested"})

    @app.route("/api/train/history")
    def train_history():
        cfg_name = request.args.get("config")
        try:
            cfg = _load_config(cfg_name)
            path = _training_log_for_config(cfg)
            if not path or not path.exists():
                return jsonify({"history": [], "path": None})
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            return jsonify(_clean_json({"history": history, "path": _rel(path)}))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/evaluate", methods=["POST"])
    def start_evaluation():
        if EVALUATION_STATE["running"]:
            return jsonify({"error": "评估正在运行"}), 400

        data = request.json or {}
        cfg_name = data.get("config")
        cfg = _request_config(data)

        EVALUATION_STATE.update({
            "running": True,
            "task_type": cfg.get("task_type"),
            "config": cfg_name,
            "report_path": None,
            "report": None,
            "start_time": time.time(),
            "error": None,
        })

        def run_evaluation():
            try:
                from evaluate import evaluate
                from evaluation.rmse_spatial import parse_lead_groups_arg

                out_dir = _safe_output_dir(data.get("output_dir"), EVAL_DIR)
                lead_groups_arg = data.get("lead_groups", "01-10,11-20,21-30,31-40")
                report = evaluate(
                    cfg,
                    _safe_optional_path(data.get("checkpoint")),
                    str(out_dir),
                    int(data.get("max_samples", 20)),
                    data.get("device", "auto"),
                    lead_min=int(data.get("lead_min", 10)),
                    lead_max=int(data.get("lead_max", 45)),
                    lead_groups=parse_lead_groups_arg(lead_groups_arg),
                    save_rmse_spatial=bool(data.get("save_rmse_maps", True)),
                    rmse_maps_dir=str(_safe_output_dir(data.get("rmse_maps_dir"), out_dir / "rmse_spatial_maps" / cfg.get("task_type", "run")))
                    if data.get("rmse_maps_dir") else None,
                )
                EVALUATION_STATE["report"] = report
                EVALUATION_STATE["report_path"] = _find_latest_report(out_dir)
            except Exception as e:
                EVALUATION_STATE["error"] = str(e)
                traceback.print_exc()
            finally:
                EVALUATION_STATE["running"] = False

        threading.Thread(target=run_evaluation, daemon=True).start()
        return jsonify({"status": "started", "task_type": cfg.get("task_type")})

    @app.route("/api/infer", methods=["POST"])
    def start_inference():
        if INFERENCE_STATE["running"]:
            return jsonify({"error": "推理正在运行"}), 400

        data = request.json or {}
        cfg_name = data.get("config")
        cfg = _request_config(data)
        out_dir = _safe_output_dir(data.get("output_dir"), OUTPUT_DIR / cfg.get("task_type", "run"))

        INFERENCE_STATE.update({
            "running": True,
            "task_type": cfg.get("task_type"),
            "config": cfg_name,
            "results": [],
            "output_dir": str(out_dir),
            "start_time": time.time(),
            "error": None,
        })

        def run_inference():
            try:
                from infer import infer

                results = infer(
                    cfg,
                    _safe_optional_path(data.get("checkpoint")),
                    str(out_dir),
                    int(data.get("max_samples", 5)),
                    device_str=data.get("device", "auto"),
                )
                INFERENCE_STATE["results"] = results
            except Exception as e:
                INFERENCE_STATE["error"] = str(e)
                traceback.print_exc()
            finally:
                INFERENCE_STATE["running"] = False

        threading.Thread(target=run_inference, daemon=True).start()
        return jsonify({"status": "started", "task_type": cfg.get("task_type")})

    @app.route("/api/eval/report")
    def eval_report():
        rel = request.args.get("path")
        path = _safe_existing_project_path(rel) if rel else _latest_file(EVAL_DIR, "eval_report*.json")
        if not path:
            return jsonify({"error": "没有找到评估报告"}), 404
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify(_clean_json({"path": _rel(path), "report": report}))

    @app.route("/api/rmse-image")
    def rmse_image():
        rel = request.args.get("path")
        path = _safe_existing_project_path(rel)
        if not path or path.suffix.lower() != ".png":
            return jsonify({"error": "RMSE 图不存在"}), 404
        return send_file(path, mimetype="image/png")

    @app.route("/api/dataset/heatmap")
    def dataset_heatmap():
        source = request.args.get("source", "sample_cache")
        task_key = request.args.get("task", "task_b")
        if source == "demo":
            return jsonify(_clean_json(_demo_heatmap()))
        if source != "sample_cache":
            return jsonify({"error": "当前热力图接口支持 demo 与 sample_cache"}), 400
        try:
            return jsonify(_clean_json(_sample_cache_heatmap(
                task_key=task_key,
                sample_idx=int(request.args.get("sample", 0)),
                lead_idx=int(request.args.get("lead", 0)),
                var_name=request.args.get("var", "wind10m"),
                model=request.args.get("model", "mean"),
                field=request.args.get("field", "input"),
            )))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/inference/frames")
    def inference_frames():
        rel = request.args.get("path")
        path = _safe_existing_project_path(rel) if rel else _latest_file(OUTPUT_DIR, "pred_*.npz")
        if not path:
            return jsonify({"error": "没有找到推理输出 .npz"}), 404
        try:
            return jsonify(_clean_json(_prediction_frames(path)))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


def _metadata():
    return {
        "region": CHINA_REGION,
        "configs": _list_configs(),
        "datasets": _list_datasets(),
        "checkpoints": _list_checkpoints(),
        "training_logs": _list_training_logs(),
        "reports": _list_reports(),
        "rmse_images": _list_rmse_images(),
        "predictions": _list_predictions(),
    }


def _record_training_error(exc, phase):
    tb = traceback.format_exc()
    TRAINING_STATE["error"] = str(exc)
    TRAINING_STATE["traceback"] = tb
    TRAINING_STATE["phase"] = phase
    written = []
    for path in [
        PROJECT_ROOT / "frontend_last_training_error.txt",
        Path(tempfile.gettempdir()) / "downsampling_focas_last_training_error.txt",
    ]:
        try:
            path.write_text(tb, encoding="utf-8")
            written.append(str(path))
        except Exception:
            pass
    TRAINING_STATE["error_file"] = written[0] if written else None


def _write_probe(target):
    target = Path(target)
    out = {"path": str(target), "exists": target.exists(), "ok": False}
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / f".permission_probe_{os.getpid()}_{int(time.time() * 1000)}.tmp"
        probe.write_text("ok", encoding="utf-8")
        out["readback"] = probe.read_text(encoding="utf-8")
        probe.unlink()
        out["ok"] = True
    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    return out


def _load_config(name):
    if not name:
        name = "task_a.yaml"
    name = Path(str(name)).name
    if not name.endswith(".yaml"):
        name += ".yaml"
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"配置不存在: {name}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _request_config(data):
    cfg = _load_config(data.get("config") or _default_config_for_task(data.get("task_type")))
    dcfg = cfg.setdefault("data", {})
    dcfg["region"] = dict(CHINA_REGION)
    dataset_mode = data.get("dataset_mode", "raw")
    dcfg["demo_mode"] = dataset_mode == "demo" or bool(data.get("demo", False))
    dcfg["use_sample_cache"] = dataset_mode == "sample_cache"
    if dcfg["use_sample_cache"]:
        dcfg["sample_cache_root"] = str(SAMPLE_CACHE_ROOT)
        _sync_config_to_sample_cache(cfg)
    if data.get("models"):
        dcfg["models"] = data["models"]
    return cfg


def _sync_config_to_sample_cache(cfg):
    dcfg = cfg.setdefault("data", {})
    mcfg = cfg.setdefault("model", {})
    task_key = "task_a" if cfg.get("task_type") == "history2era5" else "task_b"
    meta_path = SAMPLE_CACHE_ROOT / task_key / "_meta.json"
    if not meta_path.exists():
        return
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    grid_h = int(meta.get("grid_h", dcfg.get("grid_h", 121)))
    grid_w = int(meta.get("grid_w", dcfg.get("grid_w", 240)))
    dcfg["s2s_grid_h"] = grid_h
    dcfg["s2s_grid_w"] = grid_w
    dcfg["era5_grid_h"] = grid_h
    dcfg["era5_grid_w"] = grid_w
    dcfg["grid_h"] = grid_h
    dcfg["grid_w"] = grid_w
    dcfg["s2s_vars"] = meta.get("s2s_vars", dcfg.get("s2s_vars", []))
    dcfg["era5_target_vars"] = meta.get("era5_target_vars", dcfg.get("era5_target_vars", []))
    dcfg["lead_times"] = meta.get("lead_times", dcfg.get("lead_times", []))
    dcfg["num_leads"] = int(meta.get("num_leads", dcfg.get("num_leads", 45)))
    if meta.get("history_len") is not None:
        dcfg["history_len"] = int(meta["history_len"])
        mcfg["history_len"] = int(meta["history_len"])
    mcfg["c_in"] = len(dcfg["s2s_vars"]) or int(mcfg.get("c_in", 1))
    sample_files = meta.get("samples", [])
    if sample_files:
        with np.load(SAMPLE_CACHE_ROOT / task_key / sample_files[0]) as sample:
            if "target_uv" in sample.files:
                mcfg["c_out_uv"] = int(sample["target_uv"].shape[1])


def _default_config_for_task(task_type):
    return "task_b.yaml" if task_type == "fcst2era5fit" else "task_a.yaml"


def _list_configs():
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            cfg = _load_config(path.name)
            data = cfg.get("data", {})
            train = cfg.get("training", {})
            out.append({
                "name": path.name,
                "task_type": cfg.get("task_type"),
                "target_vars": data.get("era5_target_vars", []),
                "input_vars": data.get("s2s_vars", []),
                "epochs": train.get("epochs"),
                "batch_size": train.get("batch_size"),
                "lr": train.get("lr"),
                "save_dir": train.get("save_dir"),
                "region": data.get("region", CHINA_REGION),
            })
        except Exception as e:
            out.append({"name": path.name, "error": str(e)})
    return out


def _list_datasets():
    datasets = [
        {"id": "raw", "name": "ERA5 / S2S 原始数据", "available": (PROJECT_ROOT / "ERA5").exists() and (PROJECT_ROOT / "S2S").exists()},
        {"id": "sample_cache", "name": "预构建 sample_cache", "available": SAMPLE_CACHE_ROOT.exists()},
        {"id": "demo", "name": "Demo 随机数据", "available": True},
    ]
    caches = []
    for task_key in ("task_a", "task_b"):
        meta_path = SAMPLE_CACHE_ROOT / task_key / "_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            caches.append({
                "task": task_key,
                "samples": len(meta.get("samples", [])),
                "lead_times": meta.get("lead_times", []),
                "models": meta.get("models", []),
                "s2s_vars": meta.get("s2s_vars", []),
                "target_vars": meta.get("era5_target_vars", []),
            })
    return {"modes": datasets, "sample_caches": caches}


def _list_checkpoints():
    out = []
    if not CHECKPOINT_DIR.exists():
        return out
    for path in sorted(CHECKPOINT_DIR.rglob("*.pt")):
        try:
            stat = path.stat()
            out.append({
                "name": path.name,
                "run": path.parent.name,
                "path": _rel(path),
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": stat.st_mtime,
            })
        except OSError:
            pass
    return out


def _list_training_logs():
    out = []
    if not CHECKPOINT_DIR.exists():
        return out
    for path in sorted(CHECKPOINT_DIR.rglob("training_log.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            out.append({
                "path": _rel(path),
                "run": path.parent.name,
                "epochs": len(history) if isinstance(history, list) else 0,
                "modified": path.stat().st_mtime,
            })
        except Exception:
            pass
    return out


def _training_log_for_config(cfg):
    save_dir = cfg.get("training", {}).get("save_dir")
    if save_dir:
        path = Path(save_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path / "training_log.json"
    task = cfg.get("task_type", "run")
    task_key = "task_a" if task == "history2era5" else "task_b"
    return CHECKPOINT_DIR / task_key / "training_log.json"


def _list_reports():
    out = []
    for path in sorted(EVAL_DIR.rglob("eval_report*.json")) if EVAL_DIR.exists() else []:
        try:
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
            out.append({
                "path": _rel(path),
                "name": path.name,
                "task_type": report.get("task_type"),
                "target_vars": report.get("target_vars", []),
                "num_samples": report.get("num_samples"),
                "lead_range": report.get("lead_range") or report.get("lead_times"),
                "overall_metrics": report.get("overall_metrics", {}),
                "modified": path.stat().st_mtime,
            })
        except Exception:
            pass
    return out


def _list_rmse_images():
    out = []
    root = EVAL_DIR / "rmse_spatial_maps"
    for path in sorted(root.rglob("*.png")) if root.exists() else []:
        out.append({
            "path": _rel(path),
            "name": path.name,
            "run": path.parent.name,
            "url": f"/api/rmse-image?path={_rel(path)}",
        })
    return out


def _list_predictions():
    out = []
    for path in sorted(OUTPUT_DIR.rglob("pred_*.npz")) if OUTPUT_DIR.exists() else []:
        out.append({
            "path": _rel(path),
            "name": path.name,
            "run": path.parent.name,
            "modified": path.stat().st_mtime,
        })
    return out


def _sample_cache_heatmap(task_key, sample_idx, lead_idx, var_name, model, field):
    root = SAMPLE_CACHE_ROOT / task_key
    meta_path = root / "_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"没有找到 {task_key} 的 sample_cache")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    samples = sorted(meta.get("samples", []))
    if not samples:
        raise RuntimeError("sample_cache 为空")
    sample = samples[max(0, min(sample_idx, len(samples) - 1))]
    sample_path = root / sample
    lat = np.asarray(meta.get("lat", []), dtype=np.float32)
    lon = np.asarray(meta.get("lon", []), dtype=np.float32)

    with np.load(sample_path) as data:
        if field == "target":
            target = data["target_uv"]
            li = max(0, min(lead_idx, target.shape[0] - 1))
            if target.shape[1] >= 2:
                matrix = np.sqrt(target[li, 0] ** 2 + target[li, 1] ** 2)
                label = "ERA5 target wind speed"
            else:
                matrix = target[li, 0]
                label = "ERA5 target"
        else:
            key = "x_forecast" if "x_forecast" in data.files else "x_history"
            arr = data[key]
            var_idx = _var_index(meta.get("s2s_vars", []), var_name)
            if key == "x_forecast":
                li = max(0, min(lead_idx, arr.shape[1] - 1))
                if model == "mean":
                    matrix = arr[:, li, var_idx].mean(axis=0)
                else:
                    mi = max(0, min(int(model), arr.shape[0] - 1))
                    matrix = arr[mi, li, var_idx]
            else:
                li = max(0, min(lead_idx, arr.shape[1] - 1))
                if model == "mean":
                    matrix = arr[:, li, var_idx].mean(axis=0)
                else:
                    mi = max(0, min(int(model), arr.shape[0] - 1))
                    matrix = arr[mi, li, var_idx]
            label = f"S2S {var_name}"

    matrix, lat_sub, lon_sub = _crop_to_china(matrix, lat, lon)
    payload = _matrix_payload(matrix, lat_sub, lon_sub)
    payload.update({
        "label": label,
        "sample": sample,
        "task": task_key,
        "lead": meta.get("lead_times", [])[lead_idx] if lead_idx < len(meta.get("lead_times", [])) else lead_idx + 1,
        "vars": meta.get("s2s_vars", []),
        "models": meta.get("models", []),
    })
    return payload


def _prediction_frames(path):
    frames = []
    with np.load(path) as data:
        uv = data["uv"]
        lead_ids = data["lead_ids"] if "lead_ids" in data.files else np.arange(1, uv.shape[0] + 1)
        for i in range(uv.shape[0]):
            if uv.shape[1] >= 2:
                matrix = np.sqrt(uv[i, 0] ** 2 + uv[i, 1] ** 2)
            else:
                matrix = uv[i, 0]
            lat = np.linspace(CHINA_REGION["lat_max"], CHINA_REGION["lat_min"], matrix.shape[0])
            lon = np.linspace(CHINA_REGION["lon_min"], CHINA_REGION["lon_max"], matrix.shape[1])
            frame = _matrix_payload(matrix, lat, lon, max_dim=80)
            frame["lead"] = int(lead_ids[i]) if i < len(lead_ids) else i + 1
            frames.append(frame)
    return {"path": _rel(path), "frames": frames}


def _demo_heatmap():
    lat = np.linspace(CHINA_REGION["lat_max"], CHINA_REGION["lat_min"], 64)
    lon = np.linspace(CHINA_REGION["lon_min"], CHINA_REGION["lon_max"], 96)
    lon2d, lat2d = np.meshgrid(lon, lat)
    matrix = 5.5 + 2.0 * np.sin((lon2d - 95) / 9) + 1.4 * np.cos((lat2d - 32) / 6)
    payload = _matrix_payload(matrix.astype(np.float32), lat, lon)
    payload.update({"label": "Demo wind speed", "sample": "synthetic", "task": "demo", "lead": 1})
    return payload


def _crop_to_china(matrix, lat, lon):
    matrix = np.asarray(matrix, dtype=np.float32)
    if lat.size != matrix.shape[-2] or lon.size != matrix.shape[-1]:
        lat = np.linspace(CHINA_REGION["lat_max"], CHINA_REGION["lat_min"], matrix.shape[-2])
        lon = np.linspace(CHINA_REGION["lon_min"], CHINA_REGION["lon_max"], matrix.shape[-1])
        return matrix, lat, lon
    lat_mask = (lat >= CHINA_REGION["lat_min"]) & (lat <= CHINA_REGION["lat_max"])
    lon_n = np.mod(lon, 360.0)
    lon_mask = (lon_n >= CHINA_REGION["lon_min"]) & (lon_n <= CHINA_REGION["lon_max"])
    lat_idx = np.flatnonzero(lat_mask)
    lon_idx = np.flatnonzero(lon_mask)
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return matrix, lat, lon
    return matrix[lat_idx, :][:, lon_idx], lat[lat_idx], lon[lon_idx]


def _matrix_payload(matrix, lat, lon, max_dim=96):
    matrix = np.asarray(matrix, dtype=np.float32)
    sy = max(1, math.ceil(matrix.shape[0] / max_dim))
    sx = max(1, math.ceil(matrix.shape[1] / max_dim))
    matrix = matrix[::sy, ::sx]
    lat = np.asarray(lat)[::sy]
    lon = np.asarray(lon)[::sx]
    finite = np.isfinite(matrix)
    values = np.where(finite, matrix, np.nan)
    return {
        "values": values.tolist(),
        "lat": lat.astype(float).tolist(),
        "lon": lon.astype(float).tolist(),
        "min": float(np.nanmin(values)) if finite.any() else None,
        "max": float(np.nanmax(values)) if finite.any() else None,
        "mean": float(np.nanmean(values)) if finite.any() else None,
        "shape": list(values.shape),
        "region": CHINA_REGION,
    }


def _var_index(vars_, name):
    try:
        return list(vars_).index(name)
    except ValueError:
        return 0


def _safe_output_dir(value, default):
    path = Path(value) if value else Path(default)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise ValueError("输出目录必须位于项目目录内")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_optional_path(value):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        raise ValueError("路径必须位于项目目录内")
    return str(path)


def _safe_existing_project_path(rel):
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        path = path.resolve()
    except OSError:
        return None
    if not path.exists():
        return None
    if PROJECT_ROOT not in path.parents and path != PROJECT_ROOT:
        return None
    return path


def _latest_file(root, pattern):
    if not root.exists():
        return None
    files = list(root.rglob(pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _find_latest_report(out_dir):
    path = _latest_file(Path(out_dir), "eval_report*.json")
    return _rel(path) if path else None


def _rel(path):
    return str(Path(path).resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")


def _clean_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _check_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
