"""
Evaluation script for S2S-ERA5 downscaling.

支援兩種模型分支（由 cfg.model.c_out_uv 自動判斷）：

1. Vector (c_out_uv == 2)：
   輸出 (u10, v10)，指標包含：
     u / v 各自  : RMSE, MAE, Bias, R, R²
     wind speed  : RMSE, MAE, Bias
     direction   : MAE(deg), Accuracy(±30°)
     convergence : RMSE, MAE, R
     spatial     : SSIM(u), SSIM(v)
     ACC         : 以所有 val 樣本目標平均作為 climatology 的距平相關係數

2. Scalar (c_out_uv == 1)：
   輸出單變數（wind10m / wind100m / ssrd），指標包含：
     RMSE, MAE, Bias, R, R², SSIM, ACC

指標均在「原尺度」計算（會乘回 ERA5 target divisor，例如 ssrd 會從無量綱乘 1e7
還原為 J/m²），方便直接與觀測/其他模式比較。

評估範圍：
    --lead_min / --lead_max 可指定要評估的 lead 天數區間。預設 10–45。
"""

import os
import json
import argparse
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import build_model
from datasets import build_dataset
from datasets.base_dataset import resolve_era5_target_divisors
from train import resolve_device, apply_demo_overrides
from utils.grid_config import sync_model_grids_from_data
from evaluation.rmse_spatial import (
    DEFAULT_LEAD_GROUPS,
    DEFAULT_UNITS,
    compute_all_grid_rmse_maps,
    parse_lead_groups_arg,
    plot_rmse_spatial_grid,
    plot_single_variable_rmse_maps,
    save_rmse_maps,
    get_or_build_china_land_mask,
)
from evaluation.baseline_compare import (
    compute_baseline_comparison,
    print_baseline_comparison,
    save_baseline_comparison_outputs,
)


# ============================================================
# 基礎指標
# ============================================================

def _flat(a):
    return np.asarray(a, dtype=np.float64).ravel()


def _finite_pair(pred, target):
    """返回 pred/target 同時為有限值的位置，用於支持中國陸地 mask 後的 NaN。"""
    p = _flat(pred)
    t = _flat(target)
    ok = np.isfinite(p) & np.isfinite(t)
    return p[ok], t[ok]


def _finite_triple(pred, target, clim):
    p = _flat(pred)
    t = _flat(target)
    c = _flat(clim)
    ok = np.isfinite(p) & np.isfinite(t) & np.isfinite(c)
    return p[ok], t[ok], c[ok]


def rmse(pred, target):
    p, t = _finite_pair(pred, target)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(pred, target):
    p, t = _finite_pair(pred, target)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - t)))


def bias(pred, target):
    p, t = _finite_pair(pred, target)
    if p.size == 0:
        return float("nan")
    return float(np.mean(p - t))


def pearson_r(pred, target):
    p, t = _finite_pair(pred, target)
    if p.size < 2 or p.std() < 1e-12 or t.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def r_squared(pred, target):
    p, t = _finite_pair(pred, target)
    if p.size == 0:
        return 0.0
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def ssim_2d(pred, target, C1=0.01 ** 2, C2=0.03 ** 2):
    # mask 外是 NaN，因此這裡只用中國陸地內 finite 點近似計算 SSIM 統計量
    p, t = _finite_pair(pred, target)
    if p.size == 0:
        return 0.0
    mu_p = p.mean(); mu_t = t.mean()
    sig_p = p.std(); sig_t = t.std()
    sig_pt = np.mean((p - mu_p) * (t - mu_t))
    num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (sig_p ** 2 + sig_t ** 2 + C2)
    if abs(den) < 1e-12:
        return 1.0
    return float(num / den)


def acc_score(pred, target, climatology):
    """
    Anomaly Correlation Coefficient in 2D：
        pred, target, climatology 都是 (H, W)
    mask 外 NaN 會被自動忽略。
    """
    p, t, c = _finite_triple(pred, target, climatology)
    if p.size == 0:
        return 0.0
    a_pred = p - c
    a_target = t - c
    num = np.sum(a_pred * a_target)
    den = np.sqrt(np.sum(a_pred ** 2) * np.sum(a_target ** 2))
    if den < 1e-12:
        return 0.0
    return float(num / den)


# ============================================================
# 向量 (u, v) 專用
# ============================================================

def wind_speed(u, v):
    return np.sqrt(np.asarray(u, dtype=np.float64) ** 2
                   + np.asarray(v, dtype=np.float64) ** 2)


def wind_direction_deg(u, v):
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def _direction_signed_diff_deg(pred_u, pred_v, target_u, target_v):
    """
    回傳預測風向 - 觀測風向後 wrap 到 [-180, 180] 的有限值陣列。
    處理 0°/360° 圓周性，避免風向接近 0° 時的假性大誤差。
    """
    d_pred = wind_direction_deg(pred_u, pred_v)
    d_target = wind_direction_deg(target_u, target_v)
    diff = ((d_pred - d_target) + 180.0) % 360.0 - 180.0
    return diff[np.isfinite(diff)]


def direction_mae_deg(pred_u, pred_v, target_u, target_v):
    diff = _direction_signed_diff_deg(pred_u, pred_v, target_u, target_v)
    if diff.size == 0:
        return float("nan")
    return float(np.mean(np.abs(diff)))


def direction_rmse_deg(pred_u, pred_v, target_u, target_v):
    """
    風向 RMSE（單位：度）。
    用 signed-diff wrap 到 [-180, 180] 後平方平均開根號。
    """
    diff = _direction_signed_diff_deg(pred_u, pred_v, target_u, target_v)
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


def direction_accuracy(pred_u, pred_v, target_u, target_v, threshold=30.0):
    diff = _direction_signed_diff_deg(pred_u, pred_v, target_u, target_v)
    if diff.size == 0:
        return 0.0
    return float((np.abs(diff) < threshold).mean())


def compute_convergence(u, v, dx=1.0, dy=1.0):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dudx = np.zeros_like(u)
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / dx
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / dx
    dvdy = np.zeros_like(v)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * dy)
    dvdy[0, :] = (v[1, :] - v[0, :]) / dy
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / dy
    return -(dudx + dvdy)


# ============================================================
# 單一樣本 × 單一 lead 指標
# ============================================================

def metrics_scalar(pred_2d, target_2d, clim_2d):
    """純量變數 (wind10m / wind100m / ssrd) 的全套指標"""
    return {
        "rmse": round(rmse(pred_2d, target_2d), 6),
        "mae": round(mae(pred_2d, target_2d), 6),
        "bias": round(bias(pred_2d, target_2d), 6),
        "r": round(pearson_r(pred_2d, target_2d), 4),
        "r2": round(r_squared(pred_2d, target_2d), 4),
        "ssim": round(ssim_2d(pred_2d, target_2d), 4),
        "acc": round(acc_score(pred_2d, target_2d, clim_2d), 4),
    }


def metrics_vector(pred_uv_k, target_uv_k, clim_uv_k):
    """(u10, v10) 向量場 + speed + direction + convergence + ACC"""
    pu, pv = pred_uv_k[0], pred_uv_k[1]
    tu, tv = target_uv_k[0], target_uv_k[1]
    cu, cv = clim_uv_k[0], clim_uv_k[1]

    ws_pred = wind_speed(pu, pv)
    ws_target = wind_speed(tu, tv)
    ws_clim = wind_speed(cu, cv)

    conv_pred = compute_convergence(pu, pv)
    conv_target = compute_convergence(tu, tv)

    out = {
        "u10": {
            "rmse": round(rmse(pu, tu), 5),
            "mae": round(mae(pu, tu), 5),
            "bias": round(bias(pu, tu), 5),
            "r": round(pearson_r(pu, tu), 4),
            "r2": round(r_squared(pu, tu), 4),
            "acc": round(acc_score(pu, tu, cu), 4),
        },
        "v10": {
            "rmse": round(rmse(pv, tv), 5),
            "mae": round(mae(pv, tv), 5),
            "bias": round(bias(pv, tv), 5),
            "r": round(pearson_r(pv, tv), 4),
            "r2": round(r_squared(pv, tv), 4),
            "acc": round(acc_score(pv, tv, cv), 4),
        },
        "wind_speed": {
            "rmse": round(rmse(ws_pred, ws_target), 5),
            "mae": round(mae(ws_pred, ws_target), 5),
            "bias": round(bias(ws_pred, ws_target), 5),
            "acc": round(acc_score(ws_pred, ws_target, ws_clim), 4),
        },
        "wind_direction": {
            "rmse_deg": round(direction_rmse_deg(pu, pv, tu, tv), 2),
            "mae_deg": round(direction_mae_deg(pu, pv, tu, tv), 2),
            "accuracy_30deg": round(direction_accuracy(pu, pv, tu, tv, 30.0), 4),
        },
        "convergence": {
            "rmse": round(rmse(conv_pred, conv_target), 6),
            "mae": round(mae(conv_pred, conv_target), 6),
            "r": round(pearson_r(conv_pred, conv_target), 4),
        },
        "spatial": {
            "ssim_u": round(ssim_2d(pu, tu), 4),
            "ssim_v": round(ssim_2d(pv, tv), 4),
        },
    }
    return out


# ============================================================
# 主流程
# ============================================================

def _scale_back(arr, divisor):
    """乘回 divisor，把網路輸出/標籤還原到 ERA5 原單位（例如 ssrd: ×1e7）"""
    d = float(divisor or 1.0)
    if d == 1.0:
        return arr
    return (np.asarray(arr, dtype=np.float32) * np.float32(d)).astype(np.float32)


def _bilinear_resample(arr_kchw, out_h, out_w):
    """
    arr_kchw: (K, C, H, W) numpy float — 雙線性上採樣到 (out_h, out_w)
    僅用於把 S2S baseline 升到 ERA5 原生網格以便比較
    """
    t = torch.from_numpy(np.asarray(arr_kchw, dtype=np.float32))
    K, C, H, W = t.shape
    t = t.reshape(K * C, 1, H, W)
    t = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return t.reshape(K, C, out_h, out_w).numpy()


@torch.no_grad()
def evaluate(
    cfg,
    checkpoint_path,
    output_dir="eval_results",
    max_samples=50,
    device_str=None,
    lead_min=10,
    lead_max=45,
    lead_groups=None,
    save_rmse_spatial=True,
    rmse_maps_dir=None,
    china_shapefile_dir=None,
    china_land_mask_path=None,
    rebuild_china_mask=None,
    require_china_mask=None,
    include_taiwan=None,
    taiwan_shapefile_path=None,
):
    task_type = cfg["task_type"]

    # ------------------------------------------------------------
    # China land mask config（解析優先序）：
    #   1) 函式 / CLI 顯式傳入非 None → 一律以該值為準
    #   2) cfg["data"] 內有對應欄位     → 用 cfg 的設定
    #   3) 寫死的安全預設                → 最後 fallback
    # 對應的 cfg["data"] 欄位：
    #   china_shapefile_dir, china_land_mask_path, rebuild_china_mask,
    #   require_china_mask, mask_to_china_land,
    #   include_taiwan, taiwan_shapefile_path
    # ------------------------------------------------------------
    data_cfg = cfg.get("data", {})

    def _resolve(cli_val, cfg_key, default):
        if cli_val is not None:
            return cli_val
        if cfg_key in data_cfg and data_cfg[cfg_key] is not None:
            return data_cfg[cfg_key]
        return default

    mask_to_china_land = bool(_resolve(None, "mask_to_china_land", True))
    china_shapefile_dir = _resolve(china_shapefile_dir, "china_shapefile_dir",
                                   r"D:\downsampling_focas\shapefile")
    china_land_mask_path = _resolve(china_land_mask_path, "china_land_mask_path", None)
    rebuild_china_mask = bool(_resolve(rebuild_china_mask, "rebuild_china_mask", False))
    require_china_mask = bool(_resolve(require_china_mask, "require_china_mask", True))
    include_taiwan = bool(_resolve(include_taiwan, "include_taiwan", True))
    taiwan_shapefile_path = _resolve(taiwan_shapefile_path, "taiwan_shapefile_path", None)

    device = resolve_device(device_str)

    print(f"[Eval] Task: {task_type}")
    print(f"[Eval] Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    demo = cfg["data"].get("demo_mode", False)
    if demo:
        apply_demo_overrides(cfg)
    sync_model_grids_from_data(cfg, demo=demo)

    model = build_model(cfg).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded checkpoint: {checkpoint_path}")
    else:
        print("[Eval] No checkpoint, using random weights (demo only)")

    model.eval()
    use_amp = device.type == "cuda"

    c_out_uv = int(cfg["model"].get("c_out_uv", 2))
    target_vars = list(cfg["data"].get("era5_target_vars", ["u10", "v10"]))
    target_divs = resolve_era5_target_divisors(cfg["data"])
    is_scalar = (c_out_uv == 1) or (len(target_vars) == 1)
    var_name = target_vars[0] if is_scalar else "uv"
    print(f"[Eval] Target = {target_vars}  (scalar={is_scalar})")
    print(f"[Eval] Target divisors = {target_divs}")
    print(f"[Eval] Lead range = [{lead_min}, {lead_max}] days")

    dataset = build_dataset(cfg, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)

    # ---- 收集全部 pred/target（原尺度），後續算 climatology + per-lead 統計 ----
    all_preds = []         # list of (K, C, H, W)
    all_targets = []       # list of (K, C, H, W)
    all_baseline = []      # list of (K, C, H, W) or None，沿用：multi-model mean
    all_baseline_pm = []   # list of (M, K, C, H, W) or None，新增：per-model baseline
    all_leads = []         # list of (K,)
    all_baseline_ok = task_type == "fcst2era5fit"

    s2s_var_list = list(cfg["data"].get("s2s_vars", []))
    s2s_model_names = list(cfg["data"].get("models", []))
    baseline_compat = (
        all_baseline_ok
        and s2s_var_list == target_vars
        and len(s2s_model_names) > 0
    )
    s2s_divisors = {v: float(d) for v, d
                    in (cfg["data"].get("s2s_var_divisors") or {}).items()}

    for i, batch in enumerate(loader):
        if i >= max_samples:
            break

        x_static = batch["x_static"].to(device, non_blocking=True)
        lead_ids = batch["lead_ids"].to(device, non_blocking=True)
        target_uv = batch["target_uv"].numpy()[0]  # (K, C, H, W)

        if task_type == "history2era5":
            x_input = batch["x_history"].to(device, non_blocking=True)
        else:
            x_input = batch["x_forecast"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x_input, x_static, lead_ids)
        pred_uv = pred["uv"].float().cpu().numpy()[0]  # (K, C, H, W)

        # 把 divisor 乘回去，還原成 ERA5 原尺度（每個 channel 對應一個 target var）
        for c_idx, tv in enumerate(target_vars):
            d = target_divs.get(tv, 1.0)
            if d != 1.0:
                pred_uv[:, c_idx] = _scale_back(pred_uv[:, c_idx], d)
                target_uv[:, c_idx] = _scale_back(target_uv[:, c_idx], d)

        # Task B 的 baseline（S2S 輸入網格 → 雙線性升到 ERA5 網格再比）
        baseline_sample = None      # multi-model mean (沿用)
        baseline_per_model = None   # per-model (新增)
        if baseline_compat:
            raw_fcst = batch["x_forecast"].numpy()[0]   # (M, K, C_in, Hs, Ws)
            if raw_fcst.shape[2] == pred_uv.shape[1]:
                # 還原各模式 divisor
                fcst_scaled = raw_fcst.astype(np.float32, copy=True)
                for c_idx, tv in enumerate(target_vars):
                    d = s2s_divisors.get(tv, 1.0)
                    if d != 1.0:
                        fcst_scaled[:, :, c_idx] = _scale_back(fcst_scaled[:, :, c_idx], d)

                tgt_h, tgt_w = target_uv.shape[-2], target_uv.shape[-1]
                M_, K_, Cs_, Hs_, Ws_ = fcst_scaled.shape

                if (Hs_, Ws_) != (tgt_h, tgt_w):
                    # 把 (M, K, C, Hs, Ws) 攤成 (M*K, C, Hs, Ws) 做雙線性升採樣，再 reshape 回來
                    flat = fcst_scaled.reshape(M_ * K_, Cs_, Hs_, Ws_)
                    flat_up = _bilinear_resample(flat, tgt_h, tgt_w)
                    fcst_scaled = flat_up.reshape(M_, K_, Cs_, tgt_h, tgt_w)

                baseline_per_model = fcst_scaled.astype(np.float32, copy=False)
                baseline_sample = baseline_per_model.mean(axis=0)

        lead_ids_np = batch["lead_ids"].numpy()[0]
        all_preds.append(pred_uv)
        all_targets.append(target_uv)
        all_baseline.append(baseline_sample)
        all_baseline_pm.append(baseline_per_model)
        all_leads.append(lead_ids_np)

    if not all_preds:
        raise RuntimeError("沒有可評估的樣本，請檢查資料或 max_samples 設定。")

    all_preds = np.stack(all_preds)      # (N, K, C, H, W)
    all_targets = np.stack(all_targets)  # (N, K, C, H, W)
    lead_ids_arr = np.stack(all_leads)   # (N, K)  — dataset 裡是 [0..K-1]

    N, K, C, H, W = all_targets.shape
    print(f"[Eval] Collected N={N} samples, K={K} leads, C={C}, H={H}, W={W}")

    # ---- 中國（+台灣）陸地 mask：主評估與逐格點 RMSE 圖都限定在這個範圍內 ----
    eval_lat, eval_lon = _get_eval_lat_lon(dataset, H, W)
    china_land_mask = None
    china_boundary_path = None
    taiwan_boundary_path = None
    if mask_to_china_land:
        if china_land_mask_path is None:
            china_land_mask_path = os.path.join(china_shapefile_dir, "china_land_mask.npy")
        china_land_mask, china_boundary_path, taiwan_boundary_path = get_or_build_china_land_mask(
            lat=eval_lat,
            lon=eval_lon,
            target_shape=(H, W),
            mask_path=china_land_mask_path,
            shapefile_dir=china_shapefile_dir,
            rebuild=rebuild_china_mask,
            require=require_china_mask,
            include_taiwan=include_taiwan,
            taiwan_shapefile_path=taiwan_shapefile_path,
        )
        if china_land_mask is not None:
            mask5 = china_land_mask[None, None, None, :, :]
            all_preds = np.where(mask5, all_preds, np.nan)
            all_targets = np.where(mask5, all_targets, np.nan)
            land_label = "China+Taiwan land" if include_taiwan else "China land"
            print(f"[MASK] Applied {land_label} mask to all_preds/all_targets for all evaluation metrics.")
    else:
        print("[MASK] mask_to_china_land=False; evaluation is NOT limited to China land.")

    # ---- 建立 per-lead climatology = 該 lead 的 target 於所有樣本平均 ----
    with np.errstate(all="ignore"):
        climatology = np.nanmean(all_targets, axis=0)   # (K, C, H, W)

    # ---- 推導「lead 天數」。dataset.lead_ids 為 0-based index，真實 lead day 來自 cfg ----
    cfg_leads = cfg["data"].get("lead_times")
    if cfg_leads:
        lead_days = np.asarray(cfg_leads, dtype=np.int64)
    else:
        lead_days = np.arange(1, K + 1, dtype=np.int64)  # 預設 [1..K]
    assert lead_days.shape[0] == K, (
        f"lead_times 長度 {lead_days.shape[0]} 不等於預測 K={K}"
    )

    sel_k = [k for k in range(K) if lead_min <= int(lead_days[k]) <= lead_max]
    if not sel_k:
        raise RuntimeError(
            f"沒有 lead 落在 [{lead_min}, {lead_max}] 範圍。當前 lead_times = {list(lead_days)}"
        )

    # ---- 逐樣本 × 逐 lead 計算指標 ----
    per_lead_acc = defaultdict(lambda: defaultdict(list))  # lead_day → metric_path → [values]
    sample_records = []

    for n in range(N):
        sample_entry = {"sample_idx": n, "per_lead": {}}
        for k in sel_k:
            lday = int(lead_days[k])
            if is_scalar:
                p = all_preds[n, k, 0]
                t = all_targets[n, k, 0]
                c = climatology[k, 0]
                m = metrics_scalar(p, t, c)
                # baseline skill（wind speed / ssrd 都是純量比較）
                if all_baseline[n] is not None:
                    b = all_baseline[n][k, 0]
                    b_rmse = rmse(b, t)
                    m["skill_vs_baseline"] = round(1.0 - m["rmse"] / b_rmse, 4) if b_rmse > 1e-12 else 0.0
                sample_entry["per_lead"][f"lead_{lday:02d}"] = m
                for mk, mv in m.items():
                    per_lead_acc[lday][mk].append(float(mv))
            else:
                pk = all_preds[n, k]
                tk = all_targets[n, k]
                ck = climatology[k]
                m = metrics_vector(pk, tk, ck)
                if all_baseline[n] is not None and all_baseline[n].shape[1] >= 2:
                    bu, bv = all_baseline[n][k, 0], all_baseline[n][k, 1]
                    b_rmse = rmse(wind_speed(bu, bv), wind_speed(tk[0], tk[1]))
                    ws_rmse = m["wind_speed"]["rmse"]
                    m["skill_vs_baseline"] = round(1.0 - ws_rmse / b_rmse, 4) if b_rmse > 1e-12 else 0.0
                sample_entry["per_lead"][f"lead_{lday:02d}"] = m
                # 攤平 nested dict
                for cat, cat_val in m.items():
                    if isinstance(cat_val, dict):
                        for mk, mv in cat_val.items():
                            per_lead_acc[lday][f"{cat}.{mk}"].append(float(mv))
                    else:
                        per_lead_acc[lday][cat].append(float(cat_val))
        sample_records.append(sample_entry)

        if (n + 1) % 10 == 0 or n == 0:
            if is_scalar:
                any_lead = sel_k[0]
                lday = int(lead_days[any_lead])
                ms = sample_entry["per_lead"][f"lead_{lday:02d}"]
                print(f"  Sample {n}: [{var_name}] lead{lday:02d}  "
                      f"RMSE={ms['rmse']:.4f}  R={ms['r']:.3f}  ACC={ms['acc']:.3f}")
            else:
                any_lead = sel_k[0]
                lday = int(lead_days[any_lead])
                ms = sample_entry["per_lead"][f"lead_{lday:02d}"]
                print(f"  Sample {n}: lead{lday:02d}  "
                      f"u_rmse={ms['u10']['rmse']:.4f}  "
                      f"ws_rmse={ms['wind_speed']['rmse']:.4f}  "
                      f"dir_rmse={ms['wind_direction']['rmse_deg']:.1f}°  "
                      f"dir_mae={ms['wind_direction']['mae_deg']:.1f}°  "
                      f"ACC(ws)={ms['wind_speed']['acc']:.3f}")

    # ---- 聚合 per-lead 與 overall ----
    per_lead_summary = {}
    for lday, mmap in per_lead_acc.items():
        per_lead_summary[f"lead_{int(lday):02d}"] = {
            mk: round(float(np.mean(mv)), 6) for mk, mv in mmap.items()
        }

    overall = {}
    for mmap in per_lead_acc.values():
        for mk, mv in mmap.items():
            overall.setdefault(mk, []).extend(mv)
    overall = {mk: round(float(np.mean(mv)), 6) for mk, mv in overall.items()}

    # ---- 計算「到 lead k 為止的累計 RMSE/MAE/Bias」 ----
    # pooled = sqrt(mean of squared errors over leads 1..k, all samples, all pixels)
    # mean   = mean of per-lead RMSE for leads 1..k
    # 對 vector 模式同時提供 u/v/wind_speed 三套
    sel_lead_days = [int(lead_days[k]) for k in sel_k]
    cumulative = {}

    def _pooled_rmse(err2_kchw, kk_indices):
        """err2_kchw shape: (N, K, C, H, W) — squared errors. 對 leads kk_indices 取 pooled RMSE。"""
        sub = err2_kchw[:, kk_indices]
        return float(np.sqrt(np.nanmean(sub)))

    def _pooled_mae(err_kchw, kk_indices):
        sub = np.abs(err_kchw[:, kk_indices])
        return float(np.nanmean(sub))

    def _pooled_bias(err_kchw, kk_indices):
        sub = err_kchw[:, kk_indices]
        return float(np.nanmean(sub))

    if is_scalar:
        err = all_preds[:, :, 0] - all_targets[:, :, 0]   # (N, K, H, W)
        err2 = err ** 2
        per_lead_rmse_array = np.sqrt(np.nanmean(err2, axis=(0, 2, 3)))  # (K,)
        per_lead_mae_array = np.nanmean(np.abs(err), axis=(0, 2, 3))     # (K,)
        for idx, k in enumerate(sel_k):
            lday = int(lead_days[k])
            kk = sel_k[: idx + 1]                                  # 從第一個被評估 lead 到當前
            cumulative[f"lead_{lday:02d}"] = {
                "rmse_pooled_to_lead": round(_pooled_rmse(err2, kk), 6),
                "mae_pooled_to_lead": round(_pooled_mae(err, kk), 6),
                "bias_pooled_to_lead": round(_pooled_bias(err, kk), 6),
                "rmse_mean_to_lead": round(float(per_lead_rmse_array[kk].mean()), 6),
                "mae_mean_to_lead": round(float(per_lead_mae_array[kk].mean()), 6),
            }
            # 同步寫進 per_lead_summary 方便 JSON 一站式查
            per_lead_summary[f"lead_{lday:02d}"].update(cumulative[f"lead_{lday:02d}"])
    else:
        # u/v/ws 各做一份累計
        err_u = all_preds[:, :, 0] - all_targets[:, :, 0]
        err_v = all_preds[:, :, 1] - all_targets[:, :, 1]
        ws_p = np.sqrt(all_preds[:, :, 0] ** 2 + all_preds[:, :, 1] ** 2)
        ws_t = np.sqrt(all_targets[:, :, 0] ** 2 + all_targets[:, :, 1] ** 2)
        err_ws = ws_p - ws_t

        for idx, k in enumerate(sel_k):
            lday = int(lead_days[k])
            kk = sel_k[: idx + 1]
            entry = {
                "u10.rmse_pooled_to_lead": round(_pooled_rmse(err_u ** 2, kk), 6),
                "v10.rmse_pooled_to_lead": round(_pooled_rmse(err_v ** 2, kk), 6),
                "wind_speed.rmse_pooled_to_lead": round(_pooled_rmse(err_ws ** 2, kk), 6),
                "u10.mae_pooled_to_lead": round(_pooled_mae(err_u, kk), 6),
                "v10.mae_pooled_to_lead": round(_pooled_mae(err_v, kk), 6),
                "wind_speed.mae_pooled_to_lead": round(_pooled_mae(err_ws, kk), 6),
            }
            cumulative[f"lead_{lday:02d}"] = entry
            per_lead_summary[f"lead_{lday:02d}"].update(entry)

    report = {
        "task_type": task_type,
        "target_vars": target_vars,
        "is_scalar": bool(is_scalar),
        "num_samples": N,
        "num_leads_evaluated": len(sel_k),
        "lead_days_evaluated": [int(lead_days[k]) for k in sel_k],
        "lead_range": [int(lead_min), int(lead_max)],
        "overall_metrics": overall,
        "per_lead_metrics": per_lead_summary,
    }

    tag = "_".join(target_vars) if target_vars else "uv"
    report_path = os.path.join(output_dir, f"eval_report_{task_type}_{tag}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 終端輸出 ----
    bar = "=" * 78
    hr = "-" * 78
    print("\n" + bar)
    print(f"  EVALUATION REPORT — {task_type}   target={target_vars}   leads {lead_min}-{lead_max}")
    print(bar)
    print(f"  Samples evaluated : {N}")
    print(f"  Leads evaluated   : {report['lead_days_evaluated']}")
    print(hr)
    print("  OVERALL (mean over samples & evaluated leads):")
    for k, v in overall.items():
        print(f"    {k:28s} : {v}")
    print(hr)
    print("  PER-LEAD BREAKDOWN:")

    if is_scalar:
        header = (f"  {'lead':>5}  {'RMSE':>10}  {'MAE':>10}  {'Bias':>10}  "
                  f"{'R':>6}  {'SSIM':>6}  {'ACC':>6}  "
                  f"{'cumRMSE_pool':>12}  {'cumRMSE_mean':>12}  {'cumMAE_pool':>12}")
        print(header)
        for lk in sorted(per_lead_summary.keys()):
            m = per_lead_summary[lk]
            lday = int(lk.split("_")[1])
            print(f"  {lday:5d}  "
                  f"{m.get('rmse', 0):10.5f}  "
                  f"{m.get('mae', 0):10.5f}  "
                  f"{m.get('bias', 0):10.5f}  "
                  f"{m.get('r', 0):6.3f}  "
                  f"{m.get('ssim', 0):6.3f}  "
                  f"{m.get('acc', 0):6.3f}  "
                  f"{m.get('rmse_pooled_to_lead', 0):12.5f}  "
                  f"{m.get('rmse_mean_to_lead', 0):12.5f}  "
                  f"{m.get('mae_pooled_to_lead', 0):12.5f}")
    else:
        header = (f"  {'lead':>5}  {'u_RMSE':>8}  {'v_RMSE':>8}  {'ws_RMSE':>8}  "
                  f"{'dir_RMSE':>9}  {'dir_MAE':>8}  {'ACC(ws)':>7}  "
                  f"{'cumU_pool':>10}  {'cumV_pool':>10}  {'cumWS_pool':>10}")
        print(header)
        for lk in sorted(per_lead_summary.keys()):
            m = per_lead_summary[lk]
            lday = int(lk.split("_")[1])
            print(f"  {lday:5d}  "
                  f"{m.get('u10.rmse', 0):8.4f}  "
                  f"{m.get('v10.rmse', 0):8.4f}  "
                  f"{m.get('wind_speed.rmse', 0):8.4f}  "
                  f"{m.get('wind_direction.rmse_deg', 0):9.2f}  "
                  f"{m.get('wind_direction.mae_deg', 0):8.2f}  "
                  f"{m.get('wind_speed.acc', 0):7.3f}  "
                  f"{m.get('u10.rmse_pooled_to_lead', 0):10.4f}  "
                  f"{m.get('v10.rmse_pooled_to_lead', 0):10.4f}  "
                  f"{m.get('wind_speed.rmse_pooled_to_lead', 0):10.4f}")
    print(bar)
    print("  cumRMSE_pool = sqrt(mean squared error over leads 1..k);  "
          "cumRMSE_mean = mean(per-lead RMSE) for leads 1..k")
    print(f"  Report saved to: {report_path}")

    # =========================================================================
    # 多模式 baseline 對比表（只有 fcst2era5fit + s2s_vars == target_vars 才有資料）
    # =========================================================================
    if all_baseline_ok and baseline_compat and any(b is not None for b in all_baseline_pm):
        try:
            _run_baseline_comparison(
                cfg=cfg,
                all_baseline_pm=all_baseline_pm,
                all_preds=all_preds,
                all_targets=all_targets,
                lead_days=lead_days,
                target_vars=target_vars,
                sel_k=sel_k,
                model_names=s2s_model_names,
                china_land_mask=china_land_mask,
                output_dir=output_dir,
                task_type=task_type,
            )
        except Exception as e:
            import traceback
            print(f"[BASELINE] 生成 baseline 對比表時發生錯誤（不影響主要評估）: {e}")
            traceback.print_exc()
    elif all_baseline_ok and not baseline_compat:
        print(f"[BASELINE] 略過多模式對比：s2s_vars={s2s_var_list} 與 target_vars={target_vars} "
              "對不上，或 cfg.data.models 為空。")

    # =========================================================================
    # 逐格點 RMSE 空間熱力圖（不影響上面所有原本的指標 / 表）
    # =========================================================================
    if save_rmse_spatial:
        try:
            _generate_rmse_spatial_outputs(
                cfg=cfg,
                dataset=dataset,
                all_preds=all_preds,
                all_targets=all_targets,
                lead_days=lead_days,
                target_vars=target_vars,
                is_scalar=is_scalar,
                task_type=task_type,
                output_dir=output_dir,
                lead_groups=lead_groups,
                rmse_maps_dir=rmse_maps_dir,
                china_land_mask=china_land_mask,
                china_boundary_path=china_boundary_path,
                taiwan_boundary_path=taiwan_boundary_path,
            )
        except Exception as e:
            print(f"[RMSE MAP] 生成空間 RMSE 熱力圖時發生錯誤（不影響主要評估）: {e}")

    return report


def _run_baseline_comparison(
    *,
    cfg,
    all_baseline_pm,
    all_preds,
    all_targets,
    lead_days,
    target_vars,
    sel_k,
    model_names,
    china_land_mask,
    output_dir,
    task_type,
):
    """
    用收集到的 per-model baseline 與 your-model pred / target，
    產生方法比較 + lead-wise 改進率表（CSV / JSON / Markdown），
    並把終端輸出印一份對比表。
    """
    valid_indices = [i for i, b in enumerate(all_baseline_pm) if b is not None]
    if not valid_indices:
        print("[BASELINE] 無 per-model baseline 樣本，略過。")
        return

    per_model = np.stack([all_baseline_pm[i] for i in valid_indices])  # (N, M, K, C, H, W)
    targets = np.asarray([all_targets[i] for i in valid_indices])      # (N, K, C, H, W)
    my_pred = np.asarray([all_preds[i] for i in valid_indices])

    if china_land_mask is not None:
        mask6 = china_land_mask[None, None, None, None, :, :]
        per_model = np.where(mask6, per_model, np.nan)
        # all_preds / all_targets 已經是 mask 後的，per_model 額外 mask 一致

    sel_lead_days = [int(lead_days[k]) for k in sel_k]
    baseline_for_reduction = cfg.get("evaluation", {}).get("baseline_for_reduction", "ECMWF")

    comparison = compute_baseline_comparison(
        per_model_baseline=per_model,
        targets=targets,
        my_pred=my_pred,
        lead_days=lead_days,
        selected_lead_days=sel_lead_days,
        model_names=model_names,
        target_vars=target_vars,
        baseline_for_reduction=baseline_for_reduction,
    )

    print_baseline_comparison(comparison)
    sub_dir = os.path.join(output_dir, "baseline_comparison")
    tag = "_".join(target_vars) if target_vars else "uv"
    save_baseline_comparison_outputs(
        comparison, sub_dir, prefix=f"{task_type}_{tag}_baseline_comparison",
    )


# =============================================================================
# 逐格點 RMSE 熱力圖：把 evaluate 收集到的 (N,K,C,H,W) 拆成多變數後丟給 evaluation.rmse_spatial
# =============================================================================

def _resolve_region_extent(cfg):
    """從 cfg.data.region 取得繪圖用的 (lon_min, lon_max, lat_min, lat_max)，否則 None。"""
    r = cfg.get("data", {}).get("region")
    if not r or not r.get("enabled", True):
        return None
    try:
        return (
            float(r["lon_min"]), float(r["lon_max"]),
            float(r["lat_min"]), float(r["lat_max"]),
        )
    except Exception:
        return None


def _get_eval_lat_lon(dataset, target_h, target_w):
    """嘗試從 dataset 拿到中國區域裁切後的 (lat_1d, lon_1d)；失敗則退回索引座標。"""
    try:
        _, _, elat_sub, elon_sub = dataset._resolve_era5_crop_idx()
        lat = np.asarray(elat_sub).astype(float)
        lon = np.asarray(elon_sub).astype(float)
        if lat.size == target_h and lon.size == target_w:
            return lat, lon
        print(f"[RMSE MAP] WARNING: dataset lat/lon size {lat.size}/{lon.size} "
              f"與 target {target_h}/{target_w} 不一致，使用索引座標。")
    except Exception as e:
        print(f"[RMSE MAP] WARNING: 取得 dataset lat/lon 失敗（{e}），使用索引座標。")
    return np.arange(target_h, dtype=float), np.arange(target_w, dtype=float)


def _generate_rmse_spatial_outputs(
    *,
    cfg,
    dataset,
    all_preds,        # (N, K, C, H, W) 已 inverse-scaled 到 ERA5 原單位
    all_targets,      # (N, K, C, H, W)
    lead_days,        # (K,) 真實 lead day
    target_vars,      # ['wind10m'] 或 ['u10','v10']
    is_scalar,
    task_type,
    output_dir,
    lead_groups,
    rmse_maps_dir,
    china_land_mask=None,
    china_boundary_path=None,
    taiwan_boundary_path=None,
):
    if all_preds.size == 0:
        return

    N, K, C, H, W = all_preds.shape
    lat, lon = _get_eval_lat_lon(dataset, H, W)
    region_extent = _resolve_region_extent(cfg)

    if not lead_groups:
        lead_groups = dict(DEFAULT_LEAD_GROUPS)
    print(f"[RMSE MAP] lead_groups = {lead_groups}")
    print(f"[RMSE MAP] lead_days available = {lead_days.tolist()}")

    # 把 (N,K,C,H,W) 拆成「每個變數一個 [N,K,H,W]」
    pred_dict = {}
    obs_dict = {}
    units = {}
    if is_scalar:
        var = target_vars[0]
        pred_dict[var] = all_preds[:, :, 0]
        obs_dict[var] = all_targets[:, :, 0]
        units[var] = DEFAULT_UNITS.get(var, "")
    else:
        u_var = target_vars[0] if len(target_vars) >= 1 else "u"
        v_var = target_vars[1] if len(target_vars) >= 2 else "v"
        pred_dict[u_var] = all_preds[:, :, 0]
        obs_dict[u_var] = all_targets[:, :, 0]
        units[u_var] = DEFAULT_UNITS.get(u_var, "m/s")

        if C >= 2:
            pred_dict[v_var] = all_preds[:, :, 1]
            obs_dict[v_var] = all_targets[:, :, 1]
            units[v_var] = DEFAULT_UNITS.get(v_var, "m/s")

            # 標量風速 RMSE（與既有 wind_speed 指標一致：先 sqrt(u²+v²) 再算 RMSE）
            ws_pred = np.sqrt(all_preds[:, :, 0] ** 2 + all_preds[:, :, 1] ** 2)
            ws_obs = np.sqrt(all_targets[:, :, 0] ** 2 + all_targets[:, :, 1] ** 2)
            pred_dict["wind_speed"] = ws_pred
            obs_dict["wind_speed"] = ws_obs
            units["wind_speed"] = "m/s"

    rmse_maps = compute_all_grid_rmse_maps(
        pred_dict, obs_dict, lead_groups, lead_days,
        mask=china_land_mask, verbose=True,
    )
    if not rmse_maps:
        print("[RMSE MAP] 無可用 rmse_maps，略過。")
        return

    if rmse_maps_dir is None:
        tag = target_vars[0] if is_scalar else "uv"
        rmse_maps_dir = os.path.join(output_dir, "rmse_spatial_maps", f"{task_type}_{tag}")
    os.makedirs(rmse_maps_dir, exist_ok=True)

    # 1) 原始陣列：npy + 可選 nc，留給後續分析複現
    save_rmse_maps(rmse_maps, rmse_maps_dir, lat, lon, save_nc=True)

    # 2) 每個變數一張 1×N 熱力圖（單變數 eval 就只有一張）
    for var, var_maps in rmse_maps.items():
        out_png = os.path.join(rmse_maps_dir, f"{var}_rmse_spatial.png")
        plot_single_variable_rmse_maps(
            variable=var,
            rmse_maps_var=var_maps,
            lat=lat, lon=lon,
            output_path=out_png,
            units=units.get(var, ""),
            region_extent=region_extent,
            china_boundary_path=china_boundary_path,
            taiwan_boundary_path=taiwan_boundary_path,
        )

    # 3) 該次 eval 真的有 ≥ 2 個變數（uv + wind_speed）時才畫合併大圖
    if len(rmse_maps) >= 2:
        out_grid_png = os.path.join(rmse_maps_dir, "all_variables_rmse_spatial.png")
        plot_rmse_spatial_grid(
            rmse_maps, lat, lon,
            output_path=out_grid_png,
            variable_units=units,
            region_extent=region_extent,
            china_boundary_path=china_boundary_path,
            taiwan_boundary_path=taiwan_boundary_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate S2S-ERA5 downscaling model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--lead_min", type=int, default=10, help="評估 lead 下界（含），預設 10")
    parser.add_argument("--lead_max", type=int, default=45, help="評估 lead 上界（含），預設 45")
    parser.add_argument(
        "--lead_groups", type=str, default="01-10,11-20,21-30,31-40",
        help="逐格點 RMSE 空間圖的時效段，例如 '01-10,11-20,21-30,31-40'。",
    )
    parser.add_argument(
        "--skip_rmse_maps", action="store_true",
        help="關閉逐格點 RMSE 空間熱力圖輸出（其他指標照常）。",
    )
    parser.add_argument(
        "--rmse_maps_dir", type=str, default=None,
        help="逐格點 RMSE 圖的輸出資料夾；預設 <output_dir>/rmse_spatial_maps/<task>_<var>。",
    )
    parser.add_argument(
        "--china_shapefile_dir", type=str, default=r"D:\downsampling_focas\shapefile",
        help="中國 shapefile 所在資料夾，會自動尋找 gadm41_CHN_0.shp 或其他 .shp。",
    )
    parser.add_argument(
        "--china_land_mask_path", type=str, default=None,
        help="中國陸地 mask 的 .npy 路徑；預設為 <china_shapefile_dir>/china_land_mask.npy。",
    )
    parser.add_argument(
        "--rebuild_china_mask", action="store_true", default=None,
        help="即使已有 china_land_mask.npy，也根據 shapefile 和當前 lat/lon 重新生成。"
             "未指定時優先讀 cfg.data.rebuild_china_mask，最後 fallback 為 False。",
    )
    parser.add_argument(
        "--no_require_china_mask", action="store_true",
        help="如果指定，無法生成中國 mask 時只 warning，不中斷；默認要求必須有中國 mask。",
    )
    parser.add_argument(
        "--include_taiwan", dest="include_taiwan", action="store_true", default=None,
        help="把台灣納入中國陸地 mask 與背景邊界（預設啟用）。"
             "首次使用且本地無 TWN shapefile 時，會自動透過 cartopy Natural Earth 下載。",
    )
    parser.add_argument(
        "--no_include_taiwan", dest="include_taiwan", action="store_false",
        help="不把台灣納入 mask / 邊界（沿用舊版 China-only 行為）。",
    )
    parser.add_argument(
        "--taiwan_shapefile_path", type=str, default=None,
        help="可選的台灣 shapefile 路徑（例如 gadm41_TWN_0.shp）。"
             "未指定時會在 china_shapefile_dir 內遞迴搜尋 TWN/taiwan，再 fallback 到 Natural Earth。",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
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
        china_shapefile_dir=args.china_shapefile_dir,
        china_land_mask_path=args.china_land_mask_path,
        require_china_mask=(not args.no_require_china_mask),
    )
    # 只有 CLI 顯式給才覆寫 cfg / 函式預設
    if args.include_taiwan is not None:
        evaluate_kwargs["include_taiwan"] = args.include_taiwan
    if args.rebuild_china_mask is not None:
        evaluate_kwargs["rebuild_china_mask"] = args.rebuild_china_mask
    if args.taiwan_shapefile_path:
        evaluate_kwargs["taiwan_shapefile_path"] = args.taiwan_shapefile_path

    evaluate(**evaluate_kwargs)


if __name__ == "__main__":
    main()
