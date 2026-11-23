"""
多模式 baseline 對比工具
=========================

為 fcst2era5fit 任務輸出兩張表（仿 user 截圖）：

  1) 方法比較表（method-wise comparison）
     columns: method | wind_speed_RMSE | wind_speed_MAE | wind_speed_Bias |
              u_RMSE | v_RMSE | direction_MAE | RMSE_reduction(vs ECMWF)
     - S2S 端只有 wind10m/wind100m/ssrd 純量場，沒有 u/v 分量，故 u/v RMSE 與
       direction MAE 對 S2S baseline 一律填 "N/A"。
     - 「Your model」一行的 u/v / direction 欄位只在向量模式（c_out_uv==2）會非 N/A。

  2) 逐 lead 改進率表（lead-wise improvement）
     columns: lead_day | ensemble_mean_RMSE | your_model_RMSE | improvement_pct

兩張表都同時輸出 JSON / CSV / Markdown，方便直接貼到報告中。

對外 API
-----------------------------------------------------------------------------
compute_baseline_comparison(
    per_model_baseline: np.ndarray,    # (N, M, K, C, H, W)，已 inverse-scaled、已 mask
    targets: np.ndarray,               # (N, K, C, H, W)，已 inverse-scaled、已 mask
    my_pred: np.ndarray,               # (N, K, C, H, W)，已 inverse-scaled、已 mask
    lead_days: Sequence[int],          # 長度 K，每個 lead 的真實 day
    selected_lead_days: Sequence[int], # 要評估的 lead day 子集
    model_names: Sequence[str],        # 長度 M
    target_vars: Sequence[str],        # 長度 C
    baseline_for_reduction: str = "ECMWF",  # 算 RMSE_reduction 用的比較基準
    my_pred_u: Optional[np.ndarray] = None,  # 向量模式才提供 (N,K,H,W)
    my_pred_v: Optional[np.ndarray] = None,
    target_u: Optional[np.ndarray] = None,
    target_v: Optional[np.ndarray] = None,
) -> dict
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np


# =============================================================================
# 基本指標（NaN-safe）
# =============================================================================

def _rmse_nan(pred: np.ndarray, target: np.ndarray) -> float:
    err2 = (pred - target) ** 2
    val = np.nanmean(err2)
    if not np.isfinite(val):
        return float("nan")
    return float(np.sqrt(val))


def _mae_nan(pred: np.ndarray, target: np.ndarray) -> float:
    val = np.nanmean(np.abs(pred - target))
    return float(val) if np.isfinite(val) else float("nan")


def _bias_nan(pred: np.ndarray, target: np.ndarray) -> float:
    val = np.nanmean(pred - target)
    return float(val) if np.isfinite(val) else float("nan")


def _wind_dir_deg(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0


def _dir_mae_deg(pu, pv, tu, tv) -> float:
    dp = _wind_dir_deg(pu, pv)
    dt = _wind_dir_deg(tu, tv)
    diff = ((dp - dt) + 180.0) % 360.0 - 180.0
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(np.abs(diff)))


def _dir_rmse_deg(pu, pv, tu, tv) -> float:
    dp = _wind_dir_deg(pu, pv)
    dt = _wind_dir_deg(tu, tv)
    diff = ((dp - dt) + 180.0) % 360.0 - 180.0
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(diff ** 2)))


# =============================================================================
# 主流程
# =============================================================================

def _select_leads(arr: np.ndarray, all_lead_days, selected_lead_days) -> np.ndarray:
    """從 K 軸上挑出指定 lead day 對應的索引。arr 第 1 維（除 N 之外的最內）為 K。"""
    all_lead_days = list(int(x) for x in all_lead_days)
    sel = [all_lead_days.index(int(d)) for d in selected_lead_days
           if int(d) in all_lead_days]
    if not sel:
        raise ValueError(
            f"None of selected_lead_days {selected_lead_days} are in "
            f"all_lead_days {all_lead_days}"
        )
    return arr.take(sel, axis=1) if arr.ndim >= 2 else arr


def _per_lead_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """pred / target shape: (N, K, C, H, W)；回傳長度 K 的 per-lead RMSE。
    目前我們的 metric 都看「整體 wind speed」一個通道，外面會把 C 攤平餵進來。
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    K = pred.shape[1]
    out = np.full((K,), np.nan, dtype=np.float64)
    for k in range(K):
        err2 = (pred[:, k] - target[:, k]) ** 2
        v = np.nanmean(err2)
        if np.isfinite(v):
            out[k] = float(np.sqrt(v))
    return out


def _scalar_wind_speed_field(arr: np.ndarray, target_vars: Sequence[str]) -> np.ndarray:
    """
    把 (N, K, C, H, W) 還原成單一風速場 (N, K, H, W)：
      - target_vars == ['wind10m'] / ['wind100m']：直接取 channel 0
      - target_vars == ['u10','v10'] / ['u100','v100']：sqrt(u² + v²)
      - 其他單通道（含 ssrd）：直接取 channel 0
    """
    if arr.ndim != 5:
        raise ValueError(f"expected (N,K,C,H,W), got {arr.shape}")
    C = arr.shape[2]
    if C == 1:
        return arr[:, :, 0]
    if C >= 2 and target_vars[0] in ("u10", "u100") and target_vars[1] in ("v10", "v100"):
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        return np.sqrt(u * u + v * v)
    return arr[:, :, 0]


def compute_baseline_comparison(
    per_model_baseline: np.ndarray,
    targets: np.ndarray,
    my_pred: np.ndarray,
    lead_days: Sequence[int],
    selected_lead_days: Sequence[int],
    model_names: Sequence[str],
    target_vars: Sequence[str],
    baseline_for_reduction: str = "ECMWF",
    my_pred_u: Optional[np.ndarray] = None,
    my_pred_v: Optional[np.ndarray] = None,
    target_u: Optional[np.ndarray] = None,
    target_v: Optional[np.ndarray] = None,
) -> Dict:
    """
    產生方法比較表（method-wise）+ lead-wise 改進率表。
    輸入皆為「原尺度、已 mask（China+Taiwan 外為 NaN）」的 numpy array。
    """
    if per_model_baseline.ndim != 6:
        raise ValueError(f"per_model_baseline must be (N,M,K,C,H,W), got {per_model_baseline.shape}")
    N, M, K, C, H, W = per_model_baseline.shape
    if targets.shape != (N, K, C, H, W):
        raise ValueError(f"targets shape {targets.shape} mismatch with baseline (N,K,C,H,W)=({N},{K},{C},{H},{W})")
    if my_pred.shape != (N, K, C, H, W):
        raise ValueError(f"my_pred shape {my_pred.shape} mismatch")
    if len(model_names) != M:
        raise ValueError(f"model_names length {len(model_names)} != M={M}")

    sel_day_idx = []
    all_days = list(int(x) for x in lead_days)
    for d in selected_lead_days:
        if int(d) in all_days:
            sel_day_idx.append(all_days.index(int(d)))
    if not sel_day_idx:
        raise RuntimeError("No selected leads matched.")
    sel_days = [all_days[i] for i in sel_day_idx]

    targets_sel = targets[:, sel_day_idx]
    my_pred_sel = my_pred[:, sel_day_idx]
    base_sel = per_model_baseline[:, :, sel_day_idx]

    # ---- 計算統一風速場 ----
    ws_target = _scalar_wind_speed_field(targets_sel, target_vars)            # (N, k, H, W)
    ws_my = _scalar_wind_speed_field(my_pred_sel, target_vars)                # (N, k, H, W)
    ws_models = np.stack(
        [_scalar_wind_speed_field(base_sel[:, m], target_vars) for m in range(M)],
        axis=0,
    )  # (M, N, k, H, W)
    ws_mean = ws_models.mean(axis=0)                                           # (N, k, H, W)

    methods: List[Dict] = []

    def _row(name: str, ws_pred: np.ndarray, has_uv: bool = False,
             pu=None, pv=None, tu=None, tv=None) -> Dict:
        row = {
            "method": name,
            "wind_speed_rmse": round(_rmse_nan(ws_pred, ws_target), 4),
            "wind_speed_mae": round(_mae_nan(ws_pred, ws_target), 4),
            "wind_speed_bias": round(_bias_nan(ws_pred, ws_target), 4),
        }
        if has_uv and pu is not None and tu is not None:
            row.update({
                "u_rmse": round(_rmse_nan(pu, tu), 4),
                "v_rmse": round(_rmse_nan(pv, tv), 4),
                "direction_mae_deg": round(_dir_mae_deg(pu, pv, tu, tv), 2),
                "direction_rmse_deg": round(_dir_rmse_deg(pu, pv, tu, tv), 2),
            })
        else:
            row.update({
                "u_rmse": "N/A",
                "v_rmse": "N/A",
                "direction_mae_deg": "N/A",
                "direction_rmse_deg": "N/A",
            })
        return row

    for m, name in enumerate(model_names):
        methods.append(_row(name, ws_models[m]))

    methods.append(_row(f"Multi-model mean ({'+'.join(model_names)})", ws_mean))

    has_uv = (
        my_pred_u is not None and my_pred_v is not None
        and target_u is not None and target_v is not None
    )
    if has_uv:
        pu_sel = my_pred_u[:, sel_day_idx]
        pv_sel = my_pred_v[:, sel_day_idx]
        tu_sel = target_u[:, sel_day_idx]
        tv_sel = target_v[:, sel_day_idx]
        methods.append(_row(
            "Your model",
            ws_my,
            has_uv=True,
            pu=pu_sel, pv=pv_sel, tu=tu_sel, tv=tv_sel,
        ))
    else:
        methods.append(_row("Your model", ws_my))

    # ---- RMSE reduction：相對於指定 baseline ----
    ref_name = baseline_for_reduction
    ref_value = None
    for r in methods:
        if str(r["method"]).upper() == ref_name.upper():
            ref_value = float(r["wind_speed_rmse"])
            break
    if ref_value is None or ref_value <= 0:
        for r in methods:
            r["rmse_reduction_vs_ref_pct"] = "N/A"
        ref_name = None
    else:
        for r in methods:
            if str(r["method"]).upper() == ref_name.upper():
                r["rmse_reduction_vs_ref_pct"] = "-"
            else:
                v = float(r["wind_speed_rmse"])
                r["rmse_reduction_vs_ref_pct"] = round((ref_value - v) / ref_value * 100.0, 2)

    # ---- lead-wise：multi-model mean vs your model ----
    per_lead_mean_rmse = _per_lead_rmse(ws_mean[:, :, None, ...] if False else
                                        ws_mean[:, :, np.newaxis, ...],
                                        ws_target[:, :, np.newaxis, ...])
    per_lead_my_rmse = _per_lead_rmse(ws_my[:, :, np.newaxis, ...],
                                      ws_target[:, :, np.newaxis, ...])
    lead_rows: List[Dict] = []
    improvements_pct = []
    for i, d in enumerate(sel_days):
        em = float(per_lead_mean_rmse[i])
        ym = float(per_lead_my_rmse[i])
        if em > 0 and np.isfinite(em) and np.isfinite(ym):
            imp = round((em - ym) / em * 100.0, 2)
            improvements_pct.append(imp)
        else:
            imp = "N/A"
        lead_rows.append({
            "lead_day": int(d),
            "ensemble_mean_rmse": round(em, 4),
            "your_model_rmse": round(ym, 4),
            "improvement_pct": imp,
        })

    # ---- per-model wind direction（純量 baseline 一律 N/A，僅向量 my-model 給單值） ----
    direction_summary = {}
    if has_uv:
        # 對 my-model 的方向 MAE/RMSE
        pu_sel = my_pred_u[:, sel_day_idx]
        pv_sel = my_pred_v[:, sel_day_idx]
        tu_sel = target_u[:, sel_day_idx]
        tv_sel = target_v[:, sel_day_idx]
        direction_summary = {
            "your_model_direction_mae_deg": round(_dir_mae_deg(pu_sel, pv_sel, tu_sel, tv_sel), 2),
            "your_model_direction_rmse_deg": round(_dir_rmse_deg(pu_sel, pv_sel, tu_sel, tv_sel), 2),
            "baseline_direction_mae_deg": "N/A (S2S has no u/v components)",
            "baseline_direction_rmse_deg": "N/A (S2S has no u/v components)",
            "direction_improvement_pct": "N/A (no baseline direction available)",
        }
    else:
        direction_summary = {
            "your_model_direction_mae_deg": "N/A (scalar-mode model has no u/v output)",
            "your_model_direction_rmse_deg": "N/A (scalar-mode model has no u/v output)",
            "baseline_direction_mae_deg": "N/A (S2S has no u/v components)",
            "baseline_direction_rmse_deg": "N/A (S2S has no u/v components)",
            "direction_improvement_pct": "N/A (no u/v in either baseline or your model)",
        }

    # ---- 整體 wind-speed 改進率區間（min%/mean%/max%） ----
    improvement_range = {}
    if improvements_pct:
        improvement_range = {
            "min_pct": round(float(np.min(improvements_pct)), 2),
            "max_pct": round(float(np.max(improvements_pct)), 2),
            "mean_pct": round(float(np.mean(improvements_pct)), 2),
            "lead_count": len(improvements_pct),
            "summary": f"風速 RMSE 在 lead {min(sel_days)}–{max(sel_days)} 天 "
                       f"相對 multi-model mean 降低 "
                       f"{float(np.min(improvements_pct)):.1f}%–"
                       f"{float(np.max(improvements_pct)):.1f}%（平均 "
                       f"{float(np.mean(improvements_pct)):.1f}%）",
        }
    else:
        improvement_range = {"summary": "no valid leads"}

    return {
        "baseline_for_reduction": ref_name,
        "num_samples": int(N),
        "num_models": int(M),
        "model_names": list(model_names),
        "target_vars": list(target_vars),
        "lead_days_evaluated": sel_days,
        "methods_table": methods,
        "lead_wise_table": lead_rows,
        "wind_speed_improvement_range": improvement_range,
        "direction_summary": direction_summary,
    }


# =============================================================================
# 輸出：Markdown / CSV / JSON / 終端漂亮打印
# =============================================================================

def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) >= 1e-3 else f"{v:.6f}"
    if isinstance(v, str):
        return v
    return str(v)


def _fmt_pct(v) -> str:
    if isinstance(v, (int, float)):
        return f"{v:+.2f}%"
    return str(v)


def save_baseline_comparison_outputs(
    comparison: Dict,
    output_dir: str,
    prefix: str = "baseline_comparison",
):
    """把 comparison dict 寫成 JSON + CSV + Markdown，再把 Markdown 全文 return。"""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, f"{prefix}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {json_path}")

    # ---- methods table CSV ----
    methods = comparison["methods_table"]
    methods_csv = os.path.join(output_dir, f"{prefix}_methods.csv")
    method_cols = [
        "method",
        "wind_speed_rmse", "wind_speed_mae", "wind_speed_bias",
        "u_rmse", "v_rmse",
        "direction_mae_deg", "direction_rmse_deg",
        "rmse_reduction_vs_ref_pct",
    ]
    with open(methods_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=method_cols)
        w.writeheader()
        for r in methods:
            w.writerow({k: r.get(k, "") for k in method_cols})
    print(f"[SAVE] {methods_csv}")

    # ---- lead-wise CSV ----
    lead_rows = comparison["lead_wise_table"]
    lead_csv = os.path.join(output_dir, f"{prefix}_lead_wise.csv")
    lead_cols = ["lead_day", "ensemble_mean_rmse", "your_model_rmse", "improvement_pct"]
    with open(lead_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lead_cols)
        w.writeheader()
        for r in lead_rows:
            w.writerow({k: r.get(k, "") for k in lead_cols})
    print(f"[SAVE] {lead_csv}")

    # ---- Markdown 報告 ----
    md_lines: List[str] = []
    ref = comparison.get("baseline_for_reduction") or "(unspecified)"
    md_lines.append(f"# Baseline 對比表（vs {ref}）\n")
    md_lines.append(
        f"- 樣本數 N = {comparison['num_samples']}\n"
        f"- 模式數 M = {comparison['num_models']}（{', '.join(comparison['model_names'])}）\n"
        f"- 目標變數 = {comparison['target_vars']}\n"
        f"- 評估 lead day = {comparison['lead_days_evaluated']}\n"
    )

    md_lines.append("\n## 方法比較表\n")
    md_lines.append(
        "| 方法 | u RMSE | v RMSE | wind speed RMSE | wind speed MAE | wind speed Bias | "
        "wind direction MAE | wind direction RMSE | RMSE reduction (vs " + str(ref) + ") |"
    )
    md_lines.append(
        "|---|---|---|---|---|---|---|---|---|"
    )
    for r in methods:
        md_lines.append(
            f"| {r['method']} | "
            f"{_fmt(r['u_rmse'])} | {_fmt(r['v_rmse'])} | "
            f"{_fmt(r['wind_speed_rmse'])} | {_fmt(r['wind_speed_mae'])} | {_fmt(r['wind_speed_bias'])} | "
            f"{_fmt(r['direction_mae_deg'])} | {_fmt(r['direction_rmse_deg'])} | "
            f"{_fmt_pct(r['rmse_reduction_vs_ref_pct'])} |"
        )

    md_lines.append("\n## Lead-wise 改進率（multi-model mean vs your model，wind speed RMSE）\n")
    md_lines.append("| Lead day | Ensemble mean RMSE | Your model RMSE | Improvement |")
    md_lines.append("|---:|---:|---:|---:|")
    for r in lead_rows:
        md_lines.append(
            f"| {r['lead_day']} | "
            f"{_fmt(r['ensemble_mean_rmse'])} | "
            f"{_fmt(r['your_model_rmse'])} | "
            f"{_fmt_pct(r['improvement_pct'])} |"
        )

    md_lines.append("\n## 整體結論\n")
    rng = comparison["wind_speed_improvement_range"]
    md_lines.append(f"- {rng.get('summary', '')}")

    dirs = comparison["direction_summary"]
    md_lines.append("\n### 風向 / Direction")
    md_lines.append(f"- your_model_direction_mae_deg : {dirs['your_model_direction_mae_deg']}")
    md_lines.append(f"- your_model_direction_rmse_deg: {dirs['your_model_direction_rmse_deg']}")
    md_lines.append(f"- baseline_direction_mae_deg   : {dirs['baseline_direction_mae_deg']}")
    md_lines.append(f"- baseline_direction_rmse_deg  : {dirs['baseline_direction_rmse_deg']}")
    md_lines.append(f"- direction_improvement_pct    : {dirs['direction_improvement_pct']}")

    md_text = "\n".join(md_lines) + "\n"
    md_path = os.path.join(output_dir, f"{prefix}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[SAVE] {md_path}")

    return md_text


def print_baseline_comparison(comparison: Dict, bar_width: int = 102):
    """把對比表用對齊欄位的方式印到終端。"""
    bar = "=" * bar_width
    hr = "-" * bar_width

    print("\n" + bar)
    ref = comparison.get("baseline_for_reduction") or "(unspecified)"
    print(f"  BASELINE COMPARISON — wind speed (vs {ref})")
    print(bar)
    methods = comparison["methods_table"]
    header = (
        f"  {'method':<26}  {'ws_RMSE':>10}  {'ws_MAE':>10}  {'ws_Bias':>10}  "
        f"{'u_RMSE':>8}  {'v_RMSE':>8}  {'dir_MAE':>9}  {'red%':>8}"
    )
    print(header)
    print(hr)
    for r in methods:
        print(
            f"  {str(r['method'])[:26]:<26}  "
            f"{_fmt(r['wind_speed_rmse']):>10}  "
            f"{_fmt(r['wind_speed_mae']):>10}  "
            f"{_fmt(r['wind_speed_bias']):>10}  "
            f"{_fmt(r['u_rmse']):>8}  "
            f"{_fmt(r['v_rmse']):>8}  "
            f"{_fmt(r['direction_mae_deg']):>9}  "
            f"{_fmt_pct(r['rmse_reduction_vs_ref_pct']):>8}"
        )
    print(bar)
    rng = comparison["wind_speed_improvement_range"]
    print(f"  {rng.get('summary', '')}")
    dirs = comparison["direction_summary"]
    print(f"  direction (your)   : MAE={dirs['your_model_direction_mae_deg']} | "
          f"RMSE={dirs['your_model_direction_rmse_deg']}")
    print(f"  direction (baseline): {dirs['baseline_direction_mae_deg']}")
    print(bar)

    print("  LEAD-WISE IMPROVEMENT TABLE (multi-model mean vs your model, wind speed RMSE)")
    print(hr)
    print(f"  {'lead':>5}  {'ens_mean':>12}  {'your':>12}  {'improvement':>14}")
    for r in comparison["lead_wise_table"]:
        print(
            f"  {r['lead_day']:>5}  "
            f"{_fmt(r['ensemble_mean_rmse']):>12}  "
            f"{_fmt(r['your_model_rmse']):>12}  "
            f"{_fmt_pct(r['improvement_pct']):>14}"
        )
    print(bar)
