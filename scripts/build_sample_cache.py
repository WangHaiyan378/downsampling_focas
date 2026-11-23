"""
Pre-pack per-init-date training samples (S2S inputs + ERA5-native targets + static on ERA5 grid).

Task A:
    x_history  (M, T, C_in, H_s2s, W_s2s)   — first T S2S time slices, ssrd scaled
    target_uv  (K, 2, H_era5, W_era5)      — ERA5 u10/v10 daily mean, native 0.25° grid

Task B:
    x_forecast (M, K, C_in, H_s2s, W_s2s)  — S2S at each lead day, ssrd scaled
    target_uv  (K, 2, H_era5, W_era5)

Shared:
    _static.npy  (C_static, H_era5, W_era5) — lat/lon encoding + LCZ on ERA5 lat/lon
    _meta.json

Uses np.savez_compressed to limit disk usage.

Usage:
    python scripts/build_sample_cache.py --config configs/task_a.yaml
    python scripts/build_sample_cache.py --config configs/task_b.yaml --limit 5 --force
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.base_dataset import (  # noqa: E402
    ERA5DailyCacheReader,
    compute_region_indices,
    ensure_era5_daily_cache,
    generate_static_features,
    get_region_cfg,
    get_s2s_file_list,
    load_lcz_on_s2s_grid,
    parse_init_date,
    resolve_num_leads_and_lead_times,
    resolve_s2s_var_divisors,
    scale_s2s_numpy,
)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sample_index(cfg):
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    _, lead_times = resolve_num_leads_and_lead_times(dcfg, mcfg)

    s2s_root = dcfg["s2s_root"]
    era5_root = dcfg["era5_root"]
    models = dcfg["models"]
    svars = dcfg["s2s_vars"]
    target_vars = dcfg.get("era5_target_vars", ["u10", "v10"])
    era5_cache_root = dcfg.get("era5_cache_root")

    first_model = models[0]
    first_var = svars[0]
    s2s_files = get_s2s_file_list(s2s_root, first_model, first_var)

    era5_dates_cache = {}

    def era5_dates(var_name, year):
        key = (var_name, int(year))
        if key not in era5_dates_cache:
            path = os.path.join(era5_root, f"{var_name}_{int(year)}.nc")
            if not os.path.exists(path):
                era5_dates_cache[key] = None
            else:
                _, dates_p, _ = ensure_era5_daily_cache(
                    era5_root, var_name, int(year),
                    cache_root=era5_cache_root, verbose=False,
                )
                era5_dates_cache[key] = set(str(d) for d in np.load(dates_p))
        return era5_dates_cache[key]

    samples = []
    skipped_s2s = 0
    skipped_era5 = 0
    skipped_short = 0
    max_lt = max(lead_times) if lead_times else 0
    import xarray as xr

    for f in s2s_files:
        init_date = parse_init_date(f)

        per_model = {}
        ok = True
        for m in models:
            per_model[m] = {}
            for v in svars:
                p = f.replace(
                    os.path.join(s2s_root, first_model, first_var),
                    os.path.join(s2s_root, m, v),
                )
                if not os.path.exists(p):
                    ok = False
                    break
                per_model[m][v] = p
            if not ok:
                break
        if not ok:
            skipped_s2s += 1
            continue

        try:
            init_dt = datetime.strptime(init_date, "%Y%m%d")
        except ValueError:
            continue

        target_dates = []
        ok2 = True
        for lt in lead_times:
            td = (init_dt + timedelta(days=int(lt))).date()
            target_dates.append(td)
            for tv in target_vars:
                avail = era5_dates(tv, td.year)
                if avail is None or str(td) not in avail:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            skipped_era5 += 1
            continue

        ref_nc = per_model[models[0]][svars[0]]
        with xr.open_dataset(ref_nc) as ds:
            ntime = int(ds.sizes["time"])
        if ntime <= max_lt:
            skipped_short += 1
            continue

        samples.append({
            "init_date": init_date,
            "s2s_files": per_model,
            "target_dates": target_dates,
        })

    print(
        f"[Index] valid={len(samples)}  "
        f"skipped_s2s={skipped_s2s}  skipped_era5={skipped_era5}  "
        f"skipped_short_time={skipped_short}",
        flush=True,
    )
    return samples


def build_static_on_era5(cfg, era5_lat, era5_lon):
    dcfg = cfg["data"]
    mcfg = cfg["model"]
    c_static = mcfg["c_static"]
    lcz_path = dcfg.get("lcz_path")
    lcz_band = dcfg.get("lcz_band", 1)
    lcz_resamp = dcfg.get("lcz_resampling", "average")

    static = generate_static_features(
        len(era5_lat), len(era5_lon), era5_lat, era5_lon,
    )
    if lcz_path and os.path.exists(lcz_path):
        lcz = load_lcz_on_s2s_grid(
            lcz_path, era5_lat, era5_lon,
            band_index=lcz_band, resampling=lcz_resamp,
        )
        static = np.concatenate([static, lcz[None, ...]], axis=0)

    if c_static > static.shape[0]:
        pad = np.zeros(
            (c_static - static.shape[0], *static.shape[1:]),
            dtype=np.float32,
        )
        static = np.concatenate([static, pad], axis=0)
    elif c_static < static.shape[0]:
        static = static[:c_static]
    return static.astype(np.float32)


def precompute_era5_native(cfg, samples, era5_crop=None):
    """All (var, date) -> float32 (H_era5, W_era5) native daily mean fields (cropped if requested)."""
    dcfg = cfg["data"]
    era5_root = dcfg["era5_root"]
    era5_cache_root = dcfg.get("era5_cache_root")
    target_vars = dcfg.get("era5_target_vars", ["u10", "v10"])

    pairs = set()
    for s in samples:
        for td in s["target_dates"]:
            for tv in target_vars:
                pairs.add((tv, td))

    print(f"[ERA5 native] caching {len(pairs)} (var, date) fields", flush=True)
    out = {}
    t0 = time.time()
    for i, (tv, td) in enumerate(sorted(pairs, key=lambda x: (x[0], str(x[1])))):
        reader = ERA5DailyCacheReader.get(
            era5_root, tv, td.year, cache_root=era5_cache_root,
        )
        _, _, day_mean = reader.daily_mean(td)
        day_mean = np.asarray(day_mean, dtype=np.float32)
        if era5_crop is not None:
            lat_idx, lon_idx = era5_crop
            day_mean = day_mean[lat_idx, :][:, lon_idx]
        day_mean = np.nan_to_num(day_mean, nan=0.0, posinf=0.0, neginf=0.0)
        out[(tv, td)] = day_mean
        if (i + 1) % 200 == 0 or i == len(pairs) - 1:
            print(
                f"  [{i + 1}/{len(pairs)}] elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return out


def pack_task_a(cfg, samples, out_dir, era5_native, divisors, force=False,
                s2s_crop=None):
    import xarray as xr

    dcfg = cfg["data"]
    models = dcfg["models"]
    svars = dcfg["s2s_vars"]
    target_vars = dcfg.get("era5_target_vars", ["u10", "v10"])
    history_len = int(dcfg["history_len"])

    manifest = []
    t0 = time.time()
    for si, s in enumerate(samples):
        init_date = s["init_date"]
        out_path = os.path.join(out_dir, f"{init_date}.npz")
        if os.path.exists(out_path) and not force:
            manifest.append(f"{init_date}.npz")
            continue

        model_data = []
        for m in models:
            var_slices = []
            for v in svars:
                with xr.open_dataset(s["s2s_files"][m][v]) as ds:
                    arr = ds[v].values
                sl = arr[:history_len]
                if sl.shape[0] < history_len:
                    pad = np.zeros(
                        (history_len - sl.shape[0], *sl.shape[1:]),
                        dtype=np.float32,
                    )
                    sl = np.concatenate([sl, pad], axis=0)
                if s2s_crop is not None:
                    lat_idx, lon_idx = s2s_crop
                    sl = sl[:, lat_idx, :][:, :, lon_idx]
                sl = scale_s2s_numpy(v, sl, divisors)
                sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
                var_slices.append(sl.astype(np.float32, copy=False))
            model_data.append(np.stack(var_slices, axis=1))
        x_history = np.stack(model_data, axis=0).astype(np.float32)

        target_uv = []
        for td in s["target_dates"]:
            per_var = [era5_native[(tv, td)] for tv in target_vars]
            target_uv.append(np.stack(per_var, axis=0))
        target_uv = np.stack(target_uv, axis=0).astype(np.float32)

        np.savez_compressed(out_path, x_history=x_history, target_uv=target_uv)
        manifest.append(f"{init_date}.npz")

        if (si + 1) % 10 == 0 or si == len(samples) - 1:
            print(
                f"  [task_a] packed {si + 1}/{len(samples)}  "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return manifest


def pack_task_b(cfg, samples, out_dir, era5_native, divisors, force=False,
                s2s_crop=None):
    import xarray as xr

    dcfg = cfg["data"]
    models = dcfg["models"]
    svars = dcfg["s2s_vars"]
    target_vars = dcfg.get("era5_target_vars", ["u10", "v10"])
    _, lead_times = resolve_num_leads_and_lead_times(dcfg, cfg["model"])

    manifest = []
    t0 = time.time()
    for si, s in enumerate(samples):
        init_date = s["init_date"]
        out_path = os.path.join(out_dir, f"{init_date}.npz")
        if os.path.exists(out_path) and not force:
            manifest.append(f"{init_date}.npz")
            continue

        model_data = []
        for m in models:
            lead_vars = []
            for v in svars:
                with xr.open_dataset(s["s2s_files"][m][v]) as ds:
                    arr = ds[v].values
                per_lead = []
                for lt in lead_times:
                    tidx = min(int(lt), arr.shape[0] - 1)
                    slab = arr[tidx]
                    if s2s_crop is not None:
                        lat_idx, lon_idx = s2s_crop
                        slab = slab[lat_idx, :][:, lon_idx]
                    slab = scale_s2s_numpy(v, slab, divisors).astype(np.float32, copy=False)
                    slab = np.nan_to_num(slab, nan=0.0, posinf=0.0, neginf=0.0)
                    per_lead.append(slab)
                lead_vars.append(np.stack(per_lead, axis=0))
            model_data.append(np.stack(lead_vars, axis=1))
        x_forecast = np.stack(model_data, axis=0).astype(np.float32)

        target_uv = []
        for td in s["target_dates"]:
            per_var = [era5_native[(tv, td)] for tv in target_vars]
            target_uv.append(np.stack(per_var, axis=0))
        target_uv = np.stack(target_uv, axis=0).astype(np.float32)

        np.savez_compressed(out_path, x_forecast=x_forecast, target_uv=target_uv)
        manifest.append(f"{init_date}.npz")

        if (si + 1) % 10 == 0 or si == len(samples) - 1:
            print(
                f"  [task_b] packed {si + 1}/{len(samples)}  "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import xarray as xr

    cfg = load_cfg(args.config)
    dcfg = cfg["data"]
    mcfg = cfg["model"]

    nl, lt = resolve_num_leads_and_lead_times(dcfg, mcfg)
    dcfg["num_leads"] = nl
    dcfg["lead_times"] = lt
    mcfg["num_leads"] = nl
    mcfg["history_len"] = int(dcfg.get("history_len", mcfg.get("history_len", 45)))

    task_type = cfg["task_type"]

    out_root = args.out_root or dcfg.get("sample_cache_root")
    if not out_root:
        raise SystemExit("No sample_cache_root in config and --out-root not given.")
    task_key = "task_a" if task_type == "history2era5" else "task_b"
    out_dir = os.path.join(out_root, task_key)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Config] task={task_type}  out_dir={out_dir}", flush=True)

    t_all = time.time()
    divisors = resolve_s2s_var_divisors(dcfg)
    print(f"[S2S scale] divisors={divisors}", flush=True)

    samples = build_sample_index(cfg)
    if args.limit is not None:
        samples = samples[: args.limit]
        print(f"[Limit] restricted to first {len(samples)} samples", flush=True)
    if not samples:
        raise SystemExit("No valid samples — check S2S / ERA5 data paths.")

    s0 = samples[0]
    init_dt = datetime.strptime(s0["init_date"], "%Y%m%d")
    first_td = (init_dt + timedelta(days=int(lt[0]))).date()
    tvar0 = dcfg.get("era5_target_vars", ["u10", "v10"])[0]
    reader0 = ERA5DailyCacheReader.get(
        dcfg["era5_root"], tvar0, first_td.year, cache_root=dcfg.get("era5_cache_root"),
    )
    elat_full, elon_full, _ = reader0.daily_mean(first_td)
    print(f"[Grid] ERA5 native lat={len(elat_full)} lon={len(elon_full)}", flush=True)

    ref_path = s0["s2s_files"][dcfg["models"][0]][dcfg["s2s_vars"][0]]
    with xr.open_dataset(ref_path) as ds0:
        s2s_lat_full = ds0["lat"].values.astype(np.float32)
        s2s_lon_full = ds0["lon"].values.astype(np.float32)
    print(f"[Grid] S2S    native lat={len(s2s_lat_full)} lon={len(s2s_lon_full)}", flush=True)

    region = get_region_cfg(dcfg)
    if region is None:
        elat, elon = elat_full, elon_full
        s2s_lat, s2s_lon = s2s_lat_full, s2s_lon_full
        era5_crop = None
        s2s_crop = None
    else:
        era_lat_idx, era_lon_idx, elat, elon = compute_region_indices(elat_full, elon_full, region)
        s2s_lat_idx, s2s_lon_idx, s2s_lat, s2s_lon = compute_region_indices(
            s2s_lat_full, s2s_lon_full, region,
        )
        era5_crop = (era_lat_idx, era_lon_idx)
        s2s_crop = (s2s_lat_idx, s2s_lon_idx)
        print(
            f"[Region] {region}  "
            f"ERA5 crop→{len(elat)}×{len(elon)}  S2S crop→{len(s2s_lat)}×{len(s2s_lon)}",
            flush=True,
        )

    static = build_static_on_era5(cfg, elat, elon)
    static_path = os.path.join(out_dir, "_static.npy")
    np.save(static_path, static)
    print(f"[Static] saved {static_path}  shape={static.shape}", flush=True)

    era5_native = precompute_era5_native(cfg, samples, era5_crop=era5_crop)

    if task_type == "history2era5":
        manifest = pack_task_a(
            cfg, samples, out_dir, era5_native, divisors,
            force=args.force, s2s_crop=s2s_crop,
        )
    elif task_type == "fcst2era5fit":
        manifest = pack_task_b(
            cfg, samples, out_dir, era5_native, divisors,
            force=args.force, s2s_crop=s2s_crop,
        )
    else:
        raise SystemExit(f"Unknown task_type: {task_type}")

    meta = {
        "task_type": task_type,
        "lead_times": lt,
        "num_leads": nl,
        "history_len": int(dcfg["history_len"]),
        "models": list(dcfg["models"]),
        "s2s_vars": list(dcfg["s2s_vars"]),
        "era5_target_vars": list(dcfg.get("era5_target_vars", ["u10", "v10"])),
        "s2s_var_divisors": divisors,
        "s2s_grid_h": int(len(s2s_lat)),
        "s2s_grid_w": int(len(s2s_lon)),
        "era5_grid_h": int(len(elat)),
        "era5_grid_w": int(len(elon)),
        "region": region,
        "lat_s2s": np.asarray(s2s_lat).tolist(),
        "lon_s2s": np.asarray(s2s_lon).tolist(),
        "lat_era5": np.asarray(elat).astype(float).tolist(),
        "lon_era5": np.asarray(elon).astype(float).tolist(),
        "samples": sorted(manifest),
    }
    meta_path = os.path.join(out_dir, "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[Meta] saved {meta_path}", flush=True)

    print(
        f"\n[Done] task={task_type}  samples={len(manifest)}  "
        f"total={time.time() - t_all:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
