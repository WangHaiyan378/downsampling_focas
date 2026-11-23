"""
Task B Dataset: Forecast2ERA5Fit
Loads multi-model future forecasts and ERA5 targets.

Real data: loads S2S forecast data at future lead times + ERA5 ground truth
Demo data: generates random tensors
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .base_dataset import (
    compute_region_indices,
    generate_static_features,
    get_region_cfg,
    load_era5_target_field,
    resolve_era5_source_vars,
    resolve_era5_target_divisors,
    resolve_num_leads_and_lead_times,
    resolve_s2s_var_divisors,
    scale_era5_numpy,
    scale_s2s_numpy,
)


class DummyForecast2ERA5FitDataset(Dataset):
    """Dummy dataset for Task B: generates random tensors matching expected shapes."""

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        dcfg = cfg["data"]
        mcfg = cfg["model"]
        self.num_models = mcfg["num_models"]
        self.num_leads, self.lead_times = resolve_num_leads_and_lead_times(dcfg, mcfg)
        self.c_in = mcfg["c_in"]
        self.c_static = mcfg["c_static"]
        self.H = dcfg.get("demo_grid_h", dcfg.get("grid_h", 16))
        self.W = dcfg.get("demo_grid_w", dcfg.get("grid_w", 32))

        n = 40 if split == "train" else 10
        self.length = n
        static = generate_static_features(self.H, self.W)
        if self.c_static > static.shape[0]:
            extra = np.zeros((self.c_static - static.shape[0], self.H, self.W), dtype=np.float32)
            static = np.concatenate([static, extra], axis=0)
        elif self.c_static < static.shape[0]:
            static = static[:self.c_static]
        self.static = torch.from_numpy(static)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_forecast = torch.randn(self.num_models, self.num_leads, self.c_in, self.H, self.W)
        x_static = self.static.clone()
        lead_ids = torch.tensor(self.lead_times, dtype=torch.long)
        target_uv = torch.randn(self.num_leads, 2, self.H, self.W)

        return {
            "x_forecast": x_forecast,
            "x_static": x_static,
            "lead_ids": lead_ids,
            "target_uv": target_uv,
        }


class Forecast2ERA5FitDataset(Dataset):
    """
    Real dataset for Task B.
    For each sample (init_date):
    - Loads S2S forecasts at lead_times from each model
    - Loads ERA5 u10/v10 at matching dates as targets
    """

    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        dcfg = cfg["data"]
        mcfg = cfg["model"]
        self.era5_root = dcfg["era5_root"]
        self.s2s_root = dcfg["s2s_root"]
        self.model_names = dcfg["models"]
        self.s2s_vars = dcfg["s2s_vars"]
        self.num_leads, self.lead_times = resolve_num_leads_and_lead_times(dcfg, mcfg)
        self.s2s_h = int(dcfg.get("s2s_grid_h", dcfg.get("grid_h", 121)))
        self.s2s_w = int(dcfg.get("s2s_grid_w", dcfg.get("grid_w", 240)))
        self.divisors = resolve_s2s_var_divisors(dcfg)
        self.c_static = mcfg["c_static"]
        self.lcz_path = dcfg.get("lcz_path")
        self.lcz_band = dcfg.get("lcz_band", 1)
        self.lcz_resampling = dcfg.get("lcz_resampling", "average")
        self.era5_target_vars = dcfg.get("era5_target_vars", ["u10", "v10"])
        self.era5_target_divisors = resolve_era5_target_divisors(dcfg)

        self.era5_cache_root = dcfg.get("era5_cache_root")
        self.region = get_region_cfg(dcfg)
        self._era5_date_cache = {}
        self._s2s_crop_idx = None
        self._era5_crop_idx = None
        self.samples = self._build_sample_index(split)
        self.static = self._build_static_features()

    def _era5_available_dates(self, var_name, year):
        """Ensure (var, year) daily cache is built; return set of available dates."""
        import numpy as _np
        from .base_dataset import era5_yearly_file, ensure_era5_daily_cache

        key = (var_name, int(year))
        if key in self._era5_date_cache:
            return self._era5_date_cache[key]
        path = era5_yearly_file(self.era5_root, var_name, year)
        if not os.path.exists(path):
            self._era5_date_cache[key] = None
            return None
        try:
            _, dates_p, _ = ensure_era5_daily_cache(
                self.era5_root, var_name, year, cache_root=self.era5_cache_root
            )
            dates = _np.load(dates_p)
            self._era5_date_cache[key] = set(str(d) for d in dates)
        except Exception as e:
            print(f"[WARN] ERA5 cache build failed for {var_name}_{year}: {e}")
            self._era5_date_cache[key] = None
        return self._era5_date_cache[key]

    def _build_sample_index(self, split):
        from datetime import datetime, timedelta
        import pandas as pd
        import xarray as xr
        from .base_dataset import get_s2s_file_list, parse_init_date

        first_model = self.model_names[0]
        first_var = self.s2s_vars[0]
        all_files = get_s2s_file_list(self.s2s_root, first_model, first_var)

        samples = []
        for f in all_files:
            init_date = parse_init_date(f)

            per_model_files = {}
            valid = True
            for model in self.model_names:
                per_model_files[model] = {}
                for var in self.s2s_vars:
                    pattern = f.replace(
                        os.path.join(self.s2s_root, first_model, first_var),
                        os.path.join(self.s2s_root, model, var),
                    )
                    if os.path.exists(pattern):
                        per_model_files[model][var] = pattern
                    else:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            try:
                init_dt = datetime.strptime(init_date, "%Y%m%d")
            except ValueError:
                continue

            ok = True
            for lt in self.lead_times:
                target_date = (init_dt + timedelta(days=int(lt))).date()
                year = target_date.year
                date_key = str(pd.Timestamp(target_date).date())
                for tvar in self.era5_target_vars:
                    for raw in resolve_era5_source_vars(tvar):
                        avail = self._era5_available_dates(raw, year)
                        if avail is None or date_key not in avail:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            ref_nc = per_model_files[first_model][self.s2s_vars[0]]
            with xr.open_dataset(ref_nc) as ds:
                ntime = int(ds.sizes["time"])
            if ntime <= max(self.lead_times):
                continue

            samples.append({
                "init_date": init_date,
                "s2s_files": per_model_files,
            })

        n = len(samples)
        split_idx = int(n * 0.8)
        if split == "train":
            return samples[:split_idx]
        return samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def _resolve_era5_crop_idx(self):
        if self._era5_crop_idx is not None:
            return self._era5_crop_idx
        from datetime import datetime, timedelta
        s0 = self.samples[0]
        init_dt = datetime.strptime(s0["init_date"], "%Y%m%d")
        first_td = (init_dt + timedelta(days=int(self.lead_times[0]))).date()
        tvar0 = self.era5_target_vars[0]
        elat, elon, _ = load_era5_target_field(
            self.era5_root, tvar0, first_td, first_td.year,
            cache_root=self.era5_cache_root,
        )
        lat_idx, lon_idx, elat_sub, elon_sub = compute_region_indices(elat, elon, self.region)
        self._era5_crop_idx = (lat_idx, lon_idx, elat_sub, elon_sub)
        return self._era5_crop_idx

    def _resolve_s2s_crop_idx(self):
        if self._s2s_crop_idx is not None:
            return self._s2s_crop_idx
        import xarray as xr
        s0 = self.samples[0]
        fpath = s0["s2s_files"][self.model_names[0]][self.s2s_vars[0]]
        with xr.open_dataset(fpath) as ds:
            slat = ds["lat"].values
            slon = ds["lon"].values
        lat_idx, lon_idx, _, _ = compute_region_indices(slat, slon, self.region)
        self._s2s_crop_idx = (lat_idx, lon_idx)
        return self._s2s_crop_idx

    def _build_static_features(self):
        from .base_dataset import load_lcz_on_s2s_grid

        if len(self.samples) == 0:
            era5_h = int(self.cfg["data"].get("era5_grid_h", self.cfg["data"].get("grid_h", 721)))
            era5_w = int(self.cfg["data"].get("era5_grid_w", self.cfg["data"].get("grid_w", 1440)))
            static = generate_static_features(era5_h, era5_w)
        else:
            _, _, elat_sub, elon_sub = self._resolve_era5_crop_idx()
            static = generate_static_features(len(elat_sub), len(elon_sub), elat_sub, elon_sub)
            if self.lcz_path:
                lcz = load_lcz_on_s2s_grid(
                    self.lcz_path, elat_sub, elon_sub,
                    band_index=self.lcz_band,
                    resampling=self.lcz_resampling,
                )
                static = np.concatenate([static, lcz[None, ...]], axis=0)

        if self.c_static > static.shape[0]:
            extra = np.zeros(
                (self.c_static - static.shape[0], static.shape[1], static.shape[2]),
                dtype=np.float32,
            )
            static = np.concatenate([static, extra], axis=0)
        elif self.c_static < static.shape[0]:
            static = static[:self.c_static]

        return torch.from_numpy(static.astype(np.float32))

    def __getitem__(self, idx):
        import xarray as xr
        from datetime import datetime, timedelta

        sample = self.samples[idx]

        s2s_lat_idx, s2s_lon_idx = self._resolve_s2s_crop_idx()
        era5_lat_idx, era5_lon_idx, _, _ = self._resolve_era5_crop_idx()

        model_data = []
        for model in self.model_names:
            lead_data = []
            for var in self.s2s_vars:
                fpath = sample["s2s_files"][model][var]
                with xr.open_dataset(fpath) as ds:
                    data = ds[var].values

                var_leads = []
                for lt in self.lead_times:
                    tidx = min(int(lt), data.shape[0] - 1)
                    sl = data[tidx]
                    if self.region is not None:
                        sl = sl[s2s_lat_idx, :][:, s2s_lon_idx]
                    sl = scale_s2s_numpy(var, sl, self.divisors)
                    sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
                    var_leads.append(sl)
                lead_data.append(np.stack(var_leads, axis=0).astype(np.float32, copy=False))
            stacked = np.stack(lead_data, axis=1)
            model_data.append(stacked)

        x_forecast = np.stack(model_data, axis=0)

        init_dt = datetime.strptime(sample["init_date"], "%Y%m%d")
        target_leads = []
        for lt in self.lead_times:
            target_date = (init_dt + timedelta(days=int(lt))).date()
            year = target_date.year
            per_var_fields = []
            for tvar in self.era5_target_vars:
                _, _, day_mean = load_era5_target_field(
                    self.era5_root, tvar, target_date, year,
                    cache_root=self.era5_cache_root,
                )
                day_mean = np.asarray(day_mean, dtype=np.float32)
                if self.region is not None:
                    day_mean = day_mean[era5_lat_idx, :][:, era5_lon_idx]
                day_mean = scale_era5_numpy(tvar, day_mean, self.era5_target_divisors)
                day_mean = np.nan_to_num(day_mean, nan=0.0, posinf=0.0, neginf=0.0)
                per_var_fields.append(day_mean)
            target_leads.append(np.stack(per_var_fields, axis=0))
        target_uv = np.stack(target_leads, axis=0)

        return {
            "x_forecast": torch.from_numpy(x_forecast).float(),
            "x_static": self.static.clone(),
            "lead_ids": torch.tensor(self.lead_times, dtype=torch.long),
            "target_uv": torch.from_numpy(target_uv).float(),
        }
