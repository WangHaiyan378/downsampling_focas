"""
Cached datasets that read pre-packed per-init-date .npz files produced by
`scripts/build_sample_cache.py`.

The output dict exactly matches the uncached Task A / Task B datasets so that
`train.py`, `evaluate.py`, and `infer.py` work with no changes when
`data.use_sample_cache: true` is set in the config.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .base_dataset import compute_region_indices, get_region_cfg


class _CachedBase(Dataset):
    def __init__(self, cfg, split, task_key):
        dcfg = cfg["data"]
        cache_root = dcfg.get("sample_cache_root")
        if not cache_root:
            raise ValueError("data.sample_cache_root not set in config.")
        self.root = os.path.join(cache_root, task_key)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"Sample cache not found: {self.root}. "
                f"Run: python scripts/build_sample_cache.py --config <your_config>"
            )

        meta_path = os.path.join(self.root, "_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing {meta_path}.")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.lat_idx = None
        self.lon_idx = None
        region = get_region_cfg(dcfg)
        if region is not None and "lat" in self.meta and "lon" in self.meta:
            lat = np.asarray(self.meta["lat"], dtype=np.float32)
            lon = np.asarray(self.meta["lon"], dtype=np.float32)
            self.lat_idx, self.lon_idx, _, _ = compute_region_indices(lat, lon, region)

        files = sorted(self.meta["samples"])
        n = len(files)
        if n == 0:
            raise RuntimeError(f"No samples listed in {meta_path}.")
        split_idx = int(n * 0.8)
        if split == "train":
            files = files[:split_idx]
        else:
            files = files[split_idx:]
        self.files = [os.path.join(self.root, f) for f in files]

        static_path = os.path.join(self.root, "_static.npy")
        if not os.path.exists(static_path):
            raise FileNotFoundError(f"Missing {static_path}.")
        static = np.load(static_path)
        c_static = cfg["model"]["c_static"]
        if c_static > static.shape[0]:
            pad = np.zeros(
                (c_static - static.shape[0], *static.shape[1:]), dtype=np.float32,
            )
            static = np.concatenate([static, pad], axis=0)
        elif c_static < static.shape[0]:
            static = static[:c_static]
        static = self._crop_hw(static)
        self.static = torch.from_numpy(static.astype(np.float32))
        self.lead_ids_np = np.asarray(self.meta["lead_times"], dtype=np.int64)

    def __len__(self):
        return len(self.files)

    def _crop_hw(self, array):
        if self.lat_idx is None or self.lon_idx is None:
            return array
        return np.asarray(array)[..., self.lat_idx, :][..., :, self.lon_idx]


class CachedHistory2ERA5Dataset(_CachedBase):
    def __init__(self, cfg, split="train"):
        super().__init__(cfg, split, task_key="task_a")

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            x_history = self._crop_hw(data["x_history"])
            target_uv = self._crop_hw(data["target_uv"])
        return {
            "x_history": torch.from_numpy(np.ascontiguousarray(x_history)).float(),
            "x_static": self.static.clone(),
            "lead_ids": torch.from_numpy(self.lead_ids_np).long(),
            "target_uv": torch.from_numpy(np.ascontiguousarray(target_uv)).float(),
        }


class CachedForecast2ERA5FitDataset(_CachedBase):
    def __init__(self, cfg, split="train"):
        super().__init__(cfg, split, task_key="task_b")

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as data:
            x_forecast = self._crop_hw(data["x_forecast"])
            target_uv = self._crop_hw(data["target_uv"])
        return {
            "x_forecast": torch.from_numpy(np.ascontiguousarray(x_forecast)).float(),
            "x_static": self.static.clone(),
            "lead_ids": torch.from_numpy(self.lead_ids_np).long(),
            "target_uv": torch.from_numpy(np.ascontiguousarray(target_uv)).float(),
        }
