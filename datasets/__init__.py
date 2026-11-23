"""
Dataset package.

不在模組頂層 import task_a/task_b（避免 `import datasets.base_dataset` 時
強制載入 torch）。僅在呼叫 build_dataset() 時再載入 PyTorch 相關類別。
"""


def build_dataset(cfg, split="train"):
    task = cfg["task_type"]
    dcfg = cfg["data"]
    demo = dcfg.get("demo_mode", False)

    if demo:
        from .task_a_dataset import DummyHistory2ERA5Dataset
        from .task_b_dataset import DummyForecast2ERA5FitDataset
        if task == "history2era5":
            return DummyHistory2ERA5Dataset(cfg, split)
        if task == "fcst2era5fit":
            return DummyForecast2ERA5FitDataset(cfg, split)
        raise ValueError(f"Unknown task_type: {task}")

    if dcfg.get("use_sample_cache", False):
        from .cached_dataset import (
            CachedHistory2ERA5Dataset,
            CachedForecast2ERA5FitDataset,
        )
        if task == "history2era5":
            return CachedHistory2ERA5Dataset(cfg, split)
        if task == "fcst2era5fit":
            return CachedForecast2ERA5FitDataset(cfg, split)
        raise ValueError(f"Unknown task_type: {task}")

    from .task_a_dataset import History2ERA5Dataset
    from .task_b_dataset import Forecast2ERA5FitDataset

    if task == "history2era5":
        return History2ERA5Dataset(cfg, split)
    if task == "fcst2era5fit":
        return Forecast2ERA5FitDataset(cfg, split)
    raise ValueError(f"Unknown task_type: {task}")
