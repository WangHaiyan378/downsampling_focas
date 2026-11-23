from .history2era5.model import History2ERA5Model
from .fcst2era5fit.model import Forecast2ERA5FitModel


def build_model(cfg):
    task = cfg["task_type"]
    if task == "history2era5":
        return History2ERA5Model(cfg["model"])
    elif task == "fcst2era5fit":
        return Forecast2ERA5FitModel(cfg["model"])
    else:
        raise ValueError(f"Unknown task_type: {task}")
