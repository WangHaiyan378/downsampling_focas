"""
Map data.* grid fields onto model.* so encoders and output heads agree.

- S2S inputs live on `encoder_grid_*` (default 121×240).
- ERA5 targets / predictions live on `grid_*` (default ERA5 native 721×1440).

If `era5_grid_h` / `era5_grid_w` are omitted, they default to `s2s_grid_*`
or legacy `grid_h` / `grid_w` (single-grid backward compatible).

Also syncs `num_leads` / `lead_times` / `history_len` onto `cfg["model"]`.
"""

import numpy as np

from datasets.base_dataset import (
    compute_region_indices,
    get_region_cfg,
    resolve_num_leads_and_lead_times,
)


def _canonical_latlon(n_lat: int, n_lon: int):
    """Canonical ERA5-style grid: descending lat 90→-90, ascending lon 0→360-step."""
    lat = np.linspace(90.0, -90.0, int(n_lat))
    lon = np.arange(int(n_lon)) * (360.0 / float(n_lon))
    return lat, lon


def _resolve_cropped_size(n_lat: int, n_lon: int, region):
    lat, lon = _canonical_latlon(n_lat, n_lon)
    lat_idx, lon_idx, _, _ = compute_region_indices(lat, lon, region)
    return int(len(lat_idx)), int(len(lon_idx))


def sync_model_grids_from_data(cfg, demo: bool = False) -> None:
    dcfg = cfg["data"]
    mcfg = cfg["model"]

    if demo:
        dh = dcfg.get("demo_grid_h", 16)
        dw = dcfg.get("demo_grid_w", 32)
        mcfg["encoder_grid_h"] = dh
        mcfg["encoder_grid_w"] = dw
        mcfg["grid_h"] = dh
        mcfg["grid_w"] = dw
    else:
        s2s_h = int(dcfg.get("s2s_grid_h", dcfg.get("grid_h", 121)))
        s2s_w = int(dcfg.get("s2s_grid_w", dcfg.get("grid_w", 240)))
        era5_h = int(dcfg.get("era5_grid_h", s2s_h))
        era5_w = int(dcfg.get("era5_grid_w", s2s_w))

        region = get_region_cfg(dcfg)
        if region is not None:
            s2s_h, s2s_w = _resolve_cropped_size(s2s_h, s2s_w, region)
            era5_h, era5_w = _resolve_cropped_size(era5_h, era5_w, region)

        mcfg["encoder_grid_h"] = int(s2s_h)
        mcfg["encoder_grid_w"] = int(s2s_w)
        mcfg["grid_h"] = int(era5_h)
        mcfg["grid_w"] = int(era5_w)

    nl, lt = resolve_num_leads_and_lead_times(dcfg, mcfg)
    dcfg["num_leads"] = nl
    dcfg["lead_times"] = lt
    mcfg["num_leads"] = nl
    mcfg["history_len"] = int(dcfg.get("history_len", mcfg.get("history_len", 45)))
