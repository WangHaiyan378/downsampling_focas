"""
Base dataset utilities for loading ERA5 and S2S data.
Handles resolution matching (downsampling ERA5 from 0.25° to 1.5° to match S2S).
"""

import os
import glob
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_s2s_file_list(s2s_root, model_name, var_name, years=None):
    """Get sorted list of S2S nc files for a model/variable."""
    base = os.path.join(s2s_root, model_name, var_name)
    if years is None:
        years = sorted(os.listdir(base))
    files = []
    for year in years:
        year_dir = os.path.join(base, str(year))
        if not os.path.isdir(year_dir):
            continue
        for month_dir in sorted(os.listdir(year_dir)):
            mpath = os.path.join(year_dir, month_dir)
            if os.path.isdir(mpath):
                nc_files = sorted(glob.glob(os.path.join(mpath, "*.nc")))
                files.extend(nc_files)
    return files


def get_era5_file_list(era5_root, var_name):
    """
    Legacy helper: list any ERA5 nc files matching a variable, regardless
    of layout ("monthly subdir" or "yearly flat files").
    """
    base_sub = os.path.join(era5_root, var_name)
    if os.path.isdir(base_sub):
        return sorted(glob.glob(os.path.join(base_sub, "*.nc")))
    return sorted(glob.glob(os.path.join(era5_root, f"{var_name}_*.nc")))


def era5_yearly_file(era5_root, var_name, year):
    """Return path to a yearly ERA5 file: <era5_root>/<var>_<year>.nc"""
    return os.path.join(era5_root, f"{var_name}_{int(year)}.nc")


def resolve_era5_var_coords(ds):
    """
    Return (lat_name, lon_name, time_name) for an ERA5 dataset.
    Supports both new ('latitude'/'longitude'/'valid_time') and legacy
    ('lat'/'lon'/'time') coord naming.
    """
    lat_name = next((c for c in ("latitude", "lat") if c in ds.coords), None)
    lon_name = next((c for c in ("longitude", "lon") if c in ds.coords), None)
    time_name = next((c for c in ("valid_time", "time") if c in ds.coords), None)
    return lat_name, lon_name, time_name


def load_era5_daily_mean(ds, target_date):
    """
    Slow path: extract a single day's daily-mean 2D field directly from a
    (lazily opened) ERA5 yearly dataset. Kept for fallback / debugging.
    Prefer the cached path via `ensure_era5_daily_cache` in production.
    """
    import numpy as _np
    import pandas as pd

    lat_name, lon_name, time_name = resolve_era5_var_coords(ds)
    if time_name is None:
        raise KeyError("ERA5 dataset has no recognizable time coordinate")

    var_name = next(iter(ds.data_vars))
    day_start = pd.Timestamp(target_date).normalize()
    day_end = day_start + pd.Timedelta(hours=23)

    arr = ds[var_name].sel({time_name: slice(day_start, day_end)})
    if arr.sizes.get(time_name, 0) == 0:
        raise ValueError(
            f"Target date {target_date} not found in ERA5 time range "
            f"[{ds[time_name].values[0]}..{ds[time_name].values[-1]}]"
        )
    data_2d = arr.mean(dim=time_name).values.astype(_np.float32)
    return ds[lat_name].values, ds[lon_name].values, data_2d


# ------------------------------------------------------------
# ERA5 daily-mean disk cache
# ------------------------------------------------------------

def _era5_daily_cache_dir(era5_root, cache_root=None):
    cache = cache_root if cache_root else os.path.join(era5_root, "_daily_cache")
    os.makedirs(cache, exist_ok=True)
    return cache


def era5_daily_cache_paths(era5_root, var_name, year, cache_root=None):
    """Return (data_npy, dates_npy, latlon_npz) paths for cached daily means."""
    cache = _era5_daily_cache_dir(era5_root, cache_root)
    base = f"{var_name}_{int(year)}"
    return (
        os.path.join(cache, f"{base}_daily.npy"),
        os.path.join(cache, f"{base}_dates.npy"),
        os.path.join(cache, f"{base}_latlon.npz"),
    )


def ensure_era5_daily_cache(era5_root, var_name, year, cache_root=None, verbose=True):
    """
    Build (or reuse) a daily-mean cache for a single ERA5 yearly file.

    Cache layout:
        <cache>/<var>_<year>_daily.npy   float32 (N_days, H, W)  daily mean
        <cache>/<var>_<year>_dates.npy   datetime64[D] (N_days,)
        <cache>/<var>_<year>_latlon.npz  lat, lon 1D float32 arrays

    Returns the three paths. Skips rebuild if all three files already exist.

    Implementation: reads the netCDF4 file in time-chunk-aligned blocks
    (typically ~28 days of hourly data at a time) and accumulates daily
    means in a single linear pass. This avoids re-decompressing the same
    HDF5 chunks for every day, which is the dominant cost when time chunks
    span many days (e.g., ERA5 yearly files chunked at ~674 hours).
    """
    import numpy as _np
    import pandas as pd
    import netCDF4 as nc
    import time as _time

    data_p, dates_p, latlon_p = era5_daily_cache_paths(
        era5_root, var_name, year, cache_root
    )
    if all(os.path.exists(p) for p in (data_p, dates_p, latlon_p)):
        return data_p, dates_p, latlon_p

    src = era5_yearly_file(era5_root, var_name, year)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing ERA5 yearly file: {src}")

    if verbose:
        print(f"[ERA5 cache] Building daily mean for {var_name}_{year} from {src}", flush=True)

    t0 = _time.time()
    d = nc.Dataset(src, "r")
    try:
        lat_name = next((c for c in ("latitude", "lat") if c in d.variables), None)
        lon_name = next((c for c in ("longitude", "lon") if c in d.variables), None)
        time_name = next((c for c in ("valid_time", "time") if c in d.variables), None)
        if time_name is None:
            raise KeyError(f"No time variable in {src}")
        data_var_name = next(
            v for v in d.variables
            if v not in (lat_name, lon_name, time_name, "number", "expver")
        )
        var = d[data_var_name]
        lat = _np.asarray(d[lat_name][:], dtype=_np.float32)
        lon = _np.asarray(d[lon_name][:], dtype=_np.float32)

        tvar = d[time_name]
        units = getattr(tvar, "units", None)
        cal = getattr(tvar, "calendar", "standard")
        times_num = tvar[:]
        try:
            times_dt = nc.num2date(
                times_num, units=units, calendar=cal,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
        except Exception:
            times_dt = nc.num2date(times_num, units=units, calendar=cal)
        times = pd.to_datetime(_np.asarray(times_dt))
        date_per_step = times.floor("D").values.astype("datetime64[D]")
        unique_dates, inverse = _np.unique(date_per_step, return_inverse=True)
        n_days = len(unique_dates)

        n_time = var.shape[0]
        H, W = var.shape[1], var.shape[2]
        out = _np.zeros((n_days, H, W), dtype=_np.float32)
        counts = _np.zeros((n_days,), dtype=_np.int32)

        chunking = var.chunking()
        t_chunk = int(chunking[0]) if chunking else 24
        block_size = max(24, t_chunk)

        for start in range(0, n_time, block_size):
            end = min(start + block_size, n_time)
            t_read0 = _time.time()
            block = var[start:end]
            block = _np.asarray(block, dtype=_np.float32)
            block_idx = inverse[start:end]
            for d_i in _np.unique(block_idx):
                mask = block_idx == d_i
                out[d_i] += block[mask].sum(axis=0)
                counts[d_i] += int(mask.sum())
            if verbose:
                elapsed = _time.time() - t0
                print(
                    f"  [{var_name}_{year}] read hours {start}-{end - 1} / {n_time}  "
                    f"block={end - start}h  block_t={_time.time() - t_read0:.1f}s  "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

        nonzero = counts > 0
        out[nonzero] /= counts[nonzero, None, None].astype(_np.float32)

    finally:
        d.close()

    _np.save(data_p, out)
    _np.save(dates_p, unique_dates)
    _np.savez(latlon_p, lat=lat, lon=lon)

    if verbose:
        print(
            f"[ERA5 cache] Saved {data_p}  shape={out.shape}  "
            f"total={_time.time() - t0:.1f}s",
            flush=True,
        )
    return data_p, dates_p, latlon_p


class ERA5DailyCacheReader:
    """
    Fast memmap-backed reader of cached daily means built by
    `ensure_era5_daily_cache`. Thread/fork-safe: each worker lazily mmap's its
    own view; OS page cache is shared across processes.
    """

    _instances = {}

    def __init__(self, era5_root, var_name, year, cache_root=None):
        import numpy as _np

        data_p, dates_p, latlon_p = ensure_era5_daily_cache(
            era5_root, var_name, year, cache_root=cache_root, verbose=True
        )
        self.var_name = var_name
        self.year = int(year)
        self.data = _np.load(data_p, mmap_mode="r")
        self.dates = _np.load(dates_p)
        latlon = _np.load(latlon_p)
        self.lat = latlon["lat"]
        self.lon = latlon["lon"]
        self._date_to_idx = {str(d): i for i, d in enumerate(self.dates)}

    @classmethod
    def get(cls, era5_root, var_name, year, cache_root=None):
        key = (os.path.abspath(era5_root), var_name, int(year), cache_root)
        if key not in cls._instances:
            cls._instances[key] = cls(era5_root, var_name, year, cache_root)
        return cls._instances[key]

    def daily_mean(self, target_date):
        """Return (lat, lon, field_2d float32) for target_date."""
        import numpy as _np

        key = str(_np.datetime64(target_date, "D"))
        idx = self._date_to_idx.get(key)
        if idx is None:
            raise KeyError(
                f"Date {target_date} not in {self.var_name}_{self.year} cache "
                f"(range {self.dates[0]} .. {self.dates[-1]})"
            )
        return self.lat, self.lon, _np.asarray(self.data[idx], dtype=_np.float32)


def downsample_era5_to_s2s(era5_data, era5_lat, era5_lon, s2s_lat, s2s_lon):
    """
    Downsample ERA5 data to S2S grid using bilinear interpolation.
    era5_data: (T, lat_era5, lon_era5)
    returns: (T, lat_s2s, lon_s2s)
    """
    T = era5_data.shape[0]
    out = np.zeros((T, len(s2s_lat), len(s2s_lon)), dtype=np.float32)

    era5_lat_sorted = np.sort(era5_lat)
    era5_lon_sorted = np.sort(era5_lon)
    lat_ascending = era5_lat[0] < era5_lat[-1]

    for t in range(T):
        data = era5_data[t]
        if not lat_ascending:
            data = data[::-1]
            interp = RegularGridInterpolator(
                (era5_lat_sorted, era5_lon_sorted), data,
                method="linear", bounds_error=False, fill_value=None,
            )
        else:
            interp = RegularGridInterpolator(
                (era5_lat, era5_lon), data,
                method="linear", bounds_error=False, fill_value=None,
            )
        s2s_lat_s = np.sort(s2s_lat) if s2s_lat[0] > s2s_lat[-1] else s2s_lat
        grid_lat, grid_lon = np.meshgrid(s2s_lat_s, s2s_lon, indexing="ij")
        points = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=-1)
        result = interp(points).reshape(len(s2s_lat), len(s2s_lon))
        if s2s_lat[0] > s2s_lat[-1]:
            result = result[::-1]
        out[t] = result.astype(np.float32)
    return out


def generate_static_features(H, W, lat_values=None, lon_values=None):
    """
    Generate synthetic static features:
    - Latitude encoding (sin, cos)
    - Longitude encoding (sin, cos)
    Shape: (4, H, W)
    """
    if lat_values is None:
        lat = np.linspace(-90, 90, H)
    else:
        lat = np.asarray(lat_values, dtype=np.float32)
    if lon_values is None:
        lon = np.linspace(0, 360, W, endpoint=False)
    else:
        lon = np.asarray(lon_values, dtype=np.float32)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    lat_sin = np.sin(np.deg2rad(lat_grid))
    lat_cos = np.cos(np.deg2rad(lat_grid))
    lon_sin = np.sin(np.deg2rad(lon_grid))
    lon_cos = np.cos(np.deg2rad(lon_grid))

    static = np.stack([lat_sin, lat_cos, lon_sin, lon_cos], axis=0).astype(np.float32)
    return static


def _wrap_lon_to_180(lon):
    """Wrap longitude from [0, 360) to [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def _lcz_cache_path(lcz_tif_path, s2s_lat, s2s_lon, band_index, resampling):
    """Build a deterministic cache filename for the LCZ-on-grid array."""
    import hashlib
    s2s_lat = np.asarray(s2s_lat, dtype=np.float64)
    s2s_lon = np.asarray(s2s_lon, dtype=np.float64)
    key = (
        f"{os.path.abspath(lcz_tif_path)}|band{band_index}|{resampling}|"
        f"H{len(s2s_lat)}|W{len(s2s_lon)}|"
        f"lat0{float(s2s_lat[0]):.6f}|latN{float(s2s_lat[-1]):.6f}|"
        f"lon0{float(s2s_lon[0]):.6f}|lonN{float(s2s_lon[-1]):.6f}"
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    base_dir = os.path.dirname(os.path.abspath(lcz_tif_path))
    cache_dir = os.path.join(base_dir, "_cache")
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(lcz_tif_path))[0]
    return os.path.join(
        cache_dir,
        f"{name}_b{band_index}_{resampling}_{len(s2s_lat)}x{len(s2s_lon)}_{digest}.npy",
    )


def load_lcz_on_s2s_grid(
    lcz_tif_path,
    s2s_lat,
    s2s_lon,
    band_index=1,
    fill_value=0.0,
    resampling="average",
    use_cache=True,
):
    """
    Downsample a huge LCZ GeoTIFF onto the S2S lat/lon grid using streaming
    reprojection (no full-raster read, memory-safe even for global 100 m LCZ).

    Results are cached to disk next to the source .tif so subsequent runs
    load in milliseconds.

    Returns:
        lcz_grid: (lat, lon) float32 normalized to [0, 1]
    """
    if use_cache:
        cache_path = _lcz_cache_path(lcz_tif_path, s2s_lat, s2s_lon, band_index, resampling)
        try:
            if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(lcz_tif_path):
                cached = np.load(cache_path)
                if cached.shape == (len(s2s_lat), len(s2s_lon)):
                    print(f"[LCZ] Loaded cached grid: {cache_path}")
                    return cached.astype(np.float32)
        except Exception as e:
            print(f"[LCZ] Cache read failed ({e}); recomputing.")
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import Affine
    except ImportError as exc:
        raise ImportError(
            "LCZ loading requires rasterio. Please run: pip install rasterio"
        ) from exc

    resampling_map = {
        "average": Resampling.average,
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "mode": Resampling.mode,
    }
    resamp = resampling_map.get(resampling, Resampling.average)

    s2s_lat = np.asarray(s2s_lat, dtype=np.float64)
    s2s_lon = np.asarray(s2s_lon, dtype=np.float64)
    lon_180 = _wrap_lon_to_180(s2s_lon)

    H, W = len(s2s_lat), len(lon_180)

    lat_order = np.argsort(-s2s_lat)
    lon_order = np.argsort(lon_180)
    lat_sorted = s2s_lat[lat_order]
    lon_sorted = lon_180[lon_order]

    dx = float(lon_sorted[1] - lon_sorted[0]) if W > 1 else 1.5
    dy = float(lat_sorted[1] - lat_sorted[0]) if H > 1 else -1.5
    left = float(lon_sorted[0] - dx / 2.0)
    top = float(lat_sorted[0] - dy / 2.0)
    dst_transform = Affine(dx, 0.0, left, 0.0, dy, top)

    with rasterio.open(lcz_tif_path) as src:
        if band_index < 1 or band_index > src.count:
            raise ValueError(
                f"Requested LCZ band {band_index} out of range. Available: 1..{src.count}"
            )
        dst = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, band_index),
            destination=dst,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=resamp,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    lat_inv = np.argsort(lat_order)
    lon_inv = np.argsort(lon_order)
    out = dst[lat_inv, :][:, lon_inv]

    out = np.nan_to_num(out, nan=fill_value, posinf=fill_value, neginf=fill_value)

    min_v = float(np.min(out))
    max_v = float(np.max(out))
    if max_v > min_v:
        out = (out - min_v) / (max_v - min_v)
    else:
        out = np.zeros_like(out, dtype=np.float32)

    out = out.astype(np.float32)

    if use_cache:
        try:
            np.save(cache_path, out)
            print(f"[LCZ] Saved grid cache: {cache_path}")
        except Exception as e:
            print(f"[LCZ] Cache write failed ({e}); continuing without cache.")

    return out


def parse_init_date(filename):
    """Extract init date string YYYYMMDD from filename like 20250102_ys_mean.nc"""
    basename = os.path.basename(filename)
    return basename.split("_")[0]


def date_to_month_day(date_str):
    """Convert YYYYMMDD to (year, month, day)."""
    return int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])


def resolve_s2s_var_divisors(dcfg):
    """
    Per-variable positive divisors applied as: x_scaled = x / divisor.
    Keeps magnitudes O(1) for AMP-friendly training (esp. ssrd ~1e6–1e7).
    """
    svars = dcfg.get("s2s_vars", [])
    raw = dcfg.get("s2s_var_divisors") or {}
    out = {}
    for v in svars:
        if v in raw and raw[v] is not None:
            out[v] = float(raw[v])
        elif v == "ssrd":
            out[v] = float(dcfg.get("s2s_ssrd_divisor", 1.0e7))
        else:
            out[v] = 1.0
    return out


def scale_s2s_numpy(var_name, arr, divisors):
    """Apply 1/divisor in float32 for a numpy array of any shape (spatial trailing dims ok)."""
    div = float(divisors.get(var_name, 1.0))
    x = np.asarray(arr, dtype=np.float32)
    if div != 1.0:
        x = (x / np.float32(div)).astype(np.float32)
    return x


def resolve_era5_target_divisors(dcfg):
    """Per-variable positive divisors applied to ERA5 target fields before loss."""
    tvars = dcfg.get("era5_target_vars", ["u10", "v10"])
    raw = dcfg.get("era5_target_divisors") or {}
    out = {}
    for v in tvars:
        if v in raw and raw[v] is not None:
            out[v] = float(raw[v])
        elif v == "ssrd":
            out[v] = float(dcfg.get("era5_ssrd_divisor", 1.0e7))
        else:
            out[v] = 1.0
    return out


def scale_era5_numpy(var_name, arr, divisors):
    """Mirror of scale_s2s_numpy for ERA5 target fields."""
    div = float(divisors.get(var_name, 1.0))
    x = np.asarray(arr, dtype=np.float32)
    if div != 1.0:
        x = (x / np.float32(div)).astype(np.float32)
    return x


# ---- ERA5 target resolution: supports derived variables (wind10m / wind100m) ----

ERA5_DERIVED_SOURCES = {
    # name → (source var for u, source var for v)  — magnitude = sqrt(u^2 + v^2)
    "wind10m": ("u10", "v10"),
    "wind100m": ("u100", "v100"),
}


def resolve_era5_source_vars(target_var):
    """Return list of underlying ERA5 variable names that must exist on disk."""
    if target_var in ERA5_DERIVED_SOURCES:
        u, v = ERA5_DERIVED_SOURCES[target_var]
        return [u, v]
    return [target_var]


def load_era5_target_field(era5_root, target_var, target_date, year, cache_root=None):
    """
    Return (lat, lon, 2D float32) for a (possibly derived) ERA5 target variable
    at a given target_date, using the daily-mean cache under era5_cache_root.
    """
    if target_var in ERA5_DERIVED_SOURCES:
        uvar, vvar = ERA5_DERIVED_SOURCES[target_var]
        ru = ERA5DailyCacheReader.get(era5_root, uvar, year, cache_root=cache_root)
        rv = ERA5DailyCacheReader.get(era5_root, vvar, year, cache_root=cache_root)
        lat, lon, u = ru.daily_mean(target_date)
        _, _, v = rv.daily_mean(target_date)
        u = np.asarray(u, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        field = np.sqrt(u * u + v * v).astype(np.float32)
        return lat, lon, field
    r = ERA5DailyCacheReader.get(era5_root, target_var, year, cache_root=cache_root)
    return r.daily_mean(target_date)


def _lat_mask(lat, lat_min, lat_max):
    lat = np.asarray(lat, dtype=np.float64)
    return (lat >= float(lat_min)) & (lat <= float(lat_max))


def _lon_mask(lon, lon_min, lon_max):
    """Handle longitudes given in [0, 360) or [-180, 180). Supports wrap-around."""
    lon = np.asarray(lon, dtype=np.float64)
    lon_n = np.mod(lon, 360.0)
    lo = float(lon_min) % 360.0
    hi = float(lon_max) % 360.0
    if lo == hi:
        return np.ones_like(lon_n, dtype=bool)
    if lo < hi:
        return (lon_n >= lo) & (lon_n <= hi)
    return (lon_n >= lo) | (lon_n <= hi)


def compute_region_indices(lat, lon, region):
    """
    Return (lat_idx, lon_idx) arrays of integer indices (monotonic) and the
    cropped (lat_sub, lon_sub) arrays for a bounding box region dict with
    lat_min / lat_max / lon_min / lon_max (degrees).
    If region is None or disabled, returns the full indices.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if not region or not region.get("enabled", True):
        return (
            np.arange(len(lat)),
            np.arange(len(lon)),
            lat.copy(),
            lon.copy(),
        )
    lm = _lat_mask(lat, region["lat_min"], region["lat_max"])
    wm = _lon_mask(lon, region["lon_min"], region["lon_max"])
    lat_idx = np.flatnonzero(lm)
    lon_idx = np.flatnonzero(wm)
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        raise ValueError(
            f"compute_region_indices: empty crop. "
            f"region={region}  "
            f"lat range=[{lat.min():.3f},{lat.max():.3f}]  "
            f"lon range=[{lon.min():.3f},{lon.max():.3f}]"
        )
    return lat_idx, lon_idx, lat[lat_idx], lon[lon_idx]


def get_region_cfg(dcfg):
    """Read the optional data.region block. Returns dict or None."""
    r = dcfg.get("region") if dcfg else None
    if not r:
        return None
    if not r.get("enabled", True):
        return None
    for k in ("lat_min", "lat_max", "lon_min", "lon_max"):
        if k not in r:
            raise ValueError(f"data.region missing '{k}'")
    return r


def resolve_num_leads_and_lead_times(dcfg, mcfg):
    """Return (num_leads, lead_times list). If lead_times omitted, use 1..num_leads."""
    nl = int(dcfg.get("num_leads", mcfg.get("num_leads", 45)))
    lt = dcfg.get("lead_times")
    if lt is None or (isinstance(lt, (list, tuple)) and len(lt) == 0):
        lt = list(range(1, nl + 1))
    lt = [int(x) for x in lt]
    if len(lt) != nl:
        raise ValueError(
            f"data.num_leads ({nl}) must match len(data.lead_times) ({len(lt)}). "
            f"Either set lead_times explicitly or omit it to auto-fill 1..num_leads."
        )
    return nl, lt
