"""
逐格點 RMSE 空間熱力圖工具
=============================

計算邏輯（每個 lead group 對「init_time + group 內 leads」一起做 mean）：

    pred_sel: [N, len(lead_indices), H, W]
    obs_sel:  [N, len(lead_indices), H, W]
    err2 = (pred_sel - obs_sel) ** 2
    rmse_map[H, W] = sqrt(nanmean(err2, axis=(0, 1)))

注意：這是 mean (不是 sum)、不是累計 RMSE。
逐格點 RMSE 不做 cos(lat) 面積加權；面積加權只用於做全域/區域平均。

對外 API
-----------------------------------------------------------------------------
compute_grid_rmse_map(pred, obs, lead_indices, mask=None) -> [H, W]
compute_all_grid_rmse_maps(pred_dict, obs_dict, lead_groups, lead_days,
                           mask=None) -> dict[var][group_name] = [H, W]
plot_single_variable_rmse_maps(variable, rmse_maps_var, lat, lon,
                               output_path, units="", region_extent=None) -> None
plot_rmse_spatial_grid(rmse_maps, lat, lon, output_path,
                       variable_units=None, region_extent=None) -> None
save_rmse_maps(rmse_maps, output_dir, lat, lon, save_nc=True) -> None

支援 numpy 與 xarray.DataArray 輸入：對 xarray 自動 .values 化。
cartopy 可用就用，否則 fallback 到 matplotlib pcolormesh。
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# 工具：嘗試載入 cartopy（沒裝會 fallback）
# =============================================================================

def _try_import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def _to_numpy(x):
    """支援 xarray.DataArray 與 numpy.ndarray。"""
    if x is None:
        return None
    if hasattr(x, "values"):
        return np.asarray(x.values)
    return np.asarray(x)


# =============================================================================
# 中國陸地 mask / 邊界工具
# =============================================================================

def find_china_shapefile(shapefile_dir: str) -> str:
    """
    在 shapefile_dir 下遞歸查找中國邊界 .shp。
    優先選擇檔名包含 CHN_0 的國家級邊界。
    """
    import glob

    if not shapefile_dir:
        raise FileNotFoundError("shapefile_dir is empty.")

    pattern = os.path.join(shapefile_dir, "**", "*.shp")
    shp_files = glob.glob(pattern, recursive=True)
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found under: {shapefile_dir}")

    def _score(path: str) -> int:
        name = os.path.basename(path).lower()
        if "chn_0" in name:
            return 0
        if "chn" in name and "_0" in name:
            return 1
        if "chn" in name:
            return 2
        return 3

    shp_files = sorted(shp_files, key=lambda p: (_score(p), p))
    shp_path = shp_files[0]
    print(f"[MASK] Found China shapefile: {shp_path}")
    return shp_path


def find_taiwan_shapefile(shapefile_dir: Optional[str]) -> Optional[str]:
    """
    在 shapefile_dir 下尋找台灣邊界 .shp（例如 gadm41_TWN_0.shp）。
    找不到時回傳 None（不 raise，方便後續 fallback 到 Natural Earth）。
    """
    import glob

    if not shapefile_dir or not os.path.exists(shapefile_dir):
        return None

    pattern = os.path.join(shapefile_dir, "**", "*.shp")
    shp_files = glob.glob(pattern, recursive=True)
    candidates = []
    for p in shp_files:
        name = os.path.basename(p).lower()
        if "twn" in name or "taiwan" in name:
            candidates.append(p)

    if not candidates:
        return None

    def _score(p: str) -> int:
        n = os.path.basename(p).lower()
        if "twn_0" in n:
            return 0
        if "twn" in n and "_0" in n:
            return 1
        return 2

    candidates = sorted(candidates, key=lambda p: (_score(p), p))
    print(f"[MASK] Found Taiwan shapefile: {candidates[0]}")
    return candidates[0]


def _load_geometry_from_shp(shp_path: str):
    """讀取單一 .shp，返回 EPSG:4326 union geometry。優先用 geopandas、退回 cartopy。"""
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"shapefile not found: {shp_path}")

    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            raise ValueError(f"Empty shapefile: {shp_path}")
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
        return gdf.geometry.unary_union
    except Exception as e:
        print(f"[MASK] geopandas failed for {shp_path} ({e}); trying cartopy shapereader...")

    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        reader = shpreader.Reader(shp_path)
        return unary_union(list(reader.geometries()))
    except Exception as e:
        raise RuntimeError(
            f"Failed to read shapefile {shp_path}. Please install geopandas/shapely "
            "or cartopy, and check that .shp/.shx/.dbf are complete."
        ) from e


def load_china_geometry(shp_path: str):
    """讀取中國邊界 geometry（EPSG:4326）。保留舊 API 名稱。"""
    geom = _load_geometry_from_shp(shp_path)
    print("[MASK] Loaded China geometry; CRS=EPSG:4326.")
    return geom


def load_taiwan_geometry(
    shp_path: Optional[str] = None,
    fallback_natural_earth: bool = True,
    natural_earth_resolution: str = "10m",
):
    """
    優先用本地 Taiwan shapefile；不存在或讀取失敗時改用 cartopy Natural Earth admin_0_countries。
    回傳 (geometry_or_None, source_path_or_None)，source_path 之後可拿去畫邊界。
    """
    if shp_path and os.path.exists(shp_path):
        try:
            geom = _load_geometry_from_shp(shp_path)
            print(f"[MASK] Loaded Taiwan geometry from local shapefile: {shp_path}")
            return geom, shp_path
        except Exception as e:
            print(f"[MASK] WARNING: failed to load local Taiwan shapefile ({e}); "
                  "trying Natural Earth fallback...")

    if not fallback_natural_earth:
        return None, None

    try:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(
            category="cultural", name="admin_0_countries",
            resolution=natural_earth_resolution,
        )
        reader = shpreader.Reader(ne_path)
        for rec in reader.records():
            attrs = rec.attributes or {}
            adm = str(attrs.get("ADM0_A3") or attrs.get("ADM0_ISO") or "").upper()
            name = str(attrs.get("NAME") or attrs.get("NAME_LONG") or "").lower()
            if adm == "TWN" or "taiwan" in name:
                print(f"[MASK] Loaded Taiwan geometry from Natural Earth "
                      f"({natural_earth_resolution}, file={ne_path}).")
                return rec.geometry, ne_path
        print("[MASK] WARNING: Taiwan record not found in Natural Earth admin_0_countries.")
    except Exception as e:
        print(f"[MASK] WARNING: Natural Earth Taiwan load failed: {e}")

    return None, None


def _ensure_geometry_lonlat_mask(lat, lon):
    lat = _to_numpy(lat)
    lon = _to_numpy(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d.astype(float), lon2d.astype(float)
    if lat.ndim == 2 and lon.ndim == 2:
        return lat.astype(float), lon.astype(float)
    raise ValueError(f"lat/lon must both be 1D or both be 2D, got lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def _polygon_mask(geom, lat2d, lon2d, buffer_degree: float = 0.03) -> np.ndarray:
    """對單一 geometry 與 (lat2d, lon2d) 網格生成 bool mask。"""
    g = geom
    if buffer_degree and buffer_degree > 0:
        try:
            g = geom.buffer(float(buffer_degree))
        except Exception:
            g = geom

    mask = None
    try:
        import shapely
        if hasattr(shapely, "contains_xy"):
            mask = shapely.contains_xy(g, lon2d, lat2d)
    except Exception:
        mask = None

    if mask is None:
        try:
            from shapely import vectorized
            mask = vectorized.contains(g, lon2d, lat2d)
        except Exception:
            mask = None

    if mask is None:
        from shapely.geometry import Point
        flat = [g.contains(Point(x, y)) or g.touches(Point(x, y))
                for x, y in zip(lon2d.ravel(), lat2d.ravel())]
        mask = np.asarray(flat, dtype=bool).reshape(lat2d.shape)

    return np.asarray(mask, dtype=bool)


def build_china_land_mask(lat, lon, china_geometry, buffer_degree: float = 0.03) -> np.ndarray:
    """
    根據 lat/lon 網格和中國 geometry 生成 [H,W] bool mask。
    True 表示中國陸地範圍內。
    """
    lat2d, lon2d = _ensure_geometry_lonlat_mask(lat, lon)
    mask = _polygon_mask(china_geometry, lat2d, lon2d, buffer_degree=buffer_degree)
    valid = int(mask.sum())
    total = int(mask.size)
    print(f"[MASK] Built China land mask: shape={mask.shape}, valid_pixels={valid}, "
          f"total_pixels={total}, valid_ratio={valid / max(total, 1):.4f}")
    return mask


def build_combined_land_mask(
    lat,
    lon,
    geometries: Sequence,
    labels: Optional[Sequence[str]] = None,
    buffer_degree: float = 0.03,
) -> np.ndarray:
    """
    多個 geometry 取聯集做 mask。空 / None 的 geometry 會跳過。
    labels 用於 log 顯示（例如 ["China", "Taiwan"]）。
    """
    lat2d, lon2d = _ensure_geometry_lonlat_mask(lat, lon)
    if not geometries:
        raise ValueError("build_combined_land_mask: empty geometries list.")
    if labels is None:
        labels = [f"region_{i}" for i in range(len(geometries))]

    accum = None
    for geom, label in zip(geometries, labels):
        if geom is None:
            print(f"[MASK] skip empty geometry: {label}")
            continue
        sub = _polygon_mask(geom, lat2d, lon2d, buffer_degree=buffer_degree)
        v = int(sub.sum())
        print(f"[MASK]   {label}: valid_pixels={v}, valid_ratio={v / max(sub.size, 1):.4f}")
        accum = sub if accum is None else (accum | sub)

    if accum is None:
        raise RuntimeError("All geometries are None; cannot build land mask.")
    valid = int(accum.sum())
    total = int(accum.size)
    print(f"[MASK] Built combined land mask: shape={accum.shape}, valid_pixels={valid}, "
          f"total_pixels={total}, valid_ratio={valid / max(total, 1):.4f}")
    return accum


def _default_combined_mask_path(mask_path: Optional[str], include_taiwan: bool) -> Optional[str]:
    """當 include_taiwan=True 時，自動把 mask 檔名加上 _with_taiwan 後綴，避免誤用舊快取。"""
    if not mask_path:
        return mask_path
    if not include_taiwan:
        return mask_path
    base, ext = os.path.splitext(mask_path)
    if base.endswith("_with_taiwan"):
        return mask_path
    return base + "_with_taiwan" + (ext or ".npy")


def get_or_build_china_land_mask(
    lat,
    lon,
    target_shape: Tuple[int, int],
    mask_path: Optional[str],
    shapefile_dir: Optional[str],
    rebuild: bool = False,
    require: bool = True,
    include_taiwan: bool = True,
    taiwan_shapefile_path: Optional[str] = None,
    taiwan_natural_earth_resolution: str = "10m",
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """
    優先讀取已有 mask 檔；不存在或 rebuild=True 時用 shapefile 生成。

    include_taiwan=True 時：
        1) 在 shapefile_dir 找 TWN*.shp / taiwan*.shp；
        2) 找不到就 fallback 到 cartopy Natural Earth admin_0_countries（首次使用會自動下載）；
        3) 兩者都失敗時只給 warning（不會 raise），仍回傳 China-only mask；
        4) mask 檔名會自動加 _with_taiwan 後綴，避免和舊的 China-only 檔互相覆蓋。

    返回 (mask, china_shp_path, taiwan_boundary_path)。
    taiwan_boundary_path 可以餵給繪圖端疊邊界線；找不到時為 None。
    """
    target_shape = tuple(int(x) for x in target_shape)
    china_shp = None
    taiwan_boundary = None

    cache_path = _default_combined_mask_path(mask_path, include_taiwan)

    try:
        if cache_path and os.path.exists(cache_path) and not rebuild:
            mask = np.load(cache_path).astype(bool)
            if mask.shape != target_shape:
                raise ValueError(
                    f"Land mask shape {mask.shape} does not match target grid shape {target_shape}. "
                    "Please rebuild the mask with --rebuild_china_mask."
                )
            valid = int(mask.sum())
            total = int(mask.size)
            print(f"[MASK] Loaded existing land mask: {cache_path}")
            print(f"[MASK] shape={mask.shape}, valid_ratio={valid / max(total, 1):.4f}, "
                  f"include_taiwan={include_taiwan}")
            if shapefile_dir:
                try:
                    china_shp = find_china_shapefile(shapefile_dir)
                except Exception:
                    china_shp = None
                if include_taiwan:
                    taiwan_shp_local = taiwan_shapefile_path or find_taiwan_shapefile(shapefile_dir)
                    if taiwan_shp_local and os.path.exists(taiwan_shp_local):
                        taiwan_boundary = taiwan_shp_local
                    else:
                        try:
                            import cartopy.io.shapereader as shpreader
                            taiwan_boundary = shpreader.natural_earth(
                                category="cultural", name="admin_0_countries",
                                resolution=taiwan_natural_earth_resolution,
                            )
                        except Exception:
                            taiwan_boundary = None
            return mask, china_shp, taiwan_boundary

        if not shapefile_dir:
            raise FileNotFoundError("No shapefile_dir provided for building China land mask.")

        china_shp = find_china_shapefile(shapefile_dir)
        china_geom = load_china_geometry(china_shp)

        geometries = [china_geom]
        labels = ["China"]

        if include_taiwan:
            twn_shp = taiwan_shapefile_path or find_taiwan_shapefile(shapefile_dir)
            twn_geom, twn_src = load_taiwan_geometry(
                twn_shp,
                fallback_natural_earth=True,
                natural_earth_resolution=taiwan_natural_earth_resolution,
            )
            if twn_geom is not None:
                geometries.append(twn_geom)
                labels.append("Taiwan")
                taiwan_boundary = twn_src
            else:
                print("[MASK] WARNING: include_taiwan=True 但找不到台灣 geometry，"
                      "本次將只用中國陸地建立 mask。")

        mask = build_combined_land_mask(lat, lon, geometries, labels=labels)

        if mask.shape != target_shape:
            raise ValueError(f"Built land mask shape {mask.shape} != target grid shape {target_shape}")

        if cache_path:
            out_dir = os.path.dirname(cache_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            np.save(cache_path, mask.astype(bool))
            print(f"[MASK] Saved land mask to: {cache_path}")

        return mask.astype(bool), china_shp, taiwan_boundary

    except Exception as e:
        if require:
            raise
        print(f"[MASK] WARNING: failed to load/build land mask: {e}")
        return None, china_shp, taiwan_boundary



# =============================================================================
# 1) 單一 lead group 的逐格點 RMSE
# =============================================================================

def compute_grid_rmse_map(
    pred,
    obs,
    lead_indices: Sequence[int],
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    輸入:
        pred:         [N, L, H, W]  或 xarray.DataArray
        obs:          [N, L, H, W]
        lead_indices: 該 lead group 對應在 L 軸上的整數索引 list（不是 lead day 值）
        mask:         [H, W] bool（True=保留），None 表示不遮罩
    輸出:
        rmse_map: [H, W]，單位與輸入相同
    """
    pred = _to_numpy(pred)
    obs = _to_numpy(obs)

    if pred.ndim != 4 or obs.ndim != 4:
        raise ValueError(
            f"compute_grid_rmse_map: pred/obs 必須為 [N,L,H,W]，"
            f"目前 pred.shape={pred.shape}, obs.shape={obs.shape}"
        )
    if pred.shape != obs.shape:
        raise ValueError(
            f"compute_grid_rmse_map: pred/obs shape 不一致："
            f"{pred.shape} vs {obs.shape}"
        )

    lead_indices = np.asarray(list(lead_indices), dtype=np.int64)
    if lead_indices.size == 0:
        H, W = pred.shape[-2:]
        return np.full((H, W), np.nan, dtype=np.float64)

    L = pred.shape[1]
    if (lead_indices < 0).any() or (lead_indices >= L).any():
        raise IndexError(
            f"lead_indices 超出範圍 [0, {L - 1}]：{lead_indices}"
        )

    pred_sel = pred[:, lead_indices, :, :].astype(np.float64, copy=False)
    obs_sel = obs[:, lead_indices, :, :].astype(np.float64, copy=False)
    err2 = (pred_sel - obs_sel) ** 2

    # 任何一邊是 NaN → 該位置整體不參與該格點平均
    bad = ~np.isfinite(pred_sel) | ~np.isfinite(obs_sel)
    err2 = np.where(bad, np.nan, err2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # axis=(0,1) → 對 init_time + lead_group 一起平均
        mean_err2 = np.nanmean(err2, axis=(0, 1))
        rmse_map = np.sqrt(mean_err2)

    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != rmse_map.shape:
            raise ValueError(
                f"mask shape {m.shape} 與 rmse_map shape {rmse_map.shape} 不符"
            )
        rmse_map = np.where(m, rmse_map, np.nan)

    return rmse_map.astype(np.float32)


# =============================================================================
# 2) 多變數 × 多 lead group 一次算完
# =============================================================================

def _resolve_lead_indices(
    lead_days_arr: np.ndarray,
    lead_min: int,
    lead_max: int,
) -> np.ndarray:
    """
    lead_days_arr: 真實 lead day 值（例如 [1,2,...,45]）
    回傳該 group 在 L 軸上的整數索引（含端點）。
    """
    lead_days_arr = np.asarray(lead_days_arr).astype(int)
    sel = np.where((lead_days_arr >= lead_min) & (lead_days_arr <= lead_max))[0]
    return sel


def compute_all_grid_rmse_maps(
    pred_dict: Mapping[str, np.ndarray],
    obs_dict: Mapping[str, np.ndarray],
    lead_groups: Mapping[str, Tuple[int, int]],
    lead_days: Sequence[int],
    mask: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    輸入:
        pred_dict[var]:   [N, L, H, W]
        obs_dict[var]:    [N, L, H, W]
        lead_groups: 例如 {"01-10": (1,10), "11-20": (11,20), ...}
        lead_days:   長度 L 的真實 lead day 陣列（與 pred_dict 中 L 對齊）
        mask:        [H, W] bool，可選

    輸出:
        rmse_maps[var][group_name] = [H, W]
        同時對缺失變數 / 不足 lead 給 warning，但不會 crash。
    """
    rmse_maps: Dict[str, Dict[str, np.ndarray]] = {}
    lead_days_arr = np.asarray(lead_days, dtype=np.int64)

    for var, pred in pred_dict.items():
        if var not in obs_dict:
            print(f"[RMSE MAP] WARNING: variable={var} 缺對應 obs，跳過。")
            continue
        obs = obs_dict[var]
        pred = _to_numpy(pred)
        obs = _to_numpy(obs)
        if pred.shape != obs.shape:
            print(f"[RMSE MAP] WARNING: var={var} pred {pred.shape} vs obs {obs.shape} 不一致，跳過。")
            continue
        rmse_maps[var] = {}
        if verbose:
            print(f"[RMSE MAP] variable={var}, pred.shape={pred.shape}")

        for group_name, (lmin, lmax) in lead_groups.items():
            sel_idx = _resolve_lead_indices(lead_days_arr, lmin, lmax)
            if sel_idx.size == 0:
                print(f"[RMSE MAP] WARNING: var={var} group={group_name} "
                      f"({lmin}-{lmax} day) 在 lead_days {lead_days_arr.tolist()} 內無對應，跳過。")
                rmse_maps[var][group_name] = np.full(pred.shape[-2:], np.nan, dtype=np.float32)
                continue

            rmse_map = compute_grid_rmse_map(pred, obs, sel_idx, mask=mask)

            if verbose:
                finite = np.isfinite(rmse_map)
                if finite.any():
                    mn = float(np.nanmin(rmse_map))
                    mx = float(np.nanmax(rmse_map))
                    me = float(np.nanmean(rmse_map))
                else:
                    mn = mx = me = float("nan")
                nan_ratio = float((~finite).sum() / rmse_map.size)
                print(f"[RMSE MAP] variable={var}, group={group_name}, "
                      f"pred_sel=({pred.shape[0]},{sel_idx.size},{pred.shape[2]},{pred.shape[3]})")
                print(f"[RMSE MAP]   min={mn:.4g}  mean={me:.4g}  max={mx:.4g}  "
                      f"nan_ratio={nan_ratio:.3f}")

            rmse_maps[var][group_name] = rmse_map

    return rmse_maps


# =============================================================================
# 3) 繪圖工具
# =============================================================================

def _ensure_meshgrid(lat, lon):
    lat = _to_numpy(lat)
    lon = _to_numpy(lon)
    if lat.ndim == 1 and lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d
    if lat.ndim == 2 and lon.ndim == 2:
        return lat, lon
    raise ValueError(f"lat/lon 必須都是 1D 或都是 2D：lat.ndim={lat.ndim}, lon.ndim={lon.ndim}")


def _common_vrange(group_maps: Mapping[str, np.ndarray]) -> Tuple[float, float]:
    """同一變數所有 group 共用色標：vmin=0，vmax=所有 group 95 分位數。"""
    arrs = []
    for v in group_maps.values():
        v = np.asarray(v)
        v = v[np.isfinite(v)]
        if v.size:
            arrs.append(v)
    if not arrs:
        return 0.0, 1.0
    cat = np.concatenate(arrs)
    vmax = float(np.nanpercentile(cat, 95))
    if vmax <= 0:
        vmax = float(np.nanmax(cat)) if cat.size else 1.0
    return 0.0, max(vmax, 1e-6)


def _make_cmap(cmap):
    import matplotlib.pyplot as plt
    cmap_obj = plt.get_cmap(cmap).copy()
    # mask 外 NaN 區域顯示為白色，避免矩形背景被染色
    cmap_obj.set_bad(color="white", alpha=0.0)
    return cmap_obj


def _add_china_boundary(ax, ccrs, china_boundary_path: Optional[str]):
    """在 cartopy 坐標軸上疊加中國邊界線。失敗時只 warning，不中斷繪圖。"""
    if ccrs is None or not china_boundary_path:
        return
    try:
        import cartopy.io.shapereader as shpreader
        import cartopy.feature as cfeature
        reader = shpreader.Reader(china_boundary_path)
        feature = cfeature.ShapelyFeature(
            reader.geometries(),
            ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            linewidth=0.8,
        )
        ax.add_feature(feature, zorder=5)
    except Exception as e:
        print(f"[RMSE MAP] WARNING: failed to add China boundary: {e}")


def _add_taiwan_boundary(ax, ccrs, taiwan_boundary_path: Optional[str]):
    """
    疊加台灣邊界。支援三種來源：
        1) 直接是 Taiwan .shp（gadm41_TWN_0.shp 之類，所有 geometry 都畫出來）
        2) Natural Earth admin_0_countries .shp（從 records 篩 ADM0_A3 == 'TWN'）
        3) None / 不存在 → no-op
    失敗時只 warning，不中斷繪圖。
    """
    if ccrs is None or not taiwan_boundary_path:
        return
    if not os.path.exists(taiwan_boundary_path):
        return
    try:
        import cartopy.io.shapereader as shpreader
        import cartopy.feature as cfeature

        name = os.path.basename(taiwan_boundary_path).lower()
        reader = shpreader.Reader(taiwan_boundary_path)

        if "twn" in name or "taiwan" in name:
            geoms = list(reader.geometries())
        else:
            geoms = []
            for rec in reader.records():
                attrs = rec.attributes or {}
                adm = str(attrs.get("ADM0_A3") or attrs.get("ADM0_ISO") or "").upper()
                nm = str(attrs.get("NAME") or attrs.get("NAME_LONG") or "").lower()
                if adm == "TWN" or "taiwan" in nm:
                    geoms.append(rec.geometry)
            if not geoms:
                return

        feature = cfeature.ShapelyFeature(
            geoms,
            ccrs.PlateCarree(),
            edgecolor="black",
            facecolor="none",
            linewidth=0.8,
        )
        ax.add_feature(feature, zorder=5)
    except Exception as e:
        print(f"[RMSE MAP] WARNING: failed to add Taiwan boundary: {e}")


def _draw_one(
    ax,
    lat2d,
    lon2d,
    rmse_map,
    vmin,
    vmax,
    title,
    cmap,
    ccrs,
    region_extent: Optional[Tuple[float, float, float, float]] = None,
    china_boundary_path: Optional[str] = None,
    taiwan_boundary_path: Optional[str] = None,
):
    if region_extent is None:
        region_extent = (70.0, 140.0, 15.0, 55.0)

    if ccrs is not None:
        pc = ax.pcolormesh(
            lon2d, lat2d, rmse_map,
            shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            transform=ccrs.PlateCarree(),
        )
        ax.set_extent(region_extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution="50m", linewidth=0.5, zorder=4)
        try:
            import cartopy.feature as cfeature
            ax.add_feature(cfeature.BORDERS, linewidth=0.4, zorder=4)
        except Exception:
            pass
        _add_china_boundary(ax, ccrs, china_boundary_path)
        _add_taiwan_boundary(ax, ccrs, taiwan_boundary_path)
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
        gl.top_labels = gl.right_labels = False
    else:
        pc = ax.pcolormesh(
            lon2d, lat2d, rmse_map,
            shading="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        lon_min, lon_max, lat_min, lat_max = region_extent
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=10)
    return pc


def plot_single_variable_rmse_maps(
    variable: str,
    rmse_maps_var: Mapping[str, np.ndarray],
    lat,
    lon,
    output_path: str,
    units: str = "",
    region_extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "YlOrRd",
    china_boundary_path: Optional[str] = None,
    taiwan_boundary_path: Optional[str] = None,
):
    """
    輸出 1×N（N=lead group 數）的單變數 RMSE 空間圖。
    region_extent: (lon_min, lon_max, lat_min, lat_max)，可選。
    china_boundary_path: 中國邊界 shp，用於疊加中國背景邊界。
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ccrs, _ = _try_import_cartopy()
    lat2d, lon2d = _ensure_meshgrid(lat, lon)
    cmap_obj = _make_cmap(cmap)

    if region_extent is None:
        region_extent = (70.0, 140.0, 15.0, 55.0)

    group_names = list(rmse_maps_var.keys())
    n = len(group_names)
    if n == 0:
        print(f"[RMSE MAP] {variable}: 無可繪 group。")
        return

    vmin, vmax = _common_vrange(rmse_maps_var)

    fig_w = max(4 * n, 6)
    if ccrs is not None:
        fig, axes = plt.subplots(
            1, n, figsize=(fig_w, 4.2),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    pc = None
    for ax, gname in zip(axes, group_names):
        rm = np.asarray(rmse_maps_var[gname], dtype=float)
        title = f"{variable}  RMSE | {gname} days"
        pc = _draw_one(
            ax, lat2d, lon2d, rm, vmin, vmax, title, cmap_obj, ccrs,
            region_extent=region_extent,
            china_boundary_path=china_boundary_path,
            taiwan_boundary_path=taiwan_boundary_path,
        )

    if pc is not None:
        cbar = fig.colorbar(pc, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label(f"RMSE  [{units}]" if units else "RMSE")

    fig.suptitle(f"Per-grid RMSE over China land — {variable}", fontsize=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {output_path}")


def plot_rmse_spatial_grid(
    rmse_maps: Mapping[str, Mapping[str, np.ndarray]],
    lat,
    lon,
    output_path: str,
    variable_units: Optional[Mapping[str, str]] = None,
    region_extent: Optional[Tuple[float, float, float, float]] = None,
    cmap: str = "YlOrRd",
    china_boundary_path: Optional[str] = None,
    taiwan_boundary_path: Optional[str] = None,
):
    """
    多變數 × 多 lead group 大圖：行=變數，列=lead group。
    每個變數一條獨立 colorbar（vmin=0，vmax=該變數所有 group 的 95 分位）。
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ccrs, _ = _try_import_cartopy()
    lat2d, lon2d = _ensure_meshgrid(lat, lon)
    variable_units = dict(variable_units or {})
    cmap_obj = _make_cmap(cmap)

    if region_extent is None:
        region_extent = (70.0, 140.0, 15.0, 55.0)

    variables = [v for v in rmse_maps if rmse_maps[v]]
    if not variables:
        print("[RMSE MAP] 無可繪資料，略過 grid plot。")
        return
    # 列名取所有 var 的聯集，按已知順序排序
    group_order = []
    for v in variables:
        for g in rmse_maps[v].keys():
            if g not in group_order:
                group_order.append(g)

    n_rows = len(variables)
    n_cols = len(group_order)
    fig_w = max(4 * n_cols, 8)
    fig_h = max(3.6 * n_rows, 4)

    if ccrs is not None:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True, squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(fig_w, fig_h),
            constrained_layout=True, squeeze=False,
        )

    for ri, var in enumerate(variables):
        var_maps = rmse_maps[var]
        vmin, vmax = _common_vrange(var_maps)
        units = variable_units.get(var, "")
        last_pc = None
        for ci, gname in enumerate(group_order):
            ax = axes[ri][ci]
            if gname not in var_maps:
                ax.axis("off")
                continue
            rm = np.asarray(var_maps[gname], dtype=float)
            title = f"{var}  | {gname} days"
            pc = _draw_one(
                ax, lat2d, lon2d, rm, vmin, vmax, title, cmap_obj, ccrs,
                region_extent=region_extent,
                china_boundary_path=china_boundary_path,
                taiwan_boundary_path=taiwan_boundary_path,
            )
            last_pc = pc
        if last_pc is not None:
            cbar = fig.colorbar(last_pc, ax=axes[ri].tolist(), shrink=0.85, pad=0.02)
            cbar.set_label(f"RMSE  [{units}]" if units else "RMSE")

    fig.suptitle("Per-grid RMSE over China land  (rows = variables, cols = lead groups)", fontsize=12)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {output_path}")


# =============================================================================
# 4) 儲存 npy / NetCDF
# =============================================================================

def save_rmse_maps(
    rmse_maps: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str,
    lat,
    lon,
    save_nc: bool = True,
):
    """
    儲存：
      <output_dir>/rmse_map_<variable>_<group>.npy
      <output_dir>/rmse_spatial_maps.nc  (若 xarray 可用)
    """
    os.makedirs(output_dir, exist_ok=True)
    lat = _to_numpy(lat)
    lon = _to_numpy(lon)

    for var, var_maps in rmse_maps.items():
        for gname, rm in var_maps.items():
            out_path = os.path.join(output_dir, f"rmse_map_{var}_{gname.replace('-', '_')}.npy")
            np.save(out_path, rm)
            print(f"[SAVE] {out_path}")

    if not save_nc:
        return
    try:
        import xarray as xr
    except Exception:
        print("[RMSE MAP] xarray 不可用，略過 NetCDF 輸出。")
        return

    variables = list(rmse_maps.keys())
    if not variables:
        return
    group_order = []
    for v in variables:
        for g in rmse_maps[v].keys():
            if g not in group_order:
                group_order.append(g)

    H = lat.size if lat.ndim == 1 else lat.shape[0]
    W = lon.size if lon.ndim == 1 else lon.shape[-1]
    arr = np.full((len(variables), len(group_order), H, W), np.nan, dtype=np.float32)
    for vi, v in enumerate(variables):
        for gi, g in enumerate(group_order):
            if g in rmse_maps[v]:
                arr[vi, gi] = rmse_maps[v][g]

    ds = xr.Dataset(
        {"rmse": (("variable", "lead_group", "lat", "lon"), arr)},
        coords={
            "variable": variables,
            "lead_group": group_order,
            "lat": lat if lat.ndim == 1 else np.arange(H),
            "lon": lon if lon.ndim == 1 else np.arange(W),
        },
        attrs={"description": "Per-grid RMSE maps by variable and lead-time group"},
    )
    nc_path = os.path.join(output_dir, "rmse_spatial_maps.nc")
    ds.to_netcdf(nc_path)
    print(f"[SAVE] {nc_path}")


# =============================================================================
# 5) lead group 解析 helper
# =============================================================================

def parse_lead_groups_arg(s: str) -> Dict[str, Tuple[int, int]]:
    """
    解析 CLI 字串，例如：
        "01-10,11-20,21-30,31-40"
    回傳 {"01-10": (1, 10), ...}
    """
    s = (s or "").strip()
    if not s:
        return {}
    out = {}
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            lo, hi = token.split("-")
            lo_i = int(lo)
            hi_i = int(hi)
        except Exception:
            raise ValueError(f"無法解析 lead_groups 區段：'{token}'，預期格式如 '01-10'。")
        out[f"{lo_i:02d}-{hi_i:02d}"] = (lo_i, hi_i)
    return out


DEFAULT_LEAD_GROUPS: Dict[str, Tuple[int, int]] = {
    "01-10": (1, 10),
    "11-20": (11, 20),
    "21-30": (21, 30),
    "31-40": (31, 40),
}


# 各變數預設單位
DEFAULT_UNITS: Dict[str, str] = {
    "u10": "m/s",
    "v10": "m/s",
    "u100": "m/s",
    "v100": "m/s",
    "wind10m": "m/s",
    "wind100m": "m/s",
    "wind_speed_10m": "m/s",
    "wind_speed_100m": "m/s",
    "ssrd": "J/m²",
}
