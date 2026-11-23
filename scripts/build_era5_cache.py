"""
Pre-build ERA5 daily-mean cache for the yearly files.

Reads each <era5_root>/<var>_<year>.nc hourly file in time-chunk-aligned
blocks (~28 days at a time) and writes a daily-mean cache:

    <era5_root>/_daily_cache/<var>_<year>_daily.npy    (N_days, H, W) float32
    <era5_root>/_daily_cache/<var>_<year>_dates.npy    datetime64[D]
    <era5_root>/_daily_cache/<var>_<year>_latlon.npz   lat, lon 1D float32

Usage:
    python scripts/build_era5_cache.py \
        --era5-root D:/downsampling_focas/ERA5/hourly \
        --vars u10 v10 \
        --years 2025 2026
"""
import argparse
import os
import sys
import time

# allow `python scripts/build_era5_cache.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.base_dataset import ensure_era5_daily_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--era5-root",
        default="D:/downsampling_focas/ERA5/hourly",
        help="Directory containing <var>_<year>.nc yearly files.",
    )
    parser.add_argument(
        "--vars",
        nargs="+",
        default=["u10", "v10"],
        help="Variables to build cache for.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2025, 2026],
        help="Years to build cache for.",
    )
    parser.add_argument(
        "--cache-root",
        default=None,
        help="Optional override for the cache directory (defaults to "
             "<era5-root>/_daily_cache).",
    )
    args = parser.parse_args()

    t_all = time.time()
    for v in args.vars:
        for y in args.years:
            path = os.path.join(args.era5_root, f"{v}_{y}.nc")
            if not os.path.exists(path):
                print(f"[skip] missing {path}")
                continue
            print(f"\n### {v}_{y} ###", flush=True)
            t0 = time.time()
            ensure_era5_daily_cache(
                args.era5_root, v, y, cache_root=args.cache_root, verbose=True
            )
            print(f"### done {v}_{y} in {time.time() - t0:.1f}s", flush=True)
    print(f"\nAll done in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
