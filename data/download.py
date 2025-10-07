#!/usr/bin/env python3
"""
ERA5 single-level (e.g., 2m_temperature) bulk downloader using CDS API.

Security notes:
- Do NOT hardcode API keys in the source code.
- Provide keys via environment variable `CDSAPI_KEYS` as:
    alias1:key1,alias2:key2,...
  or configure the standard ~/.cdsapirc and omit `CDSAPI_KEYS`.

Example:
    export CDSAPI_KEYS="acc1:00000000-0000-0000-0000-000000000000,acc2:11111111-1111-1111-1111-111111111111"
"""

import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional

import cdsapi
import argparse

API_URL = "https://cds.climate.copernicus.eu/api"

# ---------- Helpers ----------

def _parse_keys_from_env(env_var: str = "CDSAPI_KEYS") -> List[Tuple[str, str]]:
    """
    Parse `alias:key` pairs from env var, comma-separated.
    Returns empty list if not set (the cdsapi client will fall back to ~/.cdsapirc).
    """
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return []
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid CDSAPI_KEYS entry (missing colon): {chunk}")
        alias, key = chunk.split(":", 1)
        alias = alias.strip()
        key = key.strip()
        if not alias or not key:
            raise ValueError(f"Invalid CDSAPI_KEYS entry (empty alias/key): {chunk}")
        pairs.append((alias, key))
    return pairs


def submit_era5_single_level(
    name: Optional[str],
    api_key: Optional[str],
    year: int,
    dataset: str,
    variables: List[str],
    time_list: List[str],
    area: List[float],
    grid: List[float],
    output_dir: str,
    api_url: str = API_URL,
    skip_future_years: bool = False,
) -> None:
    """
    Create a CDS client (optionally with explicit url/key) and submit a yearly request.
    Output file: {year}.nc. If the file already exists, it will be skipped.
    """
    # Optionally skip future years
    if skip_future_years and year > datetime.now().year:
        print(f"[SKIP] {year} is in the future, skipping")
        return

    # Initialize client. If api_key is None, cdsapi will use ~/.cdsapirc
    if api_key:
        client = cdsapi.Client(url=api_url, key=api_key)
    else:
        client = cdsapi.Client()

    # Output target
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{year}.nc"
    target = os.path.join(output_dir, filename)
    if os.path.exists(target):
        print(f"[SKIP] {filename} already exists")
        return

    # Build request for a whole year at 1°×1° global grid
    req = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": variables,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": time_list,
        "area": area,   # [N, W, S, E]
        "grid": grid,   # [lat_res, lon_res]
    }

    key_alias = name or "~/.cdsapirc"
    print(f"[SUBMIT] {year} → {filename}  (using key '{key_alias}')")
    client.retrieve(dataset, req, target)
    print(f"[DONE]   {year} request submitted")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="ERA5 Single Levels bulk downloader (safe, no hardcoded keys).")
    parser.add_argument("--dataset", default="reanalysis-era5-single-levels",
                        help="CDS dataset name (default: reanalysis-era5-single-levels)")
    parser.add_argument("--variable", default="2m_temperature",
                        help="Variable name (e.g., 2m_temperature, 2m_dewpoint_temperature)")
    parser.add_argument("--start-year", type=int, default=1951, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="End year (inclusive)")
    parser.add_argument("--output-dir", default="ERA5global/t2m", help="Directory to store .nc files")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--grid-lat", type=float, default=1.0, help="Latitude resolution in degrees")
    parser.add_argument("--grid-lon", type=float, default=1.0, help="Longitude resolution in degrees")
    parser.add_argument("--skip-future", action="store_true", help="Skip future years")
    parser.add_argument("--api-url", default=API_URL, help="CDS API URL (override if needed)")
    args = parser.parse_args()

    years = list(range(args.start_year, args.end_year + 1))
    variables = [args.variable]
    time_list = [f"{h:02d}:00" for h in range(24)]
    area = [90, -180, -90, 180]  # Global
    grid = [args.grid_lat, args.grid_lon]

    # Load keys from env; if empty, client will use ~/.cdsapirc
    keys = _parse_keys_from_env("CDSAPI_KEYS")

    # Dispatch jobs with round-robin keys (or a single default client)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, year in enumerate(years):
            if keys:
                name, key = keys[idx % len(keys)]
            else:
                name, key = None, None
            executor.submit(
                submit_era5_single_level,
                name, key, year,
                dataset=args.dataset,
                variables=variables,
                time_list=time_list,
                area=area,
                grid=grid,
                output_dir=args.output_dir,
                api_url=args.api_url,
                skip_future_years=args.skip_future,
            )


if __name__ == "__main__":
    main()
