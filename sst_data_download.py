#!/usr/bin/env python3
"""
Download NOAA OISST v2.1 daily (0.25°) per-year NetCDF and convert to NPZ.

- Years processed from current year down to 1981.
- If oisst_sst_daily_YYYY.npz already exists, that year is skipped.
- Each year: download NetCDF to a temp file, convert to NPZ, then atomically rename.

Result per NPZ:
  - sst: float32, shape (time, lat, lon) with NaNs over land/ice where applicable
  - time: datetime64[ns], length = number of days available in that year
  - lat: float32, shape (nlat,)
  - lon: float32, shape (nlon,)

Dependencies: xarray, numpy, netCDF4
    pip install xarray netCDF4 numpy
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
import sys
import time as _time
import urllib.request
import requests
import numpy as np
import xarray as xr

ROOT = Path(__file__).parent

# NOAA PSL THREDDS static per-year fileserver path for OISST v2.1 daily
BASE_URL = "https://psl.noaa.gov/thredds/fileServer/Datasets/noaa.oisst.v2.highres"
FILENAME_PATTERN = "sst.day.mean.{year}.nc"

def http_download(url: str, dst_tmp: Path, dst: Path):
  """Download url to dst_tmp with simple retries."""
  if dst.exists():
    return

  print(f"Downloading {url} -> {dst_tmp}")

  try:
    req = requests.get(url, stream=True)
    size = 0
    last_size = 0
    chunk_size = 2**16

    if not req.ok:
      req.raise_for_status()

    with req, dst_tmp.open('wb') as fp:
      for chunk in req.iter_content(chunk_size=chunk_size):
        if chunk:
          fp.write(chunk)
          size += len(chunk)

          if size - last_size > 10e6:
            print(f"- {size/1e6:,.1f} MB")
            last_size = size

    if size == 0:
      raise RuntimeError(f"Downloaded file had zero size: {url}")

    dst_tmp.replace(dst)

  finally:
    dst_tmp.unlink(missing_ok=True)


def convert_nc_to_npz(nc_path: Path, out_tmp_npz: Path):
    """Open NetCDF, extract arrays, and write to NPZ (atomic temp file)."""
    # Let xarray decode CF times/scale/offset and fill values -> NaNs
    ds = xr.open_dataset(nc_path, engine="netcdf4", decode_times=True, mask_and_scale=True)
    # Ensure expected names
    for nm in ("latitude", "lat"):
        if nm in ds and "lat" not in ds:
            ds = ds.rename({nm: "lat"})
    for nm in ("longitude", "lon"):
        if nm in ds and "lon" not in ds:
            ds = ds.rename({nm: "lon"})

    # Select the variable (sst)
    if "sst" not in ds:
        raise RuntimeError("Variable 'sst' not found in dataset.")
    # Load to memory so the NetCDF can be closed before writing NPZ
    sst = ds["sst"].astype("float32").values  # (time, lat, lon)
    lat = ds["lat"].astype("float32").values
    lon = ds["lon"].astype("float32").values
    time_vals = ds["time"].values  # already decoded to datetime64

    ds.close()

    # Save as compressed NPZ
    np.savez_compressed(out_tmp_npz, sst=sst, time=time_vals, lat=lat, lon=lon)

def main():

  outdir = ROOT/'oisst_sst_daily'
  outdir.mkdir(parents=True, exist_ok=True)

  current_year = dt.date.today().year
  end_year = 1981
  # end_year = 2015

  print(f"Processing years {current_year} → {end_year} (descending)")

  for year in range(current_year, end_year - 1, -1):
    out_npz = outdir / f"{year}.npz"
    tmp_npz = out_npz.parent/('tmp.'+out_npz.name)

    out_nc  = outdir/FILENAME_PATTERN.format(year=year)
    tmp_nc = out_nc.parent/(out_nc.name+'.part')
    url = f"{BASE_URL}/{out_nc.name}"

    if out_npz.exists():
      print(f"[{year}] Exists → skip")
      continue

    http_download(url, tmp_nc, out_nc)

    # Convert NC → NPZ
    try:
      print(f"[{year}] Converting NetCDF → NPZ")
      convert_nc_to_npz(out_nc, tmp_npz)
      # Atomic rename to final target
      tmp_npz.replace(out_npz)
      print(f"[{year}] Done → {out_npz}")
    finally:
      tmp_npz.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
