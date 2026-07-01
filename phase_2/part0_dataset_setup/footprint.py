"""Canonical Phase-2 forecast scoring footprint: the fine (1.3 km) target points
that are sea AND inside the reanalysis bbox AND on the North-Sea side (lon>=-2).

This is the SHARED contract between the participant submission (point_id ->
q05/q50/q95/dir_50) and the organiser scorer's ground truth. 43 715 points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))        # kit root (for config)
import config                                     # noqa: E402

# Backward-compat module constants (resolved via config at import - set
# $PHASE2_DATA_ROOT before importing if your data lives elsewhere).
TARGET_STATIC = config.target_static()
FOOTPRINT_PATH = config.footprint_path()

# reanalysis bbox (matches reanalysis_static.nc / zone.py)
LAT_MIN, LAT_MAX = 51.0, 62.0
LON_MIN, LON_MAX = -4.0, 10.0
LON_EAST_OF = -2.0   # North-Sea side only (exclude Celtic/Irish Sea)


def footprint_mask() -> np.ndarray:
    st = xr.open_dataset(config.target_static())
    lat = st["latitude"].values
    lon = st["longitude"].values
    sea = st["seamask"].values > 0.5
    return (
        sea
        & (lat >= LAT_MIN) & (lat <= LAT_MAX)
        & (lon >= LON_MIN) & (lon <= LON_MAX)
        & (lon >= LON_EAST_OF)
    )


def build_footprint() -> pd.DataFrame:
    st = xr.open_dataset(config.target_static())
    lat = st["latitude"].values
    lon = st["longitude"].values
    m = footprint_mask()
    ys, xs = np.where(m)                       # row-major (y then x) => deterministic
    df = pd.DataFrame({
        "point_id": np.arange(ys.size, dtype=np.int32),
        "lat": lat[ys, xs].astype(np.float32),
        "lon": lon[ys, xs].astype(np.float32),
    })
    return df


def load_footprint() -> pd.DataFrame:
    """The shipped footprint_points.parquet if present (read-only safe); else
    derive it in memory from arome_static.nc - no write to the data dir."""
    fp = config.footprint_path()
    if fp.exists():
        return pd.read_parquet(fp)
    return build_footprint()
