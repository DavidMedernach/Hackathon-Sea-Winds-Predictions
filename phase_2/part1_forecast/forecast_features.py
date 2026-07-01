"""Assemble the Part 1 forecast training table.

Rows = (reanalysis grid cell × issue time). Features = reanalysis light fields at the issue
time + temporal encodings + cell lat/lon. Targets = coarsened target 125m u/v at
lead times (same hour). Rows whose target is missing/NaN are dropped.
"""
from __future__ import annotations

from datetime import date as _date_t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# NB: reanalysis_loader is imported lazily inside build_training_table so that simply
# importing FEATURE_COLS (e.g. from forecast_model) does not require the
# dataset_setup package to be on sys.path.

import sys as _sys
_HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent))           # kit root (for config)
import config                                     # noqa: E402

FEATURE_COLS = ["lat", "lon", "hour_sin", "hour_cos", "woy_sin", "woy_cos",
                "u10", "v10", "u100", "v100", "ws100"]


def _coarse_target(d: _date_t, hour: int):
    """Return (u125c, v125c) 2-D arrays at (date, hour), or (None, None)."""
    p = config.coarse_root() / f"{d.year}" / f"coarse_{d:%Y%m%d}.nc"
    if not p.exists():
        return None, None
    ds = xr.open_dataset(p)
    try:
        sel = ds.sel(time=pd.Timestamp(d) + pd.Timedelta(hours=hour))
        return (sel["u125c"].values.astype(np.float32),
                sel["v125c"].values.astype(np.float32))
    finally:
        ds.close()


def build_training_table(issue_dates, hours=(0, 6, 12, 18),
                         leads_days=(1, 7, 14)) -> pd.DataFrame:
    """Stack per-cell rows for the given issue dates/hours and lead times."""
    import reanalysis_loader  # lazy: only needed when actually building the table
    rows = []
    for d in issue_dates:
        for hour in hours:
            try:
                e5 = reanalysis_loader.load_reanalysis(d, hour)
            except FileNotFoundError:
                continue
            lon2d, lat2d = np.meshgrid(e5.lons, e5.lats)
            ts = pd.Timestamp(d) + pd.Timedelta(hours=hour)
            woy = ts.isocalendar().week
            base = {
                "time": ts,
                "lat": lat2d.ravel().astype(np.float32),
                "lon": lon2d.ravel().astype(np.float32),
                "hour_sin": np.float32(np.sin(2 * np.pi * hour / 24)),
                "hour_cos": np.float32(np.cos(2 * np.pi * hour / 24)),
                "woy_sin": np.float32(np.sin(2 * np.pi * woy / 52)),
                "woy_cos": np.float32(np.cos(2 * np.pi * woy / 52)),
                "u10": e5.u10.ravel(), "v10": e5.v10.ravel(),
                "u100": e5.u100.ravel(), "v100": e5.v100.ravel(),
                "ws100": e5.ws100.ravel().astype(np.float32),
            }
            block = pd.DataFrame(base)
            assert set(FEATURE_COLS).issubset(block.columns)  # guard FEATURE_COLS drift
            keep = np.ones(len(block), dtype=bool)
            for lead in leads_days:
                td = (pd.Timestamp(d) + pd.Timedelta(days=lead)).date()
                u_t, v_t = _coarse_target(td, hour)
                if u_t is None:
                    keep[:] = False
                    block[f"u125c_d{lead}"] = np.nan
                    block[f"v125c_d{lead}"] = np.nan
                else:
                    block[f"u125c_d{lead}"] = u_t.ravel()
                    block[f"v125c_d{lead}"] = v_t.ravel()
                    # require BOTH components finite (defends against partial writes)
                    keep &= np.isfinite(u_t.ravel()) & np.isfinite(v_t.ravel())
            if keep.any():
                rows.append(block[keep])
    if not rows:
        target_cols = [c for lead in leads_days
                       for c in (f"u125c_d{lead}", f"v125c_d{lead}")]
        return pd.DataFrame(columns=["time", *FEATURE_COLS, *target_cols])
    return pd.concat(rows, ignore_index=True)
