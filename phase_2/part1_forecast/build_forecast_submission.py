"""Assemble downscaled fine (1.3 km) forecast fields into the Phase-2 submission.

Phase-2 forecast mirrors Phase 1 with FEWER dimensions: one region (north_sea),
one level (125m target), grid only. Submission schema = Phase-1 grid rows:

    type, window, region, latitude, longitude, horizon, hour, level,
    q05, q50, q95, dir_05, dir_50, dir_95

Scored by the (scoped) Phase-1 scorer: speed Winkler on (q05,q95) + circular
Winkler on (dir_05,dir_95), over 6 dimensions (3 horizons x {speed, dir}).
Points = the 43 715-pt footprint (dataset_setup/footprint.py). See design spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "part0_dataset_setup"))
import footprint as _fp   # noqa: E402

REGION = "north_sea"
LEVEL = "125m"
ROW_TYPE = "grid"
COLS = ["type", "window", "region", "latitude", "longitude", "horizon", "hour",
        "level", "q05", "q50", "q95", "dir_05", "dir_50", "dir_95"]

_GRID = None


def _grid():
    """(mask, lat2d, lon2d) for the footprint - loaded once."""
    global _GRID
    if _GRID is None:
        st = xr.open_dataset(_fp.TARGET_STATIC)
        _GRID = (_fp.footprint_mask(), st["latitude"].values, st["longitude"].values)
    return _GRID


def field_to_rows(window: int, horizon: int, hour: int,
                  q05, q50, q95, dir05, dir50, dir95) -> pd.DataFrame:
    """One (window, horizon, hour) block -> 43 715 Phase-1-format grid rows.

    All field args are (479, 433) arrays on the target grid; only footprint
    cells are kept, in the canonical np.where(mask) order.
    """
    m, lat, lon = _grid()
    ys, xs = np.where(m)
    out = pd.DataFrame({
        "type": ROW_TYPE, "window": np.int32(window), "region": REGION,
        "latitude": lat[ys, xs].round(2).astype(np.float32),
        "longitude": lon[ys, xs].round(2).astype(np.float32),
        "horizon": np.int32(horizon), "hour": np.int32(hour), "level": LEVEL,
        "q05": q05[ys, xs].astype(np.float32),
        "q50": q50[ys, xs].astype(np.float32),
        "q95": q95[ys, xs].astype(np.float32),
        "dir_05": (dir05[ys, xs] % 360).astype(np.float32),
        "dir_50": (dir50[ys, xs] % 360).astype(np.float32),
        "dir_95": (dir95[ys, xs] % 360).astype(np.float32),
    })
    # speed: enforce monotone quantiles + non-negativity (defensive)
    q = np.sort(out[["q05", "q50", "q95"]].values, axis=1)
    out["q05"], out["q50"], out["q95"] = np.clip(q[:, 0], 0, None), q[:, 1], q[:, 2]
    return out[COLS]


def assemble(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(blocks, ignore_index=True)[COLS]


def write_submission(df: pd.DataFrame, path) -> None:
    assert list(df.columns) == COLS, df.columns.tolist()
    assert (df["q05"] <= df["q95"]).all(), "q05 > q95 present"
    if str(path).endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        out = df.copy()                       # CSV (Codabench scores CSV): round to keep it light
        out["latitude"] = out["latitude"].round(2)
        out["longitude"] = out["longitude"].round(2)
        for c in ("q05", "q50", "q95", "dir_05", "dir_50", "dir_95"):
            out[c] = out[c].round(3)
        out.to_csv(path, index=False)
