"""Phase 2 - participant pairing: reanalysis light (input) → target grid, target truth (target).

Bilinear-interpolates reanalysis light onto the target light grid and attaches the
target ground truth **only when it is available to participants** (train years).
On the eval year (2022) the target field is withheld → ``pair(...).target is None``.
This is the same masking rule as Phase 1.

The grid (lat/lon/seamask) always comes from the static target file (just
coordinates, always shipped); the per-day target truth is loaded only for
train-year dates, so this works on a participant machine that has no eval-year
target files.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date_t

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import target_loader
import reanalysis_loader
import splits

__all__ = ["DownscalingPair", "interp_reanalysis_to_target", "pair", "iter_pairs"]


@dataclass
class DownscalingPair:
    time: pd.Timestamp
    lon: np.ndarray              # target grid (y, x)
    lat: np.ndarray
    seamask: np.ndarray          # (y, x), 1=sea 0=land
    reanalysis: dict[str, np.ndarray]  # interpolated on target grid: u10,v10,u100,v100,ws10,ws100
    target: dict[str, np.ndarray] | None  # target truth, or None when withheld (eval year)


def interp_reanalysis_to_target(e5: reanalysis_loader.ReanalysisSnapshot,
                         target_lon: np.ndarray, target_lat: np.ndarray
                         ) -> dict[str, np.ndarray]:
    """Bilinear reanalysis → target grid. Points outside the reanalysis domain are ``np.nan``;
    this affects the southern target rows where the grid extends below reanalysis
    coverage (≈22% of the grid)."""
    pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=1)
    out: dict[str, np.ndarray] = {}
    for comp in ("u10", "v10", "u100", "v100"):
        f = RegularGridInterpolator((e5.lats, e5.lons), getattr(e5, comp),
                                    bounds_error=False, fill_value=np.nan)
        out[comp] = f(pts).reshape(target_lat.shape).astype(np.float32)
    out["ws10"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)
    out["ws100"] = np.sqrt(out["u100"] ** 2 + out["v100"] ** 2)
    return out


def pair(date: _date_t | str, hour: int,
         levels: tuple[str, ...] = ("125m",)) -> DownscalingPair:
    """Build a (reanalysis on target grid, target truth) pair. hour ∈ {0,6,12,18}.

    The target grid/seamask come from the static file. target truth is attached
    only when ``splits.target_available(date)`` (train years); on the eval year
    it stays ``None`` and no per-day target file is read.
    """
    static = target_loader.load_static()
    e5 = reanalysis_loader.load_reanalysis(date, hour)
    reanalysis_on_grid = interp_reanalysis_to_target(e5, static.lon, static.lat)

    target_truth: dict[str, np.ndarray] | None = None
    if splits.target_available(date):
        ar = target_loader.load_snapshot(date, hour)
        target_truth = {}
        for lvl in levels:
            fld = ar.fields.get(lvl)
            if fld:
                for k, v in fld.items():
                    target_truth[f"{lvl}_{k}"] = v
        target_truth = target_truth or None   # no requested level found → None, not {}

    return DownscalingPair(
        time=pd.Timestamp(e5.time), lon=static.lon, lat=static.lat,
        seamask=static.seamask, reanalysis=reanalysis_on_grid, target=target_truth,
    )


def iter_pairs(dates, hours=(0, 6, 12, 18), levels=("125m",)):
    """Yield DownscalingPair for every (date, hour) that has data on disk.

    Skips dates/hours whose reanalysis or target file is missing (no crash on partial
    coverage). target truth is present for train-year dates and None for 2022.
    """
    for d in dates:
        for hour in hours:
            try:
                yield pair(d, hour, levels=levels)
            except FileNotFoundError:
                continue
