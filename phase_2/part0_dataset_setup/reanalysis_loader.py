"""Phase 2 - reanalysis light loader (participant-facing).

Reads the light reanalysis dataset shipped with Phase 2 (built by
``internal/data_generation/build_phase2_reanalysis_light.py``). reanalysis is the low-res
INPUT for forecasting (Part 1) and downscaling (Part 2).

Public API
----------
- ``list_dates()``              : sorted dates available on disk
- ``load_reanalysis(date, hour)``     : one snapshot (hour ∈ {0, 6, 12, 18})
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date_t
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

__all__ = ["ReanalysisSnapshot", "list_dates", "load_reanalysis", "REANALYSIS_ROOT"]

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]
# Resolved by ``config`` (honours the unzip recorded by 0_dataset_setup /
# ``$PHASE2_DATA_ROOT``; falls back to the repo dev layout).
import sys as _sys
_sys.path.insert(0, str(_HERE.parent))           # kit root (for config)
try:
    import config as _config
    REANALYSIS_ROOT = _config.reanalysis_root()
except Exception:
    REANALYSIS_ROOT = PROJECT_ROOT / "build" / "phase2_dataset" / "reanalysis"
_HOURS = (0, 6, 12, 18)


@dataclass
class ReanalysisSnapshot:
    time: pd.Timestamp
    lats: np.ndarray   # 1-D ascending
    lons: np.ndarray   # 1-D ascending
    u10: np.ndarray    # (n_lat, n_lon)
    v10: np.ndarray
    u100: np.ndarray
    v100: np.ndarray

    @property
    def ws10(self) -> np.ndarray:  return np.sqrt(self.u10 ** 2 + self.v10 ** 2)
    @property
    def ws100(self) -> np.ndarray: return np.sqrt(self.u100 ** 2 + self.v100 ** 2)


def _path(d: _date_t, root: Path | str = REANALYSIS_ROOT) -> Path:
    return Path(root) / f"{d.year}" / f"reanalysis_{d:%Y%m%d}.nc"


def list_dates(root: Path | str = REANALYSIS_ROOT) -> list[_date_t]:
    root = Path(root)
    out: list[_date_t] = []
    for f in root.glob("*/reanalysis_*.nc"):
        stem = f.stem.replace("reanalysis_", "")
        if len(stem) == 8 and stem.isdigit():
            out.append(_date_t(int(stem[:4]), int(stem[4:6]), int(stem[6:])))
    return sorted(set(out))


def load_reanalysis(date: _date_t | str, hour: int,
              root: Path | str = REANALYSIS_ROOT) -> ReanalysisSnapshot:
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    if hour not in _HOURS:
        raise ValueError(f"reanalysis light is 6-hourly (00/06/12/18), got hour={hour}")
    path = _path(date, root)
    if not path.exists():
        raise FileNotFoundError(f"No reanalysis light file for {date} at {path}")
    ds = xr.open_dataset(path)
    try:
        # Select by coordinate (not positional) so any time-axis ordering works.
        wanted = pd.Timestamp(date) + pd.Timedelta(hours=hour)
        snap = ds.sel(time=wanted)

        def _get(v: str) -> np.ndarray:
            return snap[v].values.astype(np.float32)

        return ReanalysisSnapshot(
            time=pd.Timestamp(snap.time.values),
            lats=ds["latitude"].values.astype(np.float32),
            lons=ds["longitude"].values.astype(np.float32),
            u10=_get("u10"), v10=_get("v10"), u100=_get("u100"), v100=_get("v100"),
        )
    finally:
        ds.close()
