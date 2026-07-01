"""Phase 2 - target-only data loader (participant-facing).

Loads the light target dataset shipped with the Phase 2 hackathon.
No external dependencies on reanalysis or any reanalysis: target is self-contained.

Public API
----------
- ``list_dates()``                : sorted list of dates available on disk
- ``load_static()``                : grid coordinates + sea mask (one-time)
- ``load_day(date)``               : 8 timesteps × full grid for one day
- ``load_snapshot(date, hour)``    : single timestep - convenient for plots
- ``train_val_test_dates()``       : recommended holdout within the train years (2016-2018 train,
                                     2022 test)

Convention
----------
- target hours are multiples of 3 (00, 03, 06, 09, 12, 15, 18, 21 UTC).
- Levels (m above sea level) : 125, 1500, 5000.
- Wind speed = sqrt(u² + v²), direction (meteorological "from") =
  ``(270 - atan2(v, u) deg) % 360``.
- Sea mask : 1 = sea, 0 = land.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date_t
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ───────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[2]

# Data location is resolved by ``config`` (honours the unzip recorded by
# 0_dataset_setup / ``$PHASE2_DATA_ROOT``; falls back to the repo dev layout).
# Participants can still pass ``root=...`` to ``list_dates`` / ``load_*``.
import sys as _sys
_sys.path.insert(0, str(_HERE.parent))           # kit root (for config)
try:
    import config as _config
    TARGET_ROOT = _config.target_root()
    _TARGET_STATIC = _config.target_static()
except Exception:
    TARGET_ROOT = PROJECT_ROOT / "build" / "phase2_dataset" / "arome"
    _TARGET_STATIC = TARGET_ROOT / "arome_static.nc"

LEVELS_M: tuple[int, ...] = (125, 1500, 5000)
HOURS: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21)


# ── Listing & paths ─────────────────────────────────────────────────────

def _day_path(date: _date_t, root: Path = TARGET_ROOT) -> Path:
    return Path(root) / f"{date.year}" / f"arome_{date:%Y%m%d}.nc"


def list_dates(root: Path | str = TARGET_ROOT) -> list[_date_t]:
    """Return sorted list of dates that have a per-day file."""
    import os
    out: list[_date_t] = []
    for dirpath, _dirnames, filenames in os.walk(Path(root), followlinks=True):
        for fname in filenames:
            if not fname.startswith("arome_") or not fname.endswith(".nc"):
                continue
            stem = fname[len("arome_"):-len(".nc")]
            if len(stem) == 8 and stem.isdigit():
                out.append(_date_t(int(stem[:4]), int(stem[4:6]), int(stem[6:])))
    return sorted(set(out))


# ── Static fields (lat, lon, seamask) ───────────────────────────────────

@dataclass(frozen=True)
class StaticGrid:
    """Constant fields of the target grid (the same for every snapshot)."""
    lat:     np.ndarray   # (y, x), float32
    lon:     np.ndarray   # (y, x), float32
    seamask: np.ndarray   # (y, x), float32 - 1 = sea, 0 = land

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape

    @property
    def sea(self) -> np.ndarray:
        return self.seamask > 0.5


@lru_cache(maxsize=4)
def load_static(root_str: str = str(TARGET_ROOT)) -> StaticGrid:
    """Load the target grid coordinates + sea mask (cached)."""
    p = Path(root_str) / "arome_static.nc"
    if not p.exists():
        p = _TARGET_STATIC          # config-resolved (ship layout: static/arome_static.nc)
    if not p.exists():
        raise FileNotFoundError(
            f"Static file not found: {p}\n"
            "Run 0_dataset_setup to unzip the dataset, "
            "or pass root=... pointing at your unpacked target folder."
        )
    ds = xr.open_dataset(p)
    try:
        return StaticGrid(
            lat=ds["latitude"].values.astype(np.float32),
            lon=ds["longitude"].values.astype(np.float32),
            seamask=ds["seamask"].values.astype(np.float32),
        )
    finally:
        ds.close()


# ── Per-day & per-snapshot loaders ──────────────────────────────────────

@dataclass
class TargetSnapshot:
    """A single target snapshot at one date+hour."""
    time:    pd.Timestamp
    lat:     np.ndarray         # (y, x)
    lon:     np.ndarray         # (y, x)
    seamask: np.ndarray         # (y, x)
    fields:  dict[str, dict[str, np.ndarray]]  # fields[level_str] = {u, v, ws, wd}

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape

    def ws(self, level: str = "125m") -> np.ndarray:
        return self.fields[level]["ws"]

    def wd(self, level: str = "125m") -> np.ndarray:
        return self.fields[level]["wd"]


@dataclass
class TargetDay:
    """All 8 timesteps of one day, full grid."""
    date:    _date_t
    times:   np.ndarray         # (T,) datetime64[ns]
    lat:     np.ndarray
    lon:     np.ndarray
    seamask: np.ndarray
    u:       dict[str, np.ndarray]   # u[level_str] of shape (T, y, x)
    v:       dict[str, np.ndarray]

    def ws(self, level: str = "125m") -> np.ndarray:
        return np.sqrt(self.u[level] ** 2 + self.v[level] ** 2)

    def wd(self, level: str = "125m") -> np.ndarray:
        u, v = self.u[level], self.v[level]
        return ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(np.float32)

    def snapshot(self, hour: int) -> TargetSnapshot:
        if hour not in HOURS:
            raise ValueError(f"target hour must be in {HOURS} (got {hour})")
        t_idx = hour // 3
        ts = pd.Timestamp(self.times[t_idx])
        fields: dict[str, dict[str, np.ndarray]] = {}
        for level in self.u:
            u, v = self.u[level][t_idx], self.v[level][t_idx]
            ws = np.sqrt(u ** 2 + v ** 2).astype(np.float32)
            wd = ((270.0 - np.degrees(np.arctan2(v, u))) % 360.0).astype(np.float32)
            fields[level] = {"u": u, "v": v, "ws": ws, "wd": wd}
        return TargetSnapshot(
            time=ts, lat=self.lat, lon=self.lon, seamask=self.seamask,
            fields=fields,
        )


def load_day(date: _date_t | str,
             root: Path | str = TARGET_ROOT,
             levels: tuple[str, ...] = ("125m", "1500m", "5000m")) -> TargetDay:
    """Load all 8 target timesteps for one day (full grid)."""
    if isinstance(date, str):
        date = pd.Timestamp(date).date()
    p = _day_path(date, root=root)
    if not p.exists():
        raise FileNotFoundError(f"No target file for {date} at {p}")

    static = load_static(str(root))
    ds = xr.open_dataset(p)
    try:
        u: dict[str, np.ndarray] = {}
        v: dict[str, np.ndarray] = {}
        for level in levels:
            uname, vname = f"u{level}", f"v{level}"
            if uname not in ds.data_vars:
                continue
            u[level] = ds[uname].values.astype(np.float32)
            v[level] = ds[vname].values.astype(np.float32)
        return TargetDay(
            date=date, times=ds.time.values,
            lat=static.lat, lon=static.lon, seamask=static.seamask,
            u=u, v=v,
        )
    finally:
        ds.close()


def load_snapshot(date: _date_t | str, hour: int,
                  root: Path | str = TARGET_ROOT,
                  levels: tuple[str, ...] = ("125m", "1500m", "5000m")
                  ) -> TargetSnapshot:
    """Convenience: load a single timestep instead of the full day."""
    return load_day(date, root=root, levels=levels).snapshot(hour)


# ── Recommended train/val/test split ────────────────────────────────────

def train_val_test_dates(root: Path | str = TARGET_ROOT,
                         val_year: int = 2019,
                         test_year: int = 2020
                         ) -> tuple[list[_date_t], list[_date_t], list[_date_t]]:
    """Recommended split for Phase 2.

    Default : train = 2016-2018, val = 2019, test = 2020 - a holdout WITHIN the
    given train years (the real eval years are hidden). Year-coherent (no random
    shuffle inside a year) to avoid seasonal leakage.
    """
    dates = list_dates(root)
    train = [d for d in dates if d.year < val_year]
    val   = [d for d in dates if d.year == val_year]
    test  = [d for d in dates if d.year == test_year]
    return train, val, test


__all__ = [
    "TARGET_ROOT", "LEVELS_M", "HOURS",
    "StaticGrid", "TargetSnapshot", "TargetDay",
    "list_dates", "load_static", "load_day", "load_snapshot",
    "train_val_test_dates",
]
