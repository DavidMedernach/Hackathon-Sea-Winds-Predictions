"""Downscale a coarse-125 wind field (P1 forecast output space) to the fine
target 1.3 km grid.

Input = coarse-125 u/v on the reanalysis grid (what Part 1 predicts, or a coarse-125
history frame). It is bilinearly interpolated onto the target grid and combined
with static terrain features; a per-component LightGBM recovers sub-grid detail,
beating plain bilinear interpolation. Trained on (coarse-125 -> target-125) pairs.

Scope / honest caveats:
- The added value over bilinear is SMALL (~3% RMSE on sea, ~0.73 -> ~0.71 m/s):
  coarse-125 is just target-125 smoothed onto the reanalysis grid, so its bilinear
  interpolation is already close to the truth over open sea. (Contrast the
  reanalysis->target downscaling in 2_rf_terrain_aware.ipynb, ~-31%, where the input is
  a genuinely coarser/different product.)
- ~1/3 of target sea pixels lie SOUTH/WEST of the reanalysis footprint and have no
  coarse input there: ``downscale`` returns NaN for them (they are excluded from
  training and from ``eval_day`` RMSE). The fine field is defined only where the
  target grid overlaps the reanalysis coverage.
"""
from __future__ import annotations

from datetime import date as _date_t
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

import target_loader
import reanalysis_loader
import lightgbm as lgb
from terrain_features import compute_static_features

import sys as _sys
_HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent))           # kit root (for config)
import config                                     # noqa: E402

FEATURES = ["coarse_u", "coarse_v", "coarse_ws", "elevation_m", "dist_shore_km",
            "lat", "lon"]
_LGBM = dict(n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63,
             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)


@lru_cache(maxsize=1)
def _static():
    return target_loader.load_static(str(config.target_root()))


@lru_cache(maxsize=1)
def _reanalysis_axes():
    er = config.reanalysis_root()
    e5 = reanalysis_loader.load_reanalysis(reanalysis_loader.list_dates(er)[0], hour=0, root=er)
    return np.asarray(e5.lats, float), np.asarray(e5.lons, float)


@lru_cache(maxsize=1)
def _terrain():
    st = _static()
    return compute_static_features(st.lon, st.lat, st.seamask)


def _coarse_day(d: _date_t, hour: int):
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


def interp_coarse_to_target(cu, cv):
    """Bilinear-interp a coarse-125 (u, v) field (reanalysis grid) onto the target grid.

    NaN coarse cells are nearest-filled first so the fine field is defined wherever
    the target grid overlaps the reanalysis footprint.
    """
    lats, lons = _reanalysis_axes()
    st = _static()
    pts = np.stack([np.asarray(st.lat).ravel(), np.asarray(st.lon).ravel()], axis=1)
    out = []
    for c in (np.asarray(cu, float), np.asarray(cv, float)):
        filled = c.copy()
        if np.isnan(filled).any():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(np.isnan(filled), return_distances=False,
                                         return_indices=True)
            filled = filled[tuple(idx)]
        f = RegularGridInterpolator((lats, lons), filled, bounds_error=False,
                                    fill_value=np.nan)
        out.append(f(pts).reshape(st.lat.shape).astype(np.float32))
    return out[0], out[1]


def _features_from_coarse(cu_fine, cv_fine) -> pd.DataFrame:
    terr = _terrain()
    ws = np.sqrt(cu_fine ** 2 + cv_fine ** 2)
    return pd.DataFrame({
        "coarse_u": cu_fine.ravel(), "coarse_v": cv_fine.ravel(),
        "coarse_ws": ws.ravel(),
        "elevation_m": np.asarray(terr["elevation_m"]).ravel(),
        "dist_shore_km": np.asarray(terr["dist_shore_km"]).ravel(),
        "lat": np.asarray(terr["lat"]).ravel(),
        "lon": np.asarray(terr["lon"]).ravel(),
    })


def _sea_flat() -> np.ndarray:
    return (np.asarray(_static().seamask) > 0.5).ravel()


def train_downscaler(dates, hours=(0, 6, 12, 18), params: dict | None = None) -> dict:
    """Train per-component LightGBM on (coarse-125 -> target-125) sea pixels."""
    params = {**_LGBM, **(params or {})}
    sea = _sea_flat()
    X_rows, yu_rows, yv_rows = [], [], []
    for d in dates:
        for hour in hours:
            cu, cv = _coarse_day(d, hour)
            if cu is None:
                continue
            cu_f, cv_f = interp_coarse_to_target(cu, cv)
            feat = _features_from_coarse(cu_f, cv_f)
            snap = target_loader.load_snapshot(d, hour, root=config.target_root())
            tu = snap.fields["125m"]["u"].ravel()
            tv = snap.fields["125m"]["v"].ravel()
            keep = sea & np.isfinite(feat["coarse_u"].values) & np.isfinite(tu) & np.isfinite(tv)
            X_rows.append(feat[keep])
            yu_rows.append(tu[keep])
            yv_rows.append(tv[keep])
    X = pd.concat(X_rows, ignore_index=True)[FEATURES]
    yu = np.concatenate(yu_rows)
    yv = np.concatenate(yv_rows)
    mu = lgb.LGBMRegressor(**params).fit(X, yu)
    mv = lgb.LGBMRegressor(**params).fit(X, yv)
    return {"u": mu, "v": mv}


def downscale(models: dict, cu, cv):
    """Downscale a coarse-125 (u, v) field (reanalysis grid) -> fine target (u, v).

    Returns (fine_u, fine_v) on the target grid; non-sea pixels are NaN.
    """
    cu_f, cv_f = interp_coarse_to_target(cu, cv)
    feat = _features_from_coarse(cu_f, cv_f)[FEATURES]
    sea = _sea_flat()
    shape = _static().lat.shape
    fu = np.full(feat.shape[0], np.nan, np.float32)
    fv = np.full(feat.shape[0], np.nan, np.float32)
    fu[sea] = models["u"].predict(feat[sea]).astype(np.float32)
    fv[sea] = models["v"].predict(feat[sea]).astype(np.float32)
    return fu.reshape(shape), fv.reshape(shape)


def downscale_day(models: dict, d, hour: int):
    dd = pd.Timestamp(d).date() if not isinstance(d, _date_t) else d
    cu, cv = _coarse_day(dd, hour)
    if cu is None:
        raise FileNotFoundError(f"No coarse-125 field for {d} {hour:02d}h")
    return downscale(models, cu, cv)


def eval_day(models: dict, d, hour: int):
    """Return (downscaler_rmse, bilinear_rmse) on sea pixels vs target-125 truth (ws)."""
    dd = pd.Timestamp(d).date() if not isinstance(d, _date_t) else d
    cu, cv = _coarse_day(dd, hour)
    fu, fv = downscale(models, cu, cv)
    cu_f, cv_f = interp_coarse_to_target(cu, cv)
    snap = target_loader.load_snapshot(dd, hour, root=config.target_root())
    tws = np.sqrt(snap.fields["125m"]["u"] ** 2 + snap.fields["125m"]["v"] ** 2)
    sea = np.asarray(_static().seamask) > 0.5
    m_ws = np.sqrt(fu ** 2 + fv ** 2)
    b_ws = np.sqrt(cu_f ** 2 + cv_f ** 2)
    msk = sea & np.isfinite(m_ws) & np.isfinite(tws) & np.isfinite(b_ws)
    rmse_model = float(np.sqrt(np.mean((m_ws[msk] - tws[msk]) ** 2)))
    rmse_bilin = float(np.sqrt(np.mean((b_ws[msk] - tws[msk]) ** 2)))
    return rmse_model, rmse_bilin
