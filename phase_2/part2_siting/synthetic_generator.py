"""Phase 2 (Part 2) synthetic wind generator.

Generates realistic synthetic years of 125 m wind (u/v) on the coarse reanalysis
sea grid, from the target-coarsened-125 history (Plan 1 output). Two tiers:
Tier 0 block-bootstrap, Tier 1 autoregressive (per-cell AR(1) + joint
innovation resampling). A ``to_site_series`` helper interpolates a synthetic
field to any lat/lon and returns a (time, ws, wd) series for the PyWake
simulator (Part 4).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys as _sys
_HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(_HERE.parent))           # kit root (for config)
import config                                     # noqa: E402


@dataclass
class CoarseHistory:
    """Historical coarse-125 wind on the sea grid.

    times : 1-D DatetimeIndex of length T
    lat, lon : 1-D arrays of length C (sea cell coordinates)
    U, V : (T, C) float32 wind components (all finite)
    """
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    U: np.ndarray
    V: np.ndarray


def _sea_mask_coarse(reanalysis_lats, reanalysis_lons) -> np.ndarray:
    """Boolean (nlat, nlon) - True where the reanalysis cell is majority sea."""
    import target_loader
    import coarsen
    st = target_loader.load_static()
    sea_frac = coarsen.coarsen_field(st.seamask.astype(np.float32), st.lat, st.lon,
                                     reanalysis_lats, reanalysis_lons)
    return np.isfinite(sea_frac) & (sea_frac > 0.5)


def load_coarse_history(years=(2016, 2017, 2018, 2019, 2020)) -> CoarseHistory:
    """Load the coarsened-125 history on the sea grid for the given years."""
    import reanalysis_loader
    e5 = reanalysis_loader.load_reanalysis(reanalysis_loader.list_dates()[0], hour=0)
    lats1d, lons1d = np.asarray(e5.lats), np.asarray(e5.lons)
    sea = _sea_mask_coarse(lats1d, lons1d)

    files = []
    for y in years:
        files += sorted(glob.glob(str(config.coarse_root() / f"{y}" / "coarse_*.nc")))
    if not files:
        raise FileNotFoundError(f"No coarse-125 files for years {years} under {config.coarse_root()}")

    times, U_rows, V_rows = [], [], []
    lon2d, lat2d = np.meshgrid(lons1d, lats1d)
    for f in files:
        ds = xr.open_dataset(f)
        try:
            for ti in range(ds.sizes["time"]):
                times.append(pd.Timestamp(ds.time.values[ti]))
                U_rows.append(ds["u125c"].isel(time=ti).values)
                V_rows.append(ds["v125c"].isel(time=ti).values)
        finally:
            ds.close()

    U_all = np.stack(U_rows).reshape(len(times), -1)
    V_all = np.stack(V_rows).reshape(len(times), -1)
    sea_flat = sea.reshape(-1)
    finite_all = np.isfinite(U_all).all(0) & np.isfinite(V_all).all(0)
    keep = sea_flat & finite_all
    return CoarseHistory(
        times=pd.DatetimeIndex(times),
        lat=lat2d.reshape(-1)[keep].astype(np.float32),
        lon=lon2d.reshape(-1)[keep].astype(np.float32),
        U=U_all[:, keep].astype(np.float32),
        V=V_all[:, keep].astype(np.float32),
    )


def weibull_mom(speeds) -> tuple[float, float]:
    """Method-of-moments Weibull (k, c) from a sample of wind speeds."""
    s = np.asarray(speeds, float)
    s = s[np.isfinite(s) & (s >= 0)]
    if s.size == 0:
        return float("nan"), float("nan")
    m, sd = s.mean(), s.std()
    if m <= 0:
        return float("nan"), float("nan")
    k = (sd / m) ** -1.086
    from math import gamma
    c = m / gamma(1.0 + 1.0 / k)
    return float(k), float(c)


def summary_stats(U, V) -> dict:
    """Per-field summary used for realism comparison."""
    ws = np.sqrt(np.asarray(U) ** 2 + np.asarray(V) ** 2)
    k, c = weibull_mom(ws.ravel())
    return {"mean_ws": float(ws.mean()), "std_ws": float(ws.std()),
            "weibull_k": k, "weibull_c": c}


@dataclass
class SynthYear:
    """A synthetic year of coarse-grid wind."""
    times: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray
    U: np.ndarray            # (n_steps, C)
    V: np.ndarray


def _synth_times(n_steps: int) -> pd.DatetimeIndex:
    """A canonical 6-hourly time axis of length n_steps starting 2099-01-01.

    Year 2099 is a neutral placeholder (never a real/eval year).
    """
    return pd.date_range("2099-01-01", periods=n_steps, freq="6h")


def bootstrap_year(history: CoarseHistory, block_days: int = 14,
                   n_steps: int = 1460, seed: int = 0) -> SynthYear:
    """Tier 0: concatenate random contiguous multi-cell blocks into a year.

    Preserves the marginal wind distribution and short-term (within-block)
    spatial & temporal structure. Does not reproduce the annual seasonal
    ordering - adequate for AEP, which integrates over the marginal.
    """
    rng = np.random.default_rng(seed)
    block = block_days * 4
    T = history.U.shape[0]
    if block > T:
        block = T
    out_u, out_v = [], []
    filled = 0
    while filled < n_steps:
        start = int(rng.integers(0, T - block + 1))
        take = min(block, n_steps - filled)
        out_u.append(history.U[start:start + take])
        out_v.append(history.V[start:start + take])
        filled += take
    return SynthYear(
        times=_synth_times(n_steps),
        lat=history.lat, lon=history.lon,
        U=np.concatenate(out_u, axis=0).astype(np.float32),
        V=np.concatenate(out_v, axis=0).astype(np.float32),
    )


@dataclass
class ARGenerator:
    """Fitted per-cell AR(1) model on climatology anomalies, with the pool of
    historical joint innovation vectors (for spatially-coherent resampling)."""
    lat: np.ndarray
    lon: np.ndarray
    clim: dict                 # (woy, hour) -> (U_clim vector, V_clim vector)
    phi_u: np.ndarray          # (C,) lag-1 AR coeff per cell, U
    phi_v: np.ndarray          # (C,)
    innov_u: np.ndarray        # (T-1, C) historical innovations, U
    innov_v: np.ndarray        # (T-1, C)
    anom0_u: np.ndarray        # (T, C) historical anomalies (to seed t=0)
    anom0_v: np.ndarray


def _clim_key(ts: pd.Timestamp) -> tuple[int, int]:
    return (int(ts.isocalendar().week), int(ts.hour))


def _build_clim(history: CoarseHistory):
    """Mean [U,V] per (week-of-year, hour). Returns (clim dict, U_anom, V_anom)."""
    keys = [_clim_key(t) for t in history.times]
    clim = {}
    Ua, Va = history.U.copy(), history.V.copy()
    for key in sorted(set(keys)):
        m = np.array([k == key for k in keys])
        cu = history.U[m].mean(0)
        cv = history.V[m].mean(0)
        clim[key] = (cu.astype(np.float32), cv.astype(np.float32))
        Ua[m] -= cu
        Va[m] -= cv
    return clim, Ua, Va


def fit_ar_generator(history: CoarseHistory) -> ARGenerator:
    clim, Ua, Va = _build_clim(history)

    def _phi(A):
        a0, a1 = A[:-1], A[1:]
        num = (a0 * a1).sum(0)
        den = (a0 * a0).sum(0)
        phi = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return np.clip(phi, -0.99, 0.99).astype(np.float32)

    phi_u, phi_v = _phi(Ua), _phi(Va)
    innov_u = (Ua[1:] - phi_u * Ua[:-1]).astype(np.float32)
    innov_v = (Va[1:] - phi_v * Va[:-1]).astype(np.float32)
    return ARGenerator(history.lat, history.lon, clim, phi_u, phi_v,
                       innov_u, innov_v, Ua, Va)


def sample_ar_year(model: ARGenerator, n_steps: int = 1460, seed: int = 0) -> SynthYear:
    """Tier 1: free-run AR(1) with jointly-resampled innovations, add climatology."""
    rng = np.random.default_rng(seed)
    times = _synth_times(n_steps)
    C = model.phi_u.shape[0]
    n_innov = model.innov_u.shape[0]

    Au = np.empty((n_steps, C), np.float32)
    Av = np.empty((n_steps, C), np.float32)
    s0 = int(rng.integers(0, model.anom0_u.shape[0]))
    Au[0], Av[0] = model.anom0_u[s0], model.anom0_v[s0]
    for t in range(1, n_steps):
        j = int(rng.integers(0, n_innov))
        Au[t] = model.phi_u * Au[t - 1] + model.innov_u[j]
        Av[t] = model.phi_v * Av[t - 1] + model.innov_v[j]

    keys = list(model.clim.keys())
    U = Au.copy()
    V = Av.copy()
    for t, ts in enumerate(times):
        key = _clim_key(ts)
        if key not in model.clim:
            hour = ts.hour
            cand = [k for k in keys if k[1] == hour] or keys
            key = min(cand, key=lambda k: abs(k[0] - ts.isocalendar().week))
        cu, cv = model.clim[key]
        U[t] += cu
        V[t] += cv
    return SynthYear(times=times, lat=model.lat, lon=model.lon, U=U, V=V)


def to_site_series(synth: SynthYear, lat: float, lon: float) -> pd.DataFrame:
    """Inverse-distance interpolate the synthetic grid to (lat, lon) and return
    a (time, ws, wd) DataFrame for the PyWake simulator (Part 4)."""
    d2 = (synth.lat - lat) ** 2 + (synth.lon - lon) ** 2
    order = np.argsort(d2)[:4]
    w = 1.0 / np.maximum(d2[order], 1e-9)
    w = w / w.sum()
    u = synth.U[:, order] @ w
    v = synth.V[:, order] @ w
    ws = np.sqrt(u ** 2 + v ** 2)
    wd = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return pd.DataFrame({"time": synth.times, "ws": ws.astype(np.float32),
                         "wd": wd.astype(np.float32)})


def save_synth_year(synth: SynthYear, path) -> Path:
    """Write a synthetic year to NetCDF (gridded u/v on the sea cells)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"u125": (("time", "cell"), synth.U), "v125": (("time", "cell"), synth.V)},
        coords={"time": synth.times, "lat": ("cell", synth.lat),
                "lon": ("cell", synth.lon)},
    )
    ds.to_netcdf(path)
    return path
