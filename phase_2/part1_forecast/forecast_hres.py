"""Part 1 (HRES-primary) - MOS forecast from HRES to coarsened target 125 m.

The forecast *driver* is the ECMWF HRES forecast (``north_sea_hres_forecasts``),
which is on the **same 45x57 / 0.25 deg grid** as the target-coarse target
(``u125c``). For a forecast issued on day ``D``, the column ``fcst_*_d{L}_h{H}``
is the forecast **valid at D + L days, hour H**. We learn a MOS correction
HRES(forecast u,v) -> target-coarse(u,v) on the training years, per lead.

HRES reaches d1/d7/d10 only - there is no NWP at +14 d, so t+14 falls back to a
per-cell (week-of-year, hour) climatology built from the training years.
"""
from __future__ import annotations

import sys
from datetime import date as _date_t
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))        # kit root (for config)
import config                                     # noqa: E402

HRES_LEADS = (1, 7)          # d10 exists too; d14 has no HRES (-> climatology)
HOURS = (0, 6, 12, 18)


def _uv_from_speed_dir(speed, met_dir_deg):
    """Meteorological 'from' direction (deg) + speed -> (u, v)."""
    r = np.radians(np.asarray(met_dir_deg, float))
    s = np.asarray(speed, float)
    return -s * np.sin(r), -s * np.cos(r)


@lru_cache(maxsize=1)
def _load_hres() -> pd.DataFrame:
    """HRES driver, concatenated across BOTH datasets (Phase-1 2019-2021 +
    Phase-2 back-fill 2016-2018), de-duplicated on (time, lat, lon)."""
    paths = config.hres_parquets()
    if not paths:
        raise FileNotFoundError(
            "No HRES parquet found. Set $PHASE2_DATA_ROOT to where the Phase-1 + "
            "Phase-2 datasets are unzipped (run config.describe() to debug).")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True, sort=False)
    df["time"] = pd.to_datetime(df["time"])
    return df.drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def _reanalysis_grid():
    """(lat1d, lon1d) of the coarse/reanalysis grid, read from any available coarse file."""
    import glob
    import xarray as xr
    files = sorted(glob.glob(str(config.coarse_root() / "*" / "coarse_*.nc")))
    if not files:
        raise FileNotFoundError(f"no coarse files under {config.coarse_root()}")
    ds = xr.open_dataset(files[0])
    lat, lon = ds["latitude"].values, ds["longitude"].values
    ds.close()
    return lat, lon


@lru_cache(maxsize=512)
def _coarse_grid(date_key: str):
    """(lat_flat, lon_flat, {hour: (u_flat, v_flat)}) of target@coarse for one day.

    Training years read the prebuilt ``coarse_*.nc``. For the **hidden eval year**
    (no coarse file shipped) we coarsen the fine target on the fly - organiser-only,
    used solely for internal scoring.
    """
    import xarray as xr
    d = pd.Timestamp(date_key)
    p = config.coarse_root() / f"{d.year}" / f"coarse_{d:%Y%m%d}.nc"
    if p.exists():
        ds = xr.open_dataset(p)
        lat, lon = ds["latitude"].values, ds["longitude"].values
        LON, LAT = np.meshgrid(lon, lat)
        out = {h: (ds.sel(time=d + pd.Timedelta(hours=h))["u125c"].values.ravel(),
                   ds.sel(time=d + pd.Timedelta(hours=h))["v125c"].values.ravel())
               for h in HOURS}
        ds.close()
        return LAT.ravel(), LON.ravel(), out
    # fallback: coarsen the hidden fine target (eval-year truth, organiser-only)
    sys.path.insert(0, str(_HERE.parents[1] / "part0_dataset_setup"))
    import target_loader
    import coarsen
    lat1d, lon1d = _reanalysis_grid()
    try:
        day = target_loader.load_day(d.date(), root=config.target_root())
    except FileNotFoundError:
        return None
    LON, LAT = np.meshgrid(lon1d, lat1d)
    out = {}
    for h in HOURS:
        snap = day.snapshot(h)
        u, v = snap.fields["125m"]["u"], snap.fields["125m"]["v"]
        uc, vc = coarsen.coarsen_uv(u, v, snap.lat, snap.lon, lat1d, lon1d)
        out[h] = (uc.ravel(), vc.ravel())
    return LAT.ravel(), LON.ravel(), out


def build_hres_table(issue_dates, hours=HOURS, leads=HRES_LEADS,
                     with_truth: bool = True) -> pd.DataFrame:
    """HRES forecast features per (issue_date, lead, hour), with the target-coarse
    target attached when available.

    with_truth=True (training): require the coarse truth and drop rows without it.
    with_truth=False (inference): the eval-year truth is hidden, so emit the features
    alone (u125c/v125c = NaN) - predict_mos only consumes the features.
    """
    hres = _load_hres()
    cols = ["lat", "lon", "lead", "hour", "fcst_u", "fcst_v", "fcst_speed",
            "woy_sin", "woy_cos", "u125c", "v125c"]
    blocks = []
    for D in issue_dates:
        D = pd.Timestamp(D)
        hrow = hres[hres["time"] == D]
        if hrow.empty:
            continue
        lat = hrow["latitude"].to_numpy()
        lon = hrow["longitude"].to_numpy()
        for L in leads:
            V = D + pd.Timedelta(days=L)
            cg = _coarse_grid(f"{V:%Y-%m-%d}") if with_truth else None
            if with_truth and cg is None:
                continue
            idx = None
            if cg is not None:
                clat, clon, cyu = cg
                key = {(round(la, 3), round(lo, 3)): i
                       for i, (la, lo) in enumerate(zip(clat, clon))}
                idx = np.array([key.get((round(la, 3), round(lo, 3)), -1)
                                for la, lo in zip(lat, lon)])
            woy = V.isocalendar().week
            for H in hours:
                sp = hrow[f"fcst_speed_d{L}_h{H}"].to_numpy()
                di = hrow[f"fcst_dir_d{L}_h{H}"].to_numpy()
                fu, fv = _uv_from_speed_dir(sp, di)
                if cg is not None:
                    tu_all, tv_all = cyu[H]
                    ok = idx >= 0
                    tu = np.full(idx.shape, np.nan)
                    tv = np.full(idx.shape, np.nan)
                    tu[ok] = tu_all[idx[ok]]
                    tv[ok] = tv_all[idx[ok]]
                else:
                    tu = np.full(lat.shape, np.nan)
                    tv = np.full(lat.shape, np.nan)
                blocks.append(pd.DataFrame({
                    "lat": lat, "lon": lon, "lead": L, "hour": H,
                    "fcst_u": fu, "fcst_v": fv, "fcst_speed": sp,
                    "woy_sin": np.sin(2 * np.pi * woy / 52.0),
                    "woy_cos": np.cos(2 * np.pi * woy / 52.0),
                    "u125c": tu, "v125c": tv,
                }))
    if not blocks:
        return pd.DataFrame(columns=cols)
    df = pd.concat(blocks, ignore_index=True)
    if with_truth:
        df = df.dropna(subset=["u125c", "v125c"])
    return df.reset_index(drop=True)


FEATURES = ["fcst_u", "fcst_v", "fcst_speed", "lat", "lon",
            "woy_sin", "woy_cos"]

CLIM_CACHE = _HERE.parent / "models" / "climatology_coarse.npz"


@lru_cache(maxsize=1)
def _climatology():
    """Per-cell mean (u, v) by (week-of-year, hour) from the TRAIN years.

    Used as the t+14 forecast (no NWP that far out). Cached to ``CLIM_CACHE``.
    """
    if CLIM_CACHE.exists():
        z = np.load(CLIM_CACHE, allow_pickle=True)
        keys = [tuple(int(x) for x in k) for k in z["keys"]]
        return {k: (z[f"u_{k[0]}_{k[1]}"], z[f"v_{k[0]}_{k[1]}"]) for k in keys}
    import glob
    import xarray as xr
    files = []
    for y in TRAIN_YEARS_CLIM:
        files += sorted(glob.glob(str(config.coarse_root() / f"{y}" / "coarse_*.nc")))
    su, sv, cn = {}, {}, {}
    for f in files:
        ds = xr.open_dataset(f)
        u, v, ts = ds["u125c"].values, ds["v125c"].values, ds["time"].values
        for ti, t in enumerate(ts):
            T = pd.Timestamp(t)
            if T.hour not in HOURS:
                continue
            k = (int(T.isocalendar().week), int(T.hour))
            uu, vv = u[ti].ravel(), v[ti].ravel()
            m = np.isfinite(uu)
            if k not in su:
                su[k] = np.zeros(uu.size); sv[k] = np.zeros(uu.size)
                cn[k] = np.zeros(uu.size)
            su[k] += np.nan_to_num(uu); sv[k] += np.nan_to_num(vv); cn[k] += m
        ds.close()
    clim = {k: (su[k] / np.maximum(cn[k], 1), sv[k] / np.maximum(cn[k], 1))
            for k in su}
    CLIM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save = {"keys": np.array(list(clim.keys()))}
    for k, (uu, vv) in clim.items():
        save[f"u_{k[0]}_{k[1]}"] = uu; save[f"v_{k[0]}_{k[1]}"] = vv
    np.savez_compressed(CLIM_CACHE, **save)
    return clim


# Climatology (t+14 fallback) is built from the TRAIN years only - never the
# hidden eval years (2021+). Participants only have coarse target for 2016-2020.
TRAIN_YEARS_CLIM = (2016, 2017, 2018, 2019, 2020)


def build_climatology_forecast(issue_dates, hours=HOURS, lead=14,
                               with_truth: bool = True) -> pd.DataFrame:
    """t+`lead` forecast = per-cell (week, hour) climatology. with_truth=True attaches
    the target-coarse truth for scoring; with_truth=False (inference) emits the
    prediction alone (the eval-year truth is hidden)."""
    clim = _climatology()
    lat1d, lon1d = _reanalysis_grid()
    LON, LAT = np.meshgrid(lon1d, lat1d)
    flat_lat, flat_lon = LAT.ravel(), LON.ravel()
    blocks = []
    for D in issue_dates:
        D = pd.Timestamp(D)
        V = D + pd.Timedelta(days=lead)
        cg = _coarse_grid(f"{V:%Y-%m-%d}") if with_truth else None
        woy = int(V.isocalendar().week)
        for H in hours:
            k = (woy, H) if (woy, H) in clim else min(
                clim, key=lambda kk: (abs(kk[0] - woy), abs(kk[1] - H)))
            cu, cv = clim[k]
            if cg is not None:
                tu, tv = cg[2][H]
            else:
                tu = tv = np.full(cu.size, np.nan)
            blocks.append(pd.DataFrame({
                "lat": flat_lat, "lon": flat_lon, "lead": lead, "hour": H,
                "u_pred": cu, "v_pred": cv, "u125c": tu, "v125c": tv}))
    df = pd.concat(blocks, ignore_index=True)
    if with_truth:
        df = df.dropna(subset=["u125c", "v125c"])
    return df.reset_index(drop=True)


def train_mos(train_df, leads=HRES_LEADS, **lgbm_kw):
    """One LightGBM per (lead, component) mapping HRES forecast -> target-coarse."""
    import lightgbm as lgb
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, verbose=-1, **lgbm_kw)
    models = {}
    for L in leads:
        sub = train_df[train_df["lead"] == L]
        X = sub[FEATURES]
        models[(L, "u")] = lgb.LGBMRegressor(**params).fit(X, sub["u125c"])
        models[(L, "v")] = lgb.LGBMRegressor(**params).fit(X, sub["v125c"])
    return models


#: Predictive quantiles for the probabilistic MOS (a 90 % central interval).
QUANTILES = (0.05, 0.5, 0.95)


def _qcol(q: float) -> str:
    return f"spd_q{int(round(q * 100)):02d}"


def train_quantile_mos(train_df, leads=HRES_LEADS, quantiles=QUANTILES, **lgbm_kw):
    """Quantile LightGBM per (lead, tau) mapping the HRES forecast to target-coarse
    **wind speed** (``hypot(u125c, v125c)``). Speed - not u/v - is what the turbine
    power curve and :mod:`bidding` consume, so we model its predictive quantiles
    directly (``objective='quantile'``) for calibrated uncertainty / Winkler scoring."""
    import lightgbm as lgb
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=63,
                  subsample=0.8, colsample_bytree=0.8, verbose=-1, **lgbm_kw)
    models = {}
    for L in leads:
        sub = train_df[train_df["lead"] == L]
        X = sub[FEATURES]
        y = np.hypot(sub["u125c"].to_numpy(), sub["v125c"].to_numpy())
        for q in quantiles:
            models[(L, q)] = lgb.LGBMRegressor(
                objective="quantile", alpha=q, **params).fit(X, y)
    return models


def predict_quantile_mos(models, df, quantiles=QUANTILES, adjust=None) -> pd.DataFrame:
    """Add speed-quantile columns ``spd_q05/q50/q95`` to ``df``. Independent
    quantile models can *cross*; we sort each row so q05 ≤ q50 ≤ q95 and clip at 0.

    ``adjust`` (a per-lead conformal widening from :func:`conformal_adjust`) widens
    the outer interval [q_lo − Q_L, q_hi + Q_L] so coverage reaches the target."""
    out = df.copy()
    qs = sorted(quantiles)
    cols = [_qcol(q) for q in qs]
    for c in cols:
        out[c] = np.nan
    for L in sorted(df["lead"].unique()):
        m = (df["lead"] == L).to_numpy()
        if not m.any():
            continue
        X = df.loc[m, FEATURES]
        for q, c in zip(qs, cols):
            out.loc[m, c] = models[(L, q)].predict(X)
        if adjust and L in adjust:                            # conformal widening
            out.loc[m, cols[0]] = out.loc[m, cols[0]] - adjust[L]
            out.loc[m, cols[-1]] = out.loc[m, cols[-1]] + adjust[L]
    vals = np.sort(out[cols].to_numpy(float), axis=1)        # enforce non-crossing
    out[cols] = np.clip(vals, 0.0, None)
    return out


def conformal_adjust(qmos, calib_df, alpha=0.10, quantiles=QUANTILES) -> dict:
    """Conformalized Quantile Regression (Romano, Patterson & Candès 2019).

    On a held-out **calibration** set, the conformity score per sample is
    ``E = max(q_lo − y, y − q_hi)`` (how far outside the raw interval the truth
    falls). The (1−alpha) empirical quantile of E, added symmetrically to the
    interval, gives **finite-sample (1−alpha) coverage** without distributional
    assumptions. Returns ``{lead: Q_L}`` for :func:`predict_quantile_mos`."""
    qs = sorted(quantiles)
    lo_c, hi_c = _qcol(qs[0]), _qcol(qs[-1])
    pr = predict_quantile_mos(qmos, calib_df, quantiles)
    adj = {}
    for L in sorted(calib_df["lead"].unique()):
        s = pr[pr["lead"] == L]
        y = np.hypot(s["u125c"].to_numpy(), s["v125c"].to_numpy())
        E = np.maximum(s[lo_c].to_numpy() - y, y - s[hi_c].to_numpy())
        n = len(E)
        k = int(np.ceil((n + 1) * (1.0 - alpha)))            # conformal rank
        adj[L] = float(np.sort(E)[min(k, n) - 1])
    return adj


def predictions_to_grid(pred_df, lead, hour, col_u="u_pred", col_v="v_pred"):
    """Scatter a long-format prediction (rows per cell) back onto the reanalysis grid
    (lat1d x lon1d). Cells without a prediction stay NaN."""
    lat1d, lon1d = _reanalysis_grid()
    li = {round(float(x), 3): i for i, x in enumerate(lat1d)}
    lj = {round(float(x), 3): j for j, x in enumerate(lon1d)}
    u = np.full((lat1d.size, lon1d.size), np.nan)
    v = np.full((lat1d.size, lon1d.size), np.nan)
    s = pred_df[(pred_df["lead"] == lead) & (pred_df["hour"] == hour)]
    for la, lo, uu, vv in zip(s["lat"], s["lon"], s[col_u], s[col_v]):
        i = li.get(round(float(la), 3)); j = lj.get(round(float(lo), 3))
        if i is not None and j is not None:
            u[i, j] = uu; v[i, j] = vv
    return u, v


def predict_mos(models, df) -> pd.DataFrame:
    out = df.copy()
    out["u_pred"] = np.nan
    out["v_pred"] = np.nan
    for L in sorted(df["lead"].unique()):
        m = df["lead"] == L
        out.loc[m, "u_pred"] = models[(L, "u")].predict(df.loc[m, FEATURES])
        out.loc[m, "v_pred"] = models[(L, "v")].predict(df.loc[m, FEATURES])
    return out
