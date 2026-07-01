"""Part 1 forecast - Tier 0 baselines, metrics, and (Tier 1) LightGBM.

Targets are coarsened target 125m u/v at lead times. Tier 0 = persistence
(reanalysis 100m wind carried forward as the hub-level proxy). Metrics: vector RMSE
on (u, v) and circular MAE on the derived direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def uv_rmse(u_true, v_true, u_pred, v_pred) -> float:
    """Root mean squared vector error sqrt(mean(Δu² + Δv²)). Assumes finite inputs."""
    u_true, v_true = np.asarray(u_true, float), np.asarray(v_true, float)
    u_pred, v_pred = np.asarray(u_pred, float), np.asarray(v_pred, float)
    err2 = (u_pred - u_true) ** 2 + (v_pred - v_true) ** 2
    return float(np.sqrt(np.mean(err2)))


def winkler_score(y, lo, hi, alpha: float = 0.10) -> float:
    """Winkler interval score for a central (1−alpha) prediction interval [lo, hi].

    width + (2/alpha)·(lo−y) if y<lo  + (2/alpha)·(y−hi) if y>hi, averaged.
    Rewards sharp intervals but penalises misses ∝ how far outside they fall.
    Lower is better; a perfectly-calibrated sharp forecast minimises it."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    width = hi - lo
    below = (lo - y) * (y < lo)
    above = (y - hi) * (y > hi)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def coverage_fraction(y, lo, hi) -> float:
    """Empirical coverage: fraction of y inside [lo, hi]. Target = 1−alpha (0.90)."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def speed_rmse(u_true, v_true, u_pred, v_pred) -> float:
    """RMSE of wind *speed* sqrt(mean((|pred| - |true|)²)) - interpretable in m/s
    and directly comparable to single-field downscaling RMSE. Assumes finite inputs."""
    st = np.hypot(np.asarray(u_true, float), np.asarray(v_true, float))
    sp = np.hypot(np.asarray(u_pred, float), np.asarray(v_pred, float))
    return float(np.sqrt(np.mean((sp - st) ** 2)))


def _dir_from_uv(u, v) -> np.ndarray:
    """Meteorological 'from' direction in degrees."""
    return (270.0 - np.degrees(np.arctan2(np.asarray(v, float), np.asarray(u, float)))) % 360.0


def circular_mae_from_uv(u_true, v_true, u_pred, v_pred) -> float:
    """Mean absolute circular error (degrees) between derived directions."""
    d_true = _dir_from_uv(u_true, v_true)
    d_pred = _dir_from_uv(u_pred, v_pred)
    diff = np.abs(d_pred - d_true) % 360.0
    diff = np.minimum(diff, 360.0 - diff)
    return float(np.mean(diff))


def persistence_forecast(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Tier 0 persistence: reanalysis 100m wind at issue time as the lead forecast."""
    return df["u100"].to_numpy(float), df["v100"].to_numpy(float)


# ── Tier 1 : per-target LightGBM ────────────────────────────────────────
import lightgbm as lgb  # noqa: E402
import joblib  # noqa: E402
from pathlib import Path  # noqa: E402

from forecast_features import FEATURE_COLS  # noqa: E402

_LGBM_PARAMS = dict(n_estimators=300, max_depth=7, learning_rate=0.05,
                    num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                    n_jobs=-1, verbose=-1)


def _target_cols(leads_days) -> list[str]:
    cols = []
    for lead in leads_days:
        cols += [f"u125c_d{lead}", f"v125c_d{lead}"]
    return cols


def train_lgbm(train_df, leads_days=(1, 7, 14), params: dict | None = None) -> dict:
    """Train one LightGBM regressor per (lead × component) target column."""
    params = {**_LGBM_PARAMS, **(params or {})}
    X = train_df[FEATURE_COLS]
    models: dict[str, lgb.LGBMRegressor] = {}
    for col in _target_cols(leads_days):
        m = lgb.LGBMRegressor(**params)
        m.fit(X, train_df[col])
        models[col] = m
    return models


def predict_lgbm(models: dict, df) -> dict:
    """Predict every target column; returns {col: 1-D array}."""
    X = df[FEATURE_COLS]
    return {col: m.predict(X) for col, m in models.items()}


def save_models(models: dict, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for col, m in models.items():
        joblib.dump(m, out_dir / f"lgbm_{col}.joblib")


def load_models(in_dir, leads_days=(1, 7, 14)) -> dict:
    in_dir = Path(in_dir)
    out = {}
    for col in _target_cols(leads_days):
        p = in_dir / f"lgbm_{col}.joblib"
        if p.exists():
            out[col] = joblib.load(p)
    return out
