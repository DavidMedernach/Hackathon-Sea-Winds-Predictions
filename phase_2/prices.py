"""Day-ahead electricity prices loader for Phase 2 bidding.

Loads the EPEX SPOT 2022 hourly day-ahead prices downloaded by
``internal/data_generation/download_epex_prices.py``.  Provides utilities
for resampling to match the wind series cadence (typically 6-hourly to
match reanalysis) and a flat baseline for ablation studies.

Source: energy-charts.info / SMARD.de (CC BY 4.0).

Usage
-----
::

    from prices import load_prices, align_to_wind

    prices = load_prices(zone='NL', flat=False)        # full 2022, 8760 h
    aligned = align_to_wind(prices, wind_times)         # match the wind cadence
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_PRICES_DIR = _PROJECT_ROOT / "data" / "wind_data" / "electricity"

DEFAULT_YEAR = 2022
DEFAULT_ZONE = "NL"  # Netherlands - the closest EPEX zone to NS offshore wind

# Heuristic offshore-wind imbalance penalties used by the bidding scorer when
# the user does not provide their own. They reflect a typical asymmetric
# regime where short-balancing (buying back power at higher prices) is more
# expensive than long-balancing (selling excess at a discount).
DEFAULT_PI_UP_EUR_PER_MWH:   float = 90.0  # paid when short:    bid > actual
DEFAULT_PI_DOWN_EUR_PER_MWH: float = 30.0  # paid when long:     actual > bid


def list_zones(year: int = DEFAULT_YEAR) -> list[str]:
    """Return all zones available in the downloaded file."""
    df = _read_raw(year, flat=False)
    return sorted(df["zone"].unique())


def _read_raw(year: int, flat: bool) -> pd.DataFrame:
    suffix = "_flat" if flat else ""
    path = _PRICES_DIR / f"epex_dayahead_{year}{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Prices file not found: {path}\n"
            "Run `python internal/data_generation/download_epex_prices.py` first."
        )
    return pd.read_parquet(path)


def load_prices(
    *, zone: str = DEFAULT_ZONE, year: int = DEFAULT_YEAR, flat: bool = False,
) -> pd.DataFrame:
    """Load 1 year of EPEX day-ahead prices for one bidding zone.

    Parameters
    ----------
    zone
        Bidding zone code (e.g. ``'DE-LU'``, ``'FR'``, ``'NL'``, ``'BE'``,
        ``'DK1'``).
    year
        Calendar year (default 2022).
    flat
        If True, return the *flat* version where the price is replaced by
        the zone's annual mean. Useful for ablation studies that isolate
        the effect of price-wind covariance from sheer wind variability.

    Returns
    -------
    DataFrame with columns ``time`` (UTC, hourly), ``zone``,
    ``price_eur_per_mwh``.
    """
    df = _read_raw(year, flat=flat)
    sub = df[df["zone"] == zone].copy()
    if sub.empty:
        raise ValueError(
            f"Zone '{zone}' not found. Available: {sorted(df['zone'].unique())}"
        )
    sub["time"] = pd.to_datetime(sub["time"])
    return sub.sort_values("time").reset_index(drop=True)


def align_to_wind(
    prices: pd.DataFrame, wind_times: Iterable[pd.Timestamp],
    method: str = "mean",
) -> np.ndarray:
    """Aggregate hourly prices to match a coarser wind-time index.

    The Phase 2 simulator runs on the reanalysis cadence (6-hourly). The bidding
    decision and the realised price are both bin-averaged so the bid quantity
    represents an energy commitment over the 6-h block, settled against the
    average market price for that block.

    Parameters
    ----------
    prices
        DataFrame from ``load_prices``.
    wind_times
        Sequence of timestamps marking the START of each wind step. The
        function aggregates prices in ``[t, t + dt)`` where ``dt`` is the
        median spacing of ``wind_times``.
    method
        Aggregation: ``'mean'`` (default), ``'max'``, ``'min'``, ``'first'``.

    Returns
    -------
    1-D array of aggregated prices, length ``len(wind_times)``.
    """
    wind_times = pd.to_datetime(pd.Series(list(wind_times)))
    if len(wind_times) < 2:
        raise ValueError("Need at least 2 wind times to infer the bin width")
    dt = wind_times.diff().median()

    p = prices.copy()
    p["time"] = pd.to_datetime(p["time"])
    if p["time"].dt.tz is not None:
        p["time"] = p["time"].dt.tz_convert(None)
    p = p.set_index("time")["price_eur_per_mwh"]
    if wind_times.dt.tz is not None:
        wind_times = wind_times.dt.tz_localize(None)

    out = np.empty(len(wind_times))
    for i, t0 in enumerate(wind_times):
        window = p.loc[t0: t0 + dt - pd.Timedelta(seconds=1)]
        if len(window) == 0:
            out[i] = float(p.asof(t0)) if not p.empty else np.nan
            continue
        if method == "mean":
            out[i] = float(window.mean())
        elif method == "max":
            out[i] = float(window.max())
        elif method == "min":
            out[i] = float(window.min())
        elif method == "first":
            out[i] = float(window.iloc[0])
        else:
            raise ValueError(f"Unknown method: {method!r}")
    return out
