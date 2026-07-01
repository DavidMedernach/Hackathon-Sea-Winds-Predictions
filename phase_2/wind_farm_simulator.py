"""Phase 2 wind farm simulator (PyWake / Bastankhah).

Wraps a small set of design choices that participants do not need to
re-implement:

    - Wake model:     Bastankhah Gaussian (industry standard, imposed)
    - Wind input:     a (time, wind_speed, wind_direction) DataFrame either
                      reanalysis/target-derived or participant-predicted
    - Site model:     time-series-based `XRSite` so each timestep contributes
                      to AEP / power profile
    - Turbulence (TI): per-direction-sector empirical TI estimated from the
                      wind series itself (TI_d = std(ws) / mean(ws) per 30°
                      sector, computed on the reference window). No hard-coded
                      6%, no hidden hyper-parameter.
    - Geometry:       turbines specified in local Cartesian (x_m, y_m) around
                      the farm centre. Latitude/longitude conversion is the
                      caller's job.

Two simulation modes:
    simulate_day(...)   - high-resolution power profile for ~24 h of wind
    simulate_year(...)  - time-series-based AEP integration over an arbitrary
                         multi-month window (typically 1 year of reanalysis).

Constraints (max area, max number of turbines) are checked outside the
simulator: see `validate_layout` for a helper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from py_wake.wind_turbines import WindTurbine
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.site.xrsite import XRSite


# ── Dataclasses ─────────────────────────────────────────────────────────

@dataclass
class FarmLayout:
    """Turbine positions inside the farm box, plus a turbine model.

    x_m, y_m are arrays of equal length; the farm centre sits at (0, 0).
    """
    x_m: np.ndarray
    y_m: np.ndarray
    turbine: WindTurbine

    def __post_init__(self):
        self.x_m = np.asarray(self.x_m, dtype=float)
        self.y_m = np.asarray(self.y_m, dtype=float)
        if self.x_m.shape != self.y_m.shape:
            raise ValueError("x_m and y_m must have the same shape")

    @property
    def n_turbines(self) -> int:
        return int(self.x_m.size)


@dataclass
class WindSeries:
    """A time-series of wind at the farm hub height.

    Stored as a (time, ws_hub, wd) DataFrame in m/s and degrees.

    The simulator never extrapolates from one altitude to another inside this
    class - the caller is expected to have shifted ws to hub height beforehand
    (with a power-law factor or target/reanalysis vertical interpolation). This
    avoids a hidden assumption about boundary-layer stability.
    """
    df: pd.DataFrame  # cols: time, ws, wd

    def __post_init__(self):
        for col in ("time", "ws", "wd"):
            if col not in self.df.columns:
                raise ValueError(f"WindSeries dataframe missing column: {col}")
        self.df = self.df.copy()
        self.df["time"] = pd.to_datetime(self.df["time"])
        self.df = self.df.sort_values("time").reset_index(drop=True)

    @property
    def n_steps(self) -> int:
        return int(len(self.df))

    @property
    def duration_hours(self) -> float:
        if self.n_steps < 2:
            return 0.0
        dt = (self.df["time"].iloc[-1] - self.df["time"].iloc[0]).total_seconds() / 3600
        # Add the trailing step's representative hour
        avg_step = dt / max(self.n_steps - 1, 1)
        return dt + avg_step


@dataclass
class SimulationResult:
    """Output of a simulation run."""
    farm_power_mw: np.ndarray         # shape (n_steps,) - aggregated farm power
    per_turbine_power_mw: np.ndarray  # shape (n_steps, n_turbines)
    times: np.ndarray                 # shape (n_steps,)
    aep_gwh: float                    # extrapolated to one year
    capacity_factor: float            # AEP / (rated * 8760)
    wake_loss_fraction: float         # 1 - (waked AEP / wake-free AEP)
    rated_capacity_mw: float
    n_turbines: int

    def to_dataframe(self) -> pd.DataFrame:
        """Return a per-timestep DataFrame (time, farm_power_mw)."""
        return pd.DataFrame({"time": self.times, "farm_power_mw": self.farm_power_mw})


# ── Public helpers ──────────────────────────────────────────────────────

WAKE_MODEL_NAME = "Bastankhah Gaussian (imposed)"
N_DIRECTION_SECTORS = 12  # 30° each; matches typical PyWake convention


def _charnock_ti(mean_ws_ms: float, a: float = 0.05, b: float = 0.4) -> float:
    """Offshore TI as a function of mean wind speed (Charnock-style).

    TI ≈ a + b/U  - turbulence is roughly constant at high winds and grows
    at low winds (relative to a near-constant absolute sigma). The default
    coefficients (a=0.05, b=0.4) bracket the IEC 61400-3 offshore class.
    """
    if mean_ws_ms < 0.5:
        return a + b / 0.5
    return a + b / mean_ws_ms


def derive_ti_per_sector(
    ws: np.ndarray, wd: np.ndarray, n_sectors: int = N_DIRECTION_SECTORS,
    min_ti: float = 0.03, max_ti: float = 0.18, default_ti: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical TI per 30° sector, derived from the wind series itself.

    The right way to estimate turbulence intensity needs sub-hourly wind data
    (typically 10-minute std / 10-minute mean). reanalysis / target analyses are
    6-hourly / 3-hourly, so std-over-mean of that series captures *seasonal*
    variability, not turbulence. We instead use an **offshore Charnock-style
    proxy** TI(U) = a + b/U applied to the per-sector mean wind speed:
    sectors that are usually windier get a slightly lower TI, in line with
    real offshore measurements.

    Returns (sector_centres_deg, ti_per_sector). Sectors with too few points
    fall back to ``default_ti`` (~6%). Values clipped to [min_ti, max_ti].
    """
    ws = np.asarray(ws, dtype=float)
    wd = np.asarray(wd, dtype=float) % 360
    sector_width = 360 / n_sectors
    centres = np.arange(n_sectors) * sector_width + sector_width / 2
    ti = np.full(n_sectors, default_ti)
    for i, c in enumerate(centres):
        lo = (c - sector_width / 2) % 360
        hi = (c + sector_width / 2) % 360
        if lo < hi:
            mask = (wd >= lo) & (wd < hi)
        else:
            mask = (wd >= lo) | (wd < hi)
        n = int(mask.sum())
        if n >= 30:
            mean_ws = float(np.mean(ws[mask]))
            ti[i] = _charnock_ti(mean_ws)
    return centres, np.clip(ti, min_ti, max_ti)


def _build_xrsite(wind: WindSeries, n_sectors: int = N_DIRECTION_SECTORS) -> XRSite:
    """Build a per-timestep XRSite (1 sample per row) with sector-derived TI.

    The site holds the time series of (ws, wd) plus a TI lookup that depends
    only on direction (per-sector).
    """
    ws = wind.df["ws"].to_numpy()
    wd = wind.df["wd"].to_numpy() % 360

    sector_centres, ti_per_sector = derive_ti_per_sector(ws, wd, n_sectors=n_sectors)
    # Per-step TI = the bin's TI for the step's wd
    bin_idx = np.minimum(
        (wd / (360 / n_sectors)).astype(int), n_sectors - 1
    )
    ti_per_step = ti_per_sector[bin_idx]

    n = wind.n_steps
    ds = xr.Dataset(
        data_vars={
            "WS": (("time",), ws),
            "WD": (("time",), wd),
            "TI": (("time",), ti_per_step),
            "P": (("time",), np.full(n, 1.0 / n)),  # equal weight per step
        },
        coords={"time": np.arange(n)},
    )
    return XRSite(ds=ds)


def _make_wake_model(turbine: WindTurbine, site: XRSite):
    """Return a Bastankhah Gaussian PropagateDownwind model.

    `BastankhahGaussianDeficit` is the imposed wake model for the challenge.
    `PropagateDownwind` is the standard engineering integration scheme in
    PyWake (sequential, linear sum of waked deficits).
    """
    return PropagateDownwind(
        site=site,
        windTurbines=turbine,
        wake_deficitModel=BastankhahGaussianDeficit(),
    )


# ── Simulation entry points ─────────────────────────────────────────────

def simulate(
    layout: FarmLayout, wind: WindSeries,
    annualisation: bool = True,
) -> SimulationResult:
    """Run a wake simulation across the entire wind series.

    Parameters
    ----------
    layout
        Turbine x/y positions (m) and turbine model.
    wind
        Time-series DataFrame at hub height (ws, wd in m/s, deg).
    annualisation
        If True, scale the integrated energy to one year regardless of
        actual series length. Use False when the input is exactly 1 year.

    Returns
    -------
    SimulationResult with per-step farm + per-turbine power and AEP / CF.
    """
    if layout.n_turbines == 0:
        raise ValueError("FarmLayout has zero turbines")
    if wind.n_steps == 0:
        raise ValueError("WindSeries is empty")

    site = _build_xrsite(wind)
    wake = _make_wake_model(layout.turbine, site)

    ws = wind.df["ws"].to_numpy()
    wd = wind.df["wd"].to_numpy() % 360
    # Time-series API: pass ws/wd as 1D arrays of length n_steps with time=True.
    sim = wake(layout.x_m, layout.y_m, ws=ws, wd=wd, time=True)
    # Power_ijk in time mode has shape (n_turbines, n_steps).
    power_w = np.asarray(sim.Power.values)
    while power_w.ndim > 2 and power_w.shape[-1] == 1:
        power_w = power_w[..., 0]
    # Standardise to (n_steps, n_turbines).
    if power_w.shape[0] == layout.n_turbines:
        per_turbine_w = power_w.T
    elif power_w.shape[1] == layout.n_turbines:
        per_turbine_w = power_w
    else:
        raise RuntimeError(
            f"Unexpected Power shape {power_w.shape} for layout with "
            f"{layout.n_turbines} turbines and {wind.n_steps} steps"
        )
    per_turbine_mw = per_turbine_w / 1e6
    farm_mw = per_turbine_mw.sum(axis=1)

    # Wake-free reference: same wind series, single isolated turbine -> N times
    wake_free_per_turbine_mw = _wake_free_power_mw(layout.turbine, wind)
    wake_free_farm_mw = wake_free_per_turbine_mw * layout.n_turbines

    duration_hours = wind.duration_hours
    if duration_hours <= 0:
        duration_hours = wind.n_steps  # assume hourly when the trailing-step trick fails
    energy_mwh = float(np.trapezoid(farm_mw, dx=duration_hours / max(wind.n_steps, 1)))
    energy_wf_mwh = float(np.trapezoid(
        wake_free_farm_mw, dx=duration_hours / max(wind.n_steps, 1),
    ))
    if annualisation and duration_hours > 0:
        scale = 8760.0 / duration_hours
    else:
        scale = 1.0

    aep_gwh = energy_mwh * scale / 1000
    aep_wf_gwh = energy_wf_mwh * scale / 1000
    rated_capacity_mw = layout.n_turbines * (
        layout.turbine.power(np.array([15.0])) / 1e6
    ).item()
    capacity_factor = aep_gwh * 1000 / (rated_capacity_mw * 8760) if rated_capacity_mw > 0 else 0.0
    wake_loss = 1 - aep_gwh / aep_wf_gwh if aep_wf_gwh > 0 else 0.0

    return SimulationResult(
        farm_power_mw=farm_mw,
        per_turbine_power_mw=per_turbine_mw,
        times=wind.df["time"].to_numpy(),
        aep_gwh=float(aep_gwh),
        capacity_factor=float(capacity_factor),
        wake_loss_fraction=float(wake_loss),
        rated_capacity_mw=float(rated_capacity_mw),
        n_turbines=layout.n_turbines,
    )


def simulate_day(layout: FarmLayout, wind: WindSeries) -> SimulationResult:
    """One-day simulation, no annualisation (returns the actual energy).

    Convenience wrapper: validates that the wind window is roughly 24 h then
    calls `simulate(annualisation=False)`.
    """
    duration_h = wind.duration_hours
    if not (12 <= duration_h <= 36):
        warnings.warn(
            f"simulate_day expected ~24 h of wind, got {duration_h:.1f} h. "
            "Use simulate() with annualisation=True if you meant a longer window.",
            stacklevel=2,
        )
    return simulate(layout, wind, annualisation=False)


def simulate_year(layout: FarmLayout, wind: WindSeries) -> SimulationResult:
    """One-year simulation. Annualises automatically if the window != 1 year.

    Convenience wrapper around `simulate(annualisation=True)`.
    """
    return simulate(layout, wind, annualisation=True)


# ── Internal helpers ────────────────────────────────────────────────────

def _wake_free_power_mw(turbine: WindTurbine, wind: WindSeries) -> np.ndarray:
    """Per-step power of a single isolated turbine - no wake, baseline."""
    ws = wind.df["ws"].to_numpy()
    p_w = np.asarray(turbine.power(ws), dtype=float)
    return p_w / 1e6


# ── Layout helpers ──────────────────────────────────────────────────────

def grid_layout(
    n_turbines: int, spacing_d: float, diameter_m: float,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a centred rectangular grid layout (x_m, y_m).

    Parameters
    ----------
    n_turbines
        Number of turbines to place.
    spacing_d
        Rotor-diameter spacing between adjacent turbines in the grid.
    diameter_m
        Rotor diameter, used to convert spacing_d into metres.
    rotation_deg
        Rotation angle of the grid relative to true north.
    """
    spacing_m = spacing_d * diameter_m
    n_rows = int(np.ceil(np.sqrt(n_turbines)))
    n_cols = int(np.ceil(n_turbines / n_rows))
    pts = []
    for r in range(n_rows):
        for c in range(n_cols):
            if len(pts) >= n_turbines:
                break
            pts.append((c * spacing_m, r * spacing_m))
    x = np.array([p[0] for p in pts]) - np.mean([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) - np.mean([p[1] for p in pts])
    if rotation_deg != 0:
        a = np.deg2rad(rotation_deg)
        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
    return x, y


def validate_layout(
    x_m: Iterable[float], y_m: Iterable[float],
    box_size_m: float, max_turbines: int,
    min_spacing_d: float, diameter_m: float,
) -> tuple[bool, list[str]]:
    """Validate a layout against the imposed constraints.

    Constraints (all enforced for Phase 2):
        - All turbines inside the (box_size_m × box_size_m) farm box, centred
          on the origin.
        - Number of turbines ≤ max_turbines.
        - Pairwise spacing ≥ min_spacing_d × diameter_m.

    Returns (is_valid, list_of_error_messages).
    """
    x_m = np.asarray(list(x_m), dtype=float)
    y_m = np.asarray(list(y_m), dtype=float)
    errors: list[str] = []

    if x_m.size != y_m.size:
        errors.append("x_m and y_m must have the same length")
        return False, errors

    if x_m.size > max_turbines:
        errors.append(
            f"{x_m.size} turbines exceeds max_turbines={max_turbines}"
        )

    half = box_size_m / 2
    out_of_box = (np.abs(x_m) > half) | (np.abs(y_m) > half)
    n_out = int(out_of_box.sum())
    if n_out:
        errors.append(
            f"{n_out} turbine(s) outside the {box_size_m / 1000:.1f} x "
            f"{box_size_m / 1000:.1f} km box"
        )

    min_spacing_m = min_spacing_d * diameter_m
    n = x_m.size
    too_close: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.hypot(x_m[i] - x_m[j], y_m[i] - y_m[j]))
            if d < min_spacing_m * (1 - 1e-9):   # tolerate float boundary (exactly 5D)
                too_close.append((i, j, d))
    if too_close:
        i, j, d = too_close[0]
        errors.append(
            f"{len(too_close)} pair(s) too close, e.g. turbines {i}/{j} "
            f"at {d:.0f} m < {min_spacing_m:.0f} m ({min_spacing_d}D)"
        )

    return (len(errors) == 0), errors
