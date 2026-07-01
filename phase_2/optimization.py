"""Wind-farm optimisation strategies for Phase 2.

Four separate optimisations, each operating on a single decision axis with
all other axes held fixed. They share the same objective (AEP first, LCOE
secondary) and the same constraints (max box area, max number of turbines).

    1. optimize_placement   - choose the farm CENTRE inside an allowed sea zone
                              (lat, lon), keeping layout / turbine fixed.
    2. optimize_layout      - choose turbine x/y positions inside the farm box,
                              keeping centre / turbine type fixed.
    3. optimize_turbine     - choose turbine model (catalog), keeping layout
                              positions fixed.
    4. optimize_joint       - run the 3 above sequentially with light coupling
                              (alternating projection on each axis).

All four return a `OptimizationResult` with the best config, the search log,
and a sub-call to the simulator that participants can rerun for diagnostics.

The point is to give participants a working baseline they can beat. They can:
    - swap in their own AEP simulator, kept compatible with `evaluate_config`
    - replace `differential_evolution` with anything more clever
    - plug in their own wind series (e.g. ML predictions instead of reanalysis)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from shear import SHEAR_ALPHA
from turbines_catalog import CATALOG, TurbineSpec, get_spec, load_turbine
from wind_farm_simulator import (
    FarmLayout, WindSeries, SimulationResult,
    grid_layout, simulate_year, validate_layout,
)


# ── Public dataclasses ─────────────────────────────────────────────────

@dataclass
class FarmConfig:
    """A complete farm specification (immutable view)."""
    centre_lat: float
    centre_lon: float
    turbine_key: str
    layout_x_m: np.ndarray
    layout_y_m: np.ndarray

    def n_turbines(self) -> int:
        return int(np.asarray(self.layout_x_m).size)


@dataclass
class OptimizationResult:
    """Outcome of an optimiser call."""
    best_config: FarmConfig
    best_aep_gwh: float
    best_capacity_factor: float
    best_wake_loss: float
    n_evaluations: int
    elapsed_seconds: float
    log: list[dict] = field(default_factory=list)
    sim_result: SimulationResult | None = None
    best_orientation_deg: float | None = None  # winning base orientation, if swept


# ── Wind-series cache for fast repeated calls ──────────────────────────

class WindAtPointCache:
    """Memoised access to a hub-height wind series at any (lat, lon).

    A small wrapper that loads reanalysis (or any DataFrame with columns
    `time, latitude, longitude, ws10, wd10`) once and serves nearest-grid
    time series fast for repeated optimiser calls.

    Parameters
    ----------
    grid_df
        Pre-loaded wind DataFrame at 10 m. Caller does the height correction
        (or passes already-corrected data) by setting alpha=0.
    hub_height_m
        Target hub height (m). Power-law shear is applied with `alpha`.
    alpha
        Wind shear exponent. Offshore North Sea ~0.11 (energy-weighted reanalysis 10->100m; winter peaks ~0.13). Set to 0 to disable.
    """

    def __init__(self, grid_df: pd.DataFrame, hub_height_m: float, alpha: float = SHEAR_ALPHA):
        for col in ("time", "latitude", "longitude", "ws10", "wd10"):
            if col not in grid_df.columns:
                raise ValueError(f"grid_df missing '{col}'")
        self._df = grid_df.copy()
        self._df["time"] = pd.to_datetime(self._df["time"])
        self.hub = float(hub_height_m)
        self.alpha = float(alpha)
        # Pre-compute the unique grid points for fast nearest-neighbour
        self._unique_lat = np.sort(self._df["latitude"].unique())
        self._unique_lon = np.sort(self._df["longitude"].unique())
        self._cache: dict[tuple[float, float], WindSeries] = {}

    def _snap(self, lat: float, lon: float) -> tuple[float, float]:
        nlat = float(self._unique_lat[np.argmin(np.abs(self._unique_lat - lat))])
        nlon = float(self._unique_lon[np.argmin(np.abs(self._unique_lon - lon))])
        return nlat, nlon

    def get(self, lat: float, lon: float) -> WindSeries:
        nlat, nlon = self._snap(lat, lon)
        key = (round(nlat, 4), round(nlon, 4))
        if key in self._cache:
            return self._cache[key]
        sub = self._df[(self._df["latitude"] == nlat) & (self._df["longitude"] == nlon)]
        if sub.empty:
            raise ValueError(f"No data at snapped point ({nlat}, {nlon})")
        ws_hub = sub["ws10"].to_numpy() * (self.hub / 10.0) ** self.alpha
        ws = WindSeries(pd.DataFrame({
            "time": sub["time"].to_numpy(),
            "ws":   ws_hub,
            "wd":   sub["wd10"].to_numpy(),
        }))
        self._cache[key] = ws
        return ws


# ── Core evaluation primitive ──────────────────────────────────────────

def evaluate_config(
    config: FarmConfig, wind_cache: WindAtPointCache,
) -> SimulationResult:
    """Run a 1-year simulation at the given config. Returns the SimulationResult."""
    turbine = load_turbine(config.turbine_key)
    layout = FarmLayout(
        x_m=config.layout_x_m, y_m=config.layout_y_m, turbine=turbine
    )
    wind = wind_cache.get(config.centre_lat, config.centre_lon)
    return simulate_year(layout, wind)


# ── Orientation sweep ──────────────────────────────────────────────────

#: Base layout orientations tried per candidate so a good site is not discarded
#: for a bad default rotation. 0/45/90/135° spans all distinct rectangular-grid
#: orientations (180° is symmetric to 0°), at ~4× simulator cost.
ORIENTATIONS: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)


def rotate_layout(
    x_m: np.ndarray, y_m: np.ndarray, deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a layout (x_m, y_m) about the farm centre by ``deg`` degrees."""
    if deg == 0:
        return np.asarray(x_m, float), np.asarray(y_m, float)
    a = np.deg2rad(deg)
    x = np.asarray(x_m, float)
    y = np.asarray(y_m, float)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def evaluate_best_orientation(
    config: FarmConfig, wind_cache: WindAtPointCache,
    orientations: Iterable[float] = ORIENTATIONS,
) -> tuple[SimulationResult, float]:
    """Evaluate ``config`` at each base orientation; return (best_sim, best_deg).

    The layout positions are rotated about the centre by each orientation and
    the highest-AEP variant is kept. Lets a placement/siting scan reward a site
    whose wind rose favours a rotated grid, instead of judging it on one default.
    """
    best_sim: SimulationResult | None = None
    best_deg = 0.0
    for deg in orientations:
        rx, ry = rotate_layout(config.layout_x_m, config.layout_y_m, deg)
        cfg = FarmConfig(
            centre_lat=config.centre_lat, centre_lon=config.centre_lon,
            turbine_key=config.turbine_key, layout_x_m=rx, layout_y_m=ry,
        )
        sim = evaluate_config(cfg, wind_cache)
        if best_sim is None or sim.aep_gwh > best_sim.aep_gwh:
            best_sim, best_deg = sim, float(deg)
    assert best_sim is not None
    return best_sim, best_deg


# ── Coarse placement scan (brute grid, orientation-aware) ──────────────

def scan_placement_grid(
    *, candidates: Iterable[tuple[float, float]],
    layout_x_m: np.ndarray, layout_y_m: np.ndarray, turbine_key: str,
    wind_cache: WindAtPointCache,
    orientations: Iterable[float] = ORIENTATIONS,
    is_allowed: Callable[[float, float], bool] | None = None,
) -> OptimizationResult:
    """Exhaustively score every candidate centre with a 4-orientation sweep.

    The coarse (reanalysis-resolution) AEP scan of the siting flow: each candidate
    (lat, lon) is evaluated at every base orientation and ranked by best AEP.
    The full ranking is returned in ``log`` (sorted, best first) so the caller
    can downscale + refine the top region(s).
    """
    log: list[dict] = []
    n_eval = [0]
    t0 = time.time()
    best_aep = -np.inf
    best_cfg: FarmConfig | None = None
    best_sim: SimulationResult | None = None
    best_deg = 0.0
    for lat, lon in candidates:
        if is_allowed is not None and not is_allowed(lat, lon):
            continue
        cfg = FarmConfig(
            centre_lat=float(lat), centre_lon=float(lon), turbine_key=turbine_key,
            layout_x_m=np.asarray(layout_x_m), layout_y_m=np.asarray(layout_y_m),
        )
        try:
            sim, deg = evaluate_best_orientation(cfg, wind_cache, orientations)
        except Exception as e:  # noqa: BLE001
            log.append({"lat": float(lat), "lon": float(lon), "error": str(e)})
            continue
        n_eval[0] += 1
        log.append({"lat": float(lat), "lon": float(lon),
                    "aep_gwh": sim.aep_gwh, "orientation_deg": deg,
                    "capacity_factor": sim.capacity_factor})
        if sim.aep_gwh > best_aep:
            best_aep = sim.aep_gwh
            best_deg = deg
            best_sim = sim
            rx, ry = rotate_layout(layout_x_m, layout_y_m, deg)
            best_cfg = FarmConfig(
                centre_lat=float(lat), centre_lon=float(lon),
                turbine_key=turbine_key, layout_x_m=rx, layout_y_m=ry,
            )
    elapsed = time.time() - t0
    if best_cfg is None or best_sim is None:
        raise RuntimeError("No candidate centre could be evaluated")
    log.sort(key=lambda r: r.get("aep_gwh", -np.inf), reverse=True)
    return OptimizationResult(
        best_config=best_cfg, best_aep_gwh=best_sim.aep_gwh,
        best_capacity_factor=best_sim.capacity_factor,
        best_wake_loss=best_sim.wake_loss_fraction,
        n_evaluations=n_eval[0], elapsed_seconds=elapsed,
        log=log, sim_result=best_sim, best_orientation_deg=best_deg,
    )


# ── 1. Placement optimisation ──────────────────────────────────────────

def optimize_placement(
    *, layout_x_m: np.ndarray, layout_y_m: np.ndarray, turbine_key: str,
    wind_cache: WindAtPointCache,
    lat_bounds: tuple[float, float], lon_bounds: tuple[float, float],
    is_allowed: Callable[[float, float], bool] | None = None,
    orientations: Iterable[float] | None = None,
    max_iter: int = 30, seed: int = 0,
) -> OptimizationResult:
    """Find the (lat, lon) farm CENTRE that maximises AEP.

    Parameters
    ----------
    is_allowed
        Optional callable returning True iff the point is in a valid offshore
        zone (e.g. sea, not in a protected area, not too close to shore).
        Invalid points are penalised with `-inf`.
    orientations
        If given (e.g. ``ORIENTATIONS`` = 0/45/90/135°), each candidate centre
        is scored at its best base orientation rather than the layout's default
        rotation - so a site is not discarded for a bad default. ~len× cost.
    """
    log: list[dict] = []
    n_eval = [0]
    best_deg_holder = [0.0]

    def _score(lat, lon):
        cfg = FarmConfig(
            centre_lat=lat, centre_lon=lon, turbine_key=turbine_key,
            layout_x_m=np.asarray(layout_x_m), layout_y_m=np.asarray(layout_y_m),
        )
        if orientations is not None:
            sim, deg = evaluate_best_orientation(cfg, wind_cache, orientations)
            return sim, deg
        return evaluate_config(cfg, wind_cache), 0.0

    def objective(xy):
        lat, lon = float(xy[0]), float(xy[1])
        if is_allowed is not None and not is_allowed(lat, lon):
            return 1e6  # large positive (we minimise)
        try:
            res, deg = _score(lat, lon)
        except Exception as e:
            log.append({"lat": lat, "lon": lon, "error": str(e)})
            return 1e6
        n_eval[0] += 1
        log.append({"lat": lat, "lon": lon, "aep_gwh": res.aep_gwh,
                    "orientation_deg": deg})
        return -res.aep_gwh  # minimise -AEP

    t0 = time.time()
    de = differential_evolution(
        objective, bounds=[lat_bounds, lon_bounds],
        maxiter=max_iter, popsize=12, mutation=(0.5, 1.0), tol=0.01,
        seed=seed, polish=False, init="sobol", workers=1,
    )
    elapsed = time.time() - t0

    best_lat, best_lon = float(de.x[0]), float(de.x[1])
    sim, best_deg = _score(best_lat, best_lon)
    best_deg_holder[0] = best_deg
    rx, ry = rotate_layout(layout_x_m, layout_y_m, best_deg)
    best_cfg = FarmConfig(
        centre_lat=best_lat, centre_lon=best_lon, turbine_key=turbine_key,
        layout_x_m=rx, layout_y_m=ry,
    )

    return OptimizationResult(
        best_config=best_cfg,
        best_aep_gwh=sim.aep_gwh,
        best_capacity_factor=sim.capacity_factor,
        best_wake_loss=sim.wake_loss_fraction,
        n_evaluations=n_eval[0], elapsed_seconds=elapsed,
        log=log, sim_result=sim,
        best_orientation_deg=best_deg_holder[0] if orientations is not None else None,
    )


# ── 2. Layout optimisation (turbine positions inside the farm box) ─────

def optimize_layout(
    *, centre_lat: float, centre_lon: float, turbine_key: str,
    wind_cache: WindAtPointCache,
    n_turbines: int, box_size_m: float, max_turbines: int, min_spacing_d: float,
    spacing_d_bounds: tuple[float, float] = (4.0, 12.0),
    rotation_bounds: tuple[float, float] = (-45.0, 45.0),
    max_iter: int = 30, seed: int = 0,
) -> OptimizationResult:
    """Find the best **regular grid** layout (spacing × rotation).

    A regular grid is the simplest non-trivial layout space; participants are
    free to replace this with arbitrary x/y in their own implementation.
    """
    spec = get_spec(turbine_key)
    log: list[dict] = []
    n_eval = [0]

    def objective(params):
        spacing_d, rotation = float(params[0]), float(params[1])
        x, y = grid_layout(
            n_turbines=n_turbines, spacing_d=spacing_d,
            diameter_m=spec.diameter_m, rotation_deg=rotation,
        )
        ok, _errs = validate_layout(
            x, y, box_size_m=box_size_m, max_turbines=max_turbines,
            min_spacing_d=min_spacing_d, diameter_m=spec.diameter_m,
        )
        if not ok:
            return 1e6
        cfg = FarmConfig(
            centre_lat=centre_lat, centre_lon=centre_lon,
            turbine_key=turbine_key, layout_x_m=x, layout_y_m=y,
        )
        try:
            res = evaluate_config(cfg, wind_cache)
        except Exception:
            return 1e6
        n_eval[0] += 1
        log.append({
            "spacing_d": spacing_d, "rotation": rotation,
            "aep_gwh": res.aep_gwh, "wake_loss": res.wake_loss_fraction,
        })
        return -res.aep_gwh

    t0 = time.time()
    de = differential_evolution(
        objective, bounds=[spacing_d_bounds, rotation_bounds],
        maxiter=max_iter, popsize=10, seed=seed, polish=True, init="sobol",
    )
    elapsed = time.time() - t0

    best_spacing, best_rot = float(de.x[0]), float(de.x[1])
    best_x, best_y = grid_layout(
        n_turbines=n_turbines, spacing_d=best_spacing,
        diameter_m=spec.diameter_m, rotation_deg=best_rot,
    )
    best_cfg = FarmConfig(
        centre_lat=centre_lat, centre_lon=centre_lon,
        turbine_key=turbine_key, layout_x_m=best_x, layout_y_m=best_y,
    )
    sim = evaluate_config(best_cfg, wind_cache)

    return OptimizationResult(
        best_config=best_cfg,
        best_aep_gwh=sim.aep_gwh,
        best_capacity_factor=sim.capacity_factor,
        best_wake_loss=sim.wake_loss_fraction,
        n_evaluations=n_eval[0], elapsed_seconds=elapsed,
        log=log, sim_result=sim,
    )


# ── 3. Turbine type optimisation (categorical) ─────────────────────────

def optimize_turbine(
    *, centre_lat: float, centre_lon: float,
    layout_x_m: np.ndarray, layout_y_m: np.ndarray,
    wind_cache: WindAtPointCache,
    candidate_keys: Iterable[str] | None = None,
) -> OptimizationResult:
    """Pick the best turbine model from the catalog at a fixed layout.

    Layout is taken as-is: only the turbine model varies. Categorical search,
    so we just enumerate. The result reports per-candidate AEP/CF as the log.
    """
    if candidate_keys is None:
        candidate_keys = list(CATALOG)
    log: list[dict] = []
    t0 = time.time()
    best_aep = -np.inf
    best_cfg: FarmConfig | None = None
    best_sim: SimulationResult | None = None

    for key in candidate_keys:
        cfg = FarmConfig(
            centre_lat=centre_lat, centre_lon=centre_lon, turbine_key=key,
            layout_x_m=np.asarray(layout_x_m), layout_y_m=np.asarray(layout_y_m),
        )
        try:
            res = evaluate_config(cfg, wind_cache)
        except Exception as e:
            log.append({"turbine": key, "error": str(e)})
            continue
        log.append({
            "turbine": key, "rated_mw": CATALOG[key].rated_power_mw,
            "aep_gwh": res.aep_gwh, "capacity_factor": res.capacity_factor,
            "wake_loss": res.wake_loss_fraction,
        })
        if res.aep_gwh > best_aep:
            best_aep, best_cfg, best_sim = res.aep_gwh, cfg, res

    elapsed = time.time() - t0
    if best_cfg is None or best_sim is None:
        raise RuntimeError("No candidate turbine could be evaluated")

    return OptimizationResult(
        best_config=best_cfg,
        best_aep_gwh=best_sim.aep_gwh,
        best_capacity_factor=best_sim.capacity_factor,
        best_wake_loss=best_sim.wake_loss_fraction,
        n_evaluations=len(log), elapsed_seconds=elapsed,
        log=log, sim_result=best_sim,
    )


# ── 4. Joint optimisation (alternating coordinate descent) ─────────────

def optimize_joint(
    *,
    initial_centre_lat: float, initial_centre_lon: float,
    initial_turbine_key: str, n_turbines: int,
    wind_cache: WindAtPointCache,
    lat_bounds: tuple[float, float], lon_bounds: tuple[float, float],
    box_size_m: float, max_turbines: int, min_spacing_d: float,
    is_allowed: Callable[[float, float], bool] | None = None,
    fix_turbine: bool = False,
    n_outer_iters: int = 2, max_iter_per_axis: int = 20, seed: int = 0,
) -> OptimizationResult:
    """Combine the three single-axis optimisers in a coordinate-descent loop.

    Each outer iteration:
        1. Optimise layout for the current centre + turbine
        2. Optimise turbine type for the current centre + layout
        3. Optimise centre for the current turbine + layout

    Two outer iterations is usually enough to converge on a good local
    optimum given the smoothness of AEP in (centre, layout, turbine).
    Participants can replace this with anything (Bayesian optim, GA, RL).
    """
    spec = get_spec(initial_turbine_key)
    centre_lat, centre_lon = initial_centre_lat, initial_centre_lon
    turbine_key = initial_turbine_key

    # Initial layout = simple 7D grid
    x, y = grid_layout(
        n_turbines, spacing_d=7, diameter_m=spec.diameter_m, rotation_deg=0,
    )

    log: list[dict] = []
    t0 = time.time()
    best_sim: SimulationResult | None = None
    best_cfg: FarmConfig | None = None

    for it in range(n_outer_iters):
        # 1. Layout
        spec = get_spec(turbine_key)
        layout_res = optimize_layout(
            centre_lat=centre_lat, centre_lon=centre_lon, turbine_key=turbine_key,
            wind_cache=wind_cache,
            n_turbines=n_turbines, box_size_m=box_size_m, max_turbines=max_turbines,
            min_spacing_d=min_spacing_d, max_iter=max_iter_per_axis, seed=seed + it,
        )
        x = layout_res.best_config.layout_x_m
        y = layout_res.best_config.layout_y_m
        log.append({"iter": it, "step": "layout", "aep_gwh": layout_res.best_aep_gwh})

        # 2. Turbine (skipped when the turbine is fixed by the rules, e.g. Phase 2)
        if not fix_turbine:
            turb_res = optimize_turbine(
                centre_lat=centre_lat, centre_lon=centre_lon,
                layout_x_m=x, layout_y_m=y, wind_cache=wind_cache,
            )
            if turb_res.best_aep_gwh > layout_res.best_aep_gwh:
                turbine_key = turb_res.best_config.turbine_key
            log.append({"iter": it, "step": "turbine", "key": turbine_key,
                        "aep_gwh": turb_res.best_aep_gwh})

        # 3. Centre placement
        place_res = optimize_placement(
            layout_x_m=x, layout_y_m=y, turbine_key=turbine_key,
            wind_cache=wind_cache,
            lat_bounds=lat_bounds, lon_bounds=lon_bounds,
            is_allowed=is_allowed, max_iter=max_iter_per_axis, seed=seed + 100 + it,
        )
        centre_lat = place_res.best_config.centre_lat
        centre_lon = place_res.best_config.centre_lon
        best_sim = place_res.sim_result
        best_cfg = place_res.best_config
        log.append({"iter": it, "step": "centre", "lat": centre_lat,
                    "lon": centre_lon, "aep_gwh": place_res.best_aep_gwh})

    elapsed = time.time() - t0
    assert best_cfg is not None and best_sim is not None
    return OptimizationResult(
        best_config=best_cfg,
        best_aep_gwh=best_sim.aep_gwh,
        best_capacity_factor=best_sim.capacity_factor,
        best_wake_loss=best_sim.wake_loss_fraction,
        n_evaluations=sum(
            1 for entry in log if "aep_gwh" in entry
        ),
        elapsed_seconds=elapsed,
        log=log, sim_result=best_sim,
    )


# ── Submission export ──────────────────────────────────────────────────

def export_submission(config: "FarmConfig", path, team: str = "team") -> None:
    """Write a scorer-format submission.json for a FarmConfig."""
    import json
    from pathlib import Path
    sub = {
        "team": team,
        "farm_centre_lat": float(config.centre_lat),
        "farm_centre_lon": float(config.centre_lon),
        "turbine_key": config.turbine_key,
        "layout_x_m": [float(v) for v in np.asarray(config.layout_x_m).ravel()],
        "layout_y_m": [float(v) for v in np.asarray(config.layout_y_m).ravel()],
    }
    Path(path).write_text(json.dumps(sub, indent=2))


# ── Individual placement + Quality-Diversity (MAP-Elites) ──────────────────
#
# The grid optimisers above move a *rigid* grid (spacing × rotation). At fine
# (target) resolution it pays to place turbines **individually** - and instead of
# a single optimum, illuminate a whole archive of layouts with **MAP-Elites**:
# each cell of a behaviour space (here: mean nearest-neighbour spacing × layout
# elongation) keeps its best-AEP layout. PyWake is ~7 ms/sim, so thousands of
# evaluations cost seconds.

def _layout_descriptors(x, y, diameter_m):
    """(mean nearest-neighbour spacing in D, layout elongation) behaviour pair."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy); np.fill_diagonal(dist, np.inf)
    nn = dist.min(axis=1).mean() / diameter_m
    pts = np.stack([x - x.mean(), y - y.mean()])
    ev = np.linalg.eigvalsh(np.cov(pts) + 1e-9 * np.eye(2))
    elong = float(np.sqrt(ev.max() / max(ev.min(), 1e-9)))
    return nn, elong


def _valid_layout(x, y, half_m, min_spacing_m):
    if np.any(np.abs(x) > half_m) or np.any(np.abs(y) > half_m):
        return False
    n = x.size
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    dist = np.hypot(dx, dy); np.fill_diagonal(dist, np.inf)
    return bool(dist.min() >= min_spacing_m)


@dataclass
class QDArchive:
    """MAP-Elites result: the best layout per behaviour cell, plus the elite."""
    cells: dict                       # (i, j) -> {"aep","x","y","bd","sim"}
    bd1_edges: np.ndarray             # mean-NN-spacing (D) bin edges
    bd2_edges: np.ndarray             # elongation bin edges
    best: OptimizationResult
    n_evaluations: int
    elapsed_seconds: float

    def heatmap(self):
        """2-D array of best AEP per cell (NaN where empty)."""
        h = np.full((self.bd1_edges.size - 1, self.bd2_edges.size - 1), np.nan)
        for (i, j), c in self.cells.items():
            h[i, j] = c["aep"]
        return h


def optimize_layout_qd(
    *, centre_lat: float, centre_lon: float, turbine_key: str,
    wind_cache: WindAtPointCache,
    n_turbines: int, box_size_m: float, min_spacing_d: float,
    bd1_bounds: tuple[float, float] = (5.0, 16.0),
    bd2_bounds: tuple[float, float] = (1.0, 4.0),
    n_bins: int = 12, n_init: int = 120, n_iters: int = 3000,
    sigma_d: float = 0.8, seed: int = 0,
) -> QDArchive:
    """MAP-Elites over **individual** turbine positions at a fixed centre.

    Behaviour space = (mean nearest-neighbour spacing in D, layout elongation);
    fitness = AEP. Returns the full archive (for illumination) and the single
    best layout as an ``OptimizationResult`` - a drop-in for the grid optimisers.
    """
    spec = get_spec(turbine_key)
    D = spec.diameter_m; half = box_size_m / 2; min_s = min_spacing_d * D
    rng = np.random.default_rng(seed)
    bd1_edges = np.linspace(*bd1_bounds, n_bins + 1)
    bd2_edges = np.linspace(*bd2_bounds, n_bins + 1)

    def _cell(bd1, bd2):
        i = int(np.clip(np.searchsorted(bd1_edges, bd1) - 1, 0, n_bins - 1))
        j = int(np.clip(np.searchsorted(bd2_edges, bd2) - 1, 0, n_bins - 1))
        return i, j

    def _eval(x, y):
        cfg = FarmConfig(centre_lat, centre_lon, turbine_key, x, y)
        return evaluate_config(cfg, wind_cache)

    def _random_layout():
        # jittered grid at a random spacing/rotation - a cheap source of valid seeds
        sd = rng.uniform(min_spacing_d, min_spacing_d + 4.0)
        x, y = grid_layout(n_turbines, sd, D, rotation_deg=rng.uniform(0, 90))
        x = x + rng.normal(0, 0.3 * D, x.size); y = y + rng.normal(0, 0.3 * D, y.size)
        return x, y

    cells: dict = {}
    n_eval = [0]

    def _try(x, y):
        if not _valid_layout(x, y, half, min_s):
            return
        res = _eval(x, y); n_eval[0] += 1
        bd1, bd2 = _layout_descriptors(x, y, D)
        key = _cell(bd1, bd2)
        cur = cells.get(key)
        if cur is None or res.aep_gwh > cur["aep"]:
            cells[key] = {"aep": res.aep_gwh, "x": x.copy(), "y": y.copy(),
                          "bd": (bd1, bd2), "sim": res}

    t0 = time.time()
    tries = 0
    while len(cells) < n_init and tries < n_init * 50:
        x, y = _random_layout(); _try(x, y); tries += 1
    for _ in range(n_iters):
        if not cells:
            break
        parent = cells[list(cells)[rng.integers(len(cells))]]
        x = parent["x"] + rng.normal(0, sigma_d * D, n_turbines)
        y = parent["y"] + rng.normal(0, sigma_d * D, n_turbines)
        np.clip(x, -half, half, out=x); np.clip(y, -half, half, out=y)
        _try(x, y)
    elapsed = time.time() - t0

    if not cells:
        raise RuntimeError(
            f"optimize_layout_qd: no valid layout found - the {box_size_m/1000:.0f} km "
            f"box is likely too small for {n_turbines} turbines at {min_spacing_d}D "
            f"spacing ({get_spec(turbine_key).diameter_m:.0f} m rotor).")
    best = max(cells.values(), key=lambda c: c["aep"])
    best_cfg = FarmConfig(centre_lat, centre_lon, turbine_key, best["x"], best["y"])
    best_res = OptimizationResult(
        best_config=best_cfg, best_aep_gwh=best["sim"].aep_gwh,
        best_capacity_factor=best["sim"].capacity_factor,
        best_wake_loss=best["sim"].wake_loss_fraction,
        n_evaluations=n_eval[0], elapsed_seconds=elapsed,
        log=[{"bd": c["bd"], "aep": c["aep"]} for c in cells.values()],
        sim_result=best["sim"])
    return QDArchive(cells=cells, bd1_edges=bd1_edges, bd2_edges=bd2_edges,
                     best=best_res, n_evaluations=n_eval[0], elapsed_seconds=elapsed)


# ── Fair turbine comparison at FIXED installed capacity ────────────────────

def compare_turbines_at_capacity(
    *, centre_lat: float, centre_lon: float, wind_cache: WindAtPointCache,
    target_capacity_mw: float, candidate_keys: Iterable[str] | None = None,
    box_size_m: float = 15000.0, max_turbines: int = 30,
    min_spacing_d: float = 5.0, spacing_d: float = 5.0,
) -> dict:
    """Compare turbine types at a **fixed installed capacity** (not fixed count).

    For each type the turbine **count** is sized so ``n × rated ≈ target`` (capped
    at ``max_turbines`` and the box / spacing constraints). This removes the
    trivial bias where a bigger turbine simply means more MW and therefore more
    AEP - the comparison then reflects **capacity factor, wake losses and LCOE**
    at equal installed power. Returns ``{"rows", "best", "target_capacity_mw"}``.
    """
    from cost_model import evaluate_farm
    if candidate_keys is None:
        candidate_keys = list(CATALOG)
    rows, best = [], None
    for key in candidate_keys:
        spec = CATALOG[key]
        n = int(round(target_capacity_mw / spec.rated_power_mw))
        n = max(1, min(max_turbines, n))
        cap = n * spec.rated_power_mw
        x, y = grid_layout(n, spacing_d, spec.diameter_m)
        ok, errs = validate_layout(
            x, y, box_size_m=box_size_m, max_turbines=max_turbines,
            min_spacing_d=min_spacing_d, diameter_m=spec.diameter_m)
        if not ok:
            rows.append({"turbine": key, "n_turbines": n, "capacity_mw": cap,
                         "valid": False, "error": errs[0] if errs else "invalid"})
            continue
        sim = evaluate_config(
            FarmConfig(centre_lat, centre_lon, key, x, y), wind_cache)
        cb = evaluate_farm(capacity_mw=cap, n_turbines=n, aep_gwh=sim.aep_gwh)
        row = {"turbine": key, "n_turbines": n, "capacity_mw": cap,
               "aep_gwh": sim.aep_gwh, "capacity_factor": sim.capacity_factor,
               "wake_loss": sim.wake_loss_fraction,
               "lcoe_eur_mwh": cb.lcoe_eur_per_mwh, "valid": True,
               # can this type actually deliver the target (within one turbine)?
               "reaches_target": abs(cap - target_capacity_mw) <= spec.rated_power_mw}
        rows.append(row)
    # Pick the best among turbines that ACTUALLY REACH the target capacity, so AEP
    # is compared at ~equal MW (i.e. on capacity factor / wake - not "bigger wins").
    # A type that caps out below target (max_turbines × rated < target) cannot deliver
    # the requested capacity and is not eligible; fall back only if none reach it.
    valid = [r for r in rows if r.get("valid")]
    reaching = [r for r in valid if r["reaches_target"]]
    pool = reaching or valid
    best = max(pool, key=lambda r: r["aep_gwh"]) if pool else None
    return {"rows": rows, "best": best, "target_capacity_mw": float(target_capacity_mw)}


# ── Joint siting at FIXED installed capacity ────────────────────────────────

def optimize_joint_fixed_capacity(
    *, target_capacity_mw: float,
    initial_centre_lat: float, initial_centre_lon: float,
    wind_cache: WindAtPointCache,
    lat_bounds: tuple[float, float], lon_bounds: tuple[float, float],
    box_size_m: float = 15000.0, max_turbines: int = 30, min_spacing_d: float = 5.0,
    candidate_keys: Iterable[str] | None = None,
    is_allowed: Callable[[float, float], bool] | None = None,
    orientations: Iterable[float] | None = ORIENTATIONS,
    n_outer_iters: int = 2, qd_iters: int = 2500,
    max_iter_per_axis: int = 12, seed: int = 0,
) -> OptimizationResult:
    """Joint siting at a **fixed installed capacity** rather than a fixed count.

    The turbine **count is derived** from ``target_capacity_mw / rated`` per type,
    so choosing a turbine reflects capacity factor / wake / LCOE at equal MW -
    not the trivial "biggest turbine = more MW = more AEP". Cascade per outer
    iteration: choose turbine type at fixed capacity → place the centre
    (orientation sweep) → individual placement (MAP-Elites QD) of the derived
    count. Returns an :class:`OptimizationResult`; ``log`` records the per-iter
    turbine / count / capacity / AEP so the capacity is auditable.
    """
    t0 = time.time()
    clat, clon = float(initial_centre_lat), float(initial_centre_lon)

    def _choose(lat, lon):
        cmp = compare_turbines_at_capacity(
            centre_lat=lat, centre_lon=lon, wind_cache=wind_cache,
            target_capacity_mw=target_capacity_mw, candidate_keys=candidate_keys,
            box_size_m=box_size_m, max_turbines=max_turbines,
            min_spacing_d=min_spacing_d, spacing_d=min_spacing_d)
        b = cmp["best"]
        if b is None:
            raise RuntimeError("optimize_joint_fixed_capacity: no feasible turbine "
                               f"at {target_capacity_mw} MW in the box")
        return b["turbine"], b["n_turbines"]

    key, n = _choose(clat, clon)
    log, best = [], None
    for it in range(n_outer_iters):
        spec = get_spec(key)
        # 1. place the CENTRE (orientation sweep) using a grid of the current count
        gx, gy = grid_layout(n, min_spacing_d, spec.diameter_m)
        place = optimize_placement(
            layout_x_m=gx, layout_y_m=gy, turbine_key=key, wind_cache=wind_cache,
            lat_bounds=lat_bounds, lon_bounds=lon_bounds, is_allowed=is_allowed,
            orientations=orientations, max_iter=max_iter_per_axis, seed=seed + it)
        clat, clon = place.best_config.centre_lat, place.best_config.centre_lon
        # 2. re-choose the turbine TYPE at the new centre (capacity still fixed)
        key, n = _choose(clat, clon)
        spec = get_spec(key)
        # 3. INDIVIDUAL placement (quality-diversity) of the derived count
        qd = optimize_layout_qd(
            centre_lat=clat, centre_lon=clon, turbine_key=key, wind_cache=wind_cache,
            n_turbines=n, box_size_m=box_size_m, min_spacing_d=min_spacing_d,
            n_iters=qd_iters, seed=seed + it)
        best = qd.best
        log.append({"iter": it, "turbine": key, "n_turbines": n,
                    "capacity_mw": n * spec.rated_power_mw,
                    "centre_lat": clat, "centre_lon": clon,
                    "aep_gwh": best.best_aep_gwh,
                    "capacity_factor": best.best_capacity_factor})

    assert best is not None
    return OptimizationResult(
        best_config=best.best_config, best_aep_gwh=best.best_aep_gwh,
        best_capacity_factor=best.best_capacity_factor,
        best_wake_loss=best.best_wake_loss, n_evaluations=best.n_evaluations,
        elapsed_seconds=time.time() - t0, log=log, sim_result=best.sim_result,
        best_orientation_deg=best.best_orientation_deg)
