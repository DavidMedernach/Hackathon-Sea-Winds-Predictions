"""Turbine catalog for Phase 2 wind farm optimisation.

Wraps four reference turbines suitable for offshore use, with real
power and Ct curves loaded from CSV when available. Each entry resolves
to a `py_wake.wind_turbines.WindTurbine` ready to plug into a
WindFarmModel.

Reference turbines:
    - IEA 3.4 MW    (130 m rotor, 110 m HH)  - small, mature, low LCOE
    - IEA 15 MW     (240 m rotor, 150 m HH)  - current offshore standard
    - IEA 22 MW     (284 m rotor, 170 m HH)  - next-generation prototype
    - NREL 5 MW     (126 m rotor, 90 m HH)   - widely-used research baseline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
_TURBINES_DIR = _PROJECT_ROOT / "data" / "wind_data" / "turbines"


@dataclass(frozen=True)
class TurbineSpec:
    """Static specification of a turbine model."""
    key: str
    name: str
    diameter_m: float
    hub_height_m: float
    rated_power_mw: float
    cut_in_ms: float
    cut_out_ms: float
    csv_filename: str | None  # None = use generic curve fallback

    @property
    def specific_power_w_per_m2(self) -> float:
        """Rated power per swept area, W/m². Higher = more energy per area."""
        area = np.pi * (self.diameter_m / 2) ** 2
        return self.rated_power_mw * 1e6 / area


CATALOG: dict[str, TurbineSpec] = {
    "IEA_3.4MW": TurbineSpec(
        key="IEA_3.4MW", name="IEA 3.4 MW",
        diameter_m=130, hub_height_m=110, rated_power_mw=3.4,
        cut_in_ms=3.0, cut_out_ms=25.0, csv_filename="iea_3.4mw_power_ct.csv",
    ),
    "NREL_5MW": TurbineSpec(
        key="NREL_5MW", name="NREL 5 MW",
        diameter_m=126, hub_height_m=90, rated_power_mw=5.0,
        cut_in_ms=3.0, cut_out_ms=25.0, csv_filename=None,  # use generic
    ),
    "IEA_15MW": TurbineSpec(
        key="IEA_15MW", name="IEA 15 MW",
        diameter_m=240, hub_height_m=150, rated_power_mw=15.0,
        cut_in_ms=3.0, cut_out_ms=25.0, csv_filename="iea_15mw_power_ct.csv",
    ),
    "IEA_22MW": TurbineSpec(
        key="IEA_22MW", name="IEA 22 MW",
        diameter_m=284, hub_height_m=170, rated_power_mw=22.0,
        cut_in_ms=3.0, cut_out_ms=25.0, csv_filename="iea_22mw_power_ct.csv",
    ),
}


def list_turbines() -> Iterable[TurbineSpec]:
    """Iterate over all available turbine specs."""
    return CATALOG.values()


def get_spec(key: str) -> TurbineSpec:
    """Return the TurbineSpec for `key`. Raises KeyError if unknown."""
    if key not in CATALOG:
        raise KeyError(
            f"Unknown turbine '{key}'. Available: {sorted(CATALOG)}"
        )
    return CATALOG[key]


def _generic_power_ct(spec: TurbineSpec) -> PowerCtTabular:
    """Generic cubic ramp + flat rated curve when no CSV is available.

    Used as a last-resort fallback (NREL 5 MW or any spec without CSV).
    Cubic between cut-in and rated wind speed (~12 m/s), flat at rated power
    until cut-out, then 0.
    """
    rated_ws = 12.0
    ws = np.concatenate([
        np.linspace(0, spec.cut_in_ms, 5, endpoint=False),
        np.linspace(spec.cut_in_ms, rated_ws, 25),
        np.linspace(rated_ws, spec.cut_out_ms, 15),
        np.linspace(spec.cut_out_ms, spec.cut_out_ms + 1, 3),
    ])
    power = np.zeros_like(ws)
    ct = np.zeros_like(ws)
    rated_w = spec.rated_power_mw * 1e6
    for i, w in enumerate(ws):
        if w < spec.cut_in_ms or w > spec.cut_out_ms:
            power[i], ct[i] = 0.0, 0.0
        elif w < rated_ws:
            r = (w - spec.cut_in_ms) / (rated_ws - spec.cut_in_ms)
            power[i] = rated_w * r ** 3
            # Crude Ct: max 0.85 below rated, falls toward 0.2 at cut-out
            ct[i] = min(0.85, 4.0 * r * (1 - r) + 0.55)
        else:
            power[i] = rated_w
            # Ct decays after rated (pitch regulation)
            r = (w - rated_ws) / (spec.cut_out_ms - rated_ws)
            ct[i] = max(0.2, 0.55 * (1 - r))
    return PowerCtTabular(
        ws=ws, power=power, power_unit="W", ct=ct,
        ws_cutin=spec.cut_in_ms, ws_cutout=spec.cut_out_ms,
        power_idle=0, ct_idle=0, method="linear",
    )


def _csv_power_ct(spec: TurbineSpec) -> PowerCtTabular:
    """Load real power/Ct curve from the project's CSV files."""
    csv_path = _TURBINES_DIR / spec.csv_filename
    if not csv_path.exists():
        # Caller already filtered; this is defensive.
        raise FileNotFoundError(f"Power/Ct CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    ws = df["wind_speed_ms"].to_numpy()
    power = df["power_w"].to_numpy()
    ct = df["ct"].to_numpy()
    order = np.argsort(ws)
    ws, power, ct = ws[order], power[order], ct[order]
    mask = (ws > 0) & (power >= 0)  # keep boundary points (power=0 at cut-in/out)
    return PowerCtTabular(
        ws=ws[mask], power=power[mask], power_unit="W", ct=ct[mask],
        ws_cutin=spec.cut_in_ms, ws_cutout=spec.cut_out_ms,
        power_idle=0, ct_idle=0, method="linear",
    )


def load_turbine(key: str) -> WindTurbine:
    """Return a PyWake WindTurbine for the given catalog key.

    Uses the real CSV power/Ct curve when available, otherwise a generic
    cubic-ramp curve consistent with the rated power and cut-in/out.
    """
    spec = get_spec(key)
    if spec.csv_filename and (_TURBINES_DIR / spec.csv_filename).exists():
        power_ct = _csv_power_ct(spec)
    else:
        power_ct = _generic_power_ct(spec)
    return WindTurbine(
        name=spec.name,
        diameter=spec.diameter_m,
        hub_height=spec.hub_height_m,
        powerCtFunction=power_ct,
    )


def summary_table() -> pd.DataFrame:
    """Return a DataFrame summarising every turbine in the catalog."""
    rows = []
    for spec in CATALOG.values():
        rows.append({
            "key": spec.key,
            "name": spec.name,
            "rated_power_MW": spec.rated_power_mw,
            "diameter_m": spec.diameter_m,
            "hub_height_m": spec.hub_height_m,
            "specific_power_W/m2": round(spec.specific_power_w_per_m2, 1),
            "cut_in_m/s": spec.cut_in_ms,
            "cut_out_m/s": spec.cut_out_ms,
            "real_curve": (
                spec.csv_filename is not None
                and (_TURBINES_DIR / spec.csv_filename).exists()
            ),
        })
    return pd.DataFrame(rows).sort_values("rated_power_MW").reset_index(drop=True)
