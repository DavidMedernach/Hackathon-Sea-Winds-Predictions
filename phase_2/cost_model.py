"""Wind-farm cost model - CAPEX, OPEX, LCOE.

A simple, auditable engineering cost model for offshore wind farms. The
defaults are derived from public sources (NREL ATB 2024, IRENA 2024 cost
of wind power, BVG Associates) for offshore fixed-bottom monopile
projects in 30-50 m water depth.

The MVP for Phase 2 ranks submissions on AEP first; LCOE is computed and
displayed as a secondary diagnostic. Once the kit matures we can promote
LCOE to the primary axis without changing this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── Default cost parameters (offshore, 2024 €) ──────────────────────────

@dataclass(frozen=True)
class CostParameters:
    """Cost assumptions for a fixed-bottom offshore wind farm."""
    # CAPEX components, in €/kW unless otherwise noted
    turbine_eur_per_kw:      float = 1_100   # turbine + tower, EPC
    foundation_eur_per_kw:   float = 600     # monopile + transition piece
    electrical_eur_per_kw:   float = 350     # inter-array + export cables
    installation_eur_per_kw: float = 350     # vessels, port, weather risk
    development_eur_per_kw:  float = 200     # consenting, surveys, mgmt
    contingency_pct:         float = 0.10    # of CAPEX subtotal

    # Depth scaling for foundation (€ / kW per metre water depth above ref)
    foundation_depth_ref_m:  float = 30
    foundation_depth_slope_eur_per_kw_per_m: float = 15

    # OPEX (O&M + insurance + leases). Maintenance access scales with distance:
    # farther offshore -> longer vessel transit, narrower weather windows, more
    # helicopter use -> higher O&M. Base value is at opex_distance_ref_km.
    opex_eur_per_kw_year:    float = 90      # base O&M at the reference distance
    opex_distance_ref_km:    float = 30      # distance at which the base O&M holds
    opex_distance_slope_eur_per_kw_year_per_km: float = 0.3  # +O&M per km beyond ref

    # Financing
    discount_rate:           float = 0.06    # WACC
    project_lifetime_years:  int   = 25

    # Cable export length (rough proxy: distance from farm to shore)
    distance_to_shore_km:    float = 60      # default Dogger Bank-ish
    export_cable_eur_per_km_per_kw: float = 8

    # Siting eligibility (fixed-bottom / "hard-mounted") thresholds.
    # Depth is the binding constraint; there is NO maximum distance (fixed-bottom
    # farms reach 130+ km offshore where the seabed stays shallow, e.g. Dogger Bank).
    max_fixed_bottom_depth_m: float = 60     # monopile ~0-40 m, jacket ~40-60 m; deeper -> floating
    min_distance_to_shore_km: float = 5.6    # 3 NM coastal-waters boundary (minimal setback)

    @property
    def crf(self) -> float:
        """Capital Recovery Factor used to annualise CAPEX."""
        i = self.discount_rate
        n = self.project_lifetime_years
        return i * (1 + i) ** n / ((1 + i) ** n - 1)


# ── CAPEX / OPEX / LCOE ─────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    """Full cost decomposition of a farm configuration."""
    capacity_mw: float
    n_turbines: int
    water_depth_m: float
    distance_to_shore_km: float
    capex_eur: float
    capex_eur_per_kw: float
    opex_eur_per_year: float
    aep_mwh_per_year: float
    capacity_factor: float
    lcoe_eur_per_mwh: float
    lcoe_components: dict[str, float]      # €/MWh contribution per category

    def summary(self) -> str:
        return (
            f"Capacity:   {self.capacity_mw:.0f} MW ({self.n_turbines} turbines)\n"
            f"CAPEX:      {self.capex_eur/1e6:.0f} M€  ({self.capex_eur_per_kw:.0f} €/kW)\n"
            f"OPEX:       {self.opex_eur_per_year/1e6:.1f} M€/year\n"
            f"AEP:        {self.aep_mwh_per_year/1e3:.0f} GWh/year (CF {self.capacity_factor*100:.1f}%)\n"
            f"LCOE:       {self.lcoe_eur_per_mwh:.1f} €/MWh"
        )


def is_eligible_fixed_bottom(
    water_depth_m: float, distance_to_shore_km: float,
    params: CostParameters | None = None,
) -> tuple[bool, str]:
    """Whether a site can host a fixed-bottom ("hard-mounted") farm.

    A site is eligible iff it is shallow enough for a fixed foundation AND far
    enough from shore. Returns (eligible, reason). Beyond the depth limit a
    floating foundation would be required (out of scope for this fixed-bottom
    cost model); inside the setback the site is excluded for visual/regulatory
    reasons.
    """
    if params is None:
        params = CostParameters()
    if not (water_depth_m > 0):
        return False, "not at sea (depth <= 0)"
    if water_depth_m > params.max_fixed_bottom_depth_m:
        return False, f"too deep for fixed-bottom ({water_depth_m:.0f} m > {params.max_fixed_bottom_depth_m:.0f} m -> floating)"
    if distance_to_shore_km < params.min_distance_to_shore_km:
        return False, f"inside coastal setback ({distance_to_shore_km:.1f} km < {params.min_distance_to_shore_km:.0f} km)"
    return True, "eligible"


def compute_capex(
    capacity_mw: float, water_depth_m: float, distance_to_shore_km: float,
    params: CostParameters | None = None,
) -> tuple[float, dict[str, float]]:
    """Return total CAPEX (€) and per-category breakdown (€)."""
    if params is None:
        params = CostParameters()
    kW = capacity_mw * 1000

    foundation_per_kw = params.foundation_eur_per_kw + max(
        0.0, water_depth_m - params.foundation_depth_ref_m
    ) * params.foundation_depth_slope_eur_per_kw_per_m

    export_cable_per_kw = (
        max(0.0, distance_to_shore_km) * params.export_cable_eur_per_km_per_kw
    )

    components = {
        "turbines":     kW * params.turbine_eur_per_kw,
        "foundation":   kW * foundation_per_kw,
        "electrical":   kW * params.electrical_eur_per_kw,
        "export_cable": kW * export_cable_per_kw,
        "installation": kW * params.installation_eur_per_kw,
        "development":  kW * params.development_eur_per_kw,
    }
    subtotal = sum(components.values())
    components["contingency"] = subtotal * params.contingency_pct
    total = subtotal * (1 + params.contingency_pct)
    return float(total), {k: float(v) for k, v in components.items()}


def compute_opex(capacity_mw: float, distance_to_shore_km: float | None = None,
                 params: CostParameters | None = None) -> float:
    """Return annual OPEX (€/year), including distance-scaled maintenance.

    opex_per_kw = base + slope × max(0, distance − ref). Farther sites cost more
    to maintain (access logistics), so distance is penalised in OPEX as well as
    in the export-cable CAPEX - there is no hard maximum distance, the economics
    just degrade with it.
    """
    if params is None:
        params = CostParameters()
    distance = distance_to_shore_km if distance_to_shore_km is not None else params.distance_to_shore_km
    opex_per_kw = params.opex_eur_per_kw_year + max(
        0.0, distance - params.opex_distance_ref_km
    ) * params.opex_distance_slope_eur_per_kw_year_per_km
    return float(capacity_mw * 1000 * opex_per_kw)


def compute_lcoe(
    aep_mwh_per_year: float, capex_eur: float, opex_eur_per_year: float,
    params: CostParameters | None = None,
) -> tuple[float, dict[str, float]]:
    """Return LCOE (€/MWh) and contribution decomposition.

    LCOE = (CAPEX × CRF + OPEX) / AEP
         = lcoe_capex + lcoe_opex
    """
    if params is None:
        params = CostParameters()
    if aep_mwh_per_year <= 0:
        return float("inf"), {"capex": float("inf"), "opex": float("inf")}
    lcoe_capex = capex_eur * params.crf / aep_mwh_per_year
    lcoe_opex = opex_eur_per_year / aep_mwh_per_year
    return float(lcoe_capex + lcoe_opex), {
        "capex": float(lcoe_capex),
        "opex":  float(lcoe_opex),
    }


def evaluate_farm(
    *, capacity_mw: float, n_turbines: int,
    aep_gwh: float, water_depth_m: float = 30,
    distance_to_shore_km: float | None = None,
    params: CostParameters | None = None,
) -> CostBreakdown:
    """One-call cost evaluation for a farm configuration.

    Parameters
    ----------
    capacity_mw
        Total rated capacity (n_turbines × turbine_rated_mw).
    n_turbines
        Number of turbines.
    aep_gwh
        Annual energy production from the simulator (GWh/year).
    water_depth_m
        Mean water depth at the farm site (used by the foundation model).
    distance_to_shore_km
        If None, falls back to params.distance_to_shore_km.
    """
    if params is None:
        params = CostParameters()
    distance = distance_to_shore_km if distance_to_shore_km is not None else params.distance_to_shore_km

    capex, _comp = compute_capex(capacity_mw, water_depth_m, distance, params)
    opex = compute_opex(capacity_mw, distance, params)
    aep_mwh = aep_gwh * 1000
    lcoe, lcoe_components = compute_lcoe(aep_mwh, capex, opex, params)
    cf = aep_mwh / max(capacity_mw * 8760, 1e-9)
    return CostBreakdown(
        capacity_mw=capacity_mw, n_turbines=n_turbines,
        water_depth_m=water_depth_m, distance_to_shore_km=distance,
        capex_eur=capex, capex_eur_per_kw=capex / max(capacity_mw * 1000, 1),
        opex_eur_per_year=opex,
        aep_mwh_per_year=aep_mwh, capacity_factor=cf,
        lcoe_eur_per_mwh=lcoe, lcoe_components=lcoe_components,
    )
