"""Single source of truth for the offshore power-law wind-shear exponent.

α was MEASURED from reanalysis (10 m → 100 m, North-Sea zone): the energy-weighted
(AEP-relevant) value is ≈0.11 (median ~0.10), strongly seasonal (DJF 0.13,
SON 0.06). We use the annual energy-weighted value everywhere so the optimiser,
the synthetic-wind cache, the Pangu driver and the official scorer never drift.
"""
from __future__ import annotations

#: Offshore power-law shear exponent (North Sea, annual energy-weighted reanalysis).
SHEAR_ALPHA: float = 0.11


def power_law_factor(h_from_m: float, h_to_m: float, alpha: float = SHEAR_ALPHA) -> float:
    """Multiplicative speed factor for the power-law profile h_from → h_to."""
    return (float(h_to_m) / float(h_from_m)) ** float(alpha)
