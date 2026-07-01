"""Phase 2 - Wind Farm Simulator & Optimisation kit.

Exposes the four key public modules so participants can do:

    from phase2 import wind_farm_simulator, turbines_catalog, optimization, bidding, cost_model
"""

from . import bidding, cost_model, optimization, prices, turbines_catalog, wind_farm_simulator

__all__ = [
    "bidding", "cost_model", "optimization", "prices",
    "turbines_catalog", "wind_farm_simulator",
]
