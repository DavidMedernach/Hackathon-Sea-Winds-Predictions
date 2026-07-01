"""Bridge P3 synthetic years to the Part 4 optimiser.

``SynthWindCache`` duck-types ``optimization.WindAtPointCache``: it exposes
``.get(lat, lon) -> WindSeries`` so it can be passed straight to
``evaluate_config`` / ``optimize_*``. Wind comes from a synthetic year via
``synthetic_generator.to_site_series`` (125 m), sheared to the hub height.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent          # part2_siting
for _p in (_HERE, _HERE.parent):                 # part2_siting (synthetic_generator) + kit root (shared modules)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shear import SHEAR_ALPHA                     # noqa: E402
from wind_farm_simulator import WindSeries        # noqa: E402
from synthetic_generator import to_site_series    # noqa: E402


class SynthWindCache:
    """Serve hub-height WindSeries at any (lat, lon) from a synthetic year."""

    def __init__(self, synth, hub_height_m: float = 150.0,
                 ref_height_m: float = 125.0, alpha: float = SHEAR_ALPHA):
        self.synth = synth
        self.hub = float(hub_height_m)
        self.ref = float(ref_height_m)
        self.alpha = float(alpha)
        self._cache: dict[tuple[float, float], WindSeries] = {}

    def get(self, lat: float, lon: float) -> WindSeries:
        key = (round(float(lat), 3), round(float(lon), 3))
        if key in self._cache:
            return self._cache[key]
        df = to_site_series(self.synth, lat, lon)
        ws_hub = df["ws"].to_numpy() * (self.hub / self.ref) ** self.alpha
        ws = WindSeries(pd.DataFrame({"time": df["time"].to_numpy(),
                                      "ws": ws_hub.astype(np.float32),
                                      "wd": df["wd"].to_numpy()}))
        self._cache[key] = ws
        return ws
