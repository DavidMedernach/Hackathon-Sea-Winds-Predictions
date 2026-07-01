"""Phase 2 - water depth & siting eligibility (bathymetry).

Two data backends, tried in order:

1. **EMODnet eligibility layer** (preferred) - the precomputed grid built by
   ``internal/data_generation/build_phase2_bathymetry.py`` from the EMODnet
   Bathymetry DTM 2022 (~115 m, resampled to ~1 km over the North-Sea bbox).
   It carries ``water_depth_m``, ``dist_coast_km`` and a boolean ``eligible``
   layer, so depth **and** distance-to-coast are both available.

2. **Raw GEBCO/ETOPO NetCDF** (fallback) - an ``elevation`` grid (m, negative
   below sea level) dropped in ``data/wind_data/bathymetry/``. Depth only,
   no distance-to-coast.

Eligibility (fixed-bottom / "hard-mounted") uses the agreed thresholds:

    eligible = sea  AND  water_depth <= 60 m  AND  dist_to_coast >= 10 km

Monopiles/jackets are practically limited to ~60 m; deeper sites need
**floating** foundations (out of scope). The 10 km coastal setback is a
visual/regulatory minimum. All public functions degrade gracefully when no
data is present (``available()`` is False, depth NaN, ``is_eligible`` True so
the constraint stays inactive) - importing this module never breaks anything.
"""
from __future__ import annotations

import glob
from functools import lru_cache
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3]
BATHY_DIR = ROOT / "data" / "wind_data" / "bathymetry"               # GEBCO fallback
EMODNET_CANDIDATES = [                                               # preferred layer
    ROOT / "build" / "phase2_dataset" / "bathymetry" / "emodnet_northsea_1km.nc",
    _HERE.parents[1] / "data" / "bathymetry" / "emodnet_northsea_1km.nc",
]

#: Fixed-bottom siting thresholds (agreed Phase 2 values).
#: Depth is the binding constraint; NO maximum distance (fixed-bottom reaches
#: 130+ km where the seabed stays shallow, e.g. Dogger Bank).
DEFAULT_MAX_DEPTH_M = 60.0   # monopile ~0-40 m, jacket ~40-60 m; deeper -> floating
DEFAULT_MIN_DIST_KM = 5.6    # 3 NM coastal-waters boundary (minimal setback)

# GEBCO region clip (so even a global grid loads only the North-Sea window).
_REGION_LAT = (49.0, 63.0)
_REGION_LON = (-6.0, 12.0)


# ── EMODnet precomputed layer (preferred) ────────────────────────────────

@lru_cache(maxsize=1)
def _load_emodnet():
    """Return the EMODnet eligibility xarray Dataset, or None."""
    for c in EMODNET_CANDIDATES:
        if c.exists():
            import xarray as xr
            return xr.open_dataset(c)
    return None


# ── GEBCO/ETOPO raw elevation grid (fallback) ────────────────────────────

@lru_cache(maxsize=1)
def _load_gebco():
    """(lat1d, lon1d, elevation_2d) for the North-Sea window, or None."""
    files = sorted(glob.glob(str(BATHY_DIR / "*.nc")))
    if not files:
        return None
    import xarray as xr
    ds = xr.open_dataset(files[0])
    try:
        evar = next((v for v in ("elevation", "z", "altitude", "topo", "Band1",
                                  "depth") if v in ds.data_vars), None)
        if evar is None:
            import warnings
            warnings.warn(
                f"bathymetry: no elevation variable in {Path(files[0]).name} "
                f"(found {list(ds.data_vars)}); ignoring. Use the GEBCO "
                "'ice surface elevation' grid, not the TID grid.")
            return None
        latn = next(c for c in ds.coords if "lat" in c.lower())
        lonn = next(c for c in ds.coords if "lon" in c.lower())
        lat_asc = float(ds[latn].values[1]) > float(ds[latn].values[0])
        lon_asc = float(ds[lonn].values[1]) > float(ds[lonn].values[0])
        lat_sl = slice(*(_REGION_LAT if lat_asc else _REGION_LAT[::-1]))
        lon_sl = slice(*(_REGION_LON if lon_asc else _REGION_LON[::-1]))
        sub = ds[evar].sel({latn: lat_sl, lonn: lon_sl})
        lat = np.asarray(sub[latn].values, float)
        lon = np.asarray(sub[lonn].values, float)
        elev = np.asarray(sub.values, float)
        if evar == "depth":
            elev = -np.abs(elev)
        if not lat_asc:
            lat = lat[::-1]; elev = elev[::-1, :]
        if not lon_asc:
            lon = lon[::-1]; elev = elev[:, ::-1]
        return lat, lon, elev
    finally:
        ds.close()


# ── Public API ───────────────────────────────────────────────────────────

def available() -> bool:
    """True iff any bathymetry backend (EMODnet layer or GEBCO file) is present."""
    return _load_emodnet() is not None or _load_gebco() is not None


def water_depth_m(lat: float, lon: float) -> float:
    """Water depth (m, **positive below sea level**), NaN on land / no data."""
    ds = _load_emodnet()
    if ds is not None:
        d = float(ds["water_depth_m"].sel(lat=lat, lon=lon, method="nearest").values)
        return d if np.isfinite(d) else float("nan")
    g = _load_gebco()
    if g is None:
        return float("nan")
    glat, glon, elev = g
    e = float(elev[int(np.argmin(np.abs(glat - lat))), int(np.argmin(np.abs(glon - lon)))])
    return -e if e < 0 else float("nan")


def dist_to_coast_km(lat: float, lon: float) -> float:
    """Distance to nearest coast (km). NaN when only the GEBCO fallback exists."""
    ds = _load_emodnet()
    if ds is None:
        return float("nan")
    c = float(ds["dist_coast_km"].sel(lat=lat, lon=lon, method="nearest").values)
    return c if np.isfinite(c) else float("nan")


def is_fixed_bottom(lat: float, lon: float,
                    max_depth_m: float = DEFAULT_MAX_DEPTH_M) -> bool:
    """True iff sea AND shallow enough for fixed-bottom (depth only).

    Returns True when no bathymetry is available (constraint inactive).
    """
    if not available():
        return True
    dep = water_depth_m(lat, lon)
    return bool(np.isfinite(dep) and 0.0 < dep <= max_depth_m)


def is_eligible(lat: float, lon: float,
                max_depth_m: float = DEFAULT_MAX_DEPTH_M,
                min_dist_km: float = DEFAULT_MIN_DIST_KM) -> bool:
    """Full fixed-bottom eligibility: sea AND depth<=max AND dist_coast>=min.

    Falls back to the depth-only test when distance-to-coast is unavailable
    (GEBCO backend). Returns True when no bathymetry is available at all.
    """
    if not available():
        return True
    dep = water_depth_m(lat, lon)
    if not (np.isfinite(dep) and 0.0 < dep <= max_depth_m):
        return False
    dist = dist_to_coast_km(lat, lon)
    if np.isfinite(dist):
        return bool(dist >= min_dist_km)
    return True  # depth ok, distance unknown (GEBCO fallback)


def eligibility_grid():
    """The boolean (lat, lon) eligibility DataArray, or None (EMODnet only)."""
    ds = _load_emodnet()
    return None if ds is None else ds["eligible"].astype(bool)
