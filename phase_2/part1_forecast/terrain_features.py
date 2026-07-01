"""Per-pixel terrain & static features on the target 1.3 km grid.

Cached once per call to ``compute_static_features``. The target (Lon, Lat,
SEAMASK) layers are constant across days, so the terrain features can be
computed once and reused across all training snapshots.

Returned features (all 2D arrays of shape ``(y, x)``):
    - ``elevation_m``      : surface altitude, in metres (sea points = 0)
    - ``dist_shore_km``    : Euclidean distance to the nearest land cell, km
    - ``lat``, ``lon``     : the target grid coordinates
    - ``seamask``          : 0 = land, 1 = sea (target convention)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[2]
_ELEVATION_NC = _PROJECT / "build" / "phase1_dataset" / "train" / "elevation_north_sea.nc"

# target grid resolution. Both axes are ~1.3 km; we use a single cell-edge length
# for the distance transform (cheap, off by a few %).
TARGET_PIXEL_KM = 1.3


def _load_elevation_dem() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load regional elevation as (lats_1d, lons_1d, elev_2d). Returns None if missing."""
    if not _ELEVATION_NC.exists():
        return None
    ds = xr.open_dataset(_ELEVATION_NC)
    # Auto-detect the elevation variable
    var_name = next((v for v in ("elevation", "ELEVATION", "z", "topo", "altitude")
                     if v in ds.data_vars), list(ds.data_vars)[0])
    arr = ds[var_name]
    while arr.ndim > 2:
        arr = arr.isel({arr.dims[0]: 0})
    lat_dim = next((d for d in arr.dims if "lat" in d.lower()), arr.dims[0])
    lon_dim = next((d for d in arr.dims if "lon" in d.lower()), arr.dims[1])
    lats = arr[lat_dim].values.astype(np.float32)
    lons = arr[lon_dim].values.astype(np.float32)
    values = arr.values.astype(np.float32)
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]
    return lats, lons, values


@lru_cache(maxsize=1)
def _interp_elevation_to_grid_cached(target_lat_bytes: bytes,
                                      target_lon_bytes: bytes,
                                      shape: tuple[int, int]) -> np.ndarray:
    """Interpolate the regional DEM onto the target (lat, lon) grid (cached)."""
    target_lat = np.frombuffer(target_lat_bytes, dtype=np.float32).reshape(shape)
    target_lon = np.frombuffer(target_lon_bytes, dtype=np.float32).reshape(shape)
    dem = _load_elevation_dem()
    if dem is None:
        return np.zeros(shape, dtype=np.float32)
    lats, lons, values = dem
    f = RegularGridInterpolator(
        (lats, lons), values, bounds_error=False, fill_value=0.0,
    )
    pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=1)
    elev = f(pts).reshape(shape).astype(np.float32)
    # Replace NaN (out of regional bbox) with 0 (assume sea)
    elev = np.where(np.isfinite(elev), elev, 0.0)
    return elev


def compute_static_features(target_lon: np.ndarray, target_lat: np.ndarray,
                            seamask: np.ndarray) -> dict[str, np.ndarray]:
    """Build the static-features dict on the target grid (computed once per run)."""
    target_lon = np.asarray(target_lon, dtype=np.float32)
    target_lat = np.asarray(target_lat, dtype=np.float32)
    seamask = np.asarray(seamask, dtype=np.float32)
    shape = target_lat.shape

    # Distance to shore: distance from each pixel to the nearest land cell
    # (where seamask < 0.5). Multiplied by the target pixel size to get km.
    is_land = seamask < 0.5
    dist_to_land_pixels = distance_transform_edt(~is_land)
    dist_shore_km = (dist_to_land_pixels * TARGET_PIXEL_KM).astype(np.float32)

    # Elevation: regional DEM interpolated to target grid. Cached per target_lat
    # bytes so repeated calls are O(1).
    elev = _interp_elevation_to_grid_cached(
        target_lat.tobytes(), target_lon.tobytes(), shape,
    )
    # Force sea pixels to elevation = 0 (the regional DEM may have noise there)
    elev = np.where(is_land, elev, 0.0)

    return {
        "elevation_m":   elev,
        "dist_shore_km": dist_shore_km,
        "lat":           target_lat,
        "lon":           target_lon,
        "seamask":       seamask,
    }
