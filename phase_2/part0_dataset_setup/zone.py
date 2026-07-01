"""Phase 2 allowed zone: the farm centre must lie in target ∩ reanalysis ∩ sea.

This is the valid domain for the whole challenge - points where BOTH target
(the high-res truth) and reanalysis (the low-res input) exist, restricted to sea.
On the coarse reanalysis grid this is exactly the cells that:
  (a) are ≥90% inside the target footprint. The target native grid is rotated,
      so in lat/lon its footprint is a V/fan, not a rectangle - its corners are
      cut. We require near-full coverage so we never site/predict where target
      is only partially (or not) defined;
  (b) are at least ~20% sea (keeps near-shore cells relevant for coastal
      siting, while excluding near-pure-land cells);
  (c) lie on the North-Sea side of the British Isles (lon ≥ -2°), excluding the
      Celtic/Irish Sea west of the UK.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def _grid(max_depth_m: float | None = None):
    """Return (lats1d, lons1d, allowed_mask 2D bool) on the reanalysis grid.

    If ``max_depth_m`` is given **and** a bathymetry grid is available
    (see :mod:`bathymetry`), cells deeper than that are also excluded -
    **fixed-bottom only, no floating**. With no bathymetry file the depth
    filter is inactive and the zone is unchanged.
    """
    import target_loader
    import coarsen
    import reanalysis_loader

    e5 = reanalysis_loader.load_reanalysis(reanalysis_loader.list_dates()[0], hour=0)
    lats = np.asarray(e5.lats, dtype=float)
    lons = np.asarray(e5.lons, dtype=float)
    st = target_loader.load_static()
    sea_frac = coarsen.coarsen_field(st.seamask.astype(np.float32), st.lat, st.lon,
                                     lats, lons)
    # target footprint coverage per reanalysis cell = (# target pixels in the cell) /
    # (a fully-covered cell). The target grid is a rotated, V-shaped quadrilateral
    # in lat/lon, so edge cells clip only a sliver of it; require ≥90% coverage
    # so the cell sits genuinely inside target, not on the slanted boundary.
    cell, valid, nlat, nlon = coarsen._cell_ids(st.lat, st.lon, lats, lons)
    npix = np.bincount(cell[valid], minlength=nlat * nlon).reshape(nlat, nlon)
    cover = npix / np.median(npix[npix > 0])
    allowed = (
        np.isfinite(sea_frac)        # target exists in the cell at all
        & (cover >= 0.9)             # ≥90% inside the target V-footprint
        & (sea_frac > 0.2)           # ≥~20% sea (keeps near-shore cells)
        & (lons[None, :] >= -2.0)    # North-Sea side only (no Celtic/Irish Sea)
    )
    if max_depth_m is not None:
        import bathymetry
        if bathymetry.available():
            depth_ok = np.ones_like(allowed)
            for i in range(lats.size):
                for j in range(lons.size):
                    if allowed[i, j]:
                        depth_ok[i, j] = bathymetry.is_fixed_bottom(
                            float(lats[i]), float(lons[j]), max_depth_m)
            allowed = allowed & depth_ok
    return lats, lons, allowed


def is_in_allowed_zone(lat: float, lon: float,
                       max_depth_m: float | None = None) -> bool:
    """True iff (lat, lon) snaps to an allowed (target∩reanalysis∩sea) reanalysis cell.

    Pass ``max_depth_m`` (e.g. 55) to also require fixed-bottom-feasible depth
    when a bathymetry grid is present (else the depth filter is inactive).
    """
    lats, lons, allowed = _grid(max_depth_m)
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    i = int(round((lat - lats[0]) / dlat))
    j = int(round((lon - lons[0]) / dlon))
    if not (0 <= i < lats.size and 0 <= j < lons.size):
        return False
    return bool(allowed[i, j])


def zone_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    """(lat_min, lat_max), (lon_min, lon_max) over the allowed cells."""
    lats, lons, allowed = _grid()
    ii, jj = np.where(allowed)
    return ((float(lats[ii.min()]), float(lats[ii.max()])),
            (float(lons[jj.min()]), float(lons[jj.max()])))
