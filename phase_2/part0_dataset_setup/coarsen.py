"""Coarsen an target field (fine curvilinear grid) onto the reanalysis grid.

Each target pixel is assigned to the nearest reanalysis cell centre; the coarse value
is the arithmetic mean of all target pixels in that cell. For wind, average the
u and v components separately (correct vector mean), never ws/wd directly.

reanalysis cells with no target pixel inside (e.g. north of the target footprint) are
returned as NaN.
"""
from __future__ import annotations

import numpy as np


def _cell_ids(target_lat: np.ndarray, target_lon: np.ndarray,
              reanalysis_lats: np.ndarray, reanalysis_lons: np.ndarray):
    """Flat reanalysis cell id for each target pixel; -1 for pixels outside the grid.

    Returns (cell_ids flat int array, valid bool mask, nlat, nlon).
    """
    lats = np.asarray(reanalysis_lats, dtype=float)
    lons = np.asarray(reanalysis_lons, dtype=float)
    nlat, nlon = lats.size, lons.size
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    # Nearest-centre rounding below assumes a uniform reanalysis grid.
    assert np.allclose(np.diff(lats), dlat) and np.allclose(np.diff(lons), dlon), \
        "coarsen requires a uniform (regular) reanalysis grid"

    alat = np.asarray(target_lat, dtype=float).ravel()
    alon = np.asarray(target_lon, dtype=float).ravel()
    i = np.round((alat - lats[0]) / dlat).astype(np.int64)
    j = np.round((alon - lons[0]) / dlon).astype(np.int64)
    valid = (i >= 0) & (i < nlat) & (j >= 0) & (j < nlon)
    cell = np.full(alat.shape, -1, dtype=np.int64)
    cell[valid] = i[valid] * nlon + j[valid]
    return cell, valid, nlat, nlon


def coarsen_field(target_field: np.ndarray, target_lat: np.ndarray,
                  target_lon: np.ndarray, reanalysis_lats: np.ndarray,
                  reanalysis_lons: np.ndarray) -> np.ndarray:
    """Mean-coarsen one 2-D target field onto the reanalysis (nlat, nlon) grid."""
    cell, valid, nlat, nlon = _cell_ids(target_lat, target_lon, reanalysis_lats, reanalysis_lons)
    flat = np.asarray(target_field, dtype=float).ravel()
    use = valid & np.isfinite(flat)
    ids = cell[use]
    vals = flat[use]
    n = nlat * nlon
    sums = np.bincount(ids, weights=vals, minlength=n)
    counts = np.bincount(ids, minlength=n)
    out = np.full(n, np.nan, dtype=np.float32)
    nz = counts > 0
    out[nz] = (sums[nz] / counts[nz]).astype(np.float32)
    return out.reshape(nlat, nlon)


def coarsen_uv(u: np.ndarray, v: np.ndarray, target_lat: np.ndarray,
               target_lon: np.ndarray, reanalysis_lats: np.ndarray,
               reanalysis_lons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coarsen u and v components separately (vector mean)."""
    uc = coarsen_field(u, target_lat, target_lon, reanalysis_lats, reanalysis_lons)
    vc = coarsen_field(v, target_lat, target_lon, reanalysis_lats, reanalysis_lons)
    return uc, vc
