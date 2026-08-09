"""Phase-2 data-root resolution - makes the kit run from a
clean participant unzip of the **two datasets** (Phase-1 + Phase-2), with no
hardcoded paths.

Resolution order for every resource:
  1. ``$PHASE2_DATA_ROOT`` (one path, or several separated by ``os.pathsep``)
     - and the **parent** of each, so pointing at the Phase-2 dir also finds the
     Phase-1 dir unzipped next to it.
  2. the current working directory, ``participant_kit/``, and the repo root.
Each resolver then tries the **ship layout** (``train/…``, ``static/…``) first,
then the **repo dev layout** (``build/phase2_dataset/…``, ``data/wind_data/nwp/…``).

The forecast driver (HRES) is read from BOTH datasets: Phase-1 ships
``train/hres_north_sea.parquet`` (2019-2021) and Phase-2 ships
``train/hres/north_sea_hres_2016_2018.parquet`` - they share the same 2565-point
grid, so the kit concatenates + de-duplicates them (see ``hres_parquets``).
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent      # kit root
_REPO = _HERE.parents[1]                      # repo root (dev fallback)


def _search_bases() -> list[Path]:
    """Ordered, de-duplicated list of base dirs to look under."""
    bases: list[Path] = []
    # 0. persisted root written by 0_dataset_setup (survives across notebooks/sessions)
    try:
        dotfile = _HERE / ".phase2_data_root"
        if dotfile.exists():
            for line in dotfile.read_text().splitlines():
                line = line.strip()
                if line:
                    p = Path(line).expanduser()
                    bases += [p, p.parent]
    except OSError:
        pass
    for v in os.environ.get("PHASE2_DATA_ROOT", "").split(os.pathsep):
        if v:
            p = Path(v).expanduser()
            bases += [p, p.parent]            # the root AND its parent (sibling datasets)
    bases += [Path.cwd(), _HERE.parent, _REPO]
    seen, out = set(), []
    for b in bases:
        rb = b.resolve()
        if rb not in seen:
            seen.add(rb)
            out.append(b)
    return out


def _first(cands) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


def _flatten(rel_lists) -> list[Path]:
    return [base / rel for base in _search_bases() for rel in rel_lists]


# ── Phase-2 resources (ship layout first, then repo dev layout) ───────────

def target_root() -> Path:
    p = _first(_flatten(["train/arome", "arome", "build/phase2_dataset/arome"]))
    return p or (_REPO / "build" / "phase2_dataset" / "arome")


def target_static() -> Path:
    p = _first(_flatten([
        "static/arome_static.nc", "train/arome/arome_static.nc",
        "arome/arome_static.nc", "build/phase2_dataset/arome/arome_static.nc"]))
    return p or (target_root() / "arome_static.nc")


def coarse_root() -> Path:
    p = _first(_flatten([
        "train/arome_coarse125", "arome_coarse125",
        "build/phase2_dataset/arome_coarse125"]))
    return p or (_REPO / "build" / "phase2_dataset" / "arome_coarse125")


def reanalysis_root() -> Path:
    p = _first(_flatten([
        "train/reanalysis", "reanalysis", "build/phase2_dataset/reanalysis"]))
    return p or (_REPO / "build" / "phase2_dataset" / "reanalysis")


def inference_root() -> Path | None:
    return _first(_flatten(["inference", "build/phase2_dataset/inference"]))


def footprint_path() -> Path:
    p = _first(_flatten([
        "footprint_points.parquet", "static/footprint_points.parquet",
        "build/phase2_dataset/footprint_points.parquet"]))
    return p or (_REPO / "build" / "phase2_dataset" / "footprint_points.parquet")


def hres_parquets() -> list[Path]:
    """All HRES forecast parquets across the unzipped datasets + repo, de-duplicated
    by resolved path. Phase-1 (``hres_north_sea.parquet``, 2019-2021) + Phase-2
    (``north_sea_hres_2016_2018.parquet``) + the repo merged file all share the
    same grid, so :func:`forecast_hres._load_hres` concatenates + drops duplicate
    (time, lat, lon) rows."""
    pats = [
        "train/hres/*.parquet",                                # Phase-2 ship
        "train/hres_north_sea.parquet",                        # Phase-1 ship
        "phase1_dataset/train/hres_north_sea.parquet",         # Phase-1 unzipped as a dir
        "inference/window_*/context_hres_north_sea.parquet",   # eval-year driver (issued at context_end)
        "build/phase2_dataset/inference/window_*/context_hres_north_sea.parquet",  # dev
        "data/wind_data/nwp/north_sea_hres_forecasts.parquet", # repo merged (dev)
        "data/wind_data/nwp/north_sea_hres_2016_2018.parquet", # repo back-fill (dev)
    ]
    out, seen = [], set()
    for base in _search_bases():
        for pat in pats:
            for c in sorted(base.glob(pat)):
                if c.name.startswith("._"):    # skip macOS AppleDouble sidecar files
                    continue
                rc = c.resolve()
                if rc not in seen:
                    seen.add(rc)
                    out.append(c)
    return out


def describe() -> str:
    """One-line-per-resource summary - handy as a notebook sanity check."""
    hp = hres_parquets()
    return "\n".join([
        f"target_root   : {target_root()}",
        f"target_static : {target_static()}",
        f"coarse_root  : {coarse_root()}",
        f"reanalysis_root    : {reanalysis_root()}",
        f"inference    : {inference_root()}",
        f"footprint    : {footprint_path()}",
        f"hres parquets: {len(hp)} -> {[p.name for p in hp]}",
    ])
