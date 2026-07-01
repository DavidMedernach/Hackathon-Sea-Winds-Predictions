"""Phase-2 forecast pipeline - orchestrates the existing engines into the
Phase-1-style probabilistic submission (6 dims: 3 horizons x {speed, dir},
north_sea, 125 m). The two participant notebooks are thin wrappers over this.

Flow:  reanalysis+HRES --MOS--> target-coarse speed quantiles + det u/v (dir_50)
       --downscale--> target 1.3 km  --> q05/q50/q95 + dir_05/dir_50/dir_95.
Intervals are calibrated to ~90% coverage on the fine target truth of the pooled
training years (2016-2020); 2021+ is hidden eval.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "part0_dataset_setup"))

import forecast_hres as fh          # noqa: E402
import downscaling as dn            # noqa: E402
import splits                       # noqa: E402
import build_forecast_submission as bfs  # noqa: E402

HOURS = (0, 6, 12, 18)
LEADS = (1, 7, 14)
DET_LEADS = (1, 7)                  # HRES MOS; 14 = climatology


def train_dates(freq: str = "6D") -> pd.DatetimeIndex:
    # train years = 2016-2020 (target given); 2021 public eval, 2022 final (hidden).
    return pd.to_datetime(pd.date_range("2016-01-01", "2020-12-31", freq=freq))


def _circ_abs_deg(a, b):
    d = np.abs((a - b + 180) % 360 - 180)
    return d


def fit_forecast(train: pd.DatetimeIndex):
    """Train det MOS, quantile-speed MOS (+conformal), and per-lead dir offsets."""
    thr = fh.build_hres_table(train)
    mos = fh.train_mos(thr)
    calib = train[train.year == 2020]
    qmos = fh.train_quantile_mos(fh.build_hres_table(train[train.year < 2020]))
    adj = fh.conformal_adjust(qmos, fh.build_hres_table(calib))
    # direction interval half-widths: 90th pct of coarse MOS circular error on 2020
    cal = fh.predict_mos(mos, fh.build_hres_table(calib))
    dir_true = np.degrees(np.arctan2(-cal["u125c"], -cal["v125c"])) % 360
    dir_pred = np.degrees(np.arctan2(-cal["u_pred"], -cal["v_pred"])) % 360
    cal = cal.assign(_derr=_circ_abs_deg(dir_true, dir_pred))
    offs = {int(L): float(np.nanpercentile(cal.loc[cal.lead == L, "_derr"], 90))
            for L in DET_LEADS}
    offs[14] = 80.0                 # climatology direction: wide fixed half-width
    return mos, qmos, adj, offs


def coarse_fields(mos, qmos, adj, issue_date) -> dict:
    """Per (lead, hour) coarse grids: 'det' (2,45,57) u/v and 'spd' (3,45,57)."""
    te = fh.build_hres_table([issue_date], with_truth=False)
    det = fh.predict_mos(mos, te)
    qp = fh.predict_quantile_mos(qmos, te, adjust=adj)
    clim = fh.build_climatology_forecast([issue_date], lead=14, with_truth=False)
    out = {}
    for lead in LEADS:
        src = clim if lead == 14 else det
        for h in HOURS:
            U, V = fh.predictions_to_grid(src, lead, h)
            out[(lead, h, "det")] = np.stack([U, V]).astype("float32")
            if lead in DET_LEADS:
                sub = qp[(qp.lead == lead) & (qp.hour == h)]
                S = np.stack([fh.predictions_to_grid(
                        sub.assign(u_pred=sub[c], v_pred=0), lead, h)[0]
                        for c in ("spd_q05", "spd_q50", "spd_q95")])
                out[(lead, h, "spd")] = S.astype("float32")
    return out


def _speed_interval(fields, lead, h, spd50, k: float = 1.0):
    """Base downscaled speed PI (q05, q95) around spd50, widened by factor k."""
    if (lead, h, "spd") in fields:                     # leads 1,7
        cq05, cq50, cq95 = fields[(lead, h, "spd")]
        with np.errstate(divide="ignore", invalid="ignore"):
            r_lo = np.where(cq50 > 0, cq05 / cq50, 1.0)
            r_hi = np.where(cq50 > 0, cq95 / cq50, 1.0)
        f_lo = dn.interp_coarse_to_target(r_lo, np.zeros_like(r_lo))[0]
        f_hi = dn.interp_coarse_to_target(r_hi, np.zeros_like(r_hi))[0]
        q05, q95 = spd50 * np.clip(f_lo, 0.3, 1.0), spd50 * np.clip(f_hi, 1.0, 3.0)
    else:                                              # lead 14: climatology spread
        q05, q95 = spd50 * 0.55, spd50 * 1.6
    if k != 1.0:                                       # widen around the median
        q05 = np.maximum(0.0, spd50 - k * (spd50 - q05))
        q95 = spd50 + k * (q95 - spd50)
    return q05, q95


def downscale_window(dwn, fields: dict, offs: dict, window: int,
                     spd_infl: dict | None = None,
                     dir_off: dict | None = None) -> list[pd.DataFrame]:
    """Downscale one window's coarse fields -> Phase-1-schema submission blocks.

    ``spd_infl`` / ``dir_off`` (from :func:`calibrate_intervals`) widen the speed PI
    and set the direction half-width per lead for ~90% coverage on the fine truth.
    """
    spd_infl = spd_infl or {}
    dir_off = dir_off or offs
    blocks = []
    for lead in LEADS:
        for h in HOURS:
            U, V = fields[(lead, h, "det")]
            fu, fv = dn.downscale(dwn, U, V)
            spd50 = np.sqrt(fu ** 2 + fv ** 2)
            dir50 = np.degrees(np.arctan2(-fu, -fv)) % 360
            q05, q95 = _speed_interval(fields, lead, h, spd50, k=spd_infl.get(lead, 1.0))
            o = dir_off.get(lead, offs[lead])
            d05, d95 = dir50 - o, dir50 + o
            blocks.append(bfs.field_to_rows(window, lead, h, q05, spd50, q95,
                                            d05, dir50, d95))
    return blocks


def calibrate_intervals(mos, qmos, adj, dwn, offs, calib_dates=None, target=0.90):
    """Recalibrate the *downscaled* intervals on the fine target truth of the **pooled
    training years (2016-2020)**: per-lead speed widening ``k`` to reach ``target``
    coverage, and the direction half-width = ``target`` pct of the circular error.
    Pooling years captures the year-to-year spread, so the intervals generalise to the
    hidden eval year (>=90% coverage). Returns (spd_infl, dir_off) for
    :func:`downscale_window`."""
    import sys as _s
    _s.path.insert(0, str(_HERE.parent / "part0_dataset_setup"))
    import target_loader
    import config as _cfg
    import footprint as _fp
    if calib_dates is None:                              # pooled training years
        calib_dates = pd.to_datetime([f"{y}-{m:02d}-15"
                                      for y in (2016, 2017, 2018, 2019, 2020)
                                      for m in (2, 6, 10)])
    mask = _fp.footprint_mask()
    S = {L: {"mid": [], "lo": [], "hi": [], "t": []} for L in LEADS}
    derr = {L: [] for L in LEADS}
    for D in calib_dates:
        D = pd.Timestamp(D)
        flds = coarse_fields(mos, qmos, adj, D)
        for lead in LEADS:
            V = D + pd.Timedelta(days=lead)
            try:
                day = target_loader.load_day(V.date(), root=_cfg.target_root())
            except FileNotFoundError:
                continue
            for h in HOURS:
                U, Vv = flds[(lead, h, "det")]
                fu, fv = dn.downscale(dwn, U, Vv)
                spd = np.sqrt(fu ** 2 + fv ** 2)
                d50 = np.degrees(np.arctan2(-fu, -fv)) % 360
                q05, q95 = _speed_interval(flds, lead, h, spd, k=1.0)
                sn = day.snapshot(h)
                tu, tv = sn.fields["125m"]["u"], sn.fields["125m"]["v"]
                tspd = np.sqrt(tu ** 2 + tv ** 2)
                tdir = np.degrees(np.arctan2(-tu, -tv)) % 360
                m = mask & np.isfinite(spd) & np.isfinite(tspd)
                S[lead]["mid"].append(spd[m]); S[lead]["lo"].append(q05[m])
                S[lead]["hi"].append(q95[m]); S[lead]["t"].append(tspd[m])
                derr[lead].append(_circ_abs_deg(d50[m], tdir[m]))
    spd_infl, dir_off = {}, {}
    for L in LEADS:
        if not S[L]["mid"]:
            spd_infl[L], dir_off[L] = 1.0, offs[L]
            continue
        mid = np.concatenate(S[L]["mid"]); lo = np.concatenate(S[L]["lo"])
        hi = np.concatenate(S[L]["hi"]); t = np.concatenate(S[L]["t"])

        def _cov(k):
            a = np.maximum(0.0, mid - k * (mid - lo)); b = mid + k * (hi - mid)
            return float(np.mean((t >= a) & (t <= b)))

        klo, khi = 0.5, 1.0
        while _cov(khi) < target and khi < 12:
            khi *= 1.5
        for _ in range(25):
            km = 0.5 * (klo + khi)
            if _cov(km) < target:
                klo = km
            else:
                khi = km
        spd_infl[L] = round(0.5 * (klo + khi), 3)
        dir_off[L] = round(float(np.percentile(np.concatenate(derr[L]), 100 * target)), 1)
    return spd_infl, dir_off


def issue_date_of(window: dict) -> pd.Timestamp:
    return pd.Timestamp(window["context_end"])
