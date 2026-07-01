"""Deliverable 1 scorer - downscaled forecast vs the hidden target 2022 truth.

Two interchangeable coarse *drivers* feed the identical downscale path:
  - **HRES-MOS** (:mod:`forecast_hres`)  - ECMWF HRES forecast + LightGBM MOS.
  - **Pangu-MOS** (:mod:`forecast_pangu`) - Pangu-Weather foundation model + MOS.

End-to-end reference pipeline (per driver):
  driver -> MOS (t+1/t+7) / climatology (t+14)  =  coarse forecast on the reanalysis grid
  -> LightGBM terrain downscaler               =  1.3 km target-resolution forecast
  -> score vs hidden target 2022 (coarse truth coarsened on the fly; fine = target)

The downscaler (coarse->1.3 km) is driver-independent and trained once; only the
coarse MOS differs between drivers.

Run: .venv/bin/python3 part1_forecast/score_forecast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve()
for sub in ("part0_dataset_setup", "part1_forecast"):
    sys.path.insert(0, str(_HERE.parents[1] / sub))

import target_loader as al            # noqa: E402
import downscaling as dsc            # noqa: E402
import forecast_hres as fh           # noqa: E402
import forecast_pangu as fp          # noqa: E402
import forecast_model as fm          # noqa: E402

MOS_LEADS = (1, 7)
CLIM_LEAD = 14
HOURS = (0, 6, 12, 18)   # score over ALL hours, not a single one

# name -> (module, build-table fn). Both modules expose the same MOS / climatology
# / grid-scatter API, so the scorer is driver-agnostic.
DRIVERS = {
    "HRES-MOS": (fh, fh.build_hres_table),
    "Pangu-MOS": (fp, fp.build_pangu_table),
}


def _sea_keep():
    return al.load_static().seamask.astype(bool)


def train_downscaler(downscale_days):
    """Train the shared coarse->1.3 km terrain downscaler (driver-independent)."""
    return dsc.train_downscaler(downscale_days, hours=HOURS)


def fit_mos(driver: str, train_issue_dates):
    """Train the MOS (driver forecast -> target-coarse) for one driver."""
    mod, build = DRIVERS[driver]
    return mod.train_mos(build(train_issue_dates))


def _fine_metrics(mod, build, mos, dwn, eval_issue_dates, L, keep, hours=HOURS):
    """Pooled 1.3 km errors over all eval dates × hours for lead L.

    Returns (speed_RMSE m/s, vector_RMSE m/s, dir_cMAE deg, n_fields) - the speed
    RMSE is the headline (interpretable, comparable to single-field downscaling),
    not just the uv-vector RMSE.
    """
    FU, FV, TU, TV = [], [], [], []
    n = 0
    for D in eval_issue_dates:
        one_all = mod.predict_mos(mos, build([D]))
        for H in hours:
            one = one_all[(one_all["lead"] == L) & (one_all["hour"] == H)]
            if one.empty:
                continue
            U, V = mod.predictions_to_grid(one, L, H)
            fu, fv = dsc.downscale(dwn, U, V)
            # FOOTPRINT mask: ~1/3 of target sea pixels lie SOUTH/WEST of the reanalysis
            # coverage and have no coarse input - the downscaler extrapolates garbage
            # there. Score only where the coarse field maps validly (interp finite),
            # exactly as the deck's downscaling figures do.
            iu, iv = dsc.interp_coarse_to_target(U, V)
            foot = np.isfinite(iu) & np.isfinite(iv)
            Vd = (pd.Timestamp(D) + pd.Timedelta(days=L)).date()
            try:
                snap = al.load_snapshot(Vd, H)
            except FileNotFoundError:
                continue
            tu, tv = snap.fields["125m"]["u"], snap.fields["125m"]["v"]
            m = keep & foot & np.isfinite(fu) & np.isfinite(tu)
            FU.append(fu[m]); FV.append(fv[m]); TU.append(tu[m]); TV.append(tv[m])
            n += 1
    if not FU:
        return np.nan, np.nan, np.nan, 0
    fu, fv = np.concatenate(FU), np.concatenate(FV)
    tu, tv = np.concatenate(TU), np.concatenate(TV)
    return (fm.speed_rmse(tu, tv, fu, fv), fm.uv_rmse(tu, tv, fu, fv),
            fm.circular_mae_from_uv(tu, tv, fu, fv), n)


def score_driver(driver: str, eval_issue_dates, mos, dwn, leads=MOS_LEADS):
    """Per-lead metrics for one driver, pooled over all eval dates × hours:
    coarse uv-RMSE/cMAE + 1.3 km **speed** RMSE / vector RMSE / direction cMAE."""
    mod, build = DRIVERS[driver]
    keep = _sea_keep()
    rows = []
    for L in leads:
        te = pd.concat([build([D]) for D in eval_issue_dates], ignore_index=True)
        te = te[(te["lead"] == L) & (te["hour"].isin(HOURS))]
        pr = mod.predict_mos(mos, te)
        f_spd, f_vec, f_cmae, nf = _fine_metrics(
            mod, build, mos, dwn, eval_issue_dates, L, keep)
        rows.append({"driver": driver, "lead": f"t+{L}d",
                     "coarse_uvRMSE": round(fm.uv_rmse(pr.u125c, pr.v125c,
                                                       pr.u_pred, pr.v_pred), 2),
                     "coarse_dir_cMAE": round(fm.circular_mae_from_uv(
                         pr.u125c, pr.v125c, pr.u_pred, pr.v_pred), 1),
                     "fine_speedRMSE": round(f_spd, 2),
                     "fine_uvRMSE": round(f_vec, 2),
                     "fine_dir_cMAE": round(f_cmae, 1),
                     "n_fields": nf})
    # t+14: climatology (identical for both drivers) - coarse only.
    cl = mod.build_climatology_forecast(eval_issue_dates, lead=CLIM_LEAD)
    cl = cl[cl["hour"].isin(HOURS)]
    rows.append({"driver": driver, "lead": f"t+{CLIM_LEAD}d (clim)",
                 "coarse_uvRMSE": round(fm.uv_rmse(cl.u125c, cl.v125c,
                                                   cl.u_pred, cl.v_pred), 2),
                 "coarse_dir_cMAE": round(fm.circular_mae_from_uv(
                     cl.u125c, cl.v125c, cl.u_pred, cl.v_pred), 1),
                 "fine_speedRMSE": np.nan, "fine_uvRMSE": np.nan,
                 "fine_dir_cMAE": np.nan, "n_fields": 0})
    return pd.DataFrame(rows)


def score_raw_hres(eval_issue_dates, leads=MOS_LEADS):
    """Baseline: RAW HRES forecast (no MOS) vs target-coarse truth - shows the
    value the LightGBM MOS adds on top of the raw driver."""
    rows = []
    for L in leads:
        te = pd.concat([fh.build_hres_table([D]) for D in eval_issue_dates],
                       ignore_index=True)
        te = te[(te["lead"] == L) & (te["hour"].isin(HOURS))]
        # baseline = RAW HRES forecast (no MOS) vs target-coarse truth -> shows the
        # value the LightGBM MOS adds on top of the raw driver.
        rows.append({"driver": "HRES-raw (no MOS)", "lead": f"t+{L}d",
                     "coarse_uvRMSE": round(fm.uv_rmse(te.u125c, te.v125c,
                                                       te.fcst_u, te.fcst_v), 2),
                     "coarse_dir_cMAE": round(fm.circular_mae_from_uv(
                         te.u125c, te.v125c, te.fcst_u, te.fcst_v), 1),
                     "fine_speedRMSE": np.nan, "fine_uvRMSE": np.nan,
                     "fine_dir_cMAE": np.nan, "n_fields": 0})
    return pd.DataFrame(rows)


def score(eval_issue_dates, mos, dwn, leads=MOS_LEADS):
    """Backward-compatible single-driver (HRES) scorer."""
    return score_driver("HRES-MOS", eval_issue_dates, mos, dwn, leads=leads)


def score_quantiles(train_dates, calib_dates, eval_dates, leads=MOS_LEADS, alpha=0.10):
    """Probabilistic HRES-MOS benchmark: 90 %-PI **coverage** + **Winkler** score,
    raw quantile MOS vs **conformal (CQR)** calibration. Honest protocol: the
    quantile models train on ``train_dates``, CQR calibrates on ``calib_dates``,
    and both are scored on the held-out ``eval_dates``."""
    qmos = fh.train_quantile_mos(fh.build_hres_table(train_dates))
    adj = fh.conformal_adjust(qmos, fh.build_hres_table(calib_dates), alpha=alpha)
    ev = fh.build_hres_table(eval_dates)
    raw = fh.predict_quantile_mos(qmos, ev)
    cal = fh.predict_quantile_mos(qmos, ev, adjust=adj)
    rows = []
    for L in leads:
        for tag, pr in (("raw", raw), ("CQR", cal)):
            s = pr[pr["lead"] == L]
            y = np.hypot(s.u125c.to_numpy(), s.v125c.to_numpy())
            rows.append({"quantiles": tag, "lead": f"t+{L}d",
                         "coverage90": round(fm.coverage_fraction(y, s.spd_q05, s.spd_q95), 3),
                         "winkler": round(fm.winkler_score(y, s.spd_q05, s.spd_q95), 2),
                         "PI_width": round(float(np.mean(s.spd_q95 - s.spd_q05)), 2)})
    return pd.DataFrame(rows)


def _eval_windows():
    """12 mid-month issue dates spanning all of 2022 (not just March) so the
    benchmark is seasonally representative, not anchored to one stormy window."""
    return pd.date_range("2022-01-15", "2022-12-15", freq="MS") + pd.Timedelta(days=14)


def main(drivers=("HRES-MOS", "Pangu-MOS"), eval_dates=None):
    train = pd.date_range("2019-01-03", "2021-12-25", freq="4D")
    dwn_days = [d.date() for d in pd.date_range("2021-01-05", "2021-12-25",
                                                freq="5D")]
    eval_dates = _eval_windows() if eval_dates is None else eval_dates
    dwn = train_downscaler(dwn_days)
    boards = [score_raw_hres(eval_dates)]
    for drv in drivers:
        mos = fit_mos(drv, train)
        boards.append(score_driver(drv, eval_dates, mos, dwn))
    board = pd.concat(boards, ignore_index=True)
    # probabilistic benchmark (train 2019-20 / calibrate 2021 / test = eval)
    qb = score_quantiles(pd.date_range("2019-01-03", "2020-12-27", freq="6D"),
                         pd.date_range("2021-01-05", "2021-12-25", freq="7D"),
                         eval_dates)
    print("\n=== probabilistic forecast (90%-PI coverage + Winkler) ===")
    print(qb.to_string(index=False))
    qb.to_csv(_HERE.parent / "forecast_quantile_benchmark.csv", index=False)
    out = _HERE.parent / "forecast_benchmark.csv"
    board.to_csv(out, index=False)
    print(f"\n=== Deliverable 1 - forecast benchmark "
          f"({len(eval_dates)} windows across 2022, all hours) ===")
    print(board.to_string(index=False))
    print(f"\nwrote {out}")
    return board


if __name__ == "__main__":
    main()
