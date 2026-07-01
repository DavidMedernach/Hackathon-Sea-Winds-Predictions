"""Phase 2 train/eval splits and target masking.

Design (anti-probing leaderboard):
- **Train years** (target truth given to participants): 2016-2020.
- **Public eval** (live leaderboard, open during the challenge): **2021** - target hidden.
- **Final eval** (revealed only the LAST WEEK, to limit leaderboard probing): **2022** - target hidden.
- **Wind-farm siting eval**: a secret dataset (year withheld from participants).
- Forecast eval windows: the 8 phase-1 rolling windows, shifted to the eval year.
"""
from __future__ import annotations

import json
import sys
from datetime import date as _date_t
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # kit root
import config                                                  # noqa: E402

TRAIN_YEARS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020)
EVAL_PUBLIC_YEAR: int = 2021   # live leaderboard
EVAL_FINAL_YEAR: int = 2022    # last-week final (limits probing)
SITING_YEARS: tuple[int, ...] = ()   # secret dataset - siting year withheld in the participant kit
EVAL_YEAR: int = EVAL_PUBLIC_YEAR   # back-compat: the currently-active forecast eval

# All target years withheld from participants (any eval/siting year).
HIDDEN_YEARS: tuple[int, ...] = (EVAL_PUBLIC_YEAR, EVAL_FINAL_YEAR, *SITING_YEARS)

_BASE_WINDOW_YEAR = 2022   # the year the phase-1 WINDOWS are defined in


def _as_date(d: _date_t | str) -> _date_t:
    return pd.Timestamp(d).date()


def target_available(d: _date_t | str) -> bool:
    """True if target ground truth for ``d`` is shipped to participants.

    Given only for the TRAIN_YEARS (2016-2020). Withheld for every eval/siting
    year (2021 public, 2022 final, and the secret siting dataset).
    """
    return _as_date(d).year in TRAIN_YEARS


def _shift_windows(windows: list[dict], year: int) -> list[dict]:
    """Re-date the phase-1 windows (defined in 2022) to ``year`` by year-substitution.

    the eval years are non-leap, so day-of-year is preserved with no Feb-29 issue.
    """
    src, dst = str(_BASE_WINDOW_YEAR), str(year)
    out = []
    for w in windows:
        nw = dict(w)
        for k in ("context_start", "context_end", "predict_start", "predict_end"):
            if k in nw and nw[k] is not None:
                nw[k] = str(nw[k]).replace(src, dst)
        sd = nw.get("score_days")
        if isinstance(sd, dict):
            nw["score_days"] = {k: str(v).replace(src, dst) for k, v in sd.items()}
        out.append(nw)
    return out


# The 8 rolling windows, defined in the base year 2022 (shifted to the eval year).
_BASE_WINDOWS = json.loads('''[
    {
        "id": 1,
        "context_start": "2022-01-01",
        "context_end": "2022-01-14",
        "predict_start": "2022-01-15",
        "predict_end": "2022-01-28",
        "score_days": {
            "d1": "2022-01-15",
            "d7": "2022-01-21",
            "d14": "2022-01-28"
        }
    },
    {
        "id": 2,
        "context_start": "2022-02-12",
        "context_end": "2022-02-25",
        "predict_start": "2022-02-26",
        "predict_end": "2022-03-11",
        "score_days": {
            "d1": "2022-02-26",
            "d7": "2022-03-04",
            "d14": "2022-03-11"
        }
    },
    {
        "id": 3,
        "context_start": "2022-03-26",
        "context_end": "2022-04-08",
        "predict_start": "2022-04-09",
        "predict_end": "2022-04-22",
        "score_days": {
            "d1": "2022-04-09",
            "d7": "2022-04-15",
            "d14": "2022-04-22"
        }
    },
    {
        "id": 4,
        "context_start": "2022-05-07",
        "context_end": "2022-05-20",
        "predict_start": "2022-05-21",
        "predict_end": "2022-06-03",
        "score_days": {
            "d1": "2022-05-21",
            "d7": "2022-05-27",
            "d14": "2022-06-03"
        }
    },
    {
        "id": 5,
        "context_start": "2022-06-18",
        "context_end": "2022-07-01",
        "predict_start": "2022-07-02",
        "predict_end": "2022-07-15",
        "score_days": {
            "d1": "2022-07-02",
            "d7": "2022-07-08",
            "d14": "2022-07-15"
        }
    },
    {
        "id": 6,
        "context_start": "2022-07-30",
        "context_end": "2022-08-12",
        "predict_start": "2022-08-13",
        "predict_end": "2022-08-26",
        "score_days": {
            "d1": "2022-08-13",
            "d7": "2022-08-19",
            "d14": "2022-08-26"
        }
    },
    {
        "id": 7,
        "context_start": "2022-09-10",
        "context_end": "2022-09-23",
        "predict_start": "2022-09-24",
        "predict_end": "2022-10-07",
        "score_days": {
            "d1": "2022-09-24",
            "d7": "2022-09-30",
            "d14": "2022-10-07"
        }
    },
    {
        "id": 8,
        "context_start": "2022-10-22",
        "context_end": "2022-11-04",
        "predict_start": "2022-11-05",
        "predict_end": "2022-11-18",
        "score_days": {
            "d1": "2022-11-05",
            "d7": "2022-11-11",
            "d14": "2022-11-18"
        }
    }
]''')


def eval_windows(year: int | None = None) -> list[dict]:
    """The 8 rolling inference windows for ``year`` (default = public eval 2021),
    re-dated from the base-year spec. Self-contained (no organiser files)."""
    year = EVAL_PUBLIC_YEAR if year is None else year
    return _shift_windows(_BASE_WINDOWS, year)


def train_val_dates(seed: int = 42, val_fraction: float = 0.2):
    """Deterministic random train/val split over the TRAIN_YEARS target dates.

    Never includes any hidden year. Returns (train_dates, val_dates).
    """
    import numpy as np
    import target_loader

    dates = [d for d in target_loader.list_dates(config.target_root()) if d.year in TRAIN_YEARS]
    rng = np.random.default_rng(seed)
    n = len(dates)
    n_val = max(1, round(n * val_fraction))
    val_idx = set(rng.choice(n, size=n_val, replace=False).tolist())
    train = [d for i, d in enumerate(dates) if i not in val_idx]
    val = [d for i, d in enumerate(dates) if i in val_idx]
    return train, val
