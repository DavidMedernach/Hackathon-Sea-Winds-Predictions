"""Fetch + cache ENTSO-E electricity data for the Phase-2 economic analysis.

Pulls Netherlands (bidding zone ``10YNL----------L``) figures from the ENTSO-E
Transparency Platform REST API and caches them as tidy hourly CSVs so the
economic notebook runs offline after the first call:

- **Actual total load** (demand)  - documentType ``A65`` / processType ``A16``
- **Day-ahead price**             - documentType ``A44``

Auth: a personal security token read from ``~/.config/entsoe/token`` (request one
free at https://transparency.entsoe.eu/ → My Account Settings). The token is
never written to disk by this module and must never be committed.

Gotchas handled here:
- ENTSO-E timestamps are **UTC**; positions index from the Period's
  ``timeInterval.start`` at the stated ``resolution``.
- The A44 price curve is **compressed**: a missing position repeats the previous
  point's value (forward-fill within each Period).
- Load is published at **PT15M**; we resample to hourly means to match the
  hourly price and the hourly farm-production series.
- Requests are chunked **monthly** (well inside the 1-year API limit) and
  overlapping boundary hours are de-duplicated.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://web-api.tp.entsoe.eu/api"
NL_EIC = "10YNL----------L"
TOKEN_PATH = Path(os.path.expanduser("~/.config/entsoe/token"))

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "data" / "entsoe"

_RES_MIN = {"PT60M": 60, "PT30M": 30, "PT15M": 15}


def _read_token() -> str:
    if not TOKEN_PATH.exists():
        raise FileNotFoundError(
            f"ENTSO-E token not found at {TOKEN_PATH}. Request a free token at "
            "https://transparency.entsoe.eu/ (My Account Settings) and save it there."
        )
    return TOKEN_PATH.read_text().strip()


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _month_chunks(year: int):
    """(periodStart, periodEnd) UTC strings 'YYYYMMDDHHMM' for each month."""
    edges = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="MS", tz="UTC")
    for a, b in zip(edges[:-1], edges[1:]):
        yield a.strftime("%Y%m%d%H%M"), b.strftime("%Y%m%d%H%M")


def _request(params: dict, retries: int = 3) -> str:
    tok = _read_token()
    for attempt in range(retries):
        r = requests.get(API, params={**params, "securityToken": tok}, timeout=120)
        if r.status_code == 200:
            return r.text
        # 429/5xx -> back off and retry; otherwise surface the body
        if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"ENTSO-E API {r.status_code}: {r.text[:400]}")
    raise RuntimeError("ENTSO-E API: exhausted retries")


def _parse_periods(xml_text: str, value_tag: str) -> pd.DataFrame:
    """Flatten every Period/Point into a (time UTC, value) frame.

    ``value_tag`` is the value element's local name - ``quantity`` (load) or
    ``price.amount`` (price; the dot is part of the element name, not an XML
    namespace). Missing positions inside a Period are forward-filled (ENTSO-E
    curve compression).
    """
    root = ET.fromstring(xml_text.encode("utf-8"))
    rows_t: list[pd.Timestamp] = []
    rows_v: list[float] = []
    for period in root.iter():
        if _localname(period.tag) != "Period":
            continue
        start = res = None
        _pos = None
        points: dict[int, float] = {}
        for el in period.iter():
            name = _localname(el.tag)
            if name == "start":
                start = pd.Timestamp(el.text)
            elif name == "resolution":
                res = el.text
            elif name == "position":
                _pos = int(el.text)
            elif name == value_tag and _pos is not None:
                points[_pos] = float(el.text)
        if start is None or res is None or not points:
            continue
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")
        step = pd.Timedelta(minutes=_RES_MIN[res])
        n = max(points)
        last = np.nan
        for k in range(1, n + 1):
            last = points.get(k, last)        # forward-fill compressed gaps
            rows_t.append(start + (k - 1) * step)
            rows_v.append(last)
    df = pd.DataFrame({"time": rows_t, "value": rows_v})
    return df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _fetch_year(params_base: dict, value_tag: str, year: int) -> pd.DataFrame:
    parts = []
    for ps, pe in _month_chunks(year):
        xml = _request({**params_base, "periodStart": ps, "periodEnd": pe})
        parts.append(_parse_periods(xml, value_tag))
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_load(year: int = 2022, *, eic: str = NL_EIC,
               refresh: bool = False) -> pd.DataFrame:
    """Actual total load (demand), hourly mean MW. Cached to CSV.

    Returns a frame with columns ``time`` (UTC) and ``load_mw``.
    """
    cache = CACHE_DIR / f"nl_load_{year}.csv"
    if cache.exists() and not refresh:
        out = pd.read_csv(cache, parse_dates=["time"])
        out["time"] = pd.to_datetime(out["time"], utc=True)
        return out
    raw = _fetch_year({"documentType": "A65", "processType": "A16",
                       "outBiddingZone_Domain": eic}, "quantity", year)
    hourly = (raw.set_index("time")["value"].resample("1h").mean()
              .rename("load_mw").reset_index())
    hourly = hourly[hourly["time"].dt.year == year].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache, index=False)
    return hourly


def fetch_price(year: int = 2022, *, eic: str = NL_EIC,
                refresh: bool = False) -> pd.DataFrame:
    """Day-ahead price, hourly EUR/MWh. Cached to CSV.

    Returns a frame with columns ``time`` (UTC) and ``price_eur_mwh``.
    """
    cache = CACHE_DIR / f"nl_price_{year}.csv"
    if cache.exists() and not refresh:
        out = pd.read_csv(cache, parse_dates=["time"])
        out["time"] = pd.to_datetime(out["time"], utc=True)
        return out
    raw = _fetch_year({"documentType": "A44", "in_Domain": eic, "out_Domain": eic},
                      "price.amount", year)
    hourly = (raw.set_index("time")["value"].resample("1h").mean()
              .rename("price_eur_mwh").reset_index())
    hourly = hourly[hourly["time"].dt.year == year].reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache, index=False)
    return hourly


def load_demand_and_price(year: int = 2022) -> pd.DataFrame:
    """Convenience: hourly (time, load_mw, price_eur_mwh) inner-joined on the hour."""
    d = fetch_load(year)
    p = fetch_price(year)
    return d.merge(p, on="time", how="inner")


if __name__ == "__main__":
    d = fetch_load(2022)
    p = fetch_price(2022)
    print(f"load  {len(d)} h | mean {d.load_mw.mean():.0f} MW "
          f"| peak {d.load_mw.max():.0f} MW")
    print(f"price {len(p)} h | mean {p.price_eur_mwh.mean():.1f} EUR/MWh "
          f"| max {p.price_eur_mwh.max():.0f}")
