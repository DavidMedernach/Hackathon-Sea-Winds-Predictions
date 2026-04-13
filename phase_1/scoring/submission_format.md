# Phase 1 Submission Format

Your submission is a single CSV file named **`predictions.csv`**, packaged in a ZIP for upload to Codabench.

---

## Column schema

| Column | Type | Grid rows | Station rows | Description |
|---|---|:---:|:---:|---|
| `type` | string | ✓ | ✓ | `"grid"` or `"station"` |
| `window` | int | ✓ | ✓ | Inference window ID, 1–8 |
| `region` | string | ✓ | ✓ | `"north_sea"` or `"east_china_sea"` |
| `latitude` | float | ✓ | — | Grid latitude, rounded to 2 decimals |
| `longitude` | float | ✓ | — | Grid longitude, rounded to 2 decimals |
| `station` | string | — | ✓ | Station ID, e.g. `"NS_01"`, `"ECS_03"` |
| `horizon` | int | ✓ | ✓ | Forecast horizon in days: `1`, `7`, or `14` |
| `hour` | int | ✓ | ✓ | Hour of day (UTC): `0`, `6`, `12`, or `18` |
| `level` | string | ✓ | — | Vertical level, one of `"10m"`, `"100m"`, `"1000"`, `"925"`, `"850"`, `"700"`, `"500"` |
| `q05` | float | ✓ | ✓ | 5th percentile wind speed (m/s) |
| `q50` | float | ✓ | ✓ | 50th percentile (median) wind speed (m/s) |
| `q95` | float | ✓ | ✓ | 95th percentile wind speed (m/s) |
| `dir_05` | float | ✓ | ✓ | Lower bound of the 90% direction PI (degrees, meteorological convention, 0 = North) |
| `dir_50` | float | ✓ | ✓ | Median wind direction (degrees) — informational, not used by the scorer |
| `dir_95` | float | ✓ | ✓ | Upper bound of the 90% direction PI (degrees) |

**Grid row columns not applicable to stations** (`latitude`, `longitude`, `level`) should be left empty or NaN on station rows. **Station column** (`station`) should be left empty on grid rows. 

---

## Scoring — 36-dimensional rank-based evaluation

Your submission is scored on **36 independent dimensions**, defined by the cross-product:

- **3 horizons**: `d1`, `d7`, `d14`
- **2 problems**: wind **speed** (Winkler score on `q05`/`q95`) and wind **direction** (circular Winkler score on `dir_05`/`dir_95`)
- **3 ground truths**:
  - `stations` — in-situ buoy / mast / platform measurements (8 sites in North Sea, 7 in East China Sea)
  - `surface` — reanalysis pooled over `10m` and `100m` levels
  - `pressure` — reanalysis pooled over `1000`, `925`, `850`, `700`, `500` hPa levels
- **2 regions**: North Sea (`ns`) and East China Sea (`ecs`) — scored independently

That gives **3 × 2 × 3 × 2 = 36** dimensions. The column-key convention is `{problem}_{gt}_d{horizon}_{region}`, e.g. `speed_surface_d7_ns` or `dir_pressure_d14_ecs`. The leaderboard publishes all 36 raw numeric scores plus two aggregate columns:

### Per-dimension metrics

Both metrics are **interval scores on the 90% prediction interval** .

**Speed — Winkler interval score** on `(q05, q95)`:

$$
\text{WS} = (q_{95} - q_{05}) + \tfrac{2}{\alpha}\max(0, q_{05} - y) + \tfrac{2}{\alpha}\max(0, y - q_{95}), \quad \alpha = 0.1
$$

**Direction — circular Winkler score** on `(dir_05, dir_95)`:

$$
\text{cWS} = w + \tfrac{2}{\alpha}\,d_{\text{miss}}\cdot\mathbb{1}[\text{actual} \notin \text{arc}], \quad \alpha = 0.1
$$

---

## Packaging

1. Produce `predictions.csv` (one of the starting kits generates it for you).
2. Zip it: `zip submission.zip predictions.csv`
3. Upload `submission.zip` to Codabench.

## Final Scoring

- **`mean_rank`** — the mean of your per-dimension ranks across submissions is computed by a separate post-processor that runs periodically. **This is the final ranking metric for the competition.**
