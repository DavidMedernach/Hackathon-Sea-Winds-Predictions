# Phase 2 - canonical siting submission

The scored siting submission is a fixed **55 × IEA 22 MW** (1210 MW) farm, **fixed-bottom**
(water depth ≤ 50 m), produced by `part2_siting/3b_farm_optimization_refined.ipynb`.

| field | value |
|---|---|
| file | `part2_siting/submission.json` |
| turbine | **IEA 22 MW × 55** (= 1210 MW), fixed-bottom monopile |
| centre | a shallow site (≤ 50 m), e.g. **53.5 °N, 1.5 °E** (26 m) |
| ranked on | **capacity factor** (yield); see `internal/simulator/score_phase2.py` |
| kit baseline (plain grid at 53.5°N, 1.5°E, illustrative) | **CF 53.8 % · AEP 5707 GWh · LCOE 82.0 €/MWh · CAPEX 4347 M€ · wake 5.9 % · depth 26 m** |

Regenerate the scores with `python internal/simulator/score_phase2.py <in> <out>`
(submission at `<in>/res/submission.json`, lsm at `<in>/ref/`; bathymetry is read from the kit).

## Notes
- Enforced constraints: exactly **55 × IEA_22MW**, water depth **≤ 50 m** (fixed-bottom, no
  floating), **15 × 15 km** box, **≥ 5 D** spacing, centre in the allowed zone over sea.
- The economic analysis (`part3_economics/1_economic_analysis.ipynb`) reads this submission.
- Shear exponent is single-sourced in `shear.py` (α = 0.11).
- The baseline above uses a plain grid; the optimiser notebooks improve on it.
