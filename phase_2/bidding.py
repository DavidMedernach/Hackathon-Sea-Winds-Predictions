"""Pinson-style optimal bidding from probabilistic wind power forecasts.

Reference: Pinson, Chevallier & Kariniotakis (2007),
    "Trading Wind Generation From Short-Term Probabilistic Forecasts of
    Wind Power", IEEE Transactions on Power Systems 22(3), 1148-1156.

Key result: under linear up/down imbalance penalties pi^+ / pi^-, the
expected-revenue-maximising bid is the empirical quantile of the
predicted distribution at level

    tau* = pi^- / (pi^- + pi^+)

i.e. underestimate the production whenever down-balancing (excess) is
penalised harder than up-balancing (shortfall), and vice versa. This
module provides:

    - `optimal_bid_quantile`: returns tau* for given asymmetric prices.
    - `interpolate_quantile_bid`: turn (q05, q50, q95) into the optimal bid.
    - `score_bidding_revenue`: compute realised revenue given bid + actual
      production + day-ahead and imbalance prices.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Optimal-quantile selection ──────────────────────────────────────────

def optimal_bid_quantile(price_imbalance_up: float, price_imbalance_down: float) -> float:
    """Pinson 2007 closed-form optimal quantile.

    Parameters
    ----------
    price_imbalance_up
        Penalty per MWh paid when realised production < bid (i.e. we need
        to buy back power at a higher price). pi^+ in the paper.
    price_imbalance_down
        Penalty per MWh when realised production > bid (excess power dumped
        at a lower price). pi^- in the paper.

    Returns
    -------
    tau* in [0, 1].
    """
    if price_imbalance_up <= 0 and price_imbalance_down <= 0:
        return 0.5
    return float(price_imbalance_down / (price_imbalance_down + price_imbalance_up))


def interpolate_quantile_bid(
    q05: float | np.ndarray, q50: float | np.ndarray, q95: float | np.ndarray,
    tau: float | np.ndarray,
) -> np.ndarray:
    """Interpolate from a (q05, q50, q95) triplet to an arbitrary tau in [0, 1].

    Piecewise linear in (0.05, 0.5, 0.95). Outside [0.05, 0.95] we extrapolate
    linearly using the nearest pair of quantiles - intentionally conservative
    rather than introducing parametric tail assumptions. Vectorised over
    arrays of any compatible shape.
    """
    q05 = np.asarray(q05, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q95 = np.asarray(q95, dtype=float)
    tau = np.asarray(tau, dtype=float)

    bid = np.where(
        tau < 0.05,
        q05 + (tau - 0.05) / (0.50 - 0.05) * (q50 - q05),
        np.where(
            tau <= 0.50,
            q05 + (tau - 0.05) / (0.50 - 0.05) * (q50 - q05),
            np.where(
                tau <= 0.95,
                q50 + (tau - 0.50) / (0.95 - 0.50) * (q95 - q50),
                q50 + (tau - 0.50) / (0.95 - 0.50) * (q95 - q50),
            ),
        ),
    )
    # Bid energy can never be negative
    return np.maximum(bid, 0.0)


# ── Revenue scoring ─────────────────────────────────────────────────────

@dataclass
class BiddingResult:
    """Aggregate revenue + diagnostics for a bidding strategy."""
    revenue_eur: float                    # net revenue over the window
    revenue_per_mwh_eur: float            # per produced MWh
    energy_produced_mwh: float
    energy_bid_mwh: float
    imbalance_cost_eur: float             # always non-negative
    naive_q50_revenue_eur: float          # if you had bid q50 every step
    flat_price_revenue_eur: float         # revenue under price = const(mean)
    bid_quantile_used: float              # tau*

    def summary(self) -> str:
        return (
            f"Bidding revenue: {self.revenue_eur/1e6:.2f} M€  "
            f"(naive q50: {self.naive_q50_revenue_eur/1e6:.2f}, "
            f"flat-price ref: {self.flat_price_revenue_eur/1e6:.2f})\n"
            f"Imbalance cost: {self.imbalance_cost_eur/1e6:.2f} M€  "
            f"({self.imbalance_cost_eur / max(self.revenue_eur, 1) * 100:+.1f}% of net)\n"
            f"Energy: produced {self.energy_produced_mwh/1e3:.1f} GWh, "
            f"bid {self.energy_bid_mwh/1e3:.1f} GWh, "
            f"tau* = {self.bid_quantile_used:.3f}"
        )


def score_bidding_revenue(
    bid_mwh: np.ndarray,
    actual_mwh: np.ndarray,
    price_da_eur_per_mwh: np.ndarray,
    price_imb_up_eur_per_mwh: np.ndarray | float,
    price_imb_down_eur_per_mwh: np.ndarray | float,
    tau_used: float = 0.5,
) -> BiddingResult:
    """Realised revenue under day-ahead bid + imbalance settlement.

    For each timestep t:
        revenue_t = price_DA_t * bid_t
                  - pi_up_t   * max(bid_t   - actual_t, 0)
                  - pi_down_t * max(actual_t - bid_t,   0)

    Compared against two references:
        - naive_q50_revenue: same revenue formula with bid = q50 (= mean if
          symmetric). Tells you how much the optimal-quantile choice helped.
        - flat_price_revenue: same actual production but with a flat
          (mean) day-ahead price; isolates the wind-price covariance effect.
    """
    bid = np.asarray(bid_mwh, dtype=float)
    act = np.asarray(actual_mwh, dtype=float)
    pda = np.asarray(price_da_eur_per_mwh, dtype=float)
    if np.isscalar(price_imb_up_eur_per_mwh):
        pup = np.full_like(pda, float(price_imb_up_eur_per_mwh))
    else:
        pup = np.asarray(price_imb_up_eur_per_mwh, dtype=float)
    if np.isscalar(price_imb_down_eur_per_mwh):
        pdn = np.full_like(pda, float(price_imb_down_eur_per_mwh))
    else:
        pdn = np.asarray(price_imb_down_eur_per_mwh, dtype=float)

    da_revenue = (pda * bid).sum()
    short = np.maximum(bid - act, 0.0)
    excess = np.maximum(act - bid, 0.0)
    imbalance_cost = (pup * short).sum() + (pdn * excess).sum()
    revenue = float(da_revenue - imbalance_cost)
    energy_prod = float(act.sum())

    naive_q50_revenue = float(
        (pda * act).sum()  # bid = actual = perfect = no imbalance
    )
    # Revenue with prices replaced by the mean day-ahead price
    pda_flat = np.full_like(pda, pda.mean())
    flat_short = np.maximum(bid - act, 0.0)
    flat_excess = np.maximum(act - bid, 0.0)
    flat_imb = (pup * flat_short).sum() + (pdn * flat_excess).sum()
    flat_revenue = float((pda_flat * bid).sum() - flat_imb)

    return BiddingResult(
        revenue_eur=revenue,
        revenue_per_mwh_eur=revenue / max(energy_prod, 1),
        energy_produced_mwh=energy_prod,
        energy_bid_mwh=float(bid.sum()),
        imbalance_cost_eur=float(imbalance_cost),
        naive_q50_revenue_eur=naive_q50_revenue,
        flat_price_revenue_eur=flat_revenue,
        bid_quantile_used=tau_used,
    )


# ── Convenience: bid from a quantile-prediction DataFrame ───────────────

def bid_from_predictions(
    pred: pd.DataFrame,
    price_imb_up_eur_per_mwh: float | np.ndarray,
    price_imb_down_eur_per_mwh: float | np.ndarray,
) -> np.ndarray:
    """Build the optimal bid series from a prediction DataFrame.

    `pred` must have columns ``q05, q50, q95`` (in MWh per step). Returns
    a 1-D array of bid quantities, length = len(pred).
    """
    for col in ("q05", "q50", "q95"):
        if col not in pred.columns:
            raise ValueError(f"prediction frame missing '{col}'")
    if np.isscalar(price_imb_up_eur_per_mwh) and np.isscalar(price_imb_down_eur_per_mwh):
        tau = optimal_bid_quantile(
            float(price_imb_up_eur_per_mwh), float(price_imb_down_eur_per_mwh)
        )
        bid = interpolate_quantile_bid(
            pred["q05"].to_numpy(), pred["q50"].to_numpy(), pred["q95"].to_numpy(),
            tau,
        )
        return bid
    # Time-varying: per-step tau then per-step interp
    pup = np.asarray(price_imb_up_eur_per_mwh, dtype=float)
    pdn = np.asarray(price_imb_down_eur_per_mwh, dtype=float)
    denom = pup + pdn
    tau = np.where(denom > 0, pdn / np.maximum(denom, 1e-9), 0.5)
    return interpolate_quantile_bid(
        pred["q05"].to_numpy(), pred["q50"].to_numpy(), pred["q95"].to_numpy(),
        tau,
    )
