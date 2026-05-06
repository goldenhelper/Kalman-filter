"""Exercises for learning Kalman filters with finance and crypto examples.

Fill in each function, then run:

    python -m pytest

The tests are intentionally deterministic so you can compare your output with
known-good calculations while still writing the implementation yourself.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def predict_state(
    x: np.ndarray,
    P: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Kalman prediction step for a linear Gaussian state-space model.

    Model:
        x_t = F x_{t-1} + process_noise
        process_noise ~ N(0, Q)

    Args:
        x: Previous state mean with shape ``(n,)``.
        P: Previous state covariance with shape ``(n, n)``.
        F: State transition matrix with shape ``(n, n)``.
        Q: Process noise covariance with shape ``(n, n)``.

    Returns:
        ``(x_pred, P_pred)``.
    """
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    
    return x_pred, P_pred

def update_state(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kalman measurement update step.

    Model:
        z_t = H x_t + measurement_noise
        measurement_noise ~ N(0, R)

    Args:
        x_pred: Predicted state mean with shape ``(n,)``.
        P_pred: Predicted state covariance with shape ``(n, n)``.
        z: Observation vector with shape ``(m,)``.
        H: Observation matrix with shape ``(m, n)``.
        R: Measurement noise covariance with shape ``(m, m)``.

    Returns:
        ``(x_updated, P_updated, K)``, where ``K`` is the Kalman gain.
    """
    raise NotImplementedError("Exercise: implement the Kalman update step.")


def run_scalar_random_walk_filter(
    observations: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_state: float,
    initial_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a latent fair value from noisy observed prices.

    This is the simplest finance-flavored Kalman filter:

        fair_value_t = fair_value_{t-1} + process_noise
        observed_price_t = fair_value_t + measurement_noise

    Think of ``observations`` as noisy mid-prices for a liquid crypto pair.

    Returns:
        ``(state_estimates, variance_estimates)`` arrays with one value per
        observation.
    """
    raise NotImplementedError("Exercise: implement the scalar random-walk filter.")


def estimate_time_varying_beta(
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_beta: float,
    initial_var: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a time-varying beta for an asset versus the market.

    Observation model:

        asset_return_t = beta_t * market_return_t + measurement_noise

    State model:

        beta_t = beta_{t-1} + process_noise

    This is a common building block for adaptive hedging and pairs trading.

    Returns:
        ``(beta_estimates, variance_estimates)`` arrays.
    """
    raise NotImplementedError("Exercise: implement the time-varying beta filter.")


def kalman_spread_zscore(
    exchange_a_prices: np.ndarray,
    exchange_b_prices: np.ndarray,
    process_var: float,
    measurement_var: float,
    initial_spread: float,
    initial_var: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth a cross-exchange log spread and compute residual z-scores.

    The observed spread is:

        log(exchange_a_price_t) - log(exchange_b_price_t)

    Treat the true spread mean as a random walk. After each update, compute:

        zscore_t = (observed_spread_t - estimated_spread_mean_t)
                   / sqrt(estimated_spread_variance_t + measurement_var)

    This mimics a simple crypto arbitrage signal where extreme residuals may
    indicate temporary dislocations between venues.

    Returns:
        ``(spread_mean_estimates, variance_estimates, zscores)`` arrays.
    """
    raise NotImplementedError("Exercise: implement the spread z-score filter.")

