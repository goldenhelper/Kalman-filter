# Kalman Filter Exercises

This workspace contains small test-driven exercises for learning Kalman filters,
with examples aimed at price smoothing, adaptive hedge ratios, and simple
crypto spread signals.

## Setup

Install the lightweight dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

At first, the tests should fail because the functions in
`kalman_exercises/exercises.py` raise `NotImplementedError`. Fill in one
function at a time and rerun the relevant test.

## Suggested Order

1. `predict_state`: implement the Kalman prediction equations.
2. `update_state`: implement the measurement update equations.
3. `run_scalar_random_walk_filter`: smooth noisy crypto mid-prices.
4. `estimate_time_varying_beta`: estimate an adaptive asset beta versus a market
   benchmark.
5. `kalman_spread_zscore`: build a basic cross-exchange spread dislocation
   signal.

## Running One Exercise

Use pytest's `-k` option to focus on a single topic:

```bash
python -m pytest -k random_walk
python -m pytest -k beta
python -m pytest -k spread
```

## Hints

For a linear Kalman filter:

```text
x_pred = F x
P_pred = F P F.T + Q

y = z - H x_pred
S = H P_pred H.T + R
K = P_pred H.T inv(S)
x_updated = x_pred + K y
P_updated = (I - K H) P_pred
```

For scalar filters, the same equations reduce to a few multiplications and
divisions. That is useful for finance problems where the latent state is one
number, such as a fair value, beta, or spread mean.

