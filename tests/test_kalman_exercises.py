import numpy as np

from kalman_exercises.exercises import (
    estimate_time_varying_beta,
    kalman_spread_zscore,
    predict_state,
    run_scalar_random_walk_filter,
    update_state,
)


def test_predict_state_handles_position_velocity_model():
    x = np.array([100.0, 2.0])
    P = np.array([[4.0, 0.5], [0.5, 1.0]])
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.25, 0.0], [0.0, 0.10]])

    expected_x = np.array([102.0, 2.0])
    expected_P = np.array([[6.25, 1.50], [1.50, 1.10]])
    actual_x, actual_P = predict_state(x, P, F, Q)

    np.testing.assert_allclose(actual_x, expected_x)
    np.testing.assert_allclose(actual_P, expected_P)


def test_update_state_combines_prior_with_noisy_price_and_velocity_observation():
    x_pred = np.array([102.0, 2.0])
    P_pred = np.array([[6.25, 1.5], [1.5, 1.1]])
    z = np.array([101.25, 1.2])
    H = np.array([[1.0, 0.0], [0.0, 1.0]])
    R = np.array([[1.0, 0.0], [0.0, 0.25]])

    expected_x = np.array([101.22512438, 1.35505804])
    expected_P = np.array([[0.82089552, 0.04975124], [0.04975124, 0.18988391]])
    expected_K = np.array([[0.82089552, 0.19900498], [0.04975124, 0.75953566]])
    actual_x, actual_P, actual_K = update_state(x_pred, P_pred, z, H, R)

    np.testing.assert_allclose(actual_x, expected_x)
    np.testing.assert_allclose(actual_P, expected_P)
    np.testing.assert_allclose(actual_K, expected_K)


def test_random_walk_filter_smooths_noisy_crypto_mid_prices():
    observations = np.array([100.0, 101.2, 100.7, 102.5, 103.1, 102.8])

    expected_states = np.array(
        [99.89743590, 100.52153589, 100.58874724, 101.22719751, 101.81642166, 102.11677996]
    )
    expected_vars = np.array(
        [0.31794872, 0.19165275, 0.15064394, 0.13361922, 0.12584865, 0.12214921]
    )
    actual_states, actual_vars = run_scalar_random_walk_filter(
        observations,
        process_var=0.05,
        measurement_var=0.40,
        initial_state=99.5,
        initial_var=1.50,
    )

    np.testing.assert_allclose(actual_states, expected_states)
    np.testing.assert_allclose(actual_vars, expected_vars)
    assert actual_states.shape == observations.shape


def test_random_walk_filter_trusts_price_more_when_measurement_noise_is_low():
    observations = np.array([20_000.0, 20_050.0, 20_010.0])

    smooth_states, _ = run_scalar_random_walk_filter(
        observations,
        process_var=1.0,
        measurement_var=10_000.0,
        initial_state=19_900.0,
        initial_var=100.0,
    )
    reactive_states, _ = run_scalar_random_walk_filter(
        observations,
        process_var=1.0,
        measurement_var=10.0,
        initial_state=19_900.0,
        initial_var=100.0,
    )

    smooth_error = abs(smooth_states[-1] - observations[-1])
    reactive_error = abs(reactive_states[-1] - observations[-1])
    assert reactive_error < smooth_error


def test_time_varying_beta_estimates_adaptive_eth_beta_to_btc_market():
    market_returns = np.array([0.010, -0.020, 0.015, 0.005, -0.010, 0.030])
    asset_returns = np.array([0.014, -0.031, 0.021, 0.008, -0.017, 0.046])

    expected_betas = np.array(
        [1.04601770, 1.20951214, 1.24042606, 1.24722821, 1.28102640, 1.38637202]
    )
    expected_vars = np.array(
        [0.46017699, 0.32440512, 0.28851225, 0.30267605, 0.29858915, 0.18556874]
    )
    actual_betas, actual_vars = estimate_time_varying_beta(
        asset_returns,
        market_returns,
        process_var=0.02,
        measurement_var=0.0004,
        initial_beta=1.0,
        initial_var=0.50,
    )

    np.testing.assert_allclose(actual_betas, expected_betas)
    np.testing.assert_allclose(actual_vars, expected_vars)
    assert actual_betas[-1] > 1.0


def test_spread_zscore_flags_cross_exchange_dislocation():
    exchange_a_prices = np.array([100.0, 100.2, 100.4, 103.2, 100.6, 100.5])
    exchange_b_prices = np.array([100.1, 100.1, 100.3, 100.4, 100.4, 100.6])

    expected_means = np.array(
        [
            -0.00096109,
            0.00001208,
            0.00034955,
            0.00765216,
            0.00636604,
            0.00488384,
        ]
    )
    expected_vars = np.array(
        [0.00038463, 0.00019865, 0.00013712, 0.00010756, 0.00009086, 0.00008055]
    )
    expected_zscores = np.array(
        [-0.00137107, 0.04031585, 0.02791525, 0.88128150, -0.19751437, -0.26815685]
    )

    actual_means, actual_vars, actual_zscores = kalman_spread_zscore(
        exchange_a_prices,
        exchange_b_prices,
        process_var=0.00001,
        measurement_var=0.0004,
        initial_spread=0.0,
        initial_var=0.01,
    )

    np.testing.assert_allclose(actual_means, expected_means)
    np.testing.assert_allclose(actual_vars, expected_vars)
    np.testing.assert_allclose(actual_zscores, expected_zscores)
    assert int(np.argmax(np.abs(actual_zscores))) == 3

