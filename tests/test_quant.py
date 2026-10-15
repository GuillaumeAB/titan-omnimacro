"""
Comprehensive offline unit tests for Titan OmniMacro.
Uses synthetic DGPs with known ground truth to verify mathematical correctness.
Zero external network calls required.
"""

import numpy as np
import pandas as pd
import pytest

from titan.models.covariance import ledoit_wolf, nearest_psd
from titan.models.factors import HybridFactorModel
from titan.models.rates import DynamicNelsonSiegel
from titan.models.regime import GaussianHMM
from titan.portfolio.rebalancing import evaluate_rebalancing
from titan.sim.simulator import RegimeFactorSimulator


def test_nearest_psd_and_ledoit_wolf():
    rng = np.random.default_rng(42)
    # Generate non-PSD matrix with negative eigenvalue
    a = rng.normal(0, 1, (10, 10))
    c_non_psd = (a + a.T) / 2.0
    c_non_psd[0, 0] = -10.0

    c_psd = nearest_psd(c_non_psd)
    eigvals = np.linalg.eigvalsh(c_psd)
    assert np.all(eigvals >= 0.0)

    # Ledoit-Wolf on T=100, N=10 returns
    x = rng.normal(0, 0.02, (100, 10))
    cov, shrinkage = ledoit_wolf(x)
    assert cov.shape == (10, 10)
    assert 0.0 <= shrinkage <= 1.0
    assert np.all(np.linalg.eigvalsh(cov) > 0.0)


def test_gaussian_hmm_ordering_and_filtering():
    rng = np.random.default_rng(101)
    # Synthetic 2-state process: State 0 (vol 0.01), State 1 (vol 0.05)
    t = 400
    regimes = np.zeros(t, dtype=int)
    for i in range(1, t):
        p_stay = 0.95
        regimes[i] = regimes[i - 1] if rng.uniform() < p_stay else 1 - regimes[i - 1]

    vols = np.array([0.01, 0.05])
    returns = rng.normal(0, vols[regimes]).reshape(-1, 1)

    hmm = GaussianHMM(n_states=2, n_restarts=3, seed=42)
    hmm.fit(returns)

    # Verify that State 0 has smaller variance than State 1
    assert hmm.covars_[0, 0, 0] < hmm.covars_[1, 0, 0]

    filtered = hmm.filter(returns)
    assert filtered.shape == (t, 2)
    assert np.allclose(np.sum(filtered, axis=1), 1.0)

    viterbi_path = hmm.viterbi(returns)
    assert len(viterbi_path) == t
    # State identification accuracy
    accuracy = np.mean(viterbi_path == regimes)
    assert accuracy > 0.75 or (1 - accuracy) > 0.75


def test_dynamic_nelson_siegel():
    # Synthetic term structure
    maturities = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
    dates = pd.date_range("2024-01-01", periods=50, freq="W-FRI")
    data = np.zeros((50, len(maturities)))
    for j, m in enumerate(maturities):
        # Upward sloping curve around 4.0%
        data[:, j] = 3.5 + 0.3 * np.log(m + 1.0) + np.random.normal(0, 0.02, 50)

    df_yields = pd.DataFrame(data, index=dates, columns=maturities)
    dns = DynamicNelsonSiegel(lam=0.7308)
    dns.fit(df_yields)

    assert dns.fit_rmse_bp < 30.0
    assert dns.factors_.shape == (50, 3)

    sim_paths = dns.simulate(horizon_months=12, n_paths=100)
    assert sim_paths.shape == (100, 12, len(maturities))


def test_hybrid_factor_model_and_simulator():
    rng = np.random.default_rng(77)
    dates = pd.date_range("2023-01-01", periods=120, freq="W-FRI")

    # Factors: MKT, USD, RATES
    f_df = pd.DataFrame({
        "MKT": rng.normal(0.001, 0.02, 120),
        "USD": rng.normal(0.0, 0.01, 120),
        "RATES": rng.normal(0.0, 0.05, 120),
    }, index=dates)

    # Assets: SPY (beta 1.0 to MKT), GLD (beta -0.2 to USD)
    y_df = pd.DataFrame({
        "SPY": 1.0 * f_df["MKT"] + rng.normal(0, 0.005, 120),
        "GLD": -0.4 * f_df["USD"] + rng.normal(0, 0.01, 120),
    }, index=dates)

    meta_df = pd.DataFrame([
        {"ticker": "SPY", "name": "S&P 500", "asset_class": "equity_index"},
        {"ticker": "GLD", "name": "Gold", "asset_class": "commodity"},
    ])

    hfm = HybridFactorModel(n_stat_factors=1, min_history=60)
    hfm.fit(y_df, f_df)

    assert "SPY" in hfm.betas_.index
    assert hfm.r2_["SPY"] > 0.60

    hmm = GaussianHMM(n_states=2, n_restarts=2, seed=42)
    hmm.fit(f_df.values)

    sim = RegimeFactorSimulator(
        model=hfm,
        hmm=hmm,
        hmm_input=f_df,
        asset_meta=meta_df,
        rf_annual=0.04,
    )

    out = sim.simulate(horizon_weeks=26, n_paths=500, seed=42)
    cum_log = out["cum_log"]
    assert cum_log.shape == (500, 2)

    summary = sim.summary(horizons={"6M": 26}, n_paths=500)
    assert len(summary) == 2
    for _, r in summary.iterrows():
        assert r["q05"] <= r["q50"] <= r["q95"]


def test_swensen_rebalancing_corridors():
    targets = {"SPX": 0.50, "BOND": 0.30, "GOLD": 0.20}
    # Within bands
    currents_ok = {"SPX": 0.51, "BOND": 0.29, "GOLD": 0.20}
    diag_ok = evaluate_rebalancing(targets, currents_ok)
    assert not diag_ok.is_rebalance_recommended

    # Out of band: SPX drifts from 50% to 58% (>5% absolute threshold)
    currents_drift = {"SPX": 0.58, "BOND": 0.25, "GOLD": 0.17}
    diag_drift = evaluate_rebalancing(targets, currents_drift)
    assert diag_drift.is_rebalance_recommended
    assert any(a["status"] == "OVERWEIGHT_TRIGGER" for a in diag_drift.actions)
