# quant v3 — Module Interface Contract

All agents MUST follow these signatures exactly. Python 3.14, numpy 2.5, pandas 3.0,
scipy 1.18, yfinance 1.7, requests, fastapi. **No scikit-learn / statsmodels / pyarrow /
hmmlearn** — implement with numpy/scipy. Use `pd.to_pickle` / `pd.read_pickle` for caches.
Run code from repo root: `c:\Users\gbbay\Documents\antigravity\magical-fermi`.
Every package dir needs an `__init__.py`. Config lives in `quant/config.py` (read it).

Conventions
- Weekly frequency = `W-FRI`. Returns are **log returns** unless stated otherwise.
- `^TNX` is quoted in percent (5.24 = 5.24%). The RATES factor = weekly *change* in pp.
- Tickers use Yahoo format (`BRK-B`, `NESN.SW`, `7203.T`, `CL=F`, `EURUSD=X`).
- Never fabricate data. If a source fails, raise / return empty and log a warning.

---------------------------------------------------------------------------------
## quant/data/  (owner: DATA agent)

### universe.py
`get_universe(tier: str = "core") -> pd.DataFrame`
  columns: `ticker, name, asset_class, region, country, sector, currency`
  asset_class ∈ {equity_index, equity_country, equity_sector, equity_stock, bond,
                 commodity, fx, rates_yield, crypto}
  tier "core": ~250 curated global instruments (country ETFs for ~45 countries incl.
  Africa/Middle East/Eastern Europe/LatAm/Asia, sector & thematic ETFs, bond ETFs,
  commodities futures, FX, crypto, ~120 global mega-cap stocks across US/Europe/Japan/
  Asia/EM). Must include all `config.FACTOR_PROXY_TICKERS`.
  tier "extended": core + full S&P 500 + S&P 400 (via membership.py) — hundreds more.

### loader.py
`load_prices(tickers: list[str], start: str = config.PRICE_START, refresh: bool = False) -> pd.DataFrame`
  daily auto-adjusted close, DatetimeIndex (tz-naive), columns = tickers that returned data.
  Batch yfinance downloads (chunks ~100), per-ticker pickle cache in `config.CACHE_DIR/prices/`
  with max age `config.CACHE_MAX_AGE_HOURS`. Non-positive prices -> NaN.
`to_weekly(prices: pd.DataFrame) -> pd.DataFrame`  last obs per W-FRI week.
`weekly_log_returns(prices: pd.DataFrame, yield_tickers: list[str] = ()) -> pd.DataFrame`
  log returns of weekly prices; for `yield_tickers` (asset_class rates_yield) returns the
  weekly *difference* instead. Drops all-NaN rows.

### fred.py
`load_fred_series(series_ids: list[str], start: str = "1990-01-01") -> pd.DataFrame`
  via `https://fred.stlouisfed.org/graph/fredgraph.csv?id=ID1,ID2` (no key), cached.
`load_treasury_curve(start: str = "1990-01-01") -> pd.DataFrame`
  columns are float maturities in years: [0.25, 1, 2, 3, 5, 7, 10, 20, 30]
  from DGS3MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 (percent).
`latest_risk_free() -> float`  annual decimal from DGS3MO (e.g. 0.051).

### membership.py
`get_sp500() -> pd.DataFrame`      columns `ticker, name, sector, date_added`
`get_sp400() -> pd.DataFrame`      columns `ticker, name, sector`
`get_nasdaq100() -> pd.DataFrame`  columns `ticker, name`  (Nasdaq API
   `https://api.nasdaq.com/api/quote/list-type/nasdaq100`, fallback Wikipedia)
`get_nasdaq_screener(exchange: str) -> pd.DataFrame`  exchange ∈ {nasdaq, nyse, amex}
   `https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=10000&exchange=X&download=true`
   columns `ticker, name, market_cap (float USD), last_price, volume, country, ipo_year, sector, industry, exchange`
Wikipedia: `pd.read_html(io.StringIO(html))`, browser User-Agent (config.HTTP_HEADERS),
3 retries with backoff (Wikipedia sometimes times out). Cache daily. Normalise `.`→`-`.

### ipo.py
`get_ipo_calendar(months: list[str] | None = None) -> dict[str, pd.DataFrame]`
   keys `priced, upcoming, filed`; `https://api.nasdaq.com/api/ipo/calendar?date=YYYY-MM`.
   Parse numeric price (range "15.00-17.00" -> midpoint), shares, offer amount (USD float), dates.
   Default months = previous, current, next month.

---------------------------------------------------------------------------------
## quant/models/  (owners: MODELS-A = covariance/regime/rates, MODELS-B = factors/simulator)

### covariance.py
`ledoit_wolf(X: np.ndarray) -> tuple[np.ndarray, float]`  (T×N, demeaned inside) -> (cov, shrinkage)
   Ledoit–Wolf 2004 constant-correlation OR scaled-identity target (document which).
`ewma_cov(X: np.ndarray, halflife: float) -> np.ndarray`
`nearest_psd(C: np.ndarray, eps: float = 1e-10) -> np.ndarray`

### regime.py
`class GaussianHMM(n_states=3, n_restarts=4, max_iter=200, tol=1e-6, seed=0, cov_floor=1e-8)`
  `.fit(X: np.ndarray) -> self`   Baum–Welch in log-space, full covariances, multiple restarts
                                  (k-means++-style init), keep best log-likelihood.
  `.filter(X) -> np.ndarray (T×K)`  forward filtered probs P(s_t | x_1..t)
  `.smooth(X) -> np.ndarray (T×K)`
  `.viterbi(X) -> np.ndarray (T,)`
  attributes after fit: `startprob_ (K,)`, `transmat_ (K×K)`, `means_ (K×D)`, `covars_ (K×D×D)`,
     `loglik_`, `state_order_`; states MUST be re-ordered so state 0 = lowest volatility of
     column 0 and state K-1 = highest volatility of column 0.
  `.state_names` -> e.g. ["Calm", "Normal", "Stress"] for K=3.

### rates.py
`class DynamicNelsonSiegel(lam: float = 0.7308)`  (Diebold–Li, maturities in years)
  `.fit(yields: pd.DataFrame) -> self`  monthly yields (columns float maturities). Cross-section
      OLS each month -> factors (level, slope, curvature); VAR(1) on factors; store residuals.
  `.simulate(horizon_months: int, n_paths: int, seed: int = 0) -> np.ndarray`  (n_paths, horizon_months, n_maturities)
      residual bootstrap.
  `.forecast_summary(horizons_months: dict[str,int], quantiles: list[float]) -> dict`
      {"maturities": [...], "current": [...], "horizons": {"1Y": {"q05":[...],...,"mean":[...]}}}
  `.factors_ (DataFrame)`, `.fit_rmse_bp`.

### factors.py
`build_factor_returns(weekly_prices: pd.DataFrame) -> pd.DataFrame`
   weekly factor series, columns = keys of `config.FACTOR_PROXIES`, using rules in config.
`class HybridFactorModel(n_stat_factors=config.N_STAT_FACTORS, min_history=config.MIN_HISTORY_WEEKS)`
  `.fit(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> self`
     1) per-asset OLS of returns on macro factors over overlapping non-NaN weeks
        (assets with < min_history obs are dropped -> `.dropped_`)
     2) PCA (SVD) on residuals of assets with complete coverage in the window ->
        `n_stat_factors` statistical factor return series (STAT1..STATk)
     3) regress every asset's macro residual on the STAT series -> stat betas
  attributes: `factor_names_` (macro + STAT), `factor_returns_` (DataFrame T×F, full set),
     `betas_` (DataFrame N×F), `alpha_` (Series, weekly), `resid_vol_` (Series, weekly std),
     `resid_` (DataFrame T×N, final idiosyncratic residuals, NaN where missing),
     `r2_` (Series), `stat_explained_var_` (list), `assets_` (list).

### simulator.py
`class RegimeFactorSimulator(model: HybridFactorModel, hmm: GaussianHMM, hmm_input: pd.DataFrame,
                             asset_meta: pd.DataFrame, rf_annual: float,
                             factor_premia_annual: dict = config.FACTOR_PREMIA_ANNUAL,
                             alpha_shrinkage: float = config.ALPHA_SHRINKAGE)`
   hmm_input = the (T×D) factor DataFrame the HMM was fitted on (aligned to model.factor_returns_).
   Factor draws: Markov chain on regimes starting from the last filtered probability; each
   week draws a historical factor-return vector (all F factors) from weeks whose smoothed
   state = drawn state (block length 1, posterior-weighted). Factor returns are re-centred:
   subtract unconditional historical mean, add `premia/52` (STAT factors premium 0).
   Asset weekly log return = drift_i + betas_i · f_t + eps_i,
      drift_i = carry_i + alpha_shrinkage*alpha_i_hist_excess,
      carry_i = rf/52 for equity/bond/crypto, 0 for commodity/fx;
      eps_i bootstrapped from the asset's own residuals (fallback N(0,resid_vol)).
   `.simulate(horizon_weeks: int, n_paths: int = 5000, seed: int = 0,
              scenario: dict | None = None, path_tickers: list[str] | None = None) -> dict`
       returns {"cum_log": np.ndarray (n_paths, N) at horizon,
                "paths": {ticker: np.ndarray (n_paths, horizon_weeks)} cumulative log for path_tickers,
                "factor_cum": np.ndarray (n_paths, F)}
       scenario = {"shocks": {"OIL": 0.4, "RATES": 1.0, ...}}  cumulative factor moves over the horizon
       (log for price factors, pp for RATES). Apply Gaussian conditioning on cumulative factor sums:
       S_free += Σ_fc Σ_cc^{-1}(s − S_c);  S_c = s  — then rebuild asset returns from S.
   `.summary(horizons: dict[str,int] = config.HORIZONS_WEEKS, n_paths=5000, seed=0, scenario=None) -> pd.DataFrame`
       one row per (ticker, horizon): q05..q95 (SIMPLE returns), mean, prob_loss, es95, ann_vol.
   `.fan_chart(tickers, horizon_weeks=260, n_paths=3000, step=4) -> dict`
       {ticker: {"weeks":[...], "q05":[...], "q25":[...], "q50":[...], "q75":[...], "q95":[...],
                 "max_dd_median": float}}  values = growth of 1 (simple).

---------------------------------------------------------------------------------
## quant/index/  (owner: INDEX agent)

### recon.py
`predict_nasdaq100_reconstitution(screener: pd.DataFrame, members: pd.DataFrame,
        prices: pd.DataFrame | None = None, n_sims: int = 5000, days_to_reference: int | None = None) -> pd.DataFrame`
   Universe = Nasdaq-listed, non-Financial (screener sector != "Finance"), dedupe share classes
   (keep largest per issuer name stem). Rank by market cap. Monte Carlo market caps to the
   reference date (last trading day of October; if passed, next year) with a one-factor
   lognormal model (vol from `prices` 1Y daily returns if available, else 45%), apply the
   100 / 125-buffer rule. Columns: ticker, name, market_cap, rank_now, is_member,
   prob_in_index, predicted_action ∈ {ADD, DELETE, KEEP, WATCH-ADD, WATCH-DELETE}.
`sp500_addition_candidates(screeners: pd.DataFrame, sp500: pd.DataFrame, sp400: pd.DataFrame,
        prices: pd.DataFrame | None = None, check_profitability: bool = True, top_n: int = 40) -> pd.DataFrame`
   US-domiciled non-members above ~70% of threshold, profitability via yfinance
   (sum trailing 4Q net income > 0 and latest quarter > 0, use `Ticker.quarterly_income_stmt`;
   mark "unknown" on failure), IPO seasoning, prob_above_threshold in 6 months (lognormal).
   columns: ticker, name, market_cap, exchange, in_sp400, profitable, seasoning_ok,
   prob_above_threshold_6m, score, notes.
`sp500_deletion_risk(screeners, sp500, top_n=30) -> pd.DataFrame`  smallest members by market cap
   with ratio to threshold.
`ipo_index_timeline(ipo_frames: dict[str, pd.DataFrame]) -> pd.DataFrame`
   ticker, company, exchange, expected_date, offer_amount, est_market_cap (if derivable),
   sp500_earliest_date, ndx_earliest_date, notes.

---------------------------------------------------------------------------------
## quant/backtest/walkforward.py  (owner: BACKTEST agent, wave 2)
`run_walkforward(asset_returns, factor_returns, asset_meta, horizon_weeks=52, step_weeks=13,
                 start="2013-01-01", n_paths=2000, seed=0) -> dict`
   expanding/rolling refit at each origin; outputs coverage (50/80/90), PIT histogram (10 bins),
   CRPS model vs naive (zero-drift Gaussian random walk with trailing vol), rank IC of mean
   forecast vs realised, breakdown by asset_class.

---------------------------------------------------------------------------------
## output/latest.json  (written by quant/pipeline.py, consumed by the web UI)
```json
{
 "as_of": "2026-10-12",
 "meta": {"n_assets": 0, "n_weeks": 0, "rf_annual": 0.0, "factor_premia": {}, "sources": [], "runtime_sec": 0},
 "regime": {"state_names": [], "current_probs": [], "transmat": [[]],
            "history": [{"date": "YYYY-MM-DD", "probs": []}],
            "state_stats": [{"name": "", "ann_mkt_return": 0, "ann_mkt_vol": 0, "freq": 0}]},
 "factor_model": {"factors": [], "stat_explained_var": [], "median_r2": 0,
                  "betas": {"TICKER": {"MKT": 0}}},
 "forecasts": [{"ticker": "", "name": "", "asset_class": "", "region": "", "country": "",
                "sector": "", "last_price": 0, "ann_vol": 0, "r2": 0,
                "horizons": {"1Y": {"q05":0,"q10":0,"q25":0,"q50":0,"q75":0,"q90":0,"q95":0,
                                     "mean":0,"prob_loss":0,"es95":0}}}],
 "fan_charts": {"TICKER": {"weeks": [], "q05": [], "q25": [], "q50": [], "q75": [], "q95": [], "max_dd_median": 0}},
 "scenarios": [{"id": "", "name": "", "description": "", "horizon_weeks": 52, "shocks": {},
                "results": [{"ticker": "", "name": "", "asset_class": "", "q05": 0, "q50": 0, "q95": 0, "delta_vs_base_q50": 0}]}],
 "rates": {"maturities": [], "current": [], "horizons": {"1Y": {"q05": [], "q50": [], "q95": [], "mean": []}}, "fit_rmse_bp": 0},
 "index_recon": {"reference_date": "", "nasdaq100": [], "sp500_additions": [], "sp500_deletion_risk": []},
 "ipos": {"priced": [], "upcoming": [], "filed": [], "index_timeline": []},
 "backtest": {"horizon_weeks": 52, "n_origins": 0, "coverage": {"50": 0, "80": 0, "90": 0},
              "crps_model": 0, "crps_naive": 0, "rank_ic_mean": 0, "pit_hist": [], "by_asset_class": {}}
}
```
All returns in JSON are SIMPLE returns in decimal (0.12 = +12%). Missing sections = null.
