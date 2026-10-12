"""
quant — data-driven global multi-asset forecasting engine (v3).

Design principles
-----------------
* Every input comes from a real, reproducible source (Yahoo Finance, FRED, Nasdaq).
* Distributions, not point targets: the model forecasts the *joint distribution*
  of returns across hundreds of assets, conditioned on the current regime.
* Expected returns are deliberately conservative (equilibrium + heavy shrinkage).
  Medium-term direction is close to unforecastable; risk / dependence is not.
* Every forecast is validated out-of-sample (see quant.backtest).
"""
