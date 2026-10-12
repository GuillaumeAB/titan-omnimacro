"""
OmniMacro Engine - Macroeconomic & Central Bank Financial Feeds
Captures yield curves, central bank policy trajectories, inflation, and commodity benchmarks.
"""

from typing import Dict, Any
from core.config import BASELINE_MACRO


class MacroFeed:
    """Provides live or calibrated macroeconomic baseline indicators."""

    def __init__(self):
        self.baseline = dict(BASELINE_MACRO)

    def get_current_macro_snapshot(self) -> Dict[str, Any]:
        """Returns macroeconomic state vector for multi-horizon discounting."""
        return {
            "us_10y_nominal": self.baseline["us_10y_yield"],
            "us_10y_real": self.baseline["us_10y_yield"] - self.baseline["us_cpi_inflation"],
            "fed_funds": self.baseline["fed_funds_rate"],
            "ecb_deposit": self.baseline["ecb_deposit_rate"],
            "cpi_inflation": self.baseline["us_cpi_inflation"],
            "gdp_growth_trend": self.baseline["us_real_gdp_trend"],
            "brent_crude": self.baseline["brent_crude_usd"],
            "european_gas_mwh": self.baseline["tt_gas_eur_mwh"],
            "gpr_index": self.baseline["geopolitical_risk_index"],
            "hyperscaler_capex_annual_bn": 220.0,  # Combined MSFT, GOOGL, AMZN, META AI capex
            "tsmc_monthly_rev_bn_usd": 8.5
        }
