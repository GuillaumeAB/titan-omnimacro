"""
OmniMacro Engine - Factor Transmission & Fundamental Shock Matrix
Translates geopolitical, technological, and macro shocks into constituent-level EPS and P/E impacts.
"""

from typing import Dict, Tuple
from core.models import Constituent, ScenarioConfig


class ShockTransmissionMatrix:
    """Calculates bottom-up fundamental shocks on constituent earnings and valuation multiples."""

    @staticmethod
    def compute_constituent_monthly_shocks(
        c: Constituent,
        scenario: ScenarioConfig,
        month: int
    ) -> Tuple[float, float]:
        """
        Computes (annualized_eps_growth_shock, pe_multiple_multiplier) for a constituent
        at a given month in the 5-year simulation.
        """
        # Check if scenario is active at this month
        if month < scenario.start_month or month > (scenario.start_month + scenario.duration_months):
            # Baseline mean reversion
            return 0.08, 1.0

        # --- 1. EPS REVENUE & MARGIN SHOCK ---
        # AI Productivity impact (+/-)
        ai_eps_delta = scenario.ai_productivity_multiplier * (c.ai_exposure_score * 0.14)

        # Taiwan Semiconductor disruption shock
        # High dependency means severe supply-chain halt; loss of key silicon
        taiwan_disruption = scenario.taiwan_blockade_intensity * (c.taiwan_semiconductor_dependency * -0.32)

        # Energy / Oil shock impact
        # Base oil is $77. If target is $135, delta is +75%
        oil_delta_pct = (scenario.oil_price_target - 77.5) / 77.5
        energy_margin_delta = oil_delta_pct * (c.energy_cost_elasticity * 0.12)

        # Defense rearmament boost
        # Scaled by Ukraine intensity and geopolitical stress
        defense_impulse = (
            scenario.ukraine_conflict_intensity * 0.5 + 
            scenario.taiwan_blockade_intensity * 0.3 +
            scenario.middle_east_crisis_intensity * 0.2
        )
        defense_eps_delta = defense_impulse * (c.defense_rearmament_boost * 0.24)

        # Tariff & Deglobalization drag
        tariff_drag = (scenario.tariff_level_pct / 100.0) * (c.tariff_vulnerability * -0.15)

        # Net constituent EPS growth rate (annualized)
        base_organic_growth = 0.075 * c.beta  # Baseline nominal trend
        net_eps_growth = (
            base_organic_growth +
            ai_eps_delta +
            taiwan_disruption +
            energy_margin_delta +
            defense_eps_delta +
            tariff_drag
        )

        # --- 2. VALUATION MULTIPLE (P/E) SHOCK ---
        # Rates impact (duration sensitivity)
        rate_delta_pct = (scenario.fed_funds_rate_target - 4.0) / 100.0
        rate_pe_drag = -1.0 * (rate_delta_pct * c.interest_rate_duration * 1.8)

        # Equity Risk Premium (ERP) shift
        erp_pe_drag = -1.0 * (scenario.equity_risk_premium_shift_bps / 10000.0) * 12.0 * c.beta

        # AI sentiment multiple re-rating
        ai_pe_boost = scenario.ai_productivity_multiplier * (c.ai_exposure_score * 0.18)

        # Net P/E multiplier relative to baseline
        net_pe_multiplier = max(0.40, 1.0 + rate_pe_drag + erp_pe_drag + ai_pe_boost)

        return net_eps_growth, net_pe_multiplier
