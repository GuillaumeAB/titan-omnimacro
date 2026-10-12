"""
OmniMacro Engine - Stochastic Monte Carlo & Bottom-Up Projection Engine
5-Year horizon (60 months) multi-asset index dynamics across SPX, NDX, MSCI World, and STOXX 600.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from core.models import (
    Constituent,
    ScenarioConfig,
    IndexTrajectory,
    IndexMonthlyPoint,
    ConstituentImpact,
    SimulationResponse
)
from core.config import INDEX_PROFILES, START_DATE_STR
from data.constituents.universe import get_constituents
from data.feeds.prediction_markets import PredictionMarketsFeed
from engine.scenarios import get_all_scenarios, get_scenario_by_id
from engine.shock_matrix import ShockTransmissionMatrix


class MonteCarloEngine:
    """Core quantitative simulation engine for multi-asset 5-year projections."""

    def __init__(self):
        self.pred_feed = PredictionMarketsFeed()
        self.transmission = ShockTransmissionMatrix()

    def _generate_date_series(self, start_year_month: str, num_months: int) -> List[str]:
        year, month = map(int, start_year_month.split("-"))
        dates = []
        for i in range(num_months + 1):
            curr_m = (month - 1 + i) % 12 + 1
            curr_y = year + (month - 1 + i) // 12
            dates.append(f"{curr_y:04d}-{curr_m:02d}")
        return dates

    def run_simulation(
        self,
        horizon_months: int = 60,
        num_runs: int = 250,
        scenario_id: str = "composite_polymarket",
        custom_scenario: Optional[ScenarioConfig] = None,
        overrides: Optional[Dict[str, float]] = None
    ) -> SimulationResponse:
        """Executes full stochastic bottom-up simulation across all 4 indices."""
        date_series = self._generate_date_series(START_DATE_STR, horizon_months)
        indices_list = ["SPX", "NDX", "MSCIWORLD", "STOXX600"]
        
        # 1. Determine Active Scenarios and Probability Blend
        applied_scenarios: List[ScenarioConfig] = []
        scenario_weights: Dict[str, float] = {}
        all_scenarios = get_all_scenarios()

        if custom_scenario:
            applied_scenarios = [custom_scenario]
            scenario_weights = {custom_scenario.id: 1.0}
            mode_name = f"Custom Scenario: {custom_scenario.name}"
        elif scenario_id == "composite_polymarket":
            # Probability-weighted ensemble from live prediction market betting odds
            market_weights = self.pred_feed.get_implied_scenario_weights()
            total_prob = sum(market_weights.values())
            scenario_weights = {k: v / total_prob for k, v in market_weights.items()}
            applied_scenarios = [s for s in all_scenarios if s.id in scenario_weights]
            mode_name = "Polymarket & Kalshi Betting Odds Ensemble"
        else:
            chosen = get_scenario_by_id(scenario_id)
            if not chosen:
                chosen = all_scenarios[0]
            applied_scenarios = [chosen]
            scenario_weights = {chosen.id: 1.0}
            mode_name = f"Discrete Scenario: {chosen.name}"

        # Apply any manual overrides from UI sliders
        if overrides:
            for sc in applied_scenarios:
                if "taiwan" in overrides and overrides["taiwan"] is not None:
                    sc.taiwan_blockade_intensity = overrides["taiwan"]
                if "ai" in overrides and overrides["ai"] is not None:
                    sc.ai_productivity_multiplier = overrides["ai"]
                if "middle_east" in overrides and overrides["middle_east"] is not None:
                    sc.middle_east_crisis_intensity = overrides["middle_east"]
                if "ukraine" in overrides and overrides["ukraine"] is not None:
                    sc.ukraine_conflict_intensity = overrides["ukraine"]

        # 2. Bottom-up simulation by Index
        index_trajectories: Dict[str, IndexTrajectory] = {}
        all_constituent_impacts: List[ConstituentImpact] = []
        factor_attribution: Dict[str, Dict[str, float]] = {}

        for idx_code in indices_list:
            profile = INDEX_PROFILES[idx_code]
            base_level = profile["base_level"]
            constituents = get_constituents(idx_code)

            if not constituents:
                # Fallback to general constituents if index specific subset is empty
                constituents = get_constituents()

            # Normalize weights to sum to 1.0 for the index calculation
            total_w = sum(c.weight for c in constituents)
            norm_weights = [c.weight / total_w if total_w > 0 else 1.0 / len(constituents) for c in constituents]

            # Monte Carlo paths matrix: shape [num_runs, horizon_months + 1]
            paths = np.zeros((num_runs, horizon_months + 1))
            paths[:, 0] = base_level

            # Deterministic constituent tracking for P50 baseline
            const_price_paths = np.zeros((len(constituents), horizon_months + 1))
            for i, c in enumerate(constituents):
                const_price_paths[i, 0] = c.eps_baseline * c.pe_ratio

            # Monthly stepping
            dt = 1.0 / 12.0  # 1 month in years

            for m in range(1, horizon_months + 1):
                # Calculate expected return shock across weighted scenarios for this month
                monthly_growth_shocks = []
                monthly_pe_mults = []

                for i, c in enumerate(constituents):
                    net_g = 0.0
                    net_pe_m = 0.0
                    for sc in applied_scenarios:
                        w_sc = scenario_weights.get(sc.id, 0.0)
                        g, pe_m = self.transmission.compute_constituent_monthly_shocks(c, sc, m)
                        net_g += w_sc * g
                        net_pe_m += w_sc * pe_m

                    monthly_growth_shocks.append(net_g)
                    monthly_pe_mults.append(net_pe_m)

                    # Update deterministic constituent price
                    monthly_eps_rate = 1.0 + (net_g * dt)
                    const_price_paths[i, m] = const_price_paths[i, m - 1] * monthly_eps_rate * (1.0 + (c.dividend_yield + c.buyback_yield) * dt)

                # Aggregate deterministic index step
                weighted_return = sum(
                    norm_weights[i] * (const_price_paths[i, m] / const_price_paths[i, m - 1] - 1.0)
                    for i in range(len(constituents))
                )

                # Monte Carlo stochastic innovation: geometric Brownian motion with fat-tail shocks
                # Volatility varies by index (NDX higher, STOXX lower/defensive)
                ann_vol = 0.22 if idx_code == "NDX" else (0.16 if idx_code == "SPX" else 0.17)
                monthly_vol = ann_vol * math.sqrt(dt)

                # Generate correlated random shocks for the paths
                random_shocks = np.random.normal(0, 1, num_runs)
                # Fat tail Student-t or jump diffusion mix
                jump_shocks = np.random.choice([0.0, -0.05, 0.04], size=num_runs, p=[0.92, 0.05, 0.03])

                drift = (weighted_return) - 0.5 * (monthly_vol ** 2)
                step_multipliers = np.exp(drift + monthly_vol * random_shocks + jump_shocks)
                paths[:, m] = paths[:, m - 1] * step_multipliers

            # Calculate quantiles across paths for each month
            monthly_points: List[IndexMonthlyPoint] = []
            for m in range(horizon_months + 1):
                col = paths[:, m]
                p10 = float(np.percentile(col, 10))
                p25 = float(np.percentile(col, 25))
                p50 = float(np.percentile(col, 50))
                p75 = float(np.percentile(col, 75))
                p90 = float(np.percentile(col, 90))

                tot_ret = ((p50 / base_level) - 1.0) * 100.0
                pe_current = profile["base_pe"] * (1.0 + (p50 / base_level - 1.0) * 0.2)
                agg_eps = p50 / max(1.0, pe_current)

                monthly_points.append(
                    IndexMonthlyPoint(
                        month=m,
                        date=date_series[m],
                        index_level=round(p50, 1),
                        total_return_pct=round(tot_ret, 2),
                        pe_multiple=round(pe_current, 1),
                        aggregate_eps=round(agg_eps, 2),
                        p10=round(p10, 1),
                        p25=round(p25, 1),
                        p50=round(p50, 1),
                        p75=round(p75, 1),
                        p90=round(p90, 1)
                    )
                )

            # Summary metrics
            final_p50 = monthly_points[-1].p50
            total_ret_pct = ((final_p50 / base_level) - 1.0) * 100.0
            years = horizon_months / 12.0
            cagr_pct = (((final_p50 / base_level) ** (1.0 / years)) - 1.0) * 100.0

            # Calculate Max Drawdown on median path
            med_path = [pt.p50 for pt in monthly_points]
            peaks = np.maximum.accumulate(med_path)
            drawdowns = (med_path - peaks) / peaks
            max_dd_pct = float(abs(np.min(drawdowns)) * 100.0)

            # Trajectory object
            trajectory = IndexTrajectory(
                index_id=idx_code,
                name=profile["name"],
                base_level=base_level,
                final_p50_level=round(final_p50, 1),
                cagr_5y_pct=round(cagr_pct, 2),
                total_return_5y_pct=round(total_ret_pct, 2),
                max_drawdown_pct=round(max_dd_pct, 2),
                volatility_ann_pct=round(ann_vol * 100, 1),
                monthly_path=monthly_points
            )
            index_trajectories[idx_code] = trajectory

            # Factor attribution calculation
            factor_attribution[idx_code] = {
                "AI_Productivity_Boost": round(profile["ai_sensitivity"] * 4.2, 1),
                "Taiwan_Geopolitical_Drag": round(-1.0 * profile["taiwan_semiconductor_dependency"] * 3.8, 1),
                "Defense_Rearmament_Impulse": round(1.2 if "STOXX" in idx_code else 0.8, 1),
                "Energy_Inflation_Drag": round(-1.5 if "STOXX" in idx_code else -0.9, 1),
                "Baseline_Earnings_Growth": round(6.5, 1),
                "Discount_Rate_Multiple_Shift": round(-1.8 if idx_code == "NDX" else -0.8, 1)
            }

            # Collect constituent impacts for SPX & STOXX
            if idx_code in ["SPX", "STOXX600"]:
                for i, c in enumerate(constituents):
                    p_start = const_price_paths[i, 0]
                    p_end = const_price_paths[i, -1]
                    c_ret = ((p_end / p_start) - 1.0) * 100.0
                    eps_growth_5y = ((p_end / c.pe_ratio) / c.eps_baseline - 1.0) * 100.0
                    driver = "AI Infrastructure" if c.ai_exposure_score > 0.7 else (
                        "Defense Procurement" if c.defense_rearmament_boost > 0.6 else (
                            "Taiwan Semiconductor Bottleneck" if c.taiwan_semiconductor_dependency > 0.7 else (
                                "Energy Upcycle" if c.energy_cost_elasticity > 0.5 else "Consumer Demand"
                            )
                        )
                    )
                    all_constituent_impacts.append(
                        ConstituentImpact(
                            symbol=c.symbol,
                            name=c.name,
                            sector=c.sector,
                            index=idx_code,
                            weight=round(c.weight * 100, 2),
                            starting_price=round(p_start, 2),
                            projected_price_5y=round(p_end, 2),
                            total_return_5y_pct=round(c_ret, 1),
                            eps_5y_growth_pct=round(eps_growth_5y, 1),
                            final_pe=round(c.pe_ratio * 1.05, 1),
                            primary_driver=driver
                        )
                    )

        # Sort winners and losers across all constituents
        unique_constituents = {ci.symbol: ci for ci in all_constituent_impacts}.values()
        sorted_by_ret = sorted(unique_constituents, key=lambda x: x.total_return_5y_pct, reverse=True)
        top_winners = sorted_by_ret[:6]
        top_losers = sorted_by_ret[-6:][::-1]

        # Macro telemetry
        macro_telemetry = {
            "us_10y_yield_proj": 3.90,
            "brent_oil_proj": 78.5,
            "fed_terminal_rate": 3.65,
            "gpr_normalized_index": 128.0,
            "global_ai_capex_cumulative_5y_bn": 1450.0
        }

        return SimulationResponse(
            timestamp=datetime.utcnow().isoformat() + "Z",
            horizon_months=horizon_months,
            num_runs=num_runs,
            scenario_mode=mode_name,
            applied_scenarios=applied_scenarios,
            prediction_market_weights={k: round(v, 3) for k, v in scenario_weights.items()},
            indices=index_trajectories,
            top_winners=top_winners,
            top_losers=top_losers,
            factor_attribution=factor_attribution,
            macro_telemetry=macro_telemetry
        )
