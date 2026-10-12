"""
OmniMacro Engine - Scenario Definitions & Bayesian Parameter Configurations
Institutional scenarios covering AI, Taiwan, Ukraine, Middle East, and Macro Regimes.
"""

from typing import Dict, List, Optional
from core.models import ScenarioConfig


DEFINED_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        id="baseline_soft_landing",
        name="Baseline: Orderly Trend & Steady AI Integration",
        category="Macroeconomic",
        description="Steady 2.0% US GDP growth, 1.2% EU GDP growth, gradual Fed/ECB easing, steady AI productivity ramp.",
        probability=0.38,
        start_month=1,
        duration_months=60,
        ai_productivity_multiplier=0.25,
        taiwan_blockade_intensity=0.0,
        ukraine_conflict_intensity=0.35,
        middle_east_crisis_intensity=0.20,
        oil_price_target=75.0,
        fed_funds_rate_target=3.50,
        equity_risk_premium_shift_bps=0.0,
        tariff_level_pct=10.0
    ),
    ScenarioConfig(
        id="ai_hyper_supercycle",
        name="AI Supercycle: AGI Breakthrough & Autonomous Automation",
        category="Tech_AI",
        description="Frontier models achieve autonomous software and research benchmarks. Enterprise margins expand 350 bps; power & nuclear capex boons.",
        probability=0.26,
        start_month=3,
        duration_months=57,
        ai_productivity_multiplier=1.20,
        taiwan_blockade_intensity=0.0,
        ukraine_conflict_intensity=0.25,
        middle_east_crisis_intensity=0.15,
        oil_price_target=82.0,
        fed_funds_rate_target=4.25,  # Higher real neutral rate due to capital demand
        equity_risk_premium_shift_bps=-60.0,  # Euphoria multiple expansion
        tariff_level_pct=12.0
    ),
    ScenarioConfig(
        id="ai_capex_winter",
        name="AI ROI Disappointment & Capex Digestion",
        category="Tech_AI",
        description="Enterprise AI adoption monetization fails to justify $250B+ hyperscaler capex. Chip orders canceled; tech multiples contract.",
        probability=0.20,
        start_month=6,
        duration_months=36,
        ai_productivity_multiplier=-0.35,
        taiwan_blockade_intensity=0.0,
        ukraine_conflict_intensity=0.35,
        middle_east_crisis_intensity=0.20,
        oil_price_target=68.0,
        fed_funds_rate_target=2.75,  # Rate cuts to offset tech capex slowdown
        equity_risk_premium_shift_bps=80.0,
        tariff_level_pct=10.0
    ),
    ScenarioConfig(
        id="taiwan_semiconductor_blockade",
        name="Taiwan Strait Conflict: Naval Blockade & TSMC Disruption",
        category="Geopolitical",
        description="Naval quarantine/blockade around Taiwan halts 75% of advanced global chip shipments (3nm/2nm). Unprecedented global supply chain shock.",
        probability=0.16,
        start_month=12,
        duration_months=30,
        ai_productivity_multiplier=-0.45,
        taiwan_blockade_intensity=0.88,
        ukraine_conflict_intensity=0.60,
        middle_east_crisis_intensity=0.40,
        oil_price_target=105.0,
        fed_funds_rate_target=4.75,  # Stagflationary rate trap
        equity_risk_premium_shift_bps=220.0,  # Extreme risk-off discount rate shock
        tariff_level_pct=45.0
    ),
    ScenarioConfig(
        id="mideast_hormuz_oil_shock",
        name="Middle East War: Strait of Hormuz Chokepoint Closure",
        category="Geopolitical",
        description="Direct regional war damages Persian Gulf oil infrastructure; Strait of Hormuz closed. Brent spikes to $135/bbl; stagflation hits Europe & US.",
        probability=0.18,
        start_month=4,
        duration_months=24,
        ai_productivity_multiplier=-0.10,
        taiwan_blockade_intensity=0.05,
        ukraine_conflict_intensity=0.45,
        middle_east_crisis_intensity=0.92,
        oil_price_target=135.0,
        fed_funds_rate_target=5.25,  # Emergency hiking / higher for longer
        equity_risk_premium_shift_bps=160.0,
        tariff_level_pct=15.0
    ),
    ScenarioConfig(
        id="ukraine_ceasefire_reconstruction",
        name="Ukraine Peace Settlement & European Reconstruction Boom",
        category="Geopolitical",
        description="Formal negotiated armistice and security treaties in Eastern Europe. €450B European reconstruction fund launched. European gas plunges.",
        probability=0.32,
        start_month=8,
        duration_months=52,
        ai_productivity_multiplier=0.20,
        taiwan_blockade_intensity=0.0,
        ukraine_conflict_intensity=0.05,
        middle_east_crisis_intensity=0.15,
        oil_price_target=65.0,
        fed_funds_rate_target=3.25,
        equity_risk_premium_shift_bps=-40.0,
        tariff_level_pct=8.0
    ),
    ScenarioConfig(
        id="deglobalization_tariff_war",
        name="Global Tariff War: Hawkish Protectionism & Trade Balkanization",
        category="Macroeconomic",
        description="US universal 20% tariff implemented; China and EU enact retaliatory agricultural and industrial duties. Global supply chains reshore at high cost.",
        probability=0.35,
        start_month=6,
        duration_months=48,
        ai_productivity_multiplier=0.0,
        taiwan_blockade_intensity=0.20,
        ukraine_conflict_intensity=0.40,
        middle_east_crisis_intensity=0.25,
        oil_price_target=80.0,
        fed_funds_rate_target=4.50,  # Imported cost-push inflation
        equity_risk_premium_shift_bps=90.0,
        tariff_level_pct=25.0
    )
]


def get_scenario_by_id(scenario_id: str) -> Optional[ScenarioConfig]:
    for s in DEFINED_SCENARIOS:
        if s.id == scenario_id:
            return s
    return None


def get_all_scenarios() -> List[ScenarioConfig]:
    return DEFINED_SCENARIOS
