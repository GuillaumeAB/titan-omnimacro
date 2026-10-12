"""
OmniMacro Engine - Core Domain Models & Schemas
Institutional-Grade Multi-Horizon Geopolitical & Macroeconomic Quantitative Modeling
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GeographicExposure(BaseModel):
    us: float = 0.50
    china: float = 0.15
    europe: float = 0.20
    taiwan: float = 0.05
    rest_of_world: float = 0.10


class Constituent(BaseModel):
    symbol: str
    name: str
    index: str  # 'SPX', 'NDX', 'MSCIWORLD', 'STOXX600'
    sector: str
    sub_industry: Optional[str] = ""
    country: str = "US"
    market_cap_bn: float
    weight: float  # weight in index (0.01 = 1%)
    pe_ratio: float = 22.0
    eps_baseline: float = 100.0
    beta: float = 1.0
    dividend_yield: float = 0.015
    buyback_yield: float = 0.012

    # Factor & Geopolitical Sensitivity Loadings (-1.0 to +1.0 or 0.0 to 1.0)
    ai_exposure_score: float = Field(
        default=0.0,
        description="Sensitivity to AI compute, infrastructure, or enterprise adoption. Positive = beneficiary"
    )
    taiwan_semiconductor_dependency: float = Field(
        default=0.3,
        description="Direct or indirect dependency on TSMC / Taiwan advanced packaging & fab supply chain (0 to 1)"
    )
    energy_cost_elasticity: float = Field(
        default=-0.2,
        description="Margin sensitivity to crude oil / natural gas price spikes. Negative = margin compression"
    )
    defense_rearmament_boost: float = Field(
        default=0.0,
        description="Revenue boost from NATO / global defense procurement surges"
    )
    interest_rate_duration: float = Field(
        default=1.0,
        description="Equity duration / valuation sensitivity to 10Y real interest rate shifts"
    )
    geography: GeographicExposure = Field(default_factory=GeographicExposure)
    tariff_vulnerability: float = Field(
        default=0.3,
        description="Exposure to bilateral tariff wars, deglobalization and retaliatory export controls"
    )


class ScenarioConfig(BaseModel):
    id: str
    name: str
    category: str  # 'Geopolitical', 'Tech_AI', 'Macroeconomic', 'Composite'
    description: str
    probability: float = Field(default=0.20, ge=0.0, le=1.0)
    start_month: int = Field(default=1, ge=1, le=60)
    duration_months: int = Field(default=24, ge=1, le=60)
    
    # Specific Shock Parameters
    ai_productivity_multiplier: float = Field(
        default=0.0,
        description="Shift in structural margin and revenue expansion from AI breakthrough / slowdown (-0.5 to 1.5)"
    )
    taiwan_blockade_intensity: float = Field(
        default=0.0,
        description="Severity of Taiwan Strait conflict / semiconductor supply disruption (0 to 1.0)"
    )
    ukraine_conflict_intensity: float = Field(
        default=0.5,
        description="1.0 = Major Escalation, 0.0 = Ceasefire & Peace Reconstruction (+Capex)"
    )
    middle_east_crisis_intensity: float = Field(
        default=0.3,
        description="Strait of Hormuz / Persian Gulf disruption intensity (0 to 1.0)"
    )
    oil_price_target: float = Field(
        default=78.0,
        description="Crude Oil (Brent) projection target ($/bbl)"
    )
    fed_funds_rate_target: float = Field(
        default=3.75,
        description="Fed terminal / target rate (%)"
    )
    equity_risk_premium_shift_bps: float = Field(
        default=0.0,
        description="Shift in required equity risk premium in basis points (-150 to +350)"
    )
    tariff_level_pct: float = Field(
        default=10.0,
        description="Global universal or bilateral tariff baseline percentage (0 to 60%)"
    )
    is_active: bool = True


class PredictionMarketOdds(BaseModel):
    market_id: str
    source: str  # 'Polymarket', 'Kalshi', 'Manifold', 'Ensemble'
    title: str
    category: str
    implied_probability: float  # 0.0 to 1.0
    volume_usd: float
    resolution_horizon: str
    last_updated: str
    related_scenario_id: str


class NewsIntelligenceSignal(BaseModel):
    id: str
    timestamp: str
    source: str
    headline: str
    category: str
    sentiment_score: float  # -1.0 to +1.0
    geopolitical_risk_score: float  # 0 to 100
    affected_sectors: List[str]
    confidence: float


class IndexMonthlyPoint(BaseModel):
    month: int
    date: str  # 'YYYY-MM'
    index_level: float
    total_return_pct: float
    pe_multiple: float
    aggregate_eps: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class IndexTrajectory(BaseModel):
    index_id: str  # 'SPX', 'NDX', 'MSCIWORLD', 'STOXX600'
    name: str
    base_level: float
    final_p50_level: float
    cagr_5y_pct: float
    total_return_5y_pct: float
    max_drawdown_pct: float
    volatility_ann_pct: float
    monthly_path: List[IndexMonthlyPoint]


class ConstituentImpact(BaseModel):
    symbol: str
    name: str
    sector: str
    index: str
    weight: float
    starting_price: float
    projected_price_5y: float
    total_return_5y_pct: float
    eps_5y_growth_pct: float
    final_pe: float
    primary_driver: str


class SimulationRequest(BaseModel):
    horizon_months: int = 60
    num_monte_carlo_runs: int = 250
    selected_scenario_id: Optional[str] = "composite_polymarket"
    custom_scenario: Optional[ScenarioConfig] = None
    confidence_interval: float = 0.80
    stress_taiwan_override: Optional[float] = None
    stress_ai_override: Optional[float] = None
    stress_middle_east_override: Optional[float] = None
    stress_ukraine_override: Optional[float] = None


class SimulationResponse(BaseModel):
    timestamp: str
    horizon_months: int
    num_runs: int
    scenario_mode: str
    applied_scenarios: List[ScenarioConfig]
    prediction_market_weights: Dict[str, float]
    indices: Dict[str, IndexTrajectory]
    top_winners: List[ConstituentImpact]
    top_losers: List[ConstituentImpact]
    factor_attribution: Dict[str, Dict[str, float]]
    macro_telemetry: Dict[str, Any]
