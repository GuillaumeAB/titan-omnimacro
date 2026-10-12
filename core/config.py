"""
OmniMacro Engine - Configuration, Global Index Profiles & Regional Benchmarks
Institutional parameters covering Global Core, Emerging, and Niche Regional Satellites
"""

from typing import Dict, Any

START_DATE_STR = "2026-10"
DEFAULT_HORIZON_MONTHS = 60  # 5 Years (2026 to 2031)

INDEX_PROFILES: Dict[str, Dict[str, Any]] = {
    # --- CORE DEVELOPED MARKETS ---
    "SPX": {
        "id": "SPX",
        "name": "S&P 500 Index (US Large Cap)",
        "ticker": "^GSPC",
        "category": "Core_US",
        "base_level": 5850.0,
        "base_pe": 22.8,
        "base_eps": 256.5,
        "dividend_yield": 0.014,
        "buyback_yield": 0.016,
        "us_exposure": 0.72,
        "china_exposure": 0.08,
        "eu_exposure": 0.12,
        "taiwan_semiconductor_dependency": 0.38,
        "ai_sensitivity": 0.42,
        "ter_annual_pct": 0.0003,  # 0.03% ETF Expense Ratio
        "fx_drag_annual_pct": 0.0,
        "currency": "USD"
    },
    "NDX": {
        "id": "NDX",
        "name": "Nasdaq 100 (US Tech & Growth)",
        "ticker": "^NDX",
        "category": "Core_US_Tech",
        "base_level": 20450.0,
        "base_pe": 27.5,
        "base_eps": 743.6,
        "dividend_yield": 0.008,
        "buyback_yield": 0.021,
        "us_exposure": 0.65,
        "china_exposure": 0.12,
        "eu_exposure": 0.11,
        "taiwan_semiconductor_dependency": 0.74,
        "ai_sensitivity": 0.85,
        "ter_annual_pct": 0.0020,  # 0.20%
        "fx_drag_annual_pct": 0.0,
        "currency": "USD"
    },
    "MSCIWORLD": {
        "id": "MSCIWORLD",
        "name": "MSCI World (Developed World 23 Countries)",
        "ticker": "URTH",
        "category": "Core_Global_Dev",
        "base_level": 3720.0,
        "base_pe": 20.2,
        "base_eps": 184.1,
        "dividend_yield": 0.019,
        "buyback_yield": 0.013,
        "us_exposure": 0.68,
        "china_exposure": 0.06,
        "eu_exposure": 0.17,
        "taiwan_semiconductor_dependency": 0.35,
        "ai_sensitivity": 0.38,
        "ter_annual_pct": 0.0012,  # 0.12%
        "fx_drag_annual_pct": 0.0,
        "currency": "USD"
    },
    "STOXX600": {
        "id": "STOXX600",
        "name": "STOXX Europe 600 (Pan-Europe)",
        "ticker": "^STOXX",
        "category": "Core_Europe",
        "base_level": 522.0,
        "base_pe": 14.4,
        "base_eps": 36.25,
        "dividend_yield": 0.033,
        "buyback_yield": 0.011,
        "us_exposure": 0.28,
        "china_exposure": 0.14,
        "eu_exposure": 0.46,
        "taiwan_semiconductor_dependency": 0.25,
        "ai_sensitivity": 0.22,
        "ter_annual_pct": 0.0015,  # 0.15%
        "fx_drag_annual_pct": -0.005,  # Moderate EUR/USD drift
        "currency": "EUR"
    },

    # --- SATELLITES & REGIONAL FRONTIERS ---
    "EM_BROAD": {
        "id": "EM_BROAD",
        "name": "MSCI Emerging Markets (Broad EM)",
        "ticker": "EEM",
        "category": "Satellite_Emerging",
        "base_level": 1120.0,
        "base_pe": 12.8,
        "base_eps": 87.5,
        "dividend_yield": 0.028,
        "buyback_yield": 0.006,
        "us_exposure": 0.18,
        "china_exposure": 0.26,
        "eu_exposure": 0.10,
        "taiwan_semiconductor_dependency": 0.55,  # Heavy TSMC, Foxconn, MediaTek
        "ai_sensitivity": 0.35,
        "ter_annual_pct": 0.0018,  # 0.18%
        "fx_drag_annual_pct": -0.012,
        "currency": "USD"
    },
    "AFRICA": {
        "id": "AFRICA",
        "name": "MSCI Africa / Pan-Africa (AFK / EZA)",
        "ticker": "AFK",
        "category": "Satellite_Niche",
        "base_level": 185.0,
        "base_pe": 9.4,
        "base_eps": 19.68,
        "dividend_yield": 0.048,
        "buyback_yield": 0.004,
        "us_exposure": 0.08,
        "china_exposure": 0.32,  # Major exporter of critical minerals to China
        "eu_exposure": 0.18,
        "taiwan_semiconductor_dependency": 0.05,
        "ai_sensitivity": 0.08,
        "ter_annual_pct": 0.0075,  # 0.75% high expense ratio & spreads
        "fx_drag_annual_pct": -0.042,  # Severe chronic FX depreciation (ZAR, EGP, NGN)
        "commodity_sensitivity": 0.65,  # High sensitivity to metals, platinum, gold, copper
        "currency": "USD"
    },
    "MIDDLE_EAST": {
        "id": "MIDDLE_EAST",
        "name": "MSCI GCC / Middle East (Saudi KSA & UAE)",
        "ticker": "KSA",
        "category": "Satellite_Niche",
        "base_level": 425.0,
        "base_pe": 15.6,
        "base_eps": 27.24,
        "dividend_yield": 0.042,
        "buyback_yield": 0.008,
        "us_exposure": 0.15,
        "china_exposure": 0.22,  # Crude oil sales to Asia
        "eu_exposure": 0.12,
        "taiwan_semiconductor_dependency": 0.08,
        "ai_sensitivity": 0.25,  # Sovereign AI data centers (G42, Humain)
        "ter_annual_pct": 0.0050,  # 0.50%
        "fx_drag_annual_pct": 0.000,  # USD Peg (SAR & AED pegged to dollar)
        "commodity_sensitivity": 0.88,  # Supreme positive sensitivity to crude oil
        "currency": "USD"
    },
    "EASTERN_EUROPE": {
        "id": "EASTERN_EUROPE",
        "name": "MSCI Eastern Europe ex-Russia (Poland EPOL)",
        "ticker": "EPOL",
        "category": "Satellite_Niche",
        "base_level": 285.0,
        "base_pe": 8.6,
        "base_eps": 33.14,
        "dividend_yield": 0.046,
        "buyback_yield": 0.005,
        "us_exposure": 0.10,
        "china_exposure": 0.04,
        "eu_exposure": 0.68,  # Tightly integrated with German industrial supply chain
        "taiwan_semiconductor_dependency": 0.15,
        "ai_sensitivity": 0.15,
        "ter_annual_pct": 0.0055,  # 0.55%
        "fx_drag_annual_pct": -0.008,  # PLN, RON fluctuations vs EUR/USD
        "nearshoring_sensitivity": 0.85, # Massive beneficiary of EU supply chain nearshoring
        "currency": "USD"
    }
}

# Baseline Macro Environment (as of 2026)
BASELINE_MACRO = {
    "us_10y_yield": 4.10,          # %
    "fed_funds_rate": 4.25,        # %
    "ecb_deposit_rate": 3.00,      # %
    "us_cpi_inflation": 2.60,      # %
    "us_real_gdp_trend": 2.10,     # %
    "brent_crude_usd": 77.50,      # $/bbl
    "tt_gas_eur_mwh": 38.00,       # EUR/MWh
    "geopolitical_risk_index": 120 # GPR Normalized Index (Base = 100)
}
