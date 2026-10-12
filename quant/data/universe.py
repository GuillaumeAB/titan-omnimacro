"""
quant.data.universe — Curated global multi-asset universe (~250 core instruments).
All instruments are valid Yahoo Finance tickers with metadata for factor modeling,
regime analysis, risk budgeting, and index composition tracking.
"""

from typing import List, Dict, Any
import pandas as pd
from quant.config import FACTOR_PROXY_TICKERS


CORE_UNIVERSE_METADATA: List[Dict[str, Any]] = [
    # --- BROAD EQUITY INDICES / ETFS ---
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "asset_class": "equity_index", "region": "North America", "country": "US", "sector": "Blend", "currency": "USD"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust (Nasdaq-100)", "asset_class": "equity_index", "region": "North America", "country": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "asset_class": "equity_index", "region": "North America", "country": "US", "sector": "Small Cap", "currency": "USD"},
    {"ticker": "MDY", "name": "SPDR S&P MidCap 400 ETF", "asset_class": "equity_index", "region": "North America", "country": "US", "sector": "Mid Cap", "currency": "USD"},
    {"ticker": "ACWI", "name": "iShares MSCI ACWI ETF", "asset_class": "equity_index", "region": "Global", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "URTH", "name": "iShares MSCI World ETF", "asset_class": "equity_index", "region": "Developed", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "VT", "name": "Vanguard Total World Stock ETF", "asset_class": "equity_index", "region": "Global", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "EFA", "name": "iShares MSCI EAFE ETF", "asset_class": "equity_index", "region": "EAFE", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "EEM", "name": "iShares MSCI Emerging Markets ETF", "asset_class": "equity_index", "region": "Emerging", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "VWO", "name": "Vanguard FTSE Emerging Markets ETF", "asset_class": "equity_index", "region": "Emerging", "country": "Global", "sector": "Blend", "currency": "USD"},
    {"ticker": "VGK", "name": "Vanguard FTSE Europe ETF", "asset_class": "equity_index", "region": "Europe", "country": "Europe", "sector": "Blend", "currency": "USD"},
    {"ticker": "EZU", "name": "iShares MSCI Eurozone ETF", "asset_class": "equity_index", "region": "Europe", "country": "Europe", "sector": "Blend", "currency": "USD"},

    # --- SINGLE-COUNTRY ETFS (DEVELOPED & PACIFIC) ---
    {"ticker": "EWU", "name": "iShares MSCI United Kingdom ETF", "asset_class": "equity_country", "region": "Europe", "country": "GB", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWG", "name": "iShares MSCI Germany ETF", "asset_class": "equity_country", "region": "Europe", "country": "DE", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWQ", "name": "iShares MSCI France ETF", "asset_class": "equity_country", "region": "Europe", "country": "FR", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWI", "name": "iShares MSCI Italy ETF", "asset_class": "equity_country", "region": "Europe", "country": "IT", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWP", "name": "iShares MSCI Spain ETF", "asset_class": "equity_country", "region": "Europe", "country": "ES", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWL", "name": "iShares MSCI Switzerland ETF", "asset_class": "equity_country", "region": "Europe", "country": "CH", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWN", "name": "iShares MSCI Netherlands ETF", "asset_class": "equity_country", "region": "Europe", "country": "NL", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWD", "name": "iShares MSCI Sweden ETF", "asset_class": "equity_country", "region": "Europe", "country": "SE", "sector": "Blend", "currency": "USD"},
    {"ticker": "NORW", "name": "Global X MSCI Norway ETF", "asset_class": "equity_country", "region": "Europe", "country": "NO", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWJ", "name": "iShares MSCI Japan ETF", "asset_class": "equity_country", "region": "Asia-Pac", "country": "JP", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWA", "name": "iShares MSCI Australia ETF", "asset_class": "equity_country", "region": "Asia-Pac", "country": "AU", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWC", "name": "iShares MSCI Canada ETF", "asset_class": "equity_country", "region": "North America", "country": "CA", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWS", "name": "iShares MSCI Singapore ETF", "asset_class": "equity_country", "region": "Asia-Pac", "country": "SG", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWH", "name": "iShares MSCI Hong Kong ETF", "asset_class": "equity_country", "region": "Asia-Pac", "country": "HK", "sector": "Blend", "currency": "USD"},

    # --- SINGLE-COUNTRY ETFS (EMERGING ASIA) ---
    {"ticker": "EWT", "name": "iShares MSCI Taiwan ETF", "asset_class": "equity_country", "region": "Emerging", "country": "TW", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWY", "name": "iShares MSCI South Korea ETF", "asset_class": "equity_country", "region": "Emerging", "country": "KR", "sector": "Blend", "currency": "USD"},
    {"ticker": "INDA", "name": "iShares MSCI India ETF", "asset_class": "equity_country", "region": "Emerging", "country": "IN", "sector": "Blend", "currency": "USD"},
    {"ticker": "MCHI", "name": "iShares MSCI China ETF", "asset_class": "equity_country", "region": "Emerging", "country": "CN", "sector": "Blend", "currency": "USD"},
    {"ticker": "FXI", "name": "iShares China Large-Cap ETF", "asset_class": "equity_country", "region": "Emerging", "country": "CN", "sector": "Blend", "currency": "USD"},
    {"ticker": "KWEB", "name": "KraneShares CSI China Internet ETF", "asset_class": "equity_sector", "region": "Emerging", "country": "CN", "sector": "Technology", "currency": "USD"},
    {"ticker": "EWM", "name": "iShares MSCI Malaysia ETF", "asset_class": "equity_country", "region": "Emerging", "country": "MY", "sector": "Blend", "currency": "USD"},
    {"ticker": "THD", "name": "iShares MSCI Thailand ETF", "asset_class": "equity_country", "region": "Emerging", "country": "TH", "sector": "Blend", "currency": "USD"},
    {"ticker": "EIDO", "name": "iShares MSCI Indonesia ETF", "asset_class": "equity_country", "region": "Emerging", "country": "ID", "sector": "Blend", "currency": "USD"},
    {"ticker": "EPHE", "name": "iShares MSCI Philippines ETF", "asset_class": "equity_country", "region": "Emerging", "country": "PH", "sector": "Blend", "currency": "USD"},
    {"ticker": "VNM", "name": "VanEck Vietnam ETF", "asset_class": "equity_country", "region": "Frontier", "country": "VN", "sector": "Blend", "currency": "USD"},

    # --- SINGLE-COUNTRY ETFS (LATIN AMERICA) ---
    {"ticker": "EWZ", "name": "iShares MSCI Brazil ETF", "asset_class": "equity_country", "region": "LatAm", "country": "BR", "sector": "Blend", "currency": "USD"},
    {"ticker": "EWW", "name": "iShares MSCI Mexico ETF", "asset_class": "equity_country", "region": "LatAm", "country": "MX", "sector": "Blend", "currency": "USD"},
    {"ticker": "ECH", "name": "iShares MSCI Chile ETF", "asset_class": "equity_country", "region": "LatAm", "country": "CL", "sector": "Blend", "currency": "USD"},
    {"ticker": "EPU", "name": "iShares MSCI Peru ETF", "asset_class": "equity_country", "region": "LatAm", "country": "PE", "sector": "Blend", "currency": "USD"},
    {"ticker": "ARGT", "name": "Global X MSCI Argentina ETF", "asset_class": "equity_country", "region": "LatAm", "country": "AR", "sector": "Blend", "currency": "USD"},
    {"ticker": "GXG", "name": "Global X MSCI Colombia ETF", "asset_class": "equity_country", "region": "LatAm", "country": "CO", "sector": "Blend", "currency": "USD"},

    # --- SINGLE-COUNTRY ETFS (EASTERN EUROPE, MID-EAST, AFRICA) ---
    {"ticker": "EPOL", "name": "iShares MSCI Poland ETF", "asset_class": "equity_country", "region": "Eastern Europe", "country": "PL", "sector": "Blend", "currency": "USD"},
    {"ticker": "GREK", "name": "Global X MSCI Greece ETF", "asset_class": "equity_country", "region": "Europe", "country": "GR", "sector": "Blend", "currency": "USD"},
    {"ticker": "TUR", "name": "iShares MSCI Turkey ETF", "asset_class": "equity_country", "region": "Emerging", "country": "TR", "sector": "Blend", "currency": "USD"},
    {"ticker": "KSA", "name": "iShares MSCI Saudi Arabia ETF", "asset_class": "equity_country", "region": "Middle East", "country": "SA", "sector": "Blend", "currency": "USD"},
    {"ticker": "UAE", "name": "iShares MSCI UAE ETF", "asset_class": "equity_country", "region": "Middle East", "country": "AE", "sector": "Blend", "currency": "USD"},
    {"ticker": "QAT", "name": "iShares MSCI Qatar ETF", "asset_class": "equity_country", "region": "Middle East", "country": "QA", "sector": "Blend", "currency": "USD"},
    {"ticker": "EIS", "name": "iShares MSCI Israel ETF", "asset_class": "equity_country", "region": "Middle East", "country": "IL", "sector": "Blend", "currency": "USD"},
    {"ticker": "EZA", "name": "iShares MSCI South Africa ETF", "asset_class": "equity_country", "region": "Africa", "country": "ZA", "sector": "Blend", "currency": "USD"},
    {"ticker": "AFK", "name": "VanEck Africa Index ETF", "asset_class": "equity_country", "region": "Africa", "country": "Pan-Africa", "sector": "Blend", "currency": "USD"},
    {"ticker": "EGPT", "name": "VanEck Egypt Index ETF", "asset_class": "equity_country", "region": "Africa", "country": "EG", "sector": "Blend", "currency": "USD"},

    # --- SECTOR & THEMATIC ETFS ---
    {"ticker": "XLK", "name": "Technology Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Technology", "currency": "USD"},
    {"ticker": "XLF", "name": "Financial Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "XLE", "name": "Energy Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Energy", "currency": "USD"},
    {"ticker": "XLV", "name": "Health Care Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "XLI", "name": "Industrial Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "XLY", "name": "Consumer Discretionary Select SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Consumer Discretionary", "currency": "USD"},
    {"ticker": "XLP", "name": "Consumer Staples Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "XLU", "name": "Utilities Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Utilities", "currency": "USD"},
    {"ticker": "XLB", "name": "Materials Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Materials", "currency": "USD"},
    {"ticker": "XLRE", "name": "Real Estate Select Sector SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Real Estate", "currency": "USD"},
    {"ticker": "XLC", "name": "Communication Services Select SPDR", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Communication Services", "currency": "USD"},
    {"ticker": "SMH", "name": "VanEck Semiconductor ETF", "asset_class": "equity_sector", "region": "Global", "country": "US", "sector": "Semiconductors", "currency": "USD"},
    {"ticker": "ITA", "name": "iShares U.S. Aerospace & Defense ETF", "asset_class": "equity_sector", "region": "North America", "country": "US", "sector": "Defense", "currency": "USD"},
    {"ticker": "URA", "name": "Global X Uranium ETF", "asset_class": "equity_sector", "region": "Global", "country": "Global", "sector": "Uranium & Nuclear", "currency": "USD"},
    {"ticker": "LIT", "name": "Global X Lithium & Battery Tech ETF", "asset_class": "equity_sector", "region": "Global", "country": "Global", "sector": "Lithium/Batteries", "currency": "USD"},
    {"ticker": "COPX", "name": "Global X Copper Miners ETF", "asset_class": "equity_sector", "region": "Global", "country": "Global", "sector": "Copper Mining", "currency": "USD"},
    {"ticker": "GDX", "name": "VanEck Gold Miners ETF", "asset_class": "equity_sector", "region": "Global", "country": "Global", "sector": "Gold Mining", "currency": "USD"},
    {"ticker": "ICLN", "name": "iShares Global Clean Energy ETF", "asset_class": "equity_sector", "region": "Global", "country": "Global", "sector": "Clean Energy", "currency": "USD"},

    # --- BOND ETFS ---
    {"ticker": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Govt Short", "currency": "USD"},
    {"ticker": "IEI", "name": "iShares 3-7 Year Treasury Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Govt Interm", "currency": "USD"},
    {"ticker": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Govt Interm", "currency": "USD"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Govt Long", "currency": "USD"},
    {"ticker": "TIP", "name": "iShares TIPS Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Inflation Protected", "currency": "USD"},
    {"ticker": "LQD", "name": "iShares iBoxx $ Investment Grade Corp Bond", "asset_class": "bond", "region": "North America", "country": "US", "sector": "IG Corporate", "currency": "USD"},
    {"ticker": "HYG", "name": "iShares iBoxx $ High Yield Corporate Bond", "asset_class": "bond", "region": "North America", "country": "US", "sector": "High Yield", "currency": "USD"},
    {"ticker": "EMB", "name": "iShares J.P. Morgan USD Emerging Markets Bond", "asset_class": "bond", "region": "Emerging", "country": "Global", "sector": "EM Sovereign", "currency": "USD"},
    {"ticker": "BNDX", "name": "Vanguard Total International Bond ETF", "asset_class": "bond", "region": "Global", "country": "Global", "sector": "Global Aggregate", "currency": "USD"},
    {"ticker": "MUB", "name": "iShares National Muni Bond ETF", "asset_class": "bond", "region": "North America", "country": "US", "sector": "Municipal", "currency": "USD"},

    # --- COMMODITIES FUTURES ---
    {"ticker": "CL=F", "name": "Crude Oil WTI Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Energy", "currency": "USD"},
    {"ticker": "BZ=F", "name": "Brent Crude Oil Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Energy", "currency": "USD"},
    {"ticker": "NG=F", "name": "Natural Gas Futures", "asset_class": "commodity", "region": "Global", "country": "US", "sector": "Energy", "currency": "USD"},
    {"ticker": "GC=F", "name": "Gold Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Precious Metals", "currency": "USD"},
    {"ticker": "SI=F", "name": "Silver Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Precious Metals", "currency": "USD"},
    {"ticker": "HG=F", "name": "Copper Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Industrial Metals", "currency": "USD"},
    {"ticker": "PL=F", "name": "Platinum Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Precious Metals", "currency": "USD"},
    {"ticker": "PA=F", "name": "Palladium Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Precious Metals", "currency": "USD"},
    {"ticker": "ZC=F", "name": "Corn Futures", "asset_class": "commodity", "region": "Global", "country": "US", "sector": "Agriculture", "currency": "USD"},
    {"ticker": "ZW=F", "name": "Wheat Futures", "asset_class": "commodity", "region": "Global", "country": "US", "sector": "Agriculture", "currency": "USD"},
    {"ticker": "ZS=F", "name": "Soybean Futures", "asset_class": "commodity", "region": "Global", "country": "US", "sector": "Agriculture", "currency": "USD"},
    {"ticker": "KC=F", "name": "Coffee Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Agriculture", "currency": "USD"},
    {"ticker": "SB=F", "name": "Sugar Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Agriculture", "currency": "USD"},
    {"ticker": "CC=F", "name": "Cocoa Futures", "asset_class": "commodity", "region": "Global", "country": "Global", "sector": "Agriculture", "currency": "USD"},

    # --- RATES & BENCHMARK YIELDS (asset_class rates_yield) ---
    {"ticker": "^IRX", "name": "13 Week Treasury Bill Yield", "asset_class": "rates_yield", "region": "North America", "country": "US", "sector": "Short Rate", "currency": "USD"},
    {"ticker": "^FVX", "name": "Treasury Yield 5 Years", "asset_class": "rates_yield", "region": "North America", "country": "US", "sector": "Yield Curve", "currency": "USD"},
    {"ticker": "^TNX", "name": "Treasury Yield 10 Years", "asset_class": "rates_yield", "region": "North America", "country": "US", "sector": "Yield Curve", "currency": "USD"},
    {"ticker": "^TYX", "name": "Treasury Yield 30 Years", "asset_class": "rates_yield", "region": "North America", "country": "US", "sector": "Yield Curve", "currency": "USD"},

    # --- FX SPOT / CURRENCY INDEX ---
    {"ticker": "DX-Y.NYB", "name": "US Dollar Index", "asset_class": "fx", "region": "Global", "country": "US", "sector": "Currency Index", "currency": "USD"},
    {"ticker": "EURUSD=X", "name": "EUR / USD", "asset_class": "fx", "region": "Europe", "country": "EU", "sector": "FX Major", "currency": "USD"},
    {"ticker": "GBPUSD=X", "name": "GBP / USD", "asset_class": "fx", "region": "Europe", "country": "GB", "sector": "FX Major", "currency": "USD"},
    {"ticker": "JPY=X", "name": "USD / JPY", "asset_class": "fx", "region": "Asia-Pac", "country": "JP", "sector": "FX Major", "currency": "JPY"},
    {"ticker": "CHF=X", "name": "USD / CHF", "asset_class": "fx", "region": "Europe", "country": "CH", "sector": "FX Major", "currency": "CHF"},
    {"ticker": "CAD=X", "name": "USD / CAD", "asset_class": "fx", "region": "North America", "country": "CA", "sector": "FX Commodity", "currency": "CAD"},
    {"ticker": "AUDUSD=X", "name": "AUD / USD", "asset_class": "fx", "region": "Asia-Pac", "country": "AU", "sector": "FX Commodity", "currency": "USD"},
    {"ticker": "CNY=X", "name": "USD / CNY", "asset_class": "fx", "region": "Emerging", "country": "CN", "sector": "FX Emerging", "currency": "CNY"},
    {"ticker": "BRL=X", "name": "USD / BRL", "asset_class": "fx", "region": "LatAm", "country": "BR", "sector": "FX Emerging", "currency": "BRL"},
    {"ticker": "MXN=X", "name": "USD / MXN", "asset_class": "fx", "region": "LatAm", "country": "MX", "sector": "FX Emerging", "currency": "MXN"},
    {"ticker": "INR=X", "name": "USD / INR", "asset_class": "fx", "region": "Emerging", "country": "IN", "sector": "FX Emerging", "currency": "INR"},
    {"ticker": "KRW=X", "name": "USD / KRW", "asset_class": "fx", "region": "Emerging", "country": "KR", "sector": "FX Emerging", "currency": "KRW"},
    {"ticker": "ZAR=X", "name": "USD / ZAR", "asset_class": "fx", "region": "Africa", "country": "ZA", "sector": "FX Emerging", "currency": "ZAR"},
    {"ticker": "TRY=X", "name": "USD / TRY", "asset_class": "fx", "region": "Emerging", "country": "TR", "sector": "FX Emerging", "currency": "TRY"},
    {"ticker": "PLN=X", "name": "USD / PLN", "asset_class": "fx", "region": "Eastern Europe", "country": "PL", "sector": "FX Emerging", "currency": "PLN"},

    # --- CRYPTO ---
    {"ticker": "BTC-USD", "name": "Bitcoin USD", "asset_class": "crypto", "region": "Global", "country": "Global", "sector": "Digital Asset", "currency": "USD"},
    {"ticker": "ETH-USD", "name": "Ethereum USD", "asset_class": "crypto", "region": "Global", "country": "Global", "sector": "Digital Asset", "currency": "USD"},

    # --- GLOBAL MEGA-CAP STOCKS: US (~50) ---
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "AAPL", "name": "Apple Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Discretionary", "currency": "USD"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (Class A)", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Communication Services", "currency": "USD"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Communication Services", "currency": "USD"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Discretionary", "currency": "USD"},
    {"ticker": "AVGO", "name": "Broadcom Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "V", "name": "Visa Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "WMT", "name": "Walmart Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "PG", "name": "Procter & Gamble Co.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "LLY", "name": "Eli Lilly and Company", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "MA", "name": "Mastercard Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "HD", "name": "The Home Depot Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Discretionary", "currency": "USD"},
    {"ticker": "CVX", "name": "Chevron Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Energy", "currency": "USD"},
    {"ticker": "XOM", "name": "Exxon Mobil Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Energy", "currency": "USD"},
    {"ticker": "ABBV", "name": "AbbVie Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "MRK", "name": "Merck & Co. Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "KO", "name": "The Coca-Cola Company", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "COST", "name": "Costco Wholesale Corp.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Consumer Staples", "currency": "USD"},
    {"ticker": "BAC", "name": "Bank of America Corp.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "QCOM", "name": "Qualcomm Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Communication Services", "currency": "USD"},
    {"ticker": "UNH", "name": "UnitedHealth Group Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Healthcare", "currency": "USD"},
    {"ticker": "ORCL", "name": "Oracle Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "CRM", "name": "Salesforce Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "CSCO", "name": "Cisco Systems Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "TXN", "name": "Texas Instruments Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "AMAT", "name": "Applied Materials Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "LRCX", "name": "Lam Research Corp.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "MU", "name": "Micron Technology Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "INTC", "name": "Intel Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "LMT", "name": "Lockheed Martin Corp.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "RTX", "name": "RTX Corporation", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "BA", "name": "The Boeing Company", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "CAT", "name": "Caterpillar Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "GE", "name": "GE Aerospace", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Industrials", "currency": "USD"},
    {"ticker": "GS", "name": "Goldman Sachs Group Inc.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "MS", "name": "Morgan Stanley", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Financials", "currency": "USD"},
    {"ticker": "CEG", "name": "Constellation Energy Corp.", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Utilities", "currency": "USD"},
    {"ticker": "PLTR", "name": "Palantir Technologies", "asset_class": "equity_stock", "region": "North America", "country": "US", "sector": "Information Technology", "currency": "USD"},

    # --- GLOBAL MEGA-CAP STOCKS: EUROPE (~25) ---
    {"ticker": "ASML.AS", "name": "ASML Holding N.V.", "asset_class": "equity_stock", "region": "Europe", "country": "NL", "sector": "Information Technology", "currency": "EUR"},
    {"ticker": "SAP.DE", "name": "SAP SE", "asset_class": "equity_stock", "region": "Europe", "country": "DE", "sector": "Information Technology", "currency": "EUR"},
    {"ticker": "MC.PA", "name": "LVMH Moet Hennessy", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Consumer Discretionary", "currency": "EUR"},
    {"ticker": "NOVO-B.CO", "name": "Novo Nordisk A/S", "asset_class": "equity_stock", "region": "Europe", "country": "DK", "sector": "Healthcare", "currency": "DKK"},
    {"ticker": "NESN.SW", "name": "Nestle S.A.", "asset_class": "equity_stock", "region": "Europe", "country": "CH", "sector": "Consumer Staples", "currency": "CHF"},
    {"ticker": "ROG.SW", "name": "Roche Holding AG", "asset_class": "equity_stock", "region": "Europe", "country": "CH", "sector": "Healthcare", "currency": "CHF"},
    {"ticker": "NOVN.SW", "name": "Novartis AG", "asset_class": "equity_stock", "region": "Europe", "country": "CH", "sector": "Healthcare", "currency": "CHF"},
    {"ticker": "SHEL.L", "name": "Shell plc", "asset_class": "equity_stock", "region": "Europe", "country": "GB", "sector": "Energy", "currency": "GBP"},
    {"ticker": "AZN.L", "name": "AstraZeneca PLC", "asset_class": "equity_stock", "region": "Europe", "country": "GB", "sector": "Healthcare", "currency": "GBP"},
    {"ticker": "TTE.PA", "name": "TotalEnergies SE", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Energy", "currency": "EUR"},
    {"ticker": "SIE.DE", "name": "Siemens AG", "asset_class": "equity_stock", "region": "Europe", "country": "DE", "sector": "Industrials", "currency": "EUR"},
    {"ticker": "AIR.PA", "name": "Airbus SE", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Industrials", "currency": "EUR"},
    {"ticker": "SAN.PA", "name": "Sanofi SA", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Healthcare", "currency": "EUR"},
    {"ticker": "OR.PA", "name": "L'Oreal SA", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Consumer Staples", "currency": "EUR"},
    {"ticker": "RMS.PA", "name": "Hermes International", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Consumer Discretionary", "currency": "EUR"},
    {"ticker": "RHM.DE", "name": "Rheinmetall AG", "asset_class": "equity_stock", "region": "Europe", "country": "DE", "sector": "Industrials", "currency": "EUR"},
    {"ticker": "BA.L", "name": "BAE Systems plc", "asset_class": "equity_stock", "region": "Europe", "country": "GB", "sector": "Industrials", "currency": "GBP"},
    {"ticker": "BNP.PA", "name": "BNP Paribas SA", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Financials", "currency": "EUR"},
    {"ticker": "UBSG.SW", "name": "UBS Group AG", "asset_class": "equity_stock", "region": "Europe", "country": "CH", "sector": "Financials", "currency": "CHF"},
    {"ticker": "ALV.DE", "name": "Allianz SE", "asset_class": "equity_stock", "region": "Europe", "country": "DE", "sector": "Financials", "currency": "EUR"},
    {"ticker": "SU.PA", "name": "Schneider Electric SE", "asset_class": "equity_stock", "region": "Europe", "country": "FR", "sector": "Industrials", "currency": "EUR"},
    {"ticker": "IBE.MC", "name": "Iberdrola SA", "asset_class": "equity_stock", "region": "Europe", "country": "ES", "sector": "Utilities", "currency": "EUR"},

    # --- GLOBAL MEGA-CAP STOCKS: JAPAN (~10) ---
    {"ticker": "7203.T", "name": "Toyota Motor Corp", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Consumer Discretionary", "currency": "JPY"},
    {"ticker": "6758.T", "name": "Sony Group Corp", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Consumer Discretionary", "currency": "JPY"},
    {"ticker": "8035.T", "name": "Tokyo Electron Ltd", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Information Technology", "currency": "JPY"},
    {"ticker": "6861.T", "name": "Keyence Corp", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Information Technology", "currency": "JPY"},
    {"ticker": "9984.T", "name": "SoftBank Group Corp", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Communication Services", "currency": "JPY"},
    {"ticker": "8306.T", "name": "Mitsubishi UFJ Financial", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Financials", "currency": "JPY"},
    {"ticker": "7011.T", "name": "Mitsubishi Heavy Industries", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Industrials", "currency": "JPY"},
    {"ticker": "6501.T", "name": "Hitachi Ltd", "asset_class": "equity_stock", "region": "Asia-Pac", "country": "JP", "sector": "Industrials", "currency": "JPY"},

    # --- GLOBAL MEGA-CAP STOCKS: ASIA & EMERGING (~15) ---
    {"ticker": "TSM", "name": "Taiwan Semiconductor (TSMC ADR)", "asset_class": "equity_stock", "region": "Emerging", "country": "TW", "sector": "Information Technology", "currency": "USD"},
    {"ticker": "005930.KS", "name": "Samsung Electronics", "asset_class": "equity_stock", "region": "Emerging", "country": "KR", "sector": "Information Technology", "currency": "KRW"},
    {"ticker": "000660.KS", "name": "SK Hynix Inc", "asset_class": "equity_stock", "region": "Emerging", "country": "KR", "sector": "Information Technology", "currency": "KRW"},
    {"ticker": "0700.HK", "name": "Tencent Holdings Ltd", "asset_class": "equity_stock", "region": "Emerging", "country": "CN", "sector": "Communication Services", "currency": "HKD"},
    {"ticker": "9988.HK", "name": "Alibaba Group Holding", "asset_class": "equity_stock", "region": "Emerging", "country": "CN", "sector": "Consumer Discretionary", "currency": "HKD"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "asset_class": "equity_stock", "region": "Emerging", "country": "IN", "sector": "Energy", "currency": "INR"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "asset_class": "equity_stock", "region": "Emerging", "country": "IN", "sector": "Information Technology", "currency": "INR"},
    {"ticker": "VALE", "name": "Vale S.A. (ADR)", "asset_class": "equity_stock", "region": "LatAm", "country": "BR", "sector": "Materials", "currency": "USD"},
    {"ticker": "PBR", "name": "Petroleo Brasileiro (Petrobras ADR)", "asset_class": "equity_stock", "region": "LatAm", "country": "BR", "sector": "Energy", "currency": "USD"},
    {"ticker": "2222.SR", "name": "Saudi Aramco", "asset_class": "equity_stock", "region": "Middle East", "country": "SA", "sector": "Energy", "currency": "SAR"},
    {"ticker": "1120.SR", "name": "Al Rajhi Bank", "asset_class": "equity_stock", "region": "Middle East", "country": "SA", "sector": "Financials", "currency": "SAR"},
    {"ticker": "NPN.JO", "name": "Naspers Ltd", "asset_class": "equity_stock", "region": "Africa", "country": "ZA", "sector": "Communication Services", "currency": "ZAR"},
    {"ticker": "PKO.WA", "name": "PKO Bank Polski", "asset_class": "equity_stock", "region": "Eastern Europe", "country": "PL", "sector": "Financials", "currency": "PLN"},
    {"ticker": "PKN.WA", "name": "Orlen SA", "asset_class": "equity_stock", "region": "Eastern Europe", "country": "PL", "sector": "Energy", "currency": "PLN"},
]


def get_universe(tier: str = "core") -> pd.DataFrame:
    """
    Returns universe dataframe with columns:
    ticker, name, asset_class, region, country, sector, currency
    """
    df_core = pd.DataFrame(CORE_UNIVERSE_METADATA)

    # Ensure all FACTOR_PROXY_TICKERS are present in the universe
    existing = set(df_core["ticker"])
    for pt in FACTOR_PROXY_TICKERS:
        if pt not in existing:
            df_core = pd.concat([
                df_core,
                pd.DataFrame([{
                    "ticker": pt,
                    "name": f"Macro Proxy: {pt}",
                    "asset_class": "equity_index" if pt in ["ACWI", "EEM"] else ("bond" if pt in ["HYG", "IEF"] else "commodity"),
                    "region": "Global",
                    "country": "US",
                    "sector": "Macro Proxy",
                    "currency": "USD"
                }])
            ], ignore_index=True)

    if tier == "core":
        return df_core.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    if tier == "extended":
        try:
            from quant.data.membership import get_sp500, get_sp400
            sp500 = get_sp500()
            sp400 = get_sp400()

            rows = []
            for _, r in sp500.iterrows():
                rows.append({
                    "ticker": r["ticker"],
                    "name": r.get("name", r["ticker"]),
                    "asset_class": "equity_stock",
                    "region": "North America",
                    "country": "US",
                    "sector": r.get("sector", "Blend"),
                    "currency": "USD"
                })
            for _, r in sp400.iterrows():
                rows.append({
                    "ticker": r["ticker"],
                    "name": r.get("name", r["ticker"]),
                    "asset_class": "equity_stock",
                    "region": "North America",
                    "country": "US",
                    "sector": r.get("sector", "Blend"),
                    "currency": "USD"
                })
            df_ext = pd.DataFrame(rows)
            combined = pd.concat([df_core, df_ext], ignore_index=True)
            return combined.drop_duplicates(subset=["ticker"]).reset_index(drop=True)
        except Exception as e:
            # Fallback cleanly to core
            return df_core.drop_duplicates(subset=["ticker"]).reset_index(drop=True)

    return df_core
