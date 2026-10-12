"""
Unit test for quant.data layer.
Tests universe generation, price loading, FRED treasury curves, and membership loaders.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from quant.data.universe import get_universe
from quant.data.loader import load_prices, to_weekly, weekly_log_returns
from quant.data.fred import load_treasury_curve, latest_risk_free
from quant.data.membership import get_sp500, get_sp400, get_nasdaq100, get_nasdaq_screener
from quant.data.ipo import get_ipo_calendar


def test_data_pipeline():
    print("1. Testing get_universe()...")
    u = get_universe("core")
    print(f"Core universe size: {len(u)} instruments")
    assert len(u) >= 100
    assert "SPY" in u["ticker"].values
    assert "ACWI" in u["ticker"].values

    print("\n2. Testing sample price loading...")
    test_tickers = ["SPY", "QQQ", "EWJ", "BZ=F", "^TNX", "NVDA", "ASML.AS"]
    prices = load_prices(test_tickers, start="2024-01-01")
    print(f"Loaded prices shape: {prices.shape}, columns: {list(prices.columns)}")
    assert not prices.empty

    print("\n3. Testing weekly log returns...")
    weekly = to_weekly(prices)
    w_ret = weekly_log_returns(prices, yield_tickers=["^TNX"])
    print(f"Weekly returns shape: {w_ret.shape}")
    assert not w_ret.empty

    print("\n4. Testing FRED Treasury curve & risk-free...")
    curve = load_treasury_curve(start="2024-01-01")
    rf = latest_risk_free()
    print(f"Treasury curve maturities: {list(curve.columns)}, latest risk free: {rf*100:.2f}%")
    assert 0.01 <= rf <= 0.15

    print("\n5. Testing membership...")
    sp500 = get_sp500()
    print(f"S&P 500 count: {len(sp500)}")
    assert len(sp500) >= 490

    ndx = get_nasdaq100()
    print(f"Nasdaq-100 count: {len(ndx)}")
    assert len(ndx) >= 90

    print("\n6. Testing IPO calendar...")
    ipos = get_ipo_calendar()
    print(f"IPO priced count: {len(ipos['priced'])}, upcoming count: {len(ipos['upcoming'])}")

    print("\nALL DATA TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_data_pipeline()
