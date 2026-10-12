import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.monte_carlo_engine import MonteCarloEngine
from engine.scenarios import get_all_scenarios


def test_simulation_run():
    engine = MonteCarloEngine()
    print("Testing Monte Carlo simulation run with 50 iterations...")
    res = engine.run_simulation(horizon_months=60, num_runs=50, scenario_id="composite_polymarket")
    
    assert "SPX" in res.indices
    assert "NDX" in res.indices
    assert "MSCIWORLD" in res.indices
    assert "STOXX600" in res.indices
    
    spx = res.indices["SPX"]
    print(f"SPX Starting: {spx.base_level} -> 5Y P50: {spx.final_p50_level} (CAGR: {spx.cagr_5y_pct}%)")
    ndx = res.indices["NDX"]
    print(f"NDX Starting: {ndx.base_level} -> 5Y P50: {ndx.final_p50_level} (CAGR: {ndx.cagr_5y_pct}%)")
    stoxx = res.indices["STOXX600"]
    print(f"STOXX600 Starting: {stoxx.base_level} -> 5Y P50: {stoxx.final_p50_level} (CAGR: {stoxx.cagr_5y_pct}%)")
    
    print("\nTop Winners:")
    for w in res.top_winners[:3]:
        print(f"  {w.symbol} ({w.name}): +{w.total_return_5y_pct}% | Driver: {w.primary_driver}")
        
    print("\nTop Losers / Underperformers:")
    for l in res.top_losers[:3]:
        print(f"  {l.symbol} ({l.name}): {l.total_return_5y_pct}% | Driver: {l.primary_driver}")

    print("\nTest passed successfully!")


if __name__ == "__main__":
    test_simulation_run()
