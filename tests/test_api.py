"""
Verify web server API endpoints using FastAPI TestClient.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from web.server import app


def test_api_endpoints():
    client = TestClient(app)
    
    print("Testing /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200
    print("Health response:", r.json())
    
    print("Testing /api/scenarios...")
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    scenarios = r.json()
    assert len(scenarios) >= 5
    print(f"Loaded {len(scenarios)} scenarios.")
    
    print("Testing /api/prediction-markets...")
    r = client.get("/api/prediction-markets")
    assert r.status_code == 200
    markets = r.json()
    assert len(markets) >= 5
    print(f"Loaded {len(markets)} prediction markets.")
    
    print("Testing /api/news-intel...")
    r = client.get("/api/news-intel")
    assert r.status_code == 200
    signals = r.json()
    assert len(signals) >= 3
    print(f"Loaded {len(signals)} news signals.")

    print("Testing /api/constituents...")
    r = client.get("/api/constituents")
    assert r.status_code == 200
    constituents = r.json()
    assert len(constituents) >= 15
    print(f"Loaded {len(constituents)} constituents.")

    print("Testing /api/simulate...")
    sim_payload = {
        "horizon_months": 60,
        "num_monte_carlo_runs": 25,
        "selected_scenario_id": "composite_polymarket"
    }
    r = client.post("/api/simulate", json=sim_payload)
    assert r.status_code == 200
    sim_data = r.json()
    assert "indices" in sim_data
    assert "SPX" in sim_data["indices"]
    print("Simulation completed successfully! SPX 5Y CAGR:", sim_data["indices"]["SPX"]["cagr_5y_pct"])

    print("Testing root / page...")
    r = client.get("/")
    assert r.status_code == 200
    assert "TITAN OMNIMACRO" in r.text
    print("Index HTML rendered successfully!")

    print("\nALL FASTAPI ENDPOINTS VERIFIED & FUNCTIONAL!")


if __name__ == "__main__":
    test_api_endpoints()
