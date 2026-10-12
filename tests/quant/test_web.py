"""
Unit and integration tests for quant web endpoints and FastAPI server.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from fastapi.testclient import TestClient
from web.server import app


def test_quant_endpoints():
    client = TestClient(app)

    print("1. Testing GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "online"

    print("2. Testing GET /quant (Quant Terminal HTML)...")
    resp = client.get("/quant")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Quant Terminal" in resp.text or "quant" in resp.text.lower()

    print("3. Testing GET /api/v3/status...")
    resp = client.get("/api/v3/status")
    assert resp.status_code == 200
    status_data = resp.json()
    assert "status" in status_data
    assert "latest_file" in status_data

    print("4. Testing GET /api/v3/latest...")
    resp = client.get("/api/v3/latest")
    # If latest.json exists, status is 200; if not yet generated, status is 404
    assert resp.status_code in (200, 404)
    if resp.status_code == 200:
        data = resp.json()
        assert "as_of" in data
        assert "regime" in data
        assert "forecasts" in data
        print(f"Latest forecasts count: {len(data['forecasts'])}")

    print("5. Testing GET / (V2 UI)...")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]

    print("\nALL WEB TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_quant_endpoints()
