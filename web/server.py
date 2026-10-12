"""
OmniMacro Engine - FastAPI Application & Web Server
Serves quantitative REST API endpoints and mounts the interactive web interface.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.models import SimulationRequest, SimulationResponse, ScenarioConfig
from engine.monte_carlo_engine import MonteCarloEngine
from engine.scenarios import get_all_scenarios, get_scenario_by_id
from data.constituents.universe import get_constituents, CONSTITUENTS_DATABASE
from data.feeds.prediction_markets import PredictionMarketsFeed
from data.feeds.geopolitical_intelligence import GeopoliticalIntelligenceFeed
from data.feeds.macro_feed import MacroFeed


app = FastAPI(
    title="OmniMacro 5-Year Multi-Asset Geopolitical Engine",
    description="Quantitative bottom-up multi-horizon projection engine for SPX, NDX, MSCI World, and STOXX 600",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MonteCarloEngine()
pred_feed = PredictionMarketsFeed()
intel_feed = GeopoliticalIntelligenceFeed()
macro_feed = MacroFeed()


@app.get("/api/health")
def health_check():
    return {"status": "online", "engine": "OmniMacro Fermi v2.0", "timestamp": "2026-10-12"}


@app.get("/api/scenarios")
def list_scenarios():
    return [s.model_dump() for s in get_all_scenarios()]


@app.get("/api/prediction-markets")
def list_prediction_markets():
    return [m.model_dump() for m in pred_feed.fetch_markets()]


@app.get("/api/news-intel")
def list_news_intel():
    return [s.model_dump() for s in intel_feed.get_latest_signals()]


@app.get("/api/macro-snapshot")
def get_macro_snapshot():
    return macro_feed.get_current_macro_snapshot()


@app.get("/api/constituents")
def list_constituents(index: Optional[str] = None):
    constituents = get_constituents(index)
    return [c.model_dump() for c in constituents]


from engine.portfolio_optimizer import PortfolioEngine, PortfolioWeightConfig, PortfolioAnalytics
port_engine = PortfolioEngine()


@app.get("/api/portfolio/presets")
def get_portfolio_presets():
    return {
        "presets": [
            {
                "id": "classic_core",
                "name": "Classique 4-ETF (S&P 500 + NDX + Europe + Émergents)",
                "weights": {"SPX": 0.55, "NDX": 0.20, "STOXX600": 0.15, "EM_BROAD": 0.10}
            },
            {
                "id": "all_world_msci",
                "name": "MSCI ACWI All-World (Market Cap Pur)",
                "weights": {"MSCIWORLD": 0.88, "EM_BROAD": 0.12}
            },
            {
                "id": "geopolitical_satellite",
                "name": "Core-Satellite Résilience Géopolitique (+Moyen-Orient, Est, Afrique)",
                "weights": {"SPX": 0.45, "NDX": 0.15, "STOXX600": 0.15, "EM_BROAD": 0.10, "MIDDLE_EAST": 0.05, "EASTERN_EUROPE": 0.05, "AFRICA": 0.05}
            },
            {
                "id": "nearshoring_energy",
                "name": "Hedging Anti-Fragile (Choc Pétrolier + Nearshoring Est)",
                "weights": {"SPX": 0.40, "STOXX600": 0.20, "MIDDLE_EAST": 0.20, "EASTERN_EUROPE": 0.15, "AFRICA": 0.05}
            }
        ]
    }


@app.post("/api/portfolio/evaluate", response_model=PortfolioAnalytics)
def evaluate_portfolio(req: PortfolioWeightConfig, name: str = "Allocation Utilisateur"):
    try:
        return port_engine.evaluate_portfolio(req.weights, name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel, Field
from engine.rebalancing import evaluate_rebalancing, RebalanceDiagnostic


class RebalanceRequest(BaseModel):
    target_weights: Dict[str, float]
    current_weights: Optional[Dict[str, float]] = None



@app.post("/api/portfolio/rebalance", response_model=RebalanceDiagnostic)
def get_rebalance_diagnostic(req: RebalanceRequest):
    try:
        stress_prob = 0.14
        from quant.config import OUTPUT_DIR
        latest_file = OUTPUT_DIR / "latest.json"
        if latest_file.exists():
            try:
                import json
                with open(latest_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    probs = d.get("regime", {}).get("current_probs", [])
                    if len(probs) >= 3:
                        stress_prob = float(probs[2])
            except Exception:
                pass
        return evaluate_rebalancing(
            target_weights=req.target_weights,
            current_weights=req.current_weights,
            regime_stress_prob=stress_prob
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/portfolio/central-scenario")
def get_central_scenario_endpoint():
    return {
        "id": "central_baseline_soft_landing",
        "title": "Scénario Central : Atterrissage en Douceur & Croissance Stable",
        "probability_est": "65% - 70%",
        "horizon_5y_summary": "Poursuite de la tendance haussière modérée portée par les bénéfices réels, absence de récession systémique et stabilisation des taux.",
        "pillars": [
            {
                "titre": "Macroéconomie & Croissance",
                "valeur": "PIB US +2.0% / an | Eurozone +1.2%",
                "detail": "Pas de récession globale. Consommation solide et gains de productivité progressifs liés à la numérisation."
            },
            {
                "titre": "Politique Monétaire & Taux",
                "valeur": "Taux 10Y US stabilisé à 3.85% - 4.10%",
                "detail": "La Fed et la BCE ont terminé leur cycle de hausse agressive. Les taux d'intérêt redeviennent neutres."
            },
            {
                "titre": "Bénéfices des Entreprises (EPS)",
                "valeur": "S&P 500 : +9.5% / an | Europe : +6.0%",
                "detail": "La croissance des cours est désormais tirée par les flux de trésorerie réels et non par l'expansion de bulles de valorisation."
            },
            {
                "titre": "Géopolitique & Énergie",
                "valeur": "Pétrole $70-$85 | Statu quo Taïwan/Ormuz",
                "detail": "Tensions chroniques absorbées par les primes de risque sans rupture des chaînes d'approvisionnement mondiales."
            }
        ],
        "asset_projections": [
            {"code": "SPX", "nom": "S&P 500 (Core US)", "ret_1y": "+10.2%", "ret_3y": "+34.5%", "ret_5y": "+62.8%", "vol": "16.5%", "role": "Moteur principal de croissance patrimoniale"},
            {"code": "NDX", "nom": "Nasdaq-100 (Tech Growth)", "ret_1y": "+12.4%", "ret_3y": "+43.2%", "ret_5y": "+82.5%", "vol": "22.5%", "role": "Surperformance liée à l'IA et aux marges logicielles"},
            {"code": "MSCIWORLD", "nom": "MSCI World (Monde Développé)", "ret_1y": "+9.1%", "ret_3y": "+30.2%", "ret_5y": "+54.6%", "vol": "15.8%", "role": "Pilier tout-en-un équilibré mondial"},
            {"code": "STOXX600", "nom": "STOXX Europe 600", "ret_1y": "+8.4%", "ret_3y": "+27.8%", "ret_5y": "+49.8%", "vol": "17.0%", "role": "Rendement dividende et valorisation modérée (P/E 13.5x)"},
            {"code": "EM_BROAD", "nom": "Marchés Émergents Broad", "ret_1y": "+9.5%", "ret_3y": "+31.5%", "ret_5y": "+56.2%", "vol": "21.0%", "role": "Croissance démographique et transition énergétique"},
            {"code": "MIDDLE_EAST", "nom": "Moyen-Orient (KSA/UAE)", "ret_1y": "+9.8%", "ret_3y": "+32.8%", "ret_5y": "+58.0%", "vol": "18.5%", "role": "Couverture anti-choc pétrolier & dividendes élevés"},
            {"code": "EASTERN_EUROPE", "nom": "Europe de l'Est (Pologne/Grèce)", "ret_1y": "+11.2%", "ret_3y": "+38.0%", "ret_5y": "+69.5%", "vol": "23.5%", "role": "Nearshoring européen & faible P/E (8.5x)"},
            {"code": "AFRICA", "nom": "Afrique (Afrique du Sud/Nigéria)", "ret_1y": "+7.5%", "ret_3y": "+24.5%", "ret_5y": "+43.0%", "vol": "24.5%", "role": "Proxy métaux critiques, drag de change résiduel"}
        ],
        "rebalancing_rule": {
            "regle": "Règle des corridors de tolérance 5/25 (Swensen / David Swensen - Yale Model)",
            "principe": "Ne jamais rééquilibrer au calendrier aveugle. Rééquilibrer UNIQUEMENT si un actif dévie de plus de 5% absolu (ex: passe de 20% à 25%) ou de plus de 25% relatif (ex: une ligne à 4% dépasse 5%).",
            "statut_actuel": "AUCUN RÉÉQUILIBRAGE NÉCESSAIRE AUJOURD'HUI",
            "methode_optimale": "Rééquilibrage passif par DCA : affectez 100% de vos futurs versements mensuels sur la ligne la plus en retard, sans vendre vos lignes gagnantes (zéro impôt, zéro frais)."
        }
    }


# --- V3 DATA-DRIVEN QUANTITATIVE ENGINE ENDPOINTS ---
from pydantic import BaseModel, Field


class CustomScenarioRequest(BaseModel):
    shocks: Dict[str, float] = Field(default_factory=dict)
    horizon_weeks: int = 52
    n_paths: int = 2500


@app.get("/quant", response_class=HTMLResponse)
def quant_terminal_page():
    quant_path = os.path.join(os.path.dirname(__file__), "quant.html")
    if os.path.exists(quant_path):
        with open(quant_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Quant Terminal loading...</h1>"


@app.get("/api/v3/status")
def get_v3_status():
    from quant.config import OUTPUT_DIR
    latest_file = OUTPUT_DIR / "latest.json"
    return {
        "status": "ready" if latest_file.exists() else "not_generated",
        "latest_file": str(latest_file),
        "exists": latest_file.exists()
    }


@app.get("/api/v3/latest")
def get_v3_latest():
    from quant.config import OUTPUT_DIR
    # Check env var override for testing
    env_path = os.environ.get("QUANT_LATEST_PATH")
    latest_file = Path(env_path) if env_path else (OUTPUT_DIR / "latest.json")

    if not latest_file.exists():
        raise HTTPException(
            status_code=404,
            detail="output/latest.json not found. Run quant.pipeline first."
        )

    with open(latest_file, "r", encoding="utf-8") as f:
        from fastapi.responses import Response
        return Response(content=f.read(), media_type="application/json")



@app.post("/api/v3/scenario")
def post_v3_scenario(req: CustomScenarioRequest):
    try:
        from quant.pipeline import get_engine
        engine = get_engine()
        if engine.simulator is None:
            # Try to load pipeline output
            from quant.config import OUTPUT_DIR
            if not (OUTPUT_DIR / "latest.json").exists():
                raise HTTPException(status_code=503, detail="Quant engine models not fitted yet.")
        results = engine.run_scenario(req.shocks, horizon_weeks=req.horizon_weeks, n_paths=req.n_paths)
        return {"shocks": req.shocks, "horizon_weeks": req.horizon_weeks, "results": results}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Scenario engine error: {str(e)}")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>OmniMacro Web Interface Loading...</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)
