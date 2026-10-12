"""
OmniMacro Engine - Qualitative Geopolitical & Tech News Intelligence
Monitors key flashpoints, narrative shifts, and qualitative sentiment across economics, tech, and defense.
"""

from typing import List, Dict
from core.models import NewsIntelligenceSignal


LIVE_INTELLIGENCE_STREAM: List[Dict] = [
    {
        "id": "sig-001",
        "timestamp": "2026-10-12T14:30:00Z",
        "source": "Defense Intelligence & Satellite Monitor",
        "headline": "PLA Eastern Theater Command increases amphibious drills in Taiwan Strait with novel naval drone swarm doctrines",
        "category": "Taiwan / Semiconductor",
        "sentiment_score": -0.68,
        "geopolitical_risk_score": 84.0,
        "affected_sectors": ["Information Technology", "Semiconductors", "Consumer Discretionary"],
        "confidence": 0.88
    },
    {
        "id": "sig-002",
        "timestamp": "2026-10-12T12:15:00Z",
        "source": "AI Compute Research & Energy Telemetry",
        "headline": "Tier-1 Hyperscalers lock up 15-year nuclear power purchase agreements for gigawatt-scale AI cluster buildouts",
        "category": "AI / Energy",
        "sentiment_score": 0.72,
        "geopolitical_risk_score": 25.0,
        "affected_sectors": ["Utilities", "Information Technology", "Energy"],
        "confidence": 0.94
    },
    {
        "id": "sig-003",
        "timestamp": "2026-10-12T10:00:00Z",
        "source": "Maritime & Energy Flow Intelligence",
        "headline": "Tensions in Persian Gulf trigger insurance premium escalation on VLCC crude tankers transiting Strait of Hormuz",
        "category": "Middle East / Energy",
        "sentiment_score": -0.58,
        "geopolitical_risk_score": 78.0,
        "affected_sectors": ["Energy", "Industrials", "Consumer Discretionary", "Financials"],
        "confidence": 0.82
    },
    {
        "id": "sig-004",
        "timestamp": "2026-10-12T08:45:00Z",
        "source": "European Security & Defense Council",
        "headline": "EU NATO member states propose permanent 3.0% GDP defense spending target and joint ammunition procurement fund",
        "category": "Ukraine / Defense",
        "sentiment_score": 0.45,
        "geopolitical_risk_score": 65.0,
        "affected_sectors": ["Industrials", "Defense", "Financials"],
        "confidence": 0.91
    },
    {
        "id": "sig-005",
        "timestamp": "2026-10-12T07:20:00Z",
        "source": "International Trade & Customs Monitor",
        "headline": "Comprehensive 10-20% universal import tariff proposals debated in Washington with reciprocal retaliatory trade lists prepared in Beijing and Brussels",
        "category": "Trade / Macro",
        "sentiment_score": -0.52,
        "geopolitical_risk_score": 72.0,
        "affected_sectors": ["Consumer Discretionary", "Automotive", "Technology", "Industrials"],
        "confidence": 0.86
    }
]


class GeopoliticalIntelligenceFeed:
    """Provides categorized qualitative intelligence signals and GPR shocks."""

    def __init__(self):
        self.signals = [NewsIntelligenceSignal(**s) for s in LIVE_INTELLIGENCE_STREAM]

    def get_latest_signals(self, limit: int = 10) -> List[NewsIntelligenceSignal]:
        return self.signals[:limit]

    def get_aggregate_sentiment_by_category(self) -> Dict[str, float]:
        cat_sums: Dict[str, float] = {}
        cat_counts: Dict[str, int] = {}
        for s in self.signals:
            cat_sums[s.category] = cat_sums.get(s.category, 0.0) + s.sentiment_score
            cat_counts[s.category] = cat_counts.get(s.category, 0) + 1
        return {c: cat_sums[c] / cat_counts[c] for c in cat_sums}
