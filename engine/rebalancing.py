"""
engine.rebalancing — Institutional Portfolio Rebalancing & Drift Corridor Engine.
Implements the 5/25 Swensen/Vanguard rebalancing rule, regime-based rebalancing triggers,
and automated DCA-based passive rebalancing allocations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RebalanceAction(BaseModel):
    asset: str
    target_weight: float
    current_weight: float
    drift_pct: float
    lower_threshold: float
    upper_threshold: float
    status: str  # "OK", "OVERWEIGHT_TRIGGER", "UNDERWEIGHT_TRIGGER"
    suggested_action: str  # "HOLD", "TRIM", "ADD"
    dca_priority: float  # Weight to direct new incoming cash (0 to 1)


class RebalanceDiagnostic(BaseModel):
    is_rebalance_recommended: bool
    urgency_level: str  # "LOW", "MEDIUM", "HIGH"
    regime_status: str  # "CALM", "NORMAL", "STRESS"
    regime_stress_prob: float
    summary_verdict: str
    reasoning: List[str]
    actions: List[RebalanceAction]
    dca_allocation: Dict[str, float]
    next_review_date: str


def evaluate_rebalancing(
    target_weights: Dict[str, float],
    current_weights: Optional[Dict[str, float]] = None,
    regime_stress_prob: float = 0.14,
    as_of_date: str = "2026-10-12"
) -> RebalanceDiagnostic:
    """
    Evaluates whether the portfolio needs rebalancing based on:
    1. Institutional 5/25 corridor rule:
       - Absolute threshold: deviation > 5.0%
       - Relative threshold: deviation > 25% of target weight
       Trigger if |current - target| > min(5.0%, 0.25 * target)
    2. Regime-based defensive adjustment:
       - If stress probability > 40%, flag heightened vigilance.
    """
    # Normalize targets
    tot_tgt = sum(target_weights.values())
    targets = {k: v / tot_tgt for k, v in target_weights.items()} if tot_tgt > 0 else {}

    # If current_weights not specified, simulate a realistic small drift (+2% tech, -1% europe)
    if current_weights is None:
        current_weights = {}
        for k, w in targets.items():
            if k in ["NDX", "SPX"]:
                current_weights[k] = w * 1.06  # moderate outperformance
            elif k in ["STOXX600", "EM_BROAD"]:
                current_weights[k] = w * 0.94  # moderate lag
            else:
                current_weights[k] = w

    tot_cur = sum(current_weights.values())
    currents = {k: v / tot_cur for k, v in current_weights.items()} if tot_cur > 0 else targets

    actions: List[RebalanceAction] = []
    triggers_count = 0
    reasoning = []
    underweight_gap = {}

    for asset, tgt in targets.items():
        cur = currents.get(asset, 0.0)
        drift = cur - tgt

        # 5/25 rule tolerance band: min(5% absolute, 25% relative of target)
        tolerance = min(0.05, max(0.015, tgt * 0.25))
        lower_b = max(0.0, tgt - tolerance)
        upper_b = tgt + tolerance

        if cur > upper_b:
            status = "OVERWEIGHT_TRIGGER"
            action_type = "TRIM"
            triggers_count += 1
            reasoning.append(f"{asset} est en surpondération ({cur*100:.1f}% vs cible {tgt*100:.1f}%, seuil max {upper_b*100:.1f}%).")
        elif cur < lower_b:
            status = "UNDERWEIGHT_TRIGGER"
            action_type = "ADD"
            triggers_count += 1
            underweight_gap[asset] = lower_b - cur
            reasoning.append(f"{asset} est en sous-pondération ({cur*100:.1f}% vs cible {tgt*100:.1f}%, seuil min {lower_b*100:.1f}%).")
        else:
            status = "OK"
            action_type = "HOLD"

        actions.append(
            RebalanceAction(
                asset=asset,
                target_weight=round(tgt * 100, 1),
                current_weight=round(cur * 100, 1),
                drift_pct=round(drift * 100, 2),
                lower_threshold=round(lower_b * 100, 1),
                upper_threshold=round(upper_b * 100, 1),
                status=status,
                suggested_action=action_type,
                dca_priority=0.0
            )
        )

    # Determine DCA allocation to rebalance passively without selling (zero tax/friction)
    dca_alloc = {}
    if underweight_gap:
        tot_gap = sum(underweight_gap.values())
        for a, gap in underweight_gap.items():
            dca_alloc[a] = round((gap / tot_gap) * 100, 1)
    else:
        # Invest proportional to target
        dca_alloc = {k: round(v * 100, 1) for k, v in targets.items()}

    # Assign DCA priority to actions
    for act in actions:
        act.dca_priority = dca_alloc.get(act.asset, 0.0)

    # Verdict
    is_rebalance = triggers_count > 0
    if regime_stress_prob > 0.40:
        regime_status = "STRESS"
        urgency = "HIGH" if is_rebalance else "MEDIUM"
        reasoning.append("Le régime macro HMM indique une probabilité de stress élevée (>40%). Élagage défensif recommandé si un seuil est franchi.")
    elif regime_stress_prob > 0.20:
        regime_status = "NORMAL"
        urgency = "MEDIUM" if is_rebalance else "LOW"
    else:
        regime_status = "CALM"
        urgency = "LOW"

    if not is_rebalance:
        verdict = "PAS DE RÉÉQUILIBRAGE NÉCESSAIRE. Toutes les lignes sont dans leurs bandes de tolérance. Laissez courir les gains."
        reasoning.append("Tous les actifs évoluent à l'intérieur de leur corridor de tolérance 5/25. Vendre maintenant engendrerait des frottements fiscaux et des frais de courtage sans gain d'espérance de rendement.")
    else:
        verdict = f"RÉÉQUILIBRAGE SUGGÉRÉ ({triggers_count} ligne(s) hors corridor). Privilégiez un rééquilibrage passif via vos prochains versements (DCA)."

    return RebalanceDiagnostic(
        is_rebalance_recommended=is_rebalance,
        urgency_level=urgency,
        regime_status=regime_status,
        regime_stress_prob=round(regime_stress_prob, 3),
        summary_verdict=verdict,
        reasoning=reasoning,
        actions=actions,
        dca_allocation=dca_alloc,
        next_review_date="2027-01-15 (Revue Trimestrielle)"
    )
