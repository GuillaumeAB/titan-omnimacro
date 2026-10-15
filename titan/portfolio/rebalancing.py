"""
titan.portfolio.rebalancing — Swensen 5/25 Corridor Rebalancing Engine.
Evaluates portfolio drift against absolute and relative boundaries,
triggers alerts, and computes frictionless DCA passive rebalancing allocation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class RebalanceAction:
    asset: str
    target_weight: float
    current_weight: float
    drift_pct: float
    lower_threshold: float
    upper_threshold: float
    status: str  # "OK", "OVERWEIGHT_TRIGGER", "UNDERWEIGHT_TRIGGER"
    suggested_action: str  # "HOLD", "TRIM", "ADD"
    dca_priority: float


@dataclass
class RebalanceDiagnostic:
    is_rebalance_recommended: bool
    urgency_level: str  # "LOW", "MEDIUM", "HIGH"
    regime_status: str  # "CALM", "NORMAL", "STRESS"
    regime_stress_prob: float
    summary_verdict: str
    reasoning: list[str]
    actions: list[dict]
    dca_allocation: dict[str, float]


def evaluate_rebalancing(
    target_weights: dict[str, float],
    current_weights: dict[str, float] | None = None,
    regime_stress_prob: float = 0.15,
) -> RebalanceDiagnostic:
    """
    Evaluates whether the portfolio needs rebalancing based on:
    1. Institutional 5/25 corridor rule:
       - Absolute threshold: deviation > 5.0%
       - Relative threshold: deviation > 25% of target weight
       Trigger if |current - target| > min(5.0%, 0.25 * target)
    2. Regime-based defensive adjustment.
    """
    tot_tgt = sum(target_weights.values())
    targets = {k: v / tot_tgt for k, v in target_weights.items()} if tot_tgt > 0 else {}

    if current_weights is None:
        current_weights = targets.copy()

    tot_cur = sum(current_weights.values())
    currents = {k: v / tot_cur for k, v in current_weights.items()} if tot_cur > 0 else targets

    actions: list[RebalanceAction] = []
    triggers_count = 0
    reasoning = []
    underweight_gap: dict[str, float] = {}

    for asset, tgt in targets.items():
        cur = currents.get(asset, 0.0)
        drift = cur - tgt

        tolerance = min(0.05, max(0.015, tgt * 0.25))
        lower_b = max(0.0, tgt - tolerance)
        upper_b = tgt + tolerance

        if cur > upper_b:
            status = "OVERWEIGHT_TRIGGER"
            action_type = "TRIM"
            triggers_count += 1
            reasoning.append(
                f"{asset}: surpondération ({cur*100:.1f}% vs cible {tgt*100:.1f}%, max {upper_b*100:.1f}%)."
            )
        elif cur < lower_b:
            status = "UNDERWEIGHT_TRIGGER"
            action_type = "ADD"
            triggers_count += 1
            underweight_gap[asset] = lower_b - cur
            reasoning.append(
                f"{asset}: sous-pondération ({cur*100:.1f}% vs cible {tgt*100:.1f}%, min {lower_b*100:.1f}%)."
            )
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
                dca_priority=0.0,
            )
        )

    # Passive DCA allocation
    dca_alloc = {}
    if underweight_gap:
        tot_gap = sum(underweight_gap.values())
        for a, gap in underweight_gap.items():
            dca_alloc[a] = round((gap / tot_gap) * 100, 1)
    else:
        dca_alloc = {k: round(v * 100, 1) for k, v in targets.items()}

    for act in actions:
        act.dca_priority = dca_alloc.get(act.asset, 0.0)

    is_rebalance = triggers_count > 0
    if regime_stress_prob > 0.40:
        regime_status = "STRESS"
        urgency = "HIGH" if is_rebalance else "MEDIUM"
        reasoning.append("Régime macro HMM en état de stress (>40%). Élagage défensif recommandé si seuil franchi.")
    elif regime_stress_prob > 0.20:
        regime_status = "NORMAL"
        urgency = "MEDIUM" if is_rebalance else "LOW"
    else:
        regime_status = "CALM"
        urgency = "LOW"

    if not is_rebalance:
        verdict = "AUCUN RÉÉQUILIBRAGE REQUIS. Toutes les allocations demeurent dans les corridors 5/25."
    else:
        verdict = f"RÉÉQUILIBRAGE CONSEILLÉ ({triggers_count} ligne(s) hors corridor). Privilégiez un DCA passif."

    return RebalanceDiagnostic(
        is_rebalance_recommended=is_rebalance,
        urgency_level=urgency,
        regime_status=regime_status,
        regime_stress_prob=round(regime_stress_prob, 3),
        summary_verdict=verdict,
        reasoning=reasoning,
        actions=[asdict(a) for a in actions],
        dca_allocation=dca_alloc,
    )
