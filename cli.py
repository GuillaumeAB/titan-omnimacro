"""
OmniMacro Engine - Institutional Command-Line Interface (CLI)
Execution console for quantitative scenario simulation across SPX, NDX, MSCI World, and STOXX 600.
"""

import sys
import os

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.monte_carlo_engine import MonteCarloEngine
from engine.scenarios import get_all_scenarios
from data.feeds.prediction_markets import PredictionMarketsFeed
from data.feeds.geopolitical_intelligence import GeopoliticalIntelligenceFeed


console = Console(force_terminal=True, legacy_windows=False)


def print_banner():
    banner = Text(
        "================================================================================\n"
        "              TITAN OMNIMACRO - 5-YEAR MULTI-ASSET PROJECTION ENGINE           \n"
        "      Geopolitical Transmission | Bottom-Up Constituent Dynamics | MC Paths     \n"
        "================================================================================",
        style="bold cyan"
    )
    console.print(banner)


def display_prediction_markets():
    feed = PredictionMarketsFeed()
    markets = feed.fetch_markets()
    
    table = Table(title="Live Prediction Markets ('Gambling' / Collective Probabilities)", style="magenta")
    table.add_column("Market Source", style="cyan", width=12)
    table.add_column("Event / Contract Title", style="white")
    table.add_column("Implied Odds", justify="right", style="bold green", width=14)
    table.add_column("Volume USD", justify="right", style="yellow", width=14)
    table.add_column("Horizon", justify="center", style="dim", width=10)

    for m in markets:
        prob_str = f"{m.implied_probability * 100:.1f}%"
        vol_str = f"${m.volume_usd / 1e6:.1f}M"
        table.add_row(m.source, m.title, prob_str, vol_str, m.resolution_horizon)
    console.print(table)


def display_geopolitical_intel():
    feed = GeopoliticalIntelligenceFeed()
    signals = feed.get_latest_signals()

    table = Table(title="Qualitative Intelligence Stream & Geopolitical Signals", style="red")
    table.add_column("Source", style="cyan", width=22)
    table.add_column("Flashpoint / Intelligence Headline", style="white")
    table.add_column("GPR Score", justify="right", style="bold red", width=11)
    table.add_column("Sentiment", justify="right", style="yellow", width=11)

    for s in signals:
        sent_color = "green" if s.sentiment_score > 0 else "red"
        sent_text = f"[{sent_color}]{s.sentiment_score:+.2f}[/{sent_color}]"
        table.add_row(s.source, s.headline, f"{s.geopolitical_risk_score:.0f}/100", sent_text)
    console.print(table)


def run_cli_simulation(scenario_id: str, runs: int, horizon: int):
    console.print(f"\n[bold yellow]>> Initializing 5-Year Stochastic Path Engine...[/bold yellow]")
    console.print(f"Scenario: [bold cyan]{scenario_id}[/bold cyan] | Horizon: [bold cyan]{horizon} months (5 Years)[/bold cyan] | Runs: [bold cyan]{runs}[/bold cyan]")

    engine = MonteCarloEngine()
    with console.status("[bold green]Computing bottom-up constituent shocks and multi-asset paths...[/bold green]"):
        res = engine.run_simulation(horizon_months=horizon, num_runs=runs, scenario_id=scenario_id)

    # 1. Index Projections Table
    idx_table = Table(title=f"5-Year Index Dynamics Projection ({res.scenario_mode})", style="blue")
    idx_table.add_column("Index", style="cyan", width=12)
    idx_table.add_column("Base (2026)", justify="right", style="white", width=12)
    idx_table.add_column("5Y Target (P50)", justify="right", style="bold green", width=14)
    idx_table.add_column("5Y Total Ret", justify="right", style="bold yellow", width=14)
    idx_table.add_column("5Y CAGR", justify="right", style="bold magenta", width=11)
    idx_table.add_column("Max DD (P50)", justify="right", style="red", width=12)
    idx_table.add_column("Bear (P10)", justify="right", style="dim red", width=12)
    idx_table.add_column("Bull (P90)", justify="right", style="dim green", width=12)

    for code, traj in res.indices.items():
        p10_final = traj.monthly_path[-1].p10
        p90_final = traj.monthly_path[-1].p90
        idx_table.add_row(
            code,
            f"{traj.base_level:,.1f}",
            f"{traj.final_p50_level:,.1f}",
            f"{traj.total_return_5y_pct:+.1f}%",
            f"{traj.cagr_5y_pct:+.1f}%",
            f"-{traj.max_drawdown_pct:.1f}%",
            f"{p10_final:,.1f}",
            f"{p90_final:,.1f}"
        )
    console.print(idx_table)

    # 2. Constituent Outperformers & Underperformers
    perf_table = Table(title="Constituent Dislocation Matrix (Individual Component Dynamics)", style="green")
    perf_table.add_column("Symbol", style="bold white", width=10)
    perf_table.add_column("Company Name", style="white", width=24)
    perf_table.add_column("Sector", style="dim", width=22)
    perf_table.add_column("Index Weight", justify="right", width=12)
    perf_table.add_column("5Y Return", justify="right", style="bold green", width=12)
    perf_table.add_column("Primary Macro / Geopolitical Catalyst", style="yellow")

    console.print("\n[bold green][+] Top Outperforming Constituents (Beneficiaries):[/bold green]")
    for w in res.top_winners:
        perf_table.add_row(
            w.symbol,
            w.name,
            w.sector,
            f"{w.weight:.2f}%",
            f"{w.total_return_5y_pct:+.1f}%",
            w.primary_driver
        )
    console.print(perf_table)

    losers_table = Table(title="Underperforming Constituents (High Geopolitical / Shock Vulnerability)", style="red")
    losers_table.add_column("Symbol", style="bold white", width=10)
    losers_table.add_column("Company Name", style="white", width=24)
    losers_table.add_column("Sector", style="dim", width=22)
    losers_table.add_column("Index Weight", justify="right", width=12)
    losers_table.add_column("5Y Return", justify="right", style="bold red", width=12)
    losers_table.add_column("Primary Vulnerability", style="yellow")

    for l in res.top_losers:
        losers_table.add_row(
            l.symbol,
            l.name,
            l.sector,
            f"{l.weight:.2f}%",
            f"{l.total_return_5y_pct:+.1f}%",
            l.primary_driver
        )
    console.print(losers_table)


def main():
    parser = argparse.ArgumentParser(description="Titan OmniMacro Geopolitical & Financial 5-Year Projection Console")
    parser.add_argument("--scenario", type=str, default="composite_polymarket",
                        help="Scenario ID (e.g. composite_polymarket, ai_hyper_supercycle, taiwan_semiconductor_blockade, mideast_hormuz_oil_shock)")
    parser.add_argument("--runs", type=int, default=150, help="Number of Monte Carlo simulations")
    parser.add_argument("--horizon", type=int, default=60, help="Projection horizon in months (default 60 = 5 years)")
    parser.add_argument("--show-odds", action="store_true", help="Display live prediction market odds")
    parser.add_argument("--show-intel", action="store_true", help="Display qualitative geopolitical intelligence stream")
    parser.add_argument("--list-scenarios", action="store_true", help="List all pre-configured scenarios")

    args = parser.parse_args()
    print_banner()

    if args.list_scenarios:
        scenarios = get_all_scenarios()
        t = Table(title="Available Pre-Configured Scenarios")
        t.add_column("ID", style="cyan")
        t.add_column("Name", style="bold white")
        t.add_column("Category", style="magenta")
        t.add_column("Baseline Prob", justify="right", style="green")
        for s in scenarios:
            t.add_row(s.id, s.name, s.category, f"{s.probability*100:.0f}%")
        console.print(t)
        return

    if args.show_odds:
        display_prediction_markets()

    if args.show_intel:
        display_geopolitical_intel()

    run_cli_simulation(scenario_id=args.scenario, runs=args.runs, horizon=args.horizon)


if __name__ == "__main__":
    main()
