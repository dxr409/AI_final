"""
simulation_runner.py - Closed-Loop Pipeline & Terminal Demonstration
==================================================================
Connects all three AI modules in a daily closed-loop simulation:
  1. HMM infers corridor risk state & disruption probability p from noisy observations.
  2. MDP selects optimal order quantity based on current stock & inferred state,
     and computes stockout urgency U from current stock.
  3. Dynamic A* plans the cost-minimal multimodal route from Jebel Ali to Hamburg.
  4. Real-world fulfillment uses the TRUE scenario state (fixing the old ground-truth bug),
     reducing arrival rates only when the chosen route passes through chokepoints.
"""

from typing import List
import pandas as pd

from data_loader import PRESETS, SupplyChainDataGenerator
from hmm_module import SimpleHMM, risk_to_state
from mdp_planner import SimpleInventoryMDP
from graph_search import MultimodalNetwork


def format_route_mode(modes: List[str]) -> str:
    """Formats a sequence of edge transport modes into a readable summary."""
    if "Air" in modes:
        return "Air"
    if "Truck" in modes:
        return "Truck + Sea"
    return "Sea"


def run_closed_loop(
    scenario: str = "hormuz_blockage",
    days: int = 12,
    start_stock: int = 4,
    tariff: float = 2.0,
    seed: int = 42,
    strategy: str = "ai",
) -> pd.DataFrame:
    """
    Runs a closed-loop day-by-day supply chain simulation.

    Args:
      scenario:   'all_clear', 'tariff_surge', or 'hormuz_blockage'
      days:       Number of simulation days
      start_stock: Initial warehouse inventory (0..8 boxes)
      tariff:     Known import tariff surcharge per ordered box (USD)
      seed:       Random seed for noisy observations
      strategy:   'ai' (HMM + MDP + Dynamic A*) or 'baseline' (fixed order=2, fixed Hormuz+Suez)
    """
    hmm = SimpleHMM()
    mdp = SimpleInventoryMDP(max_inventory=8, tariff_per_box=tariff)
    policy, _ = mdp.solve_value_iteration()
    network = MultimodalNetwork()
    data_gen = SupplyChainDataGenerator(seed=seed)
    data = data_gen.generate_scenario(scenario, T=days)

    fixed_path = ["JEBEL_ALI", "HORMUZ", "SUEZ", "HAMBURG"]
    fixed_route = network.route_cost(fixed_path, disruption_prob=0.0, urgency_score=0.0)

    stock = int(start_stock)
    rows = []

    for day_idx in range(days):
        obs = data["observations"][day_idx]
        true_s = data["true_states"][day_idx]

        # 1. HMM belief update
        belief = hmm.update_belief(obs)
        p_risk = hmm.get_disruption_probability()
        inferred_s = hmm.get_most_likely_state()

        # 2 & 3. Decision selection (AI vs Static Baseline)
        urgency = mdp.get_urgency_score(stock)
        if strategy == "ai":
            order = int(policy[stock, inferred_s])
            route = network.a_star_search(
                start_node="JEBEL_ALI",
                goal_node="HAMBURG",
                disruption_prob=p_risk,
                urgency_score=urgency,
                tariff_per_box=tariff,
            )
            route_name = route["route_name"]
            mode = format_route_mode(route["modes"])
            route_base = route["total_base_cost"]
            has_chokepoint = len(route["chokepoints"]) > 0
        elif strategy == "baseline":
            order = 2
            route_name = fixed_route["route_name"]
            mode = format_route_mode(fixed_route["modes"])
            route_base = fixed_route["base"]
            has_chokepoint = len(fixed_route["chokepoints"]) > 0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 4. Real-world fulfillment uses TRUE ground-truth state (not inferred state)
        if has_chokepoint:
            delivered = int(round(order * mdp.delivery_success[true_s]))
        else:
            delivered = order  # Bypass or Air avoids maritime chokepoints

        available = stock + delivered
        sold = min(mdp.daily_demand, available)
        stockout = mdp.daily_demand - sold
        next_stock = min(mdp.max_inventory, available - sold)

        shipping_cost = route_base if order > 0 else 0.0
        daily_cost = (
            (mdp.cost_per_item + mdp.tariff_per_box) * order
            + shipping_cost
            + mdp.holding_cost_per_item * next_stock
            + mdp.stockout_penalty * stockout
        )

        rows.append(
            {
                "day": day_idx + 1,
                "observation": SupplyChainDataGenerator.OBS_NAMES[obs],
                "true state": SimpleHMM.STATE_NAMES[true_s],
                "inferred state": SimpleHMM.STATE_NAMES[inferred_s],
                "P(N)": float(belief[0]),
                "P(C)": float(belief[1]),
                "P(X)": float(belief[2]),
                "risk": float(p_risk),
                "stock": int(stock),
                "order": int(order),
                "urgency": float(urgency),
                "route_name": route_name,
                "mode": mode,
                "delivered": int(delivered),
                "stockout": int(stockout),
                "daily_cost": float(daily_cost),
            }
        )

        stock = next_stock

    return pd.DataFrame(rows)


def run_classroom_demonstration() -> None:
    print("=" * 82)
    print("  SUPPLY CHAIN VULNERABILITY & TARIFF ADVISORY SYSTEM (JEBEL ALI -> HAMBURG)")
    print("=" * 82)

    mdp = SimpleInventoryMDP(max_inventory=8, tariff_per_box=2.0)
    policy, _ = mdp.solve_value_iteration()
    network = MultimodalNetwork()

    # Step 1: MDP Policy Table
    print(f"\n[STEP 1: MDP VALUE ITERATION] (tariff = ${mdp.tariff_per_box:.0f}/box, converged in {mdp.iterations_used} iterations)")
    print(f"{'Inventory':<14} | {'Normal (100%)':<15} | {'Congestion (50%)':<17} | {'Crisis (25%)'}")
    print("-" * 68)
    for inv in range(mdp.max_inventory + 1):
        print(
            f"{inv} boxes       | Order {policy[inv, 0]} boxes   | "
            f"Order {policy[inv, 1]} boxes     | Order {policy[inv, 2]} boxes"
        )
    print("-" * 68)
    print(
        f"Solver summary at empty warehouse (0 boxes): Normal -> order {policy[0, 0]}, "
        f"Congestion -> order {policy[0, 1]}, Crisis -> order {policy[0, 2]}."
    )

    # Step 2: 3 Preset A* Comparisons
    print("\n" + "=" * 82)
    print("[STEP 2: DYNAMIC A* ROUTING ACROSS THE 3 PRESET SCENARIOS]")
    print("=" * 82)
    for key in ("all_clear", "tariff_surge", "hormuz_blockage"):
        preset = PRESETS[key]
        p = preset["risk_pct"] / 100.0
        u = mdp.get_urgency_score(preset["stock"])
        s_est = risk_to_state(p)
        route = network.a_star_search(
            start_node="JEBEL_ALI",
            goal_node="HAMBURG",
            disruption_prob=p,
            urgency_score=u,
            tariff_per_box=preset["tariff"],
        )
        tariff_exp = network.TARIFF_EXPOSURE * preset["tariff"] * route["total_transit_days"]
        print(f"\nPreset: {preset['label']} (Risk={preset['risk_pct']}%, Stock={preset['stock']}, Tariff=${preset['tariff']}/box)")
        print(f"  - Risk State: {SimpleHMM.STATE_NAMES[s_est]} | Urgency: {u:.1f}")
        print(f"  - A* Route:   {route['route_name']} ({' -> '.join(route['path'])})")
        print(
            f"  - Mode: {format_route_mode(route['modes'])} | Dynamic Cost: ${route['total_dynamic_cost']:.1f} "
            f"(Base ${route['total_base_cost']:.0f} + Tariff Exposure ${tariff_exp:.1f}) | Transit: {route['total_transit_days']:.1f}d | Nodes Expanded: {route['nodes_expanded']}"
        )

    # Step 3: AI vs Baseline Comparison Across All Scenarios
    print("\n" + "=" * 82)
    print("[STEP 3: CLOSED-LOOP SIMULATION SUMMARY (AI ADVISORY VS STATIC BASELINE, 15 DAYS)]")
    print("=" * 82)
    print(f"{'Scenario':<18} | {'HMM Acc':<8} | {'AI Stockout':<12} | {'Base Stockout':<14} | {'AI Total $':<11} | {'Base Total $'}")
    print("-" * 82)
    for key in ("all_clear", "tariff_surge", "hormuz_blockage"):
        tariff = PRESETS[key]["tariff"]
        df_ai = run_closed_loop(scenario=key, days=15, start_stock=4, tariff=tariff, strategy="ai")
        df_base = run_closed_loop(scenario=key, days=15, start_stock=4, tariff=tariff, strategy="baseline")
        hmm_acc = (df_ai["true state"] == df_ai["inferred state"]).mean() * 100.0
        ai_so = int((df_ai["stockout"] > 0).sum())
        base_so = int((df_base["stockout"] > 0).sum())
        ai_cost = df_ai["daily_cost"].sum()
        base_cost = df_base["daily_cost"].sum()
        print(
            f"{PRESETS[key]['label']:<18} | {hmm_acc:6.1f}%  | {ai_so:>3} days     | "
            f"{base_so:>3} days       | ${ai_cost:>8.1f}  | ${base_cost:>8.1f}"
        )
    print("-" * 82)


if __name__ == "__main__":
    run_classroom_demonstration()
