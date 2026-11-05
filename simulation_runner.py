"""
simulation_runner.py - Clean Terminal Runner for Class Demonstration
====================================================================
Runs the end-to-end simulation linking all 3 algorithms together:
  1. HMM estimates disruption risk from noisy delay/price observations.
  2. MDP decides how many items to reorder based on inventory & risk.
  3. A* picks the best multimodal route (Sea, Rail, Air) to ship the goods.
"""

from data_loader import SupplyChainDataGenerator
from hmm_module import SimpleHMM
from mdp_planner import SimpleInventoryMDP
from graph_search import MultimodalNetwork


def run_classroom_demonstration():
    print("=" * 75)
    print("  SUPPLY CHAIN ADVISORY SYSTEM - AI FUNDAMENTALS DEMO")
    print("=" * 75)

    # 1. Initialize our three algorithms
    hmm = SimpleHMM()
    mdp = SimpleInventoryMDP(max_inventory=8)
    network = MultimodalNetwork()
    data_gen = SupplyChainDataGenerator(seed=42)

    # 2. Step 1: Solve the MDP offline with Value Iteration
    print("\n[STEP 1: MDP VALUE ITERATION]")
    print("Solving optimal warehouse ordering policy using Bellman equations...")
    policy, V = mdp.solve_value_iteration()
    print("✓ Solved! Here is the manager's decision table:\n")

    print(f"{'Current Inventory':<18} | {'Normal State':<14} | {'Congestion State':<16} | {'Crisis State'}")
    print("-" * 65)
    for inv in range(0, mdp.max_inventory + 1):
        print(
            f"{inv} boxes in stock    | Order {policy[inv, 0]} boxes    | "
            f"Order {policy[inv, 1]} boxes      | Order {policy[inv, 2]} boxes"
        )
    print("-" * 65)
    print("Notice the logic: When warehouse is empty (0 boxes), we order 4-6 boxes.")
    print("During a Crisis, orders are reduced because goods get stuck at customs!")

    # 3. Step 2: Compare A* routes under different conditions
    print("\n" + "=" * 75)
    print("[STEP 2: A* DYNAMIC REROUTING UNDER DIFFERENT SCENARIOS]")
    print("=" * 75)

    scenarios = [
        ("1. Peaceful Times (Zero Risk, Normal Inventory)", 0.0, 0.0),
        ("2. Suez Canal Blocked (High Disruption, Normal Inventory)", 0.9, 0.0),
        ("3. Canal Blocked + Warehouse Empty (High Disruption + Urgent Stockout)", 0.9, 1.0),
    ]

    for title, p_risk, urgency in scenarios:
        route = network.a_star_search(
            start_node="SHANGHAI",
            goal_node="HAMBURG",
            disruption_prob=p_risk,
            urgency_score=urgency,
        )
        print(f"\nScenario: {title}")
        print(f"  • Disruption Risk: {p_risk*100:.0f}%  |  Warehouse Urgency: {urgency*100:.0f}%")
        print(f"  • A* Selected Route: {' ➔ '.join(route['path'])}")
        print(f"  • Travel Mode: {route['modes'][0]}")
        print(f"  • Total Cost: ${route['total_dynamic_cost']:.1f} (Base Shipping: ${route['total_base_cost']:.0f})")
        print(f"  • Transit Time: {route['total_transit_days']:.0f} days")

    # 4. Step 3: Run full day-by-day simulation
    print("\n" + "=" * 75)
    print("[STEP 3: FULL DAY-BY-DAY INTEGRATED SIMULATION (12 DAYS)]")
    print("=" * 75)

    data = data_gen.generate_scenario("tariff_crisis", T=12)
    current_inventory = 5  # Start with 5 boxes in warehouse
    hmm.reset()

    print(
        f"{'Day':<4} | {'Observation':<18} | {'HMM Belief':<20} | "
        f"{'Stock':<6} | {'Order':<6} | {'Urgency':<8} | {'A* Chosen Route'}"
    )
    print("-" * 88)

    for day in range(12):
        obs = data["observations"][day]
        obs_text = data_gen.OBS_NAMES[obs]

        # Algorithm 1: Update HMM belief
        belief = hmm.update_belief(obs)
        risk_prob = hmm.get_disruption_probability()
        most_likely_s = hmm.get_most_likely_state()

        # Algorithm 2: Look up MDP policy and check urgency
        order_qty = policy[current_inventory, most_likely_s]
        urgency = mdp.get_urgency_score(current_inventory)

        # Algorithm 3: Run A* search
        route = network.a_star_search(
            start_node="SHANGHAI",
            goal_node="HAMBURG",
            disruption_prob=risk_prob,
            urgency_score=urgency,
        )

        # Simple inventory update for tomorrow:
        # Items arrived today + current inventory - customer demand (2)
        arrived = int(round(order_qty * mdp.delivery_success[most_likely_s]))
        next_inventory = max(0, min(mdp.max_inventory, current_inventory + arrived - mdp.daily_demand))

        belief_str = f"[{belief[0]:.2f}, {belief[1]:.2f}, {belief[2]:.2f}]"
        path_str = f"{route['path'][1]} ({route['modes'][0]})"

        print(
            f"{day:<4} | {obs_text:<18} | {belief_str:<20} | "
            f"{current_inventory:<6} | {order_qty:<6} | {urgency:<8.1f} | {path_str}"
        )

        current_inventory = next_inventory

    print("-" * 88)
    print("Simulation finished successfully!")


if __name__ == "__main__":
    run_classroom_demonstration()
