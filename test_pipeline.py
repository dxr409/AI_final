"""
test_pipeline.py - Automated Verification Suite
===============================================
Verifies correctness of all 3 AI modules, heuristic admissibility,
and closed-loop ground-truth fulfillment:
  1. HMM Forward filtering, belief normalization, and forward_step_details.
  2. MDP Value Iteration policy sanity and Q-value breakdown identity.
  3. Dynamic A* routing decisions across 5 representative (p, U) regimes.
  4. Provable admissibility h(n) <= h*(n) across all nodes and (p, U) grid.
  5. Closed-loop simulation using TRUE ground-truth corridor states.
"""

import numpy as np

from hmm_module import SimpleHMM
from mdp_planner import SimpleInventoryMDP
from graph_search import MultimodalNetwork
from simulation_runner import run_closed_loop


def test_hmm():
    hmm = SimpleHMM()
    assert np.isclose(np.sum(hmm.current_belief), 1.0)

    # Check forward_step_details purity and normalization math
    prior = hmm.current_belief.copy()
    details = hmm.forward_step_details(prior, observation=2)
    assert np.isclose(np.sum(details["posterior"]), 1.0)
    assert np.isclose(details["Z"], np.sum(details["unnormalized"]))
    assert np.allclose(details["posterior"], details["unnormalized"] / details["Z"])
    assert np.allclose(hmm.current_belief, prior), "forward_step_details must be pure"

    # Repeated 'Expensive & Stuck' signals must push Crisis belief > 0.60
    for _ in range(4):
        belief = hmm.update_belief(2)
        assert np.isclose(np.sum(belief), 1.0)

    assert belief[2] > 0.60, "Crisis probability should dominate after repeated stuck signals!"
    print("✓ Test 1 Passed: HMM belief updates, purity, and normalization verified.")


def test_mdp():
    mdp = SimpleInventoryMDP(max_inventory=8, tariff_per_box=2.0)
    policy, V = mdp.solve_value_iteration()

    assert policy[0, 0] > 0, "Must order when stock is 0 in Normal state"
    assert policy[0, 1] > 0, "Must order when stock is 0 in Congestion state"
    assert policy[8, 0] == 0, "Must order 0 when warehouse is completely full"

    # Check Q-value breakdown identity: total == immediate + future
    for inv in (0, 2, 5):
        for s in (0, 1, 2):
            for a in mdp.actions:
                bd = mdp.q_value_breakdown(inv, s, a)
                assert np.isclose(bd["total"], bd["immediate"] + bd["future"], atol=1e-9)

    print("✓ Test 2 Passed: MDP Value Iteration policy and Bellman Q-breakdown verified.")


def test_a_star_rerouting():
    net = MultimodalNetwork()

    # 1. Calm seas, low urgency -> Hormuz + Suez (base 110, 18 days)
    r1 = net.a_star_search(disruption_prob=0.0, urgency_score=0.0)
    assert r1["path"] == ["JEBEL_ALI", "HORMUZ", "SUEZ", "HAMBURG"]
    assert np.isclose(r1["total_base_cost"], 110.0)
    assert np.isclose(r1["total_transit_days"], 18.0)

    # 2. Moderate risk (0.4), zero urgency -> Truck to Salalah + Sea via Suez (194)
    r2 = net.a_star_search(disruption_prob=0.4, urgency_score=0.0)
    assert r2["path"] == ["JEBEL_ALI", "SALALAH", "SUEZ", "HAMBURG"]
    assert np.isclose(r2["total_dynamic_cost"], 194.0)

    # 3. High Hormuz/Red Sea disruption (0.9), zero urgency -> Salalah + Cape (210)
    r3 = net.a_star_search(disruption_prob=0.9, urgency_score=0.0)
    assert r3["path"] == ["JEBEL_ALI", "SALALAH", "CAPE", "HAMBURG"]
    assert np.isclose(r3["total_dynamic_cost"], 210.0)

    # 4. High disruption (0.9) + Warehouse Empty (U=1.0) -> Air Cargo from Dubai (318)
    r4 = net.a_star_search(disruption_prob=0.9, urgency_score=1.0)
    assert r4["path"] == ["JEBEL_ALI", "DUBAI_AIRPORT", "HAMBURG"]
    assert np.isclose(r4["total_dynamic_cost"], 318.0)

    # 5. Calm seas (0.0), moderate urgency (0.4) -> Hormuz + Suez
    r5 = net.a_star_search(disruption_prob=0.0, urgency_score=0.4)
    assert r5["path"] == ["JEBEL_ALI", "HORMUZ", "SUEZ", "HAMBURG"]

    print("✓ Test 3 Passed: Dynamic A* selects expected optimal routes across all 5 regimes.")


def test_tariff_sensitivity():
    net = MultimodalNetwork()

    # Hormuz Blockage (p=0.9, U=0): stays on Salalah+Cape at tau=7, flips to Air at tau=8
    r_cape = net.a_star_search(disruption_prob=0.9, urgency_score=0.0, tariff_per_box=7.0)
    assert r_cape["path"] == ["JEBEL_ALI", "SALALAH", "CAPE", "HAMBURG"]

    r_air_block = net.a_star_search(disruption_prob=0.9, urgency_score=0.0, tariff_per_box=8.0)
    assert r_air_block["path"] == ["JEBEL_ALI", "DUBAI_AIRPORT", "HAMBURG"]

    # Tariff Surge (p=0.4, U=0): Salalah+Suez at tau=12 (140 + 54 + 0.4*12*18 = 280.4), Air at tau=17
    r_surge = net.a_star_search(disruption_prob=0.4, urgency_score=0.0, tariff_per_box=12.0)
    assert r_surge["path"] == ["JEBEL_ALI", "SALALAH", "SUEZ", "HAMBURG"]
    assert np.isclose(r_surge["total_dynamic_cost"], 280.4)

    r_air_surge = net.a_star_search(disruption_prob=0.4, urgency_score=0.0, tariff_per_box=17.0)
    assert r_air_surge["path"] == ["JEBEL_ALI", "DUBAI_AIRPORT", "HAMBURG"]

    # All Clear (p=0.05, U=0, tau=2): Hormuz+Suez
    r_clear = net.a_star_search(disruption_prob=0.05, urgency_score=0.0, tariff_per_box=2.0)
    assert r_clear["path"] == ["JEBEL_ALI", "HORMUZ", "SUEZ", "HAMBURG"]

    print("✓ Test 4 Passed: Dynamic A* responds to tariff-timing exposure across regimes.")


def test_heuristic_admissibility():
    net = MultimodalNetwork()
    for p in (0.0, 0.5, 1.0):
        for u in (0.0, 0.5, 1.0):
            for tau in (0.0, 20.0):
                true_costs = net.true_costs_to_goal(
                    "HAMBURG", disruption_prob=p, urgency_score=u, tariff_per_box=tau
                )
                for node in net.nodes:
                    h_n = net.heuristic(node, "HAMBURG")
                    h_star = true_costs[node]
                    assert h_n <= h_star + 1e-9, (
                        f"Heuristic overestimated at {node} for p={p}, U={u}, tau={tau}: h={h_n} > h*={h_star}"
                    )
    print("✓ Test 5 Passed: A* heuristic h(n) <= h*(n) verified across all nodes and (p, U, tau) grid.")


def test_closed_loop_simulation():
    chokepoint_routes = {
        "Sea via Hormuz + Suez",
        "Sea via Hormuz + Cape",
        "Truck to Salalah + Sea via Suez",
    }

    for scen in ("all_clear", "tariff_surge", "hormuz_blockage"):
        for strat in ("ai", "baseline"):
            df = run_closed_loop(scenario=scen, days=15, start_stock=4, tariff=2.0, strategy=strat)
            assert len(df) == 15
            if scen == "hormuz_blockage":
                crisis_chokepoint_rows = df[
                    (df["true state"] == "Crisis") & (df["route_name"].isin(chokepoint_routes))
                ]
                if strat == "baseline":
                    assert len(crisis_chokepoint_rows) > 0
                for _, row in crisis_chokepoint_rows.iterrows():
                    expected_delivered = int(round(row["order"] * 0.25))
                    assert row["delivered"] == expected_delivered, (
                        f"Expected ground-truth Crisis arrival {expected_delivered}, got {row['delivered']}"
                    )

    print("✓ Test 6 Passed: Closed-loop simulation runs all scenarios and enforces ground-truth states.")


if __name__ == "__main__":
    print("Running verification suite...")
    test_hmm()
    test_mdp()
    test_a_star_rerouting()
    test_tariff_sensitivity()
    test_heuristic_admissibility()
    test_closed_loop_simulation()
    print("\nALL 6 VERIFICATION SUITES PASSED!\n")
