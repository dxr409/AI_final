"""
test_pipeline.py - Verification Tests for Class Demonstration
=============================================================
Quickly verifies that all 3 algorithms work correctly:
  1. HMM probabilities sum to 100% and respond to delay signals.
  2. MDP solves without errors and orders when inventory is empty.
  3. A* routes change dynamically when disruption or urgency occurs.
"""

import numpy as np
from hmm_module import SimpleHMM
from mdp_planner import SimpleInventoryMDP
from graph_search import MultimodalNetwork


def test_hmm():
    hmm = SimpleHMM()
    # Initial belief sums to 1.0
    assert np.isclose(np.sum(hmm.current_belief), 1.0)

    # After seeing "Expensive & Stuck" (2) multiple times, crisis probability must rise
    for _ in range(4):
        belief = hmm.update_belief(2)
        assert np.isclose(np.sum(belief), 1.0)

    assert belief[2] > 0.60, "Crisis probability should dominate after repeated stuck signals!"
    print("✓ Test 1 Passed: HMM belief updates and probability normalization work.")


def test_mdp():
    mdp = SimpleInventoryMDP(max_inventory=8)
    policy, V = mdp.solve_value_iteration()

    # When warehouse is empty (0 boxes), the manager MUST order goods in Normal and Congestion states
    assert policy[0, 0] > 0, "Must order when stock is 0 in normal state"
    assert policy[0, 1] > 0, "Must order when stock is 0 in congestion state"

    # When warehouse is completely full (8 boxes), manager should order 0
    assert policy[8, 0] == 0
    print("✓ Test 2 Passed: MDP Value Iteration finds sensible reordering policies.")


def test_a_star_rerouting():
    net = MultimodalNetwork()

    # 1. Zero risk -> Should pick cheap Suez Canal sea route
    res_normal = net.a_star_search(disruption_prob=0.0, urgency_score=0.0)
    assert res_normal["path"] == ["SHANGHAI", "SUEZ_CANAL", "HAMBURG"]
    assert res_normal["total_base_cost"] == 100.0

    # 2. Suez Canal blocked (high disruption) -> Should divert to Cape detour
    res_blocked = net.a_star_search(disruption_prob=0.9, urgency_score=0.0)
    assert res_blocked["path"] == ["SHANGHAI", "CAPE_ROUTE", "HAMBURG"]

    # 3. Blocked + Warehouse Empty (urgent) -> Should take Express Rail
    res_urgent = net.a_star_search(disruption_prob=0.9, urgency_score=1.0)
    assert res_urgent["path"] == ["SHANGHAI", "TRANS_ASIA_RAIL", "HAMBURG"]

    print("✓ Test 3 Passed: A* automatically reroutes under canal shocks and urgency.")


if __name__ == "__main__":
    print("Running verification suite...")
    test_hmm()
    test_mdp()
    test_a_star_rerouting()
    print("\nALL TESTS PASSED! Ready for student defense.\n")
