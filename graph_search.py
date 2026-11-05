"""
graph_search.py - Dynamic Multimodal A* Route Planner
=====================================================
Topic: Classical Search & Heuristics

WHAT THIS DOES IN SIMPLE WORDS:
  A shipment of laptops needs to travel from SHANGHAI to HAMBURG.
  There are 4 main route choices:
    1. Standard Sea via Suez Canal: Very cheap ($100), but takes 20 days.
       -> Vulnerable to blockages and tariff bottlenecks!
    2. Sea Detour via Cape of Good Hope: Cheap ($160), safe from Suez, but takes 30 days.
    3. Overland Express Rail: Medium price ($200), fast (10 days), bypasses sea canals.
    4. Air Cargo: Expensive ($400), but lightning fast (2 days).

HOW ALGORITHM 1 & 2 CONTROL THE ROUTER:
  Dynamic Edge Cost = Base_Cost + Disruption_Penalty + Urgency_Penalty
    - Disruption_Penalty = Base_Cost * Disruption_Prob * Risk_Factor  (from HMM)
    - Urgency_Penalty = Urgency_Score * 12.0 * Transit_Days           (from MDP)

A* HEURISTIC:
  h(node) = Straight-line Euclidean distance to Hamburg * 8.0
  Because straight-line distance is always shorter than or equal to any real route,
  h(node) NEVER overestimates the true cost (Admissible Heuristic).
"""

import math
import heapq
from typing import Dict, List, Tuple, Any


class MultimodalNetwork:
    """A clean, 6-node multimodal transport network from Shanghai to Hamburg."""

    def __init__(self):
        # 2D coordinates for drawing and heuristic distance calculations
        # (x, y) coordinates on a simple grid
        self.node_positions = {
            "SHANGHAI": (0.0, 3.0),
            "SUEZ_CANAL": (6.0, 2.0),
            "CAPE_ROUTE": (5.0, -2.0),
            "TRANS_ASIA_RAIL": (5.0, 5.0),
            "AIR_CARGO": (5.0, 3.0),
            "HAMBURG": (11.0, 4.0),
        }

        # Edges connecting the network:
        # (start, end, mode, base_cost, transit_days, canal_risk_factor)
        self.edges = [
            # 1. Sea route via Suez Canal (Cheap, but high risk at Suez)
            ("SHANGHAI", "SUEZ_CANAL", "Sea", 60.0, 12.0, 2.0),
            ("SUEZ_CANAL", "HAMBURG", "Sea", 40.0, 8.0, 1.5),

            # 2. Sea detour via Cape of Good Hope (Avoids Suez, but slow)
            ("SHANGHAI", "CAPE_ROUTE", "Sea", 90.0, 18.0, 0.0),
            ("CAPE_ROUTE", "HAMBURG", "Sea", 70.0, 12.0, 0.0),

            # 3. Trans-Asia Express Rail (Bypasses sea entirely, faster)
            ("SHANGHAI", "TRANS_ASIA_RAIL", "Rail", 110.0, 5.0, 0.0),
            ("TRANS_ASIA_RAIL", "HAMBURG", "Rail", 90.0, 5.0, 0.0),

            # 4. Air Cargo (Ultra-fast, zero canal risk, higher base cost)
            ("SHANGHAI", "AIR_CARGO", "Air", 220.0, 1.0, 0.0),
            ("AIR_CARGO", "HAMBURG", "Air", 180.0, 1.0, 0.0),
        ]

    def heuristic(self, current_node: str, goal_node: str) -> float:
        """
        Admissible straight-line distance heuristic.
        h(n) <= true remaining cost to goal.
        """
        x1, y1 = self.node_positions[current_node]
        x2, y2 = self.node_positions[goal_node]
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Scaled by 8.0, which is smaller than the cheapest cost per distance unit
        return distance * 8.0

    def calculate_edge_cost(
        self,
        base_cost: float,
        transit_days: float,
        risk_factor: float,
        disruption_prob: float,
        urgency_score: float,
    ) -> float:
        """
        Dynamic edge cost combining:
          1. Base financial shipping fee
          2. Disruption penalty (from HMM belief)
          3. Time urgency penalty (from MDP inventory level)
        """
        # Extra cost if this leg passes through a disrupted canal/chokepoint
        disruption_penalty = base_cost * (disruption_prob * risk_factor * 1.5)

        # Extra penalty for slow travel days if warehouse is running out of stock
        urgency_penalty = urgency_score * 12.0 * transit_days

        return base_cost + disruption_penalty + urgency_penalty

    def a_star_search(
        self,
        start_node: str = "SHANGHAI",
        goal_node: str = "HAMBURG",
        disruption_prob: float = 0.0,
        urgency_score: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs standard A* search using a priority queue (min-heap).
        """
        # Priority queue holds tuples of: (f_score, counter, current_node, path, g_cost, days, base_cost, modes)
        counter = 0
        open_set = []
        start_h = self.heuristic(start_node, goal_node)
        heapq.heappush(open_set, (start_h, counter, start_node, [start_node], 0.0, 0.0, 0.0, []))

        visited_costs = {start_node: 0.0}

        while open_set:
            f, _, current, path, g, total_days, total_base, modes = heapq.heappop(open_set)

            # If we reached Hamburg, we found the optimal route!
            if current == goal_node:
                return {
                    "path": path,
                    "modes": modes,
                    "total_dynamic_cost": g,
                    "total_base_cost": total_base,
                    "total_transit_days": total_days,
                }

            # Check neighbors
            for u, v, mode, base, days, risk in self.edges:
                if u == current:
                    step_cost = self.calculate_edge_cost(
                        base, days, risk, disruption_prob, urgency_score
                    )
                    new_g = g + step_cost

                    if v not in visited_costs or new_g < visited_costs[v]:
                        visited_costs[v] = new_g
                        h = self.heuristic(v, goal_node)
                        new_f = new_g + h
                        counter += 1
                        heapq.heappush(
                            open_set,
                            (
                                new_f,
                                counter,
                                v,
                                path + [v],
                                new_g,
                                total_days + days,
                                total_base + base,
                                modes + [mode],
                            ),
                        )

        raise RuntimeError("No route found!")
