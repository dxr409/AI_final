"""
graph_search.py - Dynamic Multimodal A* Route Planner (Jebel Ali -> Hamburg)
=============================================================================
Topic: Classical Search & Heuristics

Models a 7-node multimodal transport network from Jebel Ali (Dubai) to
Hamburg across 5 distinct corridor alternatives:
  1. Sea via Hormuz + Suez        ($110 base, 18 days, vulnerable at Hormuz & Red Sea)
  2. Sea via Hormuz + Cape        ($185 base, 31 days, vulnerable at Hormuz)
  3. Truck to Salalah + Sea Suez  ($140 base, 18 days, bypasses Hormuz, Red Sea risk)
  4. Truck to Salalah + Sea Cape  ($210 base, 30 days, bypasses Hormuz & Suez)
  5. Air Cargo from Dubai         ($300 base, 1.5 days, fastest & zero chokepoint risk)

DYNAMIC EDGE COST:
  cost(e) = base + (base * kappa * p * r) + (beta * U * days) + (theta * tau * days)
    - kappa (RISK_WEIGHT = 1.5): sensitivity to HMM disruption probability p and edge risk r
    - beta (URGENCY_WEIGHT = 12.0): time-urgency penalty per day from MDP stockout urgency U
    - theta (TARIFF_EXPOSURE = 0.4): tariff-timing exposure per USD/box per transit day

ADMISSIBLE & CONSISTENT HEURISTIC:
  h(n) = c_min * haversine_km(n, goal)
  where c_min = min_{e in E} (base_cost(e) / haversine_km(e.u, e.v)).
  Because disruption, urgency, and tariff-timing penalties are always >= 0, every edge cost satisfies:
    cost(u, v) >= base_cost(u, v) >= c_min * haversine_km(u, v).
  By the spherical triangle inequality, haversine_km(u, goal) <= haversine_km(u, v) + haversine_km(v, goal),
  which guarantees both consistency (h(u) <= cost(u, v) + h(v)) and admissibility (h(n) <= h*(n))
  for all p >= 0, U >= 0, and tau >= 0.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Any, Dict, List, Tuple


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers between two (lon, lat) coordinates."""
    r_earth = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * r_earth * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


@dataclass(frozen=True)
class Edge:
    """Directed multimodal transport leg with geographic waypoints for offline rendering."""

    u: str
    v: str
    mode: str
    base_cost: float
    days: float
    risk_factor: float
    label: str
    waypoints: Tuple[Tuple[float, float], ...]


class MultimodalNetwork:
    """7-node multimodal logistics network from Jebel Ali (Dubai) to Hamburg."""

    RISK_WEIGHT = 1.5      # kappa
    URGENCY_WEIGHT = 12.0  # beta
    TARIFF_EXPOSURE = 0.4  # theta: ≈ 8-box shipment × 5%/day chance the tariff hits before clearance

    ROUTE_NAMES: Dict[Tuple[str, ...], str] = {
        ("JEBEL_ALI", "HORMUZ", "SUEZ", "HAMBURG"): "Sea via Hormuz + Suez",
        ("JEBEL_ALI", "HORMUZ", "CAPE", "HAMBURG"): "Sea via Hormuz + Cape",
        ("JEBEL_ALI", "SALALAH", "SUEZ", "HAMBURG"): "Truck to Salalah + Sea via Suez",
        ("JEBEL_ALI", "SALALAH", "CAPE", "HAMBURG"): "Truck to Salalah + Sea via Cape",
        ("JEBEL_ALI", "DUBAI_AIRPORT", "HAMBURG"): "Air Cargo from Dubai",
    }

    def __init__(self):
        # Nodes: {key: (display_name, lon, lat, kind)}
        self.nodes: Dict[str, Tuple[str, float, float, str]] = {
            "JEBEL_ALI": ("Jebel Ali (Dubai)", 55.03, 25.00, "port"),
            "HORMUZ": ("Strait of Hormuz", 56.45, 26.55, "chokepoint"),
            "SALALAH": ("Salalah (Oman)", 54.00, 16.95, "port"),
            "SUEZ": ("Suez Canal", 32.55, 30.00, "canal"),
            "CAPE": ("Cape of Good Hope", 18.00, -35.00, "cape"),
            "DUBAI_AIRPORT": ("Dubai Airport", 55.36, 25.25, "airport"),
            "HAMBURG": ("Hamburg", 9.99, 53.55, "port"),
        }

        self.node_positions: Dict[str, Tuple[float, float]] = {
            k: (lon, lat) for k, (_, lon, lat, _) in self.nodes.items()
        }

        red_sea_tail = (
            (45.0, 12.0),
            (43.4, 12.6),
            (42.0, 15.0),
            (39.5, 19.5),
            (37.0, 23.5),
            (34.6, 26.8),
            (33.2, 28.5),
        )
        cape_mid = (
            (47.0, -2.0),
            (42.5, -11.0),
            (40.0, -20.0),
            (36.5, -27.0),
            (31.5, -32.5),
            (25.0, -35.5),
        )
        atlantic_tail = (
            (-10.2, 43.5),
            (-6.0, 47.5),
            (-4.8, 48.8),
            (-1.0, 50.1),
            (1.5, 50.9),
            (3.0, 52.3),
            (5.5, 53.8),
            (8.3, 54.0),
        )

        self.edges: List[Edge] = [
            Edge(
                "JEBEL_ALI",
                "HORMUZ",
                "Sea",
                20.0,
                1.0,
                3.0,
                "Jebel Ali → Strait of Hormuz",
                ((55.6, 25.8),),
            ),
            Edge(
                "HORMUZ",
                "SUEZ",
                "Sea",
                50.0,
                9.0,
                2.0,
                "Hormuz → Suez (Red Sea)",
                ((57.5, 25.2), (60.2, 23.0), (58.5, 18.5), (52.0, 13.5)) + red_sea_tail,
            ),
            Edge(
                "HORMUZ",
                "CAPE",
                "Sea",
                95.0,
                17.0,
                0.0,
                "Hormuz → Cape of Good Hope",
                ((57.5, 25.2), (60.2, 23.0), (58.0, 15.0), (52.0, 8.0)) + cape_mid,
            ),
            Edge(
                "JEBEL_ALI",
                "SALALAH",
                "Truck",
                55.0,
                3.0,
                0.0,
                "Jebel Ali → Salalah (Overland Truck)",
                ((55.5, 23.5), (55.0, 20.5)),
            ),
            Edge(
                "SALALAH",
                "SUEZ",
                "Sea",
                45.0,
                7.0,
                2.0,
                "Salalah → Suez (Red Sea)",
                ((55.0, 15.5), (52.0, 13.5)) + red_sea_tail,
            ),
            Edge(
                "SALALAH",
                "CAPE",
                "Sea",
                85.0,
                14.0,
                0.0,
                "Salalah → Cape of Good Hope",
                ((55.0, 14.0), (52.0, 8.0)) + cape_mid,
            ),
            Edge(
                "SUEZ",
                "HAMBURG",
                "Sea",
                40.0,
                8.0,
                0.0,
                "Suez Canal → Hamburg",
                (
                    (32.3, 31.6),
                    (28.0, 33.5),
                    (20.0, 34.5),
                    (15.0, 36.3),
                    (11.5, 37.6),
                    (5.0, 37.6),
                    (-2.0, 36.3),
                    (-5.6, 35.95),
                    (-9.8, 36.5),
                )
                + atlantic_tail,
            ),
            Edge(
                "CAPE",
                "HAMBURG",
                "Sea",
                70.0,
                13.0,
                0.0,
                "Cape of Good Hope → Hamburg",
                (
                    (14.5, -30.0),
                    (10.0, -20.0),
                    (5.0, -5.0),
                    (-5.0, 2.0),
                    (-15.0, 8.0),
                    (-20.0, 15.0),
                    (-20.0, 25.0),
                    (-15.0, 33.0),
                    (-11.0, 40.0),
                )
                + atlantic_tail,
            ),
            Edge(
                "JEBEL_ALI",
                "DUBAI_AIRPORT",
                "Truck",
                10.0,
                0.5,
                0.0,
                "Jebel Ali → Dubai Airport (Truck)",
                (),
            ),
            Edge(
                "DUBAI_AIRPORT",
                "HAMBURG",
                "Air",
                290.0,
                1.0,
                0.0,
                "Dubai Airport → Hamburg (Air Cargo)",
                (),
            ),
        ]

        self._adj: Dict[str, List[Edge]] = {k: [] for k in self.nodes}
        self._rev_adj: Dict[str, List[Edge]] = {k: [] for k in self.nodes}
        self._edge_lookup: Dict[Tuple[str, str], Edge] = {}
        cost_ratios: List[float] = []

        for edge in self.edges:
            self._adj[edge.u].append(edge)
            self._rev_adj[edge.v].append(edge)
            self._edge_lookup[(edge.u, edge.v)] = edge
            lon1, lat1 = self.node_positions[edge.u]
            lon2, lat2 = self.node_positions[edge.v]
            dist_km = haversine_km(lon1, lat1, lon2, lat2)
            if dist_km > 0:
                cost_ratios.append(edge.base_cost / dist_km)

        # Lower-bound cost per km guarantees h(n) <= h*(n) and consistency across all (p, U)
        self.min_cost_per_km: float = min(cost_ratios)

    def heuristic(self, current_node: str, goal_node: str = "HAMBURG") -> float:
        """
        Provably admissible & consistent straight-line (great-circle) lower-bound heuristic:
          h(n) = min_cost_per_km * haversine_km(n, goal)
        """
        if current_node == goal_node:
            return 0.0
        lon1, lat1 = self.node_positions[current_node]
        lon2, lat2 = self.node_positions[goal_node]
        return self.min_cost_per_km * haversine_km(lon1, lat1, lon2, lat2)

    def edge_cost_breakdown(
        self,
        edge: Edge,
        disruption_prob: float,
        urgency_score: float,
        tariff_per_box: float = 0.0,
    ) -> Dict[str, float]:
        """Returns base, disruption, urgency, tariff exposure, and total dynamic cost for an edge."""
        base = edge.base_cost
        disruption = base * self.RISK_WEIGHT * disruption_prob * edge.risk_factor
        urgency = self.URGENCY_WEIGHT * urgency_score * edge.days
        tariff = self.TARIFF_EXPOSURE * tariff_per_box * edge.days
        return {
            "base": base,
            "disruption": disruption,
            "urgency": urgency,
            "tariff": tariff,
            "total": base + disruption + urgency + tariff,
        }

    def calculate_edge_cost(
        self,
        edge: Edge,
        disruption_prob: float,
        urgency_score: float,
        tariff_per_box: float = 0.0,
    ) -> float:
        """Dynamic edge cost combining base cost, HMM disruption, MDP urgency, and tariff exposure."""
        return self.edge_cost_breakdown(
            edge, disruption_prob, urgency_score, tariff_per_box=tariff_per_box
        )["total"]

    def get_route_name(self, path: List[str]) -> str:
        """Returns a human-readable route label for a node sequence."""
        return self.ROUTE_NAMES.get(tuple(path), " → ".join(path))

    def a_star_search(
        self,
        start_node: str = "JEBEL_ALI",
        goal_node: str = "HAMBURG",
        disruption_prob: float = 0.0,
        urgency_score: float = 0.0,
        tariff_per_box: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs standard A* search with a consistent heuristic and closed set.
        """
        counter = 0
        start_h = self.heuristic(start_node, goal_node)
        open_set: List[Tuple[float, int, str, List[str], float, float, float, List[str], List[str], List[str]]] = [
            (start_h, counter, start_node, [start_node], 0.0, 0.0, 0.0, [], [], [])
        ]
        best_g: Dict[str, float] = {start_node: 0.0}
        closed_set: set[str] = set()
        nodes_expanded = 0

        while open_set:
            (
                _f,
                _,
                current,
                path,
                g,
                total_days,
                total_base,
                modes,
                edge_labels,
                chokepoints,
            ) = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            if current == goal_node:
                return {
                    "path": path,
                    "modes": modes,
                    "edge_labels": edge_labels,
                    "total_dynamic_cost": g,
                    "total_base_cost": total_base,
                    "total_transit_days": total_days,
                    "nodes_expanded": nodes_expanded,
                    "chokepoints": chokepoints,
                    "route_name": self.get_route_name(path),
                }

            for edge in self._adj[current]:
                if edge.v in closed_set:
                    continue
                step_cost = self.calculate_edge_cost(
                    edge, disruption_prob, urgency_score, tariff_per_box=tariff_per_box
                )
                new_g = g + step_cost
                if edge.v not in best_g or new_g < best_g[edge.v]:
                    best_g[edge.v] = new_g
                    new_f = new_g + self.heuristic(edge.v, goal_node)
                    counter += 1
                    new_chokepoints = (
                        chokepoints + [edge.label] if edge.risk_factor > 0 else chokepoints
                    )
                    heapq.heappush(
                        open_set,
                        (
                            new_f,
                            counter,
                            edge.v,
                            path + [edge.v],
                            new_g,
                            total_days + edge.days,
                            total_base + edge.base_cost,
                            modes + [edge.mode],
                            edge_labels + [edge.label],
                            new_chokepoints,
                        ),
                    )

        raise RuntimeError(f"No route found from {start_node} to {goal_node}!")

    def enumerate_routes(
        self, start_node: str = "JEBEL_ALI", goal_node: str = "HAMBURG"
    ) -> List[List[str]]:
        """Finds all simple directed paths from start_node to goal_node via DFS."""
        routes: List[List[str]] = []

        def dfs(node: str, path: List[str]) -> None:
            if node == goal_node:
                routes.append(list(path))
                return
            for edge in self._adj[node]:
                if edge.v not in path:
                    path.append(edge.v)
                    dfs(edge.v, path)
                    path.pop()

        dfs(start_node, [start_node])
        return routes

    def route_cost(
        self,
        path: List[str],
        disruption_prob: float = 0.0,
        urgency_score: float = 0.0,
        tariff_per_box: float = 0.0,
    ) -> Dict[str, Any]:
        """Computes full cost breakdown across all edges of a specific path."""
        edges = [self._edge_lookup[(u, v)] for u, v in zip(path[:-1], path[1:])]
        base = 0.0
        disruption = 0.0
        urgency = 0.0
        tariff = 0.0
        days = 0.0
        modes: List[str] = []
        edge_labels: List[str] = []
        chokepoints: List[str] = []

        for edge in edges:
            bd = self.edge_cost_breakdown(
                edge, disruption_prob, urgency_score, tariff_per_box=tariff_per_box
            )
            base += bd["base"]
            disruption += bd["disruption"]
            urgency += bd["urgency"]
            tariff += bd["tariff"]
            days += edge.days
            modes.append(edge.mode)
            edge_labels.append(edge.label)
            if edge.risk_factor > 0:
                chokepoints.append(edge.label)

        return {
            "route_name": self.get_route_name(path),
            "path": path,
            "modes": modes,
            "edge_labels": edge_labels,
            "edges": edges,
            "base": base,
            "disruption": disruption,
            "urgency": urgency,
            "tariff": tariff,
            "total": base + disruption + urgency + tariff,
            "days": days,
            "chokepoints": chokepoints,
        }

    def true_costs_to_goal(
        self,
        goal_node: str = "HAMBURG",
        disruption_prob: float = 0.0,
        urgency_score: float = 0.0,
        tariff_per_box: float = 0.0,
    ) -> Dict[str, float]:
        """
        Computes exact minimum remaining cost h*(n) from every node to goal_node
        using Dijkstra on reversed edges (used to verify heuristic admissibility).
        """
        dist: Dict[str, float] = {goal_node: 0.0}
        pq: List[Tuple[float, str]] = [(0.0, goal_node)]

        while pq:
            d, node = heapq.heappop(pq)
            if d > dist.get(node, float("inf")):
                continue
            for edge in self._rev_adj[node]:
                step_cost = self.calculate_edge_cost(
                    edge, disruption_prob, urgency_score, tariff_per_box=tariff_per_box
                )
                nd = d + step_cost
                if nd < dist.get(edge.u, float("inf")):
                    dist[edge.u] = nd
                    heapq.heappush(pq, (nd, edge.u))

        return dist
