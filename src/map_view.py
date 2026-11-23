"""
map_view.py - Offline Interactive SVG Route Map (Jebel Ali -> Hamburg)
======================================================================
Renders a completely offline, self-contained HTML/SVG world map with:
  - Hand-curated coastlines (Africa, Eurasia, Great Britain, Ireland, Madagascar)
  - Sea-lane and overland waypoints (avoiding crossing landmass)
  - Dynamic Strait of Hormuz risk indicator (color gradient + pulsing ring at p >= 0.7)
  - Animated "moving trailing lines" (dashed flow + staggered comet particles via animateMotion)
  - Hover tooltips on every leg showing dynamic cost and transit days
"""

import html
from typing import Dict, List, Tuple
from src.graph_search import Edge, MultimodalNetwork

LonLat = Tuple[float, float]

AFRICA: List[LonLat] = [
    (-5.9, 35.8), (-9.8, 30.0), (-13.0, 27.5), (-16.5, 22.0), (-17.1, 14.7),
    (-16.7, 12.0), (-13.3, 9.0), (-11.0, 6.8), (-7.5, 4.4), (-3.0, 5.0),
    (1.0, 6.0), (4.5, 6.3), (8.5, 4.5), (9.5, 2.0), (9.0, -1.0),
    (11.8, -4.5), (13.2, -8.8), (12.3, -13.0), (11.8, -17.5), (14.5, -22.9),
    (15.2, -27.0), (16.5, -28.6), (18.4, -33.9), (20.0, -34.8), (22.5, -34.0),
    (25.6, -33.9), (28.0, -32.5), (31.0, -29.8), (32.9, -26.0), (35.5, -24.0),
    (35.3, -21.0), (37.0, -17.5), (40.5, -15.0), (40.5, -10.5), (39.3, -6.8),
    (40.2, -2.8), (43.0, 1.0), (47.5, 5.0), (51.2, 10.4), (51.2, 11.8),
    (48.0, 11.2), (43.4, 11.6), (42.5, 13.0), (41.0, 14.5), (39.0, 16.0),
    (37.4, 18.8), (36.5, 22.0), (35.5, 24.0), (33.8, 27.4), (32.5, 29.9),
    (32.3, 31.3), (29.5, 31.0), (25.0, 31.6), (20.0, 30.8), (19.5, 30.3),
    (15.5, 31.5), (11.0, 33.3), (10.2, 36.8), (8.5, 36.9), (3.0, 36.8),
    (-1.0, 35.3),
]

EURASIA: List[LonLat] = [
    (-5.6, 36.0), (-9.0, 37.0), (-9.5, 39.5), (-8.8, 42.5), (-8.0, 43.7),
    (-1.8, 43.4), (-1.2, 46.0), (-4.5, 48.0), (-1.5, 48.7), (1.5, 50.2),
    (2.5, 51.1), (4.0, 51.9), (4.8, 53.0), (7.0, 53.5), (8.6, 53.9),
    (8.6, 55.5), (8.1, 56.8), (10.5, 57.7), (11.8, 58.3), (11.2, 59.2),
    (10.6, 59.9), (10.2, 59.0), (8.0, 58.1), (6.5, 58.1), (5.5, 58.8),
    (5.0, 60.5), (5.0, 62.0), (76.0, 62.0), (76.0, 21.0), (70.2, 21.0),
    (68.9, 22.4), (67.3, 24.6), (66.6, 25.4), (61.6, 25.2), (57.3, 25.8),
    (56.4, 27.1), (54.0, 26.7), (51.4, 27.9), (50.2, 30.0), (48.5, 30.0),
    (47.9, 29.9), (48.5, 28.0), (49.6, 27.0), (50.2, 26.2), (50.8, 24.7),
    (51.3, 26.1), (51.6, 25.3), (51.6, 24.3), (54.0, 24.1), (55.1, 25.0),
    (56.0, 26.0), (56.4, 26.4), (56.3, 24.9), (57.0, 24.0), (58.6, 23.6),
    (59.8, 22.5), (58.5, 20.5), (57.7, 19.0), (56.6, 18.0), (55.3, 17.6),
    (54.1, 17.0), (52.2, 15.6), (49.0, 14.5), (45.0, 12.8), (43.5, 12.7),
    (42.7, 15.5), (41.5, 18.0), (39.2, 21.5), (38.4, 23.7), (37.2, 25.5),
    (35.2, 28.0), (34.6, 28.1), (34.9, 29.5), (34.2, 27.8), (33.6, 28.3),
    (32.6, 29.9), (32.3, 31.3), (34.2, 31.3), (35.5, 33.9), (36.0, 36.8),
    (32.0, 36.1), (30.5, 36.4), (27.3, 37.0), (26.3, 39.4), (26.2, 40.6),
    (23.9, 40.7), (22.9, 40.6), (23.6, 38.0), (22.5, 36.4), (21.7, 36.8),
    (21.0, 38.5), (19.4, 40.3), (19.5, 41.8), (16.0, 43.5), (13.6, 45.5),
    (12.3, 45.3), (12.4, 44.2), (14.0, 42.5), (16.0, 41.4), (18.5, 40.1),
    (16.6, 38.4), (15.7, 38.0), (15.8, 39.5), (14.5, 40.6), (12.6, 41.4),
    (10.5, 43.0), (8.8, 44.4), (7.5, 43.8), (5.0, 43.4), (3.1, 43.1),
    (3.2, 41.9), (0.8, 41.0), (-0.3, 39.4), (0.2, 38.7), (-0.7, 37.6),
    (-2.1, 36.7), (-4.4, 36.7),
]

GREAT_BRITAIN: List[LonLat] = [
    (-5.7, 50.1), (-3.0, 50.6), (1.4, 51.2), (1.7, 52.7), (0.2, 53.5),
    (-0.5, 54.5), (-1.6, 55.6), (-2.1, 57.7), (-3.1, 58.6), (-5.0, 58.6),
    (-6.2, 57.5), (-5.6, 56.0), (-4.9, 55.0), (-3.0, 54.2), (-3.1, 53.3),
    (-4.6, 53.3), (-4.5, 52.2), (-5.2, 51.7), (-3.2, 51.4), (-4.5, 51.0),
]

IRELAND: List[LonLat] = [
    (-6.0, 52.2), (-6.2, 53.4), (-5.5, 54.5), (-6.2, 55.3), (-7.3, 55.4),
    (-8.5, 54.6), (-10.0, 54.0), (-9.5, 53.0), (-10.3, 51.8), (-8.5, 51.6),
]

MADAGASCAR: List[LonLat] = [
    (49.3, -12.0), (50.5, -15.5), (47.1, -24.9), (45.0, -25.5),
    (43.3, -22.0), (44.3, -17.0), (46.3, -15.7),
]

BLACK_SEA: List[LonLat] = [
    (28.0, 41.2), (29.0, 41.2), (31.2, 41.1), (35.0, 42.0), (38.3, 40.9),
    (41.5, 41.5), (41.6, 42.6), (39.8, 43.5), (37.5, 44.8), (36.6, 45.4),
    (35.0, 45.0), (33.5, 44.5), (32.5, 45.4), (30.8, 46.6), (29.6, 45.3),
    (28.6, 44.0), (28.0, 42.5),
]

CASPIAN: List[LonLat] = [
    (47.0, 44.5), (49.0, 46.5), (52.0, 46.8), (53.0, 45.0), (51.3, 44.5),
    (52.7, 42.0), (53.9, 40.5), (53.0, 39.3), (53.9, 37.3), (51.0, 36.8),
    (49.0, 37.7), (49.5, 40.3), (48.5, 41.8), (47.5, 43.2),
]

MODE_COLORS = {
    "Sea": "#2563eb",
    "Truck": "#a16207",
    "Air": "#16a34a",
}


def _proj(lon: float, lat: float) -> Tuple[float, float]:
    """Equirectangular projection covering lon -22..74, lat -38..60 into 960x980."""
    return (lon + 22.0) * 10.0, (60.0 - lat) * 10.0


def _poly_points(coords: List[LonLat]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in (_proj(lo, la) for lo, la in coords))


def _edge_svg_segments(network: MultimodalNetwork, edge: Edge) -> Tuple[str, str]:
    """Returns (start_M_segment, continuation_segment) for an edge."""
    x1, y1 = _proj(*network.node_positions[edge.u])
    x2, y2 = _proj(*network.node_positions[edge.v])
    start = f"M {x1:.1f} {y1:.1f}"

    if edge.mode == "Air":
        cx = (x1 + x2) / 2.0 + 30.0
        cy = min(y1, y2) - 42.0
        cont = f"Q {cx:.1f} {cy:.1f} {x2:.1f} {y2:.1f}"
        return start, cont

    pts = [_proj(lo, la) for lo, la in edge.waypoints] + [(x2, y2)]
    cont = " ".join(f"L {px:.1f} {py:.1f}" for px, py in pts)
    return start, cont


def _risk_color(p: float) -> str:
    """Interpolates Hormuz marker color from green -> amber -> red as p goes 0 -> 1."""
    p = max(0.0, min(1.0, p))
    if p < 0.5:
        t = p * 2.0
        r = int(22 + (245 - 22) * t)
        g = int(163 + (158 - 163) * t)
        b = int(74 + (11 - 74) * t)
    else:
        t = (p - 0.5) * 2.0
        r = int(245 + (220 - 245) * t)
        g = int(158 + (38 - 158) * t)
        b = int(11 + (38 - 11) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def render_route_map(
    network: MultimodalNetwork,
    chosen_path: List[str],
    route_costs: Dict[str, float],
    disruption_prob: float,
    urgency_score: float = 0.0,
    tariff_per_box: float = 0.0,
) -> str:
    """
    Generates a standalone, offline HTML document containing an interactive SVG map
    with animated moving route dashes and trailing comet particles.
    """
    active_edges = set(zip(chosen_path[:-1], chosen_path[1:]))
    chosen_name = network.get_route_name(chosen_path)

    # Graticule lines
    grid_svg = []
    for lon in range(-15, 76, 15):
        gx, _ = _proj(lon, 0)
        grid_svg.append(
            f'<line x1="{gx:.1f}" y1="0" x2="{gx:.1f}" y2="980" stroke="#bfdbfe" stroke-width="0.6" stroke-dasharray="3 4"/>'
        )
    for lat in range(-30, 61, 15):
        _, gy = _proj(0, lat)
        grid_svg.append(
            f'<line x1="0" y1="{gy:.1f}" x2="960" y2="{gy:.1f}" stroke="#bfdbfe" stroke-width="0.6" stroke-dasharray="3 4"/>'
        )

    # Land & inland seas
    land_polys = [AFRICA, EURASIA, GREAT_BRITAIN, IRELAND, MADAGASCAR]
    land_svg = "\n".join(
        f'<polygon points="{_poly_points(poly)}" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.4" stroke-linejoin="round"/>'
        for poly in land_polys
    )
    inland_water_svg = "\n".join(
        f'<polygon points="{_poly_points(poly)}" fill="#dbeafe" stroke="#cbd5e1" stroke-width="1.1" stroke-linejoin="round"/>'
        for poly in (BLACK_SEA, CASPIAN)
    )

    # All network edges (background dashed)
    bg_edges_svg = []
    chosen_d_parts: List[str] = []

    edge_map = {(e.u, e.v): e for e in network.edges}
    for edge in network.edges:
        start_seg, cont_seg = _edge_svg_segments(network, edge)
        d_full = f"{start_seg} {cont_seg}"
        bd = network.edge_cost_breakdown(
            edge, disruption_prob, urgency_score, tariff_per_box=tariff_per_box
        )
        tooltip = html.escape(
            f"{edge.label} ({edge.mode}) — Cost: ${bd['total']:.1f} "
            f"(Base ${bd['base']:.0f} + Risk ${bd['disruption']:.1f} + Urgency ${bd['urgency']:.1f} + Tariff ${bd['tariff']:.1f}, {edge.days:g}d)"
        )
        color = MODE_COLORS[edge.mode]
        is_active = (edge.u, edge.v) in active_edges
        opacity = "0.22" if is_active else "0.45"
        bg_edges_svg.append(
            f'<path d="{d_full}" fill="none" stroke="{color}" stroke-width="2.4" '
            f'stroke-opacity="{opacity}" stroke-dasharray="6 5" stroke-linecap="round">'
            f"<title>{tooltip}</title></path>"
        )

    # Build continuous SVG path for chosen route in exact travel order
    for idx, (u, v) in enumerate(zip(chosen_path[:-1], chosen_path[1:])):
        edge = edge_map[(u, v)]
        start_seg, cont_seg = _edge_svg_segments(network, edge)
        if idx == 0:
            chosen_d_parts.append(start_seg)
        chosen_d_parts.append(cont_seg)
    chosen_d = " ".join(chosen_d_parts)

    # Active route overlay + comet particles
    active_route_svg = f"""
    <path d="{chosen_d}" fill="none" stroke="#ef4444" stroke-width="10" stroke-opacity="0.18" stroke-linecap="round" stroke-linejoin="round"/>
    <path id="chosen-route-path" d="{chosen_d}" fill="none" stroke="#ef4444" stroke-width="4.4" stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="14 10" class="flow-line">
        <title>{html.escape(chosen_name)}</title>
    </path>
    <g>
      <circle r="6.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5" opacity="0.96">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="0s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="5.0" fill="#ef4444" opacity="0.72">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="0.09s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="3.6" fill="#f97316" opacity="0.48">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="0.18s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="2.4" fill="#fb923c" opacity="0.28">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="0.27s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="6.5" fill="#dc2626" stroke="#ffffff" stroke-width="1.5" opacity="0.96">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="2.1s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="5.0" fill="#ef4444" opacity="0.72">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="2.19s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="3.6" fill="#f97316" opacity="0.48">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="2.28s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
      <circle r="2.4" fill="#fb923c" opacity="0.28">
        <animateMotion dur="4.2s" repeatCount="indefinite" begin="2.37s"><mpath href="#chosen-route-path"/></animateMotion>
      </circle>
    </g>
    """

    # Carefully placed node markers & non-overlapping labels
    label_offsets = {
        "JEBEL_ALI": (-11, 14, "end"),
        "DUBAI_AIRPORT": (-11, -9, "end"),
        "HORMUZ": (12, -10, "start"),
        "SALALAH": (11, 12, "start"),
        "SUEZ": (-11, -10, "end"),
        "CAPE": (12, -4, "start"),
        "HAMBURG": (12, 5, "start"),
    }

    hormuz_color = _risk_color(disruption_prob)
    risk_pct = int(round(disruption_prob * 100))

    nodes_svg = []
    for key, (name, lon, lat, kind) in network.nodes.items():
        nx, ny = _proj(lon, lat)
        is_in = key in chosen_path
        dx, dy, anchor = label_offsets.get(key, (10, -8, "start"))

        if key == "HORMUZ":
            pulse = ""
            if disruption_prob >= 0.7:
                pulse = (
                    f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="10" fill="none" stroke="#dc2626" stroke-width="2.5">'
                    f'<animate attributeName="r" values="8;24;8" dur="1.5s" repeatCount="indefinite"/>'
                    f'<animate attributeName="opacity" values="0.9;0.05;0.9" dur="1.5s" repeatCount="indefinite"/>'
                    f"</circle>"
                )
            label_text = f"Strait of Hormuz — risk {risk_pct}%"
            nodes_svg.append(
                f"{pulse}"
                f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="8" fill="{hormuz_color}" stroke="#1e293b" stroke-width="2"/>'
                f'<text x="{nx + dx:.1f}" y="{ny + dy:.1f}" text-anchor="{anchor}" class="map-label hormuz-label">'
                f"{html.escape(label_text)}</text>"
            )
        else:
            if kind == "airport" and not is_in:
                # Subtle small airport marker when Air isn't active so Jebel Ali stays uncluttered
                fill = "#16a34a"
                r = 4.5
                lbl = "DXB Airport"
            else:
                fill = "#ef4444" if is_in else "#334155"
                r = 7.2 if is_in else 5.5
                lbl = name
            nodes_svg.append(
                f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="1.8"/>'
                f'<text x="{nx + dx:.1f}" y="{ny + dy:.1f}" text-anchor="{anchor}" class="map-label">'
                f"{html.escape(lbl)}</text>"
            )

    # Sorted mini route-cost sidebar inside the card
    sorted_routes = sorted(route_costs.items(), key=lambda kv: kv[1])
    route_pills = []
    for rname, rcost in sorted_routes:
        active_cls = "route-pill active" if rname == chosen_name else "route-pill"
        badge = "✓ " if rname == chosen_name else ""
        route_pills.append(
            f'<div class="{active_cls}">'
            f'<span>{badge}{html.escape(rname)}</span>'
            f'<strong>${rcost:.1f}</strong>'
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
  body {{ margin: 0; padding: 0; background: transparent; color: #0f172a; }}
  .wrap {{
    position: relative;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
  }}
  svg {{ display: block; width: 100%; height: 590px; background: #dbeafe; }}
  .flow-line {{
    animation: dashflow 0.75s linear infinite;
  }}
  @keyframes dashflow {{
    from {{ stroke-dashoffset: 24; }}
    to   {{ stroke-dashoffset: 0; }}
  }}
  .map-label {{
    font-size: 15px;
    font-weight: 700;
    fill: #0f172a;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 4px;
    stroke-linejoin: round;
    pointer-events: none;
  }}
  .hormuz-label {{
    fill: #7f1d1d;
  }}
  .overlay-legend {{
    position: absolute;
    left: 14px;
    bottom: 14px;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 10px 13px;
    font-size: 12px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    max-width: 245px;
  }}
  .legend-title {{ font-weight: 700; margin-bottom: 6px; color: #1e293b; font-size: 12.5px; }}
  .leg-row {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .swatch {{ width: 20px; height: 4px; border-radius: 2px; display: inline-block; }}
  .swatch.dot {{ width: 10px; height: 10px; border-radius: 50%; margin: 0 5px; }}
  .overlay-routes {{
    position: absolute;
    right: 14px;
    bottom: 14px;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 12px;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    width: 265px;
  }}
  .route-pill {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 4px 7px;
    border-radius: 6px;
    margin-top: 3px;
    color: #334155;
  }}
  .route-pill.active {{
    background: #fee2e2;
    color: #991b1b;
    font-weight: 700;
    border: 1px solid #fca5a5;
  }}
  .note {{
    margin-top: 6px;
    font-size: 10.5px;
    color: #64748b;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <svg viewBox="0 0 960 980" preserveAspectRatio="xMidYMid meet">
      <rect x="0" y="0" width="960" height="980" fill="#dbeafe"/>
      {''.join(grid_svg)}
      {land_svg}
      {inland_water_svg}
      {''.join(bg_edges_svg)}
      {active_route_svg}
      {''.join(nodes_svg)}
    </svg>

    <div class="overlay-legend">
      <div class="legend-title">Corridor Map Legend</div>
      <div class="leg-row"><span class="swatch" style="background:#ef4444;height:5px;"></span> Active A* Optimal Route</div>
      <div class="leg-row"><span class="swatch" style="background:#2563eb;"></span> Maritime Sea Lane</div>
      <div class="leg-row"><span class="swatch" style="background:#a16207;"></span> Overland Truck Bypass</div>
      <div class="leg-row"><span class="swatch" style="background:#16a34a;"></span> Air Cargo Corridor</div>
      <div class="leg-row"><span class="swatch dot" style="background:{hormuz_color};"></span> Hormuz Risk ({risk_pct}%)</div>
      <div class="note">Not to scale – simplified offline map. Hover edges for leg details.</div>
    </div>

    <div class="overlay-routes">
      <div class="legend-title">Live Route Comparison (Dynamic Cost)</div>
      {''.join(route_pills)}
    </div>
  </div>
</body>
</html>"""
