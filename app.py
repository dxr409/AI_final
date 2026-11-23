"""
app.py - Interactive AI Supply Chain Advisory Dashboard (Jebel Ali -> Hamburg)
==============================================================================
100% offline Streamlit application uniting:
  1. Hidden Markov Model (HMM)       - Corridor Risk Inference
  2. Markov Decision Process (MDP)   - Optimal Warehouse Replenishment
  3. Dynamic A* Search               - Multimodal Route Selection
"""

from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_loader import PRESETS, SupplyChainDataGenerator
from graph_search import MultimodalNetwork
from hmm_module import SimpleHMM, risk_to_state
from map_view import render_route_map
from mdp_planner import SimpleInventoryMDP
from simulation_runner import format_route_mode, run_closed_loop
from ui_components import inject_css, interactive_formula, sym

st.set_page_config(
    page_title="AI Supply Chain Advisor (Hormuz → Hamburg)",
    page_icon="🚢",
    layout="wide",
)
inject_css()


# --- SESSION STATE INITIALIZATION (BEFORE ANY WIDGET) ---
def apply_preset(preset_key: str) -> None:
    preset = PRESETS[preset_key]
    st.session_state["scenario"] = preset_key
    st.session_state["risk_pct"] = preset["risk_pct"]
    st.session_state["stock"] = preset["stock"]
    st.session_state["tariff"] = preset["tariff"]


if "scenario" not in st.session_state:
    apply_preset("all_clear")


@st.cache_data
def solve_policy(tariff: float) -> Tuple[np.ndarray, np.ndarray, int]:
    mdp_solver = SimpleInventoryMDP(max_inventory=8, tariff_per_box=tariff)
    policy, V = mdp_solver.solve_value_iteration()
    return policy, V, mdp_solver.iterations_used


# --- HEADER ---
st.title("🚢 AI Supply Chain & Strait of Hormuz Advisory")
st.markdown(
    "Integrates an **HMM** (corridor risk inference), an **MDP** (optimal warehouse ordering), "
    "and **Dynamic A\\*** (multimodal routing) to keep laptop shipments moving from **Jebel Ali (Dubai)** to **Hamburg**."
)

# --- SCENARIO PRESET BUTTONS ---
st.markdown("#### Quick Scenarios")
b_cols = st.columns(3)
preset_keys = ["all_clear", "tariff_surge", "hormuz_blockage"]
for col, key in zip(b_cols, preset_keys):
    preset = PRESETS[key]
    is_active = st.session_state["scenario"] == key
    with col:
        st.button(
            preset["label"],
            key=f"btn_{key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            on_click=apply_preset,
            args=(key,),
        )
        st.caption(preset["description"])

# --- LIVE SLIDERS ---
s_col1, s_col2, s_col3 = st.columns(3)
with s_col1:
    st.slider(
        "Corridor Disruption Risk p (%)",
        min_value=0,
        max_value=100,
        key="risk_pct",
    )
with s_col2:
    st.slider(
        "Current Warehouse Stock (boxes)",
        min_value=0,
        max_value=8,
        key="stock",
    )
with s_col3:
    st.slider(
        "Import Tariff Surcharge (USD / box)",
        min_value=0,
        max_value=20,
        key="tariff",
    )

# --- DERIVED STATE & AI DECISIONS ---
p_risk = st.session_state["risk_pct"] / 100.0
current_stock = int(st.session_state["stock"])
tariff_val = float(st.session_state["tariff"])

mdp = SimpleInventoryMDP(max_inventory=8, tariff_per_box=tariff_val)
policy, V, mdp_iters = solve_policy(tariff_val)
mdp.V = V
mdp.policy = policy
mdp.iterations_used = mdp_iters

risk_state = risk_to_state(p_risk)
urgency_U = mdp.get_urgency_score(current_stock)
recommended_order = int(policy[current_stock, risk_state])

with s_col2:
    st.caption(f"→ Stockout Urgency **U = {urgency_U:.1f}** (out of 1.0)")
with s_col3:
    st.caption("Affects MDP ordering and A* routing: slow routes risk paying the new tariff (θ·τ·d)")

network = MultimodalNetwork()
best_route = network.a_star_search(
    start_node="JEBEL_ALI",
    goal_node="HAMBURG",
    disruption_prob=p_risk,
    urgency_score=urgency_U,
    tariff_per_box=tariff_val,
)

all_paths = network.enumerate_routes("JEBEL_ALI", "HAMBURG")
route_breakdowns = [
    network.route_cost(path, p_risk, urgency_U, tariff_per_box=tariff_val)
    for path in all_paths
]
route_breakdowns.sort(key=lambda r: r["total"])
route_costs_map = {r["route_name"]: r["total"] for r in route_breakdowns}

# --- KPI ROWS (3 + 3 so route names never truncate) ---
r1_1, r1_2, r1_3 = st.columns(3)
r1_1.metric("Risk State (from p)", SimpleHMM.STATE_NAMES[risk_state])
r1_2.metric("Stockout Urgency U", f"{urgency_U:.1f}")
r1_3.metric("MDP Order Qty", f"{recommended_order} boxes")

r2_1, r2_2, r2_3 = st.columns(3)
r2_1.metric("A* Chosen Route", best_route["route_name"])
r2_2.metric("Transit Time", f"{best_route['total_transit_days']:.1f} days")
r2_3.metric(
    "Dynamic Route Cost",
    f"USD {best_route['total_dynamic_cost']:.1f}",
    delta=f"Base USD {best_route['total_base_cost']:.0f}",
    delta_color="off",
)

st.markdown("---")

# --- TABS ---
tab_map, tab_calc, tab_mdp, tab_sim, tab_how = st.tabs(
    [
        "🗺️ Route Map",
        "🧮 Calculations",
        "📦 Order Policy (MDP)",
        "📈 Simulation & Results",
        "ℹ️ How it works",
    ]
)

# ================= TAB 1: ROUTE MAP =================
with tab_map:
    st.subheader("Live Multimodal Corridor Map (Jebel Ali → Hamburg)")
    st.markdown(
        f"**Selected by A\\*:** {best_route['route_name']} ({format_route_mode(best_route['modes'])}) — "
        f"**{best_route['total_transit_days']:.1f} days**, dynamic cost **USD {best_route['total_dynamic_cost']:.1f}**."
    )

    map_html = render_route_map(
        network=network,
        chosen_path=best_route["path"],
        route_costs=route_costs_map,
        disruption_prob=p_risk,
        urgency_score=urgency_U,
        tariff_per_box=tariff_val,
    )
    components.html(map_html, height=610, scrolling=False)

    st.markdown("#### All 5 Route Candidates Ranked by Dynamic Cost")
    route_table = []
    for rb in route_breakdowns:
        is_best = rb["route_name"] == best_route["route_name"]
        route_table.append(
            {
                "Status": "✅ Selected" if is_best else "—",
                "Route": rb["route_name"],
                "Mode": format_route_mode(rb["modes"]),
                "Transit (days)": rb["days"],
                "Base Cost (USD)": round(rb["base"], 1),
                "Risk Penalty (USD)": round(rb["disruption"], 1),
                "Urgency Penalty (USD)": round(rb["urgency"], 1),
                "Tariff Exposure (USD)": round(rb["tariff"], 1),
                "Total Dynamic Cost (USD)": round(rb["total"], 1),
            }
        )
    st.dataframe(pd.DataFrame(route_table), hide_index=True, use_container_width=True)

# ================= TAB 2: INTERACTIVE CALCULATIONS =================
with tab_calc:
    st.subheader("Interactive Formula Explorer")
    st.caption(
        "Hover or click any colored symbol, live number, or card below to see where it comes from and how it moves the decision."
    )

    # --- 2A. DYNAMIC A* EDGE COST ---
    st.markdown("### 1. Dynamic A\\* Route Cost & Admissible Heuristic")
    st.markdown(
        "A\\* scores every leg by adding base freight, a Hormuz/Red Sea disruption penalty from the HMM, "
        "a delay penalty from warehouse urgency, and tariff-timing exposure on slow transit."
    )

    edge_formula_html = (
        f"{sym('cost', 'cost(e)')} = {sym('b', 'b')} + "
        f"{sym('b', 'b')} · {sym('kappa', '&kappa;')} · {sym('p', 'p')} · {sym('r', 'r')} + "
        f"{sym('beta', '&beta;')} · {sym('U', 'U')} · {sym('d', 'd')} + "
        f"{sym('theta', '&theta;')} · {sym('tau', '&tau;')} · {sym('d', 'd')}"
    )

    astar_rows = []
    for rb in route_breakdowns:
        is_best = rb["route_name"] == best_route["route_name"]
        cls = "calc-row winner" if is_best else "calc-row"
        # Sum of risk-weighted base across edges (sum(b_i * r_i))
        br_sum = sum(e.base_cost * e.risk_factor for e in rb["edges"])
        base_s = f"{rb['base']:.0f}"
        days_s = f"{rb['days']:g}"
        tau_s = f"{tariff_val:g}"
        tot_s = f"USD {rb['total']:.1f}"
        expr = (
            f"{sym('b', base_s)} + "
            f"{sym('kappa', '1.5')}·{sym('p', f'{p_risk:.2f}')}·{sym('r', f'{br_sum:.0f}')} + "
            f"{sym('beta', '12')}·{sym('U', f'{urgency_U:.1f}')}·{sym('d', days_s)} + "
            f"{sym('theta', '0.4')}·{sym('tau', tau_s)}·{sym('d', days_s)}"
        )
        tag = "✅ " if is_best else ""
        astar_rows.append(
            f'<div class="{cls}">'
            f'<span class="row-name">{tag}{rb["route_name"]}</span>'
            f'<span class="row-expr">{expr}</span>'
            f'<span class="row-total">= {sym("cost", tot_s)}</span>'
            f"</div>"
        )

    astar_vars = [
        {
            "key": "cost",
            "symbol": "cost(e)",
            "name": "Dynamic Edge / Route Cost",
            "value": f"USD {best_route['total_dynamic_cost']:.1f} (best)",
            "source": "Route data",
            "effect": "A* minimizes the sum of edge costs from Jebel Ali to Hamburg.",
        },
        {
            "key": "b",
            "symbol": "b",
            "name": "Base Shipping Fee",
            "value": "USD 110–300 across routes",
            "source": "Route data",
            "effect": "Direct freight tariff; Sea via Hormuz+Suez is cheapest at calm times.",
        },
        {
            "key": "p",
            "symbol": "p",
            "name": "Corridor Disruption Risk",
            "value": f"{p_risk:.2f} ({st.session_state['risk_pct']}%)",
            "source": "HMM",
            "effect": "Higher p penalizes routes crossing Hormuz (r=3) and Red Sea (r=2), forcing bypasses.",
        },
        {
            "key": "r",
            "symbol": "r (or Σ b·r)",
            "name": "Chokepoint Risk Factor",
            "value": "Hormuz=3.0, Red Sea=2.0, Cape/Air=0",
            "source": "Route data",
            "effect": "Multiplies base cost on vulnerable chokepoints; 0 means completely safe from blockages.",
        },
        {
            "key": "U",
            "symbol": "U",
            "name": "Warehouse Stockout Urgency",
            "value": f"{urgency_U:.1f} (stock={current_stock})",
            "source": "MDP",
            "effect": "Higher U penalizes slow transit days, switching sea cargo to fast Air Cargo.",
        },
        {
            "key": "d",
            "symbol": "d",
            "name": "Transit Duration",
            "value": "1.5d (Air) to 31d (Cape)",
            "source": "Route data",
            "effect": "Multiplies both urgency penalty and tariff timing exposure so slow routes lose out under tight deadlines.",
        },
        {
            "key": "tau",
            "symbol": "&tau;",
            "name": "Import Tariff",
            "value": f"USD {tariff_val:.0f} / box",
            "source": "Input",
            "effect": "Higher τ makes slow routes (Cape, 30 days) riskier, pushing A* toward faster routes or Air.",
        },
        {
            "key": "kappa",
            "symbol": "&kappa;",
            "name": "Disruption Sensitivity",
            "value": f"{network.RISK_WEIGHT:.1f}",
            "source": "Constant",
            "effect": "Scales how strongly HMM disruption probability inflates chokepoint costs.",
        },
        {
            "key": "beta",
            "symbol": "&beta;",
            "name": "Urgency Time Penalty",
            "value": f"{network.URGENCY_WEIGHT:.0f} USD / day",
            "source": "Constant",
            "effect": "Dollar penalty per transit day at 100% urgency.",
        },
        {
            "key": "theta",
            "symbol": "&theta;",
            "name": "Tariff Timing Exposure",
            "value": f"{network.TARIFF_EXPOSURE:.1f} per USD / day",
            "source": "Constant",
            "effect": "Converts tariff level into expected cost per transit day (~8-box shipment × 5%/day tariff-hit chance).",
        },
    ]

    interactive_formula(
        title="Dynamic A* Edge & Route Cost Formula",
        formula_html=edge_formula_html,
        variables=astar_vars,
        rows_html="".join(astar_rows),
        footnote=(
            f"<strong>f(n) = g(n) + h(n)</strong>: A* expanded <strong>{best_route['nodes_expanded']}</strong> of "
            f"{len(network.nodes)} nodes using admissible heuristic h(n) = {network.min_cost_per_km:.4f} × great-circle km."
        ),
        height=610,
    )

    with st.expander("🔍 Verify A* Heuristic Admissibility: h(n) ≤ h*(n) for every node", expanded=False):
        true_costs = network.true_costs_to_goal(
            "HAMBURG", p_risk, urgency_U, tariff_per_box=tariff_val
        )
        adm_rows = []
        for node_key, (node_name, _, _, _) in network.nodes.items():
            h_val = network.heuristic(node_key, "HAMBURG")
            h_star = true_costs[node_key]
            adm_rows.append(
                {
                    "Node": node_name,
                    "Heuristic h(n) (USD)": round(h_val, 2),
                    "True Optimal Remaining h*(n) (USD)": round(h_star, 2),
                    "Admissible (h ≤ h*)": "✅ Yes" if h_val <= h_star + 1e-9 else "❌ No",
                }
            )
        st.dataframe(pd.DataFrame(adm_rows), hide_index=True, use_container_width=True)

    # --- 2B. MDP BELLMAN Q-VALUE ---
    st.markdown("### 2. MDP Bellman Order Evaluation")
    st.markdown(
        f"For your current stock (**I = {current_stock}**) and risk state (**{SimpleHMM.STATE_NAMES[risk_state]}**), "
        "Bellman Value Iteration evaluates every candidate order size and picks the lowest expected long-run cost."
    )

    ip_sym = sym("Ip", "I'")
    ev_sym = sym("EV", "&Sigma; P(s'|s)·V(I',s')")
    mdp_formula_html = (
        f"{sym('Q', 'Q(I,s,a)')} = "
        f"({sym('c', 'c')} + {sym('tau', '&tau;')})·{sym('a', 'a')} + "
        f"{sym('ch', 'c<sub>h</sub>')}·{ip_sym} + "
        f"{sym('cso', 'c<sub>so</sub>')}·{sym('short', 'short')} + "
        f"{sym('gamma', '&gamma;')}·{ev_sym}"
    )

    q_breakdowns = [mdp.q_value_breakdown(current_stock, risk_state, a) for a in mdp.actions]
    min_q = min(qb["total"] for qb in q_breakdowns)
    mdp_rows = []
    for qb in q_breakdowns:
        is_best = abs(qb["total"] - min_q) < 1e-6
        cls = "calc-row winner" if is_best else "calc-row"
        tag = "✅ " if is_best else ""
        ev_s = f"{qb['future'] / mdp.gamma:.1f}"
        q_tot_s = f"USD {qb['total']:.1f}"
        expr = (
            f"({sym('c', '4')}+{sym('tau', f'{tariff_val:g}')})·{sym('a', str(qb['action']))} + "
            f"{sym('ch', '1')}·{sym('Ip', str(qb['next_inv']))} + "
            f"{sym('cso', '30')}·{sym('short', str(qb['shortage']))} + "
            f"{sym('gamma', f'{mdp.gamma:.2f}')}·{sym('EV', ev_s)}"
        )
        mdp_rows.append(
            f'<div class="{cls}">'
            f'<span class="row-name">{tag}Order a = {qb["action"]} boxes (arrives {qb["arrived"]})</span>'
            f'<span class="row-expr">{expr}</span>'
            f'<span class="row-total">= {sym("Q", q_tot_s)}</span>'
            f"</div>"
        )

    best_qb = min(q_breakdowns, key=lambda x: x["total"])
    mdp_vars = [
        {
            "key": "Q",
            "symbol": "Q(I,s,a)",
            "name": "Expected Long-Run Cost",
            "value": f"USD {best_qb['total']:.1f} (best a={best_qb['action']})",
            "source": "MDP",
            "effect": "Action with minimum Q(I,s,a) becomes optimal policy pi*(I,s).",
        },
        {
            "key": "a",
            "symbol": "a",
            "name": "Order Quantity",
            "value": f"Best = {recommended_order} boxes",
            "source": "MDP",
            "effect": "Candidate choices {0, 2, 4, 6}; arrivals scale by state delivery rate.",
        },
        {
            "key": "tau",
            "symbol": "&tau;",
            "name": "Import Tariff Surcharge",
            "value": f"USD {tariff_val:.0f} / box",
            "source": "Input",
            "effect": "Known policy surcharge; higher tariffs discourage speculative over-ordering.",
        },
        {
            "key": "Ip",
            "symbol": "I'",
            "name": "Ending Inventory",
            "value": f"{best_qb['next_inv']} boxes (at best a)",
            "source": "MDP",
            "effect": "min(8, max(0, I + arrived - 2)); enters holding cost and tomorrow's value V(I', s').",
        },
        {
            "key": "short",
            "symbol": "short",
            "name": "Unmet Demand (Stockout)",
            "value": f"{best_qb['shortage']} boxes (at best a)",
            "source": "MDP",
            "effect": "max(0, 2 - (I + arrived)); triggers steep USD 30/box stockout penalty.",
        },
        {
            "key": "EV",
            "symbol": "E[V(I',s')]",
            "name": "Expected Future Cost",
            "value": f"USD {best_qb['future'] / mdp.gamma:.1f}",
            "source": "MDP",
            "effect": "Weighted average of tomorrow's optimal value over transition probabilities P(s'|s).",
        },
        {
            "key": "c",
            "symbol": "c",
            "name": "Wholesale Unit Cost",
            "value": "USD 4 / box",
            "source": "Constant",
            "effect": "Baseline procurement price per laptop box.",
        },
        {
            "key": "ch",
            "symbol": "c<sub>h</sub>",
            "name": "Holding Cost",
            "value": "USD 1 / box / day",
            "source": "Constant",
            "effect": "Overnight storage fee preventing needless stock hoarding.",
        },
        {
            "key": "cso",
            "symbol": "c<sub>so</sub>",
            "name": "Stockout Penalty",
            "value": "USD 30 / box",
            "source": "Constant",
            "effect": "Lost-sales penalty driving emergency orders when stock approaches zero.",
        },
        {
            "key": "gamma",
            "symbol": "&gamma;",
            "name": "Discount Factor",
            "value": f"{mdp.gamma:.2f}",
            "source": "Constant",
            "effect": f"Weights future days vs today; Value Iteration converged in {mdp_iters} sweeps.",
        },
    ]

    interactive_formula(
        title="MDP Bellman Equation Breakdown",
        formula_html=mdp_formula_html,
        variables=mdp_vars,
        rows_html="".join(mdp_rows),
        footnote=(
            f"Delivery success rates: Normal = 100%, Congestion = 50%, Crisis = 25%. "
            f"Value Iteration converged in {mdp_iters} iterations (tolerance 0.001)."
        ),
        height=560,
    )

    # --- 2C. HMM FORWARD STEP ---
    st.markdown("### 3. HMM Forward Filtering Step")
    st.markdown(
        "Instead of overreacting to one noisy report, the HMM blends yesterday's belief through the transition matrix **A** with today's observation likelihood **B**."
    )

    sim_for_hmm = SupplyChainDataGenerator(seed=42).generate_scenario(
        st.session_state["scenario"], T=15
    )
    inspect_day = st.slider(
        "Inspect Simulation Day (for live HMM Forward step)",
        min_value=1,
        max_value=15,
        value=6,
        key="hmm_inspect_day",
    )

    hmm_calc = SimpleHMM()
    prev_b = hmm_calc.initial_belief.copy()
    for d in range(inspect_day - 1):
        prev_b = hmm_calc.update_belief(sim_for_hmm["observations"][d])
    obs_today = sim_for_hmm["observations"][inspect_day - 1]
    true_today = sim_for_hmm["true_states"][inspect_day - 1]
    f_det = hmm_calc.forward_step_details(prev_b, obs_today)
    p_day = 0.4 * float(f_det["posterior"][1]) + 1.0 * float(f_det["posterior"][2])

    st.caption(
        f"Scenario **{PRESETS[st.session_state['scenario']]['label']}**, Day **{inspect_day}**: "
        f"Observed signal = **{SupplyChainDataGenerator.OBS_NAMES[obs_today]}** "
        f"(True hidden state = **{SimpleHMM.STATE_NAMES[true_today]}**)"
    )

    hmm_formula_html = (
        f"{sym('bt', 'b<sub>t</sub>(j)')} = "
        f"{sym('B', 'B[j, y<sub>t</sub>]')} · "
        f"(&Sigma;<sub>i</sub> {sym('bt1', 'b<sub>t-1</sub>(i)')} · {sym('A', 'A[i, j]')}) / {sym('Z', 'Z')}"
    )

    hmm_rows = []
    for j, sname in enumerate(SimpleHMM.STATE_NAMES):
        is_top = j == int(np.argmax(f_det["posterior"]))
        cls = "calc-row winner" if is_top else "calc-row"
        tag = "✅ " if is_top else ""
        pred_s = f"{f_det['predicted'][j]:.3f}"
        lik_s = f"{f_det['likelihood'][j]:.2f}"
        z_s = f"{f_det['Z']:.3f}"
        post_pct = f"{f_det['posterior'][j] * 100:.1f}%"
        expr = (
            f"pred={sym('A', pred_s)} × "
            f"B[{sname[:4]}]={sym('B', lik_s)} → "
            f"unnorm={f_det['unnormalized'][j]:.3f} / {sym('Z', z_s)}"
        )
        hmm_rows.append(
            f'<div class="{cls}">'
            f'<span class="row-name">{tag}State j = {sname} (prior {sym("bt1", f"{prev_b[j]:.2f}")})</span>'
            f'<span class="row-expr">{expr}</span>'
            f'<span class="row-total">= {sym("bt", post_pct)}</span>'
            f"</div>"
        )
    p_cong_s = f"{f_det['posterior'][1]:.3f}"
    p_cris_s = f"{f_det['posterior'][2]:.3f}"
    hmm_rows.append(
        f'<div class="calc-row">'
        f'<span class="row-name">Scalar Risk passed to A*</span>'
        f'<span class="row-expr">p = 0.4 · {sym("bt", p_cong_s)} + 1.0 · {sym("bt", p_cris_s)}</span>'
        f'<span class="row-total">= {sym("p_out", f"{p_day*100:.1f}%")}</span>'
        f"</div>"
    )

    hmm_vars = [
        {
            "key": "bt",
            "symbol": "b<sub>t</sub>(j)",
            "name": "Posterior State Belief",
            "value": f"[{', '.join(f'{v:.2f}' for v in f_det['posterior'])}]",
            "source": "HMM",
            "effect": "Updated probability distribution summing to 1.0; argmax feeds MDP state.",
        },
        {
            "key": "bt1",
            "symbol": "b<sub>t-1</sub>(i)",
            "name": "Yesterday's Belief",
            "value": f"[{', '.join(f'{v:.2f}' for v in prev_b)}]",
            "source": "HMM",
            "effect": "Provides temporal memory so one noisy reading does not panic the planner.",
        },
        {
            "key": "A",
            "symbol": "A[i, j]",
            "name": "Transition Matrix",
            "value": "3×3 Markov matrix",
            "source": "Constant",
            "effect": "Models persistence of Normal (85%), Congestion (70%), and Crisis (70%) regimes.",
        },
        {
            "key": "B",
            "symbol": "B[j, y<sub>t</sub>]",
            "name": "Emission Likelihood",
            "value": f"Obs '{SupplyChainDataGenerator.OBS_NAMES[obs_today]}'",
            "source": "Input",
            "effect": "Measures how likely today's delay/price signal is under each hidden state.",
        },
        {
            "key": "Z",
            "symbol": "Z",
            "name": "Normalizing Constant",
            "value": f"{f_det['Z']:.4f}",
            "source": "HMM",
            "effect": "Sum of unnormalized beliefs ensuring probabilities always total 100%.",
        },
        {
            "key": "p_out",
            "symbol": "p",
            "name": "Output Risk to A*",
            "value": f"{p_day:.3f} ({p_day*100:.1f}%)",
            "source": "HMM",
            "effect": "Weights Congestion (0.4) and Crisis (1.0) into a single score for A* routing.",
        },
    ]

    interactive_formula(
        title=f"HMM Forward Filtering Step (Day {inspect_day})",
        formula_html=hmm_formula_html,
        variables=hmm_vars,
        rows_html="".join(hmm_rows),
        height=520,
    )

# ================= TAB 3: MDP ORDER POLICY =================
with tab_mdp:
    st.subheader(f"MDP Optimal Order Policy Table (Tariff = USD {tariff_val:.0f}/box)")
    st.markdown(
        f"Your current selection (**Stock = {current_stock} boxes**, **State = {SimpleHMM.STATE_NAMES[risk_state]}**) "
        f"is highlighted below: recommended order is **{recommended_order} boxes**."
    )

    state_cols = ["Normal (100% arrival)", "Congestion (50% arrival)", "Crisis (25% arrival)"]
    policy_df = pd.DataFrame(
        {
            "Warehouse Stock": [f"{inv} boxes" for inv in range(mdp.max_inventory + 1)],
            state_cols[0]: [int(policy[inv, 0]) for inv in range(mdp.max_inventory + 1)],
            state_cols[1]: [int(policy[inv, 1]) for inv in range(mdp.max_inventory + 1)],
            state_cols[2]: [int(policy[inv, 2]) for inv in range(mdp.max_inventory + 1)],
        }
    )

    active_col = state_cols[risk_state]

    def _highlight_active(df_in: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=df_in.index, columns=df_in.columns)
        styles.loc[current_stock, active_col] = (
            "background-color: #dbeafe; color: #1e3a8a; font-weight: bold; border: 2px solid #2563eb;"
        )
        return styles

    st.dataframe(
        policy_df.style.apply(_highlight_active, axis=None),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Why the Solver Chose This Policy")
    st.markdown(
        f"- **Empty warehouse (`0` boxes):** Under Normal seas the solver orders **{policy[0, 0]}** boxes to cover daily demand (`2`/day). "
        f"Under Congestion (`50%` arrival), it scales up to **{policy[0, 1]}** boxes so enough units arrive. "
        f"Under Crisis (`25%` arrival, USD {tariff_val:.0f} tariff), it orders **{policy[0, 2]}** boxes."
    )
    first_zero_norm = next((i for i in range(mdp.max_inventory + 1) if policy[i, 0] == 0), mdp.max_inventory)
    st.markdown(
        f"- **High inventory (`>= {first_zero_norm}` boxes in Normal):** Ordering drops to **0 boxes** to avoid unnecessary USD 1/box/day holding fees."
    )

# ================= TAB 4: SIMULATION & RESULTS =================
with tab_sim:
    st.subheader("Closed-Loop Simulation: Integrated AI vs Static Baseline")
    sim_days = st.slider("Simulation Length (Days)", min_value=6, max_value=20, value=15, key="sim_days")

    active_scen = st.session_state["scenario"]
    df_ai = run_closed_loop(
        scenario=active_scen,
        days=sim_days,
        start_stock=current_stock,
        tariff=tariff_val,
        seed=42,
        strategy="ai",
    )
    df_base = run_closed_loop(
        scenario=active_scen,
        days=sim_days,
        start_stock=current_stock,
        tariff=tariff_val,
        seed=42,
        strategy="baseline",
    )

    hmm_acc = (df_ai["true state"] == df_ai["inferred state"]).mean() * 100.0
    ai_stockout_days = int((df_ai["stockout"] > 0).sum())
    base_stockout_days = int((df_base["stockout"] > 0).sum())
    ai_total_cost = float(df_ai["daily_cost"].sum())
    base_total_cost = float(df_base["daily_cost"].sum())
    cost_diff = base_total_cost - ai_total_cost

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HMM State Accuracy", f"{hmm_acc:.1f}%")
    m2.metric(
        "AI Stockout Days",
        f"{ai_stockout_days} / {sim_days}",
        delta=f"{ai_stockout_days - base_stockout_days:+d} vs Baseline",
        delta_color="inverse",
    )
    m3.metric("Baseline Stockout Days", f"{base_stockout_days} / {sim_days}")
    m4.metric(
        "AI Total Cost",
        f"USD {ai_total_cost:,.0f}",
        delta=f"Saves USD {cost_diff:,.0f}" if cost_diff >= 0 else f"+USD {-cost_diff:,.0f} vs Base",
        delta_color="normal" if cost_diff >= 0 else "inverse",
    )

    c_left, c_right = st.columns(2)
    with c_left:
        fig1, ax1 = plt.subplots(figsize=(6.4, 3.6))
        days_arr = df_ai["day"].values
        ax1.plot(days_arr, df_ai["P(N)"], label="P(Normal)", color="#16a34a", lw=2.2)
        ax1.plot(days_arr, df_ai["P(C)"], label="P(Congestion)", color="#f59e0b", lw=2.2)
        ax1.plot(days_arr, df_ai["P(X)"], label="P(Crisis)", color="#dc2626", lw=2.2)

        shade_colors = {"Normal": "#dcfce7", "Congestion": "#fef3c7", "Crisis": "#fee2e2"}
        for idx, ts in enumerate(df_ai["true state"]):
            d_val = df_ai["day"].iloc[idx]
            ax1.axvspan(d_val - 0.5, d_val + 0.5, color=shade_colors[ts], alpha=0.45, zorder=0)

        ax1.set_title("HMM Belief vs True Corridor Regime (Shaded)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Probability")
        ax1.set_ylim(-0.02, 1.05)
        ax1.set_xlim(0.5, sim_days + 0.5)
        ax1.legend(loc="best", fontsize=8.5)
        ax1.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    with c_right:
        fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
        ax2.plot(df_ai["day"], df_ai["stock"], marker="o", label="Integrated AI Advisory", color="#2563eb", lw=2.4)
        ax2.plot(df_base["day"], df_base["stock"], marker="x", linestyle="--", label="Static Baseline (Order 2, Suez)", color="#64748b", lw=2.0)
        ax2.axhline(2.0, color="#ef4444", linestyle=":", lw=1.3, label="Daily Demand (2 boxes)")
        ax2.set_title("Warehouse Inventory: AI vs Static Baseline")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Opening Stock (boxes)")
        ax2.set_ylim(-0.3, 8.5)
        ax2.legend(loc="best", fontsize=8.5)
        ax2.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("#### Day-by-Day Integrated AI Log")
    display_df = df_ai.copy()
    display_df["risk"] = (display_df["risk"] * 100).round(0).astype(int).astype(str) + "%"
    display_df["daily_cost"] = display_df["daily_cost"].round(1)
    st.dataframe(
        display_df[
            [
                "day",
                "observation",
                "true state",
                "inferred state",
                "risk",
                "stock",
                "order",
                "delivered",
                "stockout",
                "route_name",
                "mode",
                "daily_cost",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("📌 Modeling Assumptions, Limitations & Future Improvements"):
        st.markdown(
            """
            - **Same-day delivery abstraction:** The MDP evaluates replenishment with same-day partial arrival fractions (`100% / 50% / 25%`) rather than explicit multi-day pipeline lead times, keeping the state space compact (`9 × 3 = 27` states).
            - **Hand-calibrated HMM matrices:** Transition matrix **A** and emission matrix **B** are domain-defined constants; in production, Baum-Welch (EM) could learn them from historical AIS vessel tracking and freight indices.
            - **Deterministic demand:** Daily demand is fixed at `2` boxes/day; future versions can expand the MDP transition sum over a Poisson demand distribution.
            """
        )

# ================= TAB 5: HOW IT WORKS =================
with tab_how:
    st.subheader("End-to-End Closed-Loop AI Architecture")
    st.graphviz_chart(
        """
        digraph G {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, margin="0.18,0.10"];
            edge [fontname="Helvetica", fontsize=10, color="#475569"];

            Obs [label="Daily Signal\\n(Delay & Rate Report)", fillcolor="#f1f5f9", color="#94a3b8"];
            HMM [label="1. HMM Forward Filter\\n(hmm_module.py)", fillcolor="#f3e8ff", color="#9333ea"];
            Stock [label="Warehouse Stock\\n& Tariff Input", fillcolor="#dbeafe", color="#2563eb"];
            MDP [label="2. MDP Value Iteration\\n(mdp_planner.py)", fillcolor="#ffedd5", color="#ea580c"];
            AStar [label="3. Dynamic A* Search\\n(graph_search.py)", fillcolor="#ccfbf1", color="#0d9488"];
            Action [label="Decision:\\nOrder Qty + Optimal Route", fillcolor="#dcfce7", color="#16a34a"];

            Obs -> HMM [label="observation y_t"];
            HMM -> MDP [label="inferred state s"];
            HMM -> AStar [label="disruption risk p"];
            Stock -> MDP [label="inventory I, tariff"];
            MDP -> AStar [label="urgency score U"];
            MDP -> Action [label="order a*"];
            AStar -> Action [label="path & mode"];
        }
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info(
            "**1. Hidden Markov Model (HMM)**\n\n"
            "*Course Topic: Probabilistic Temporal Models*\n\n"
            "Maintains a normalized belief over hidden states *(Normal, Congestion, Crisis)* "
            "using Forward filtering so single noisy signals do not trigger false alarms."
        )
    with c2:
        st.warning(
            "**2. Markov Decision Process (MDP)**\n\n"
            "*Course Topic: Sequential Decision-Making*\n\n"
            "Solves Bellman's optimality equation via Value Iteration across `27` `(Inventory, State)` pairs, "
            "balancing holding fees, tariff surcharges, and USD 30 stockout penalties."
        )
    with c3:
        st.success(
            "**3. Dynamic A\\* Search**\n\n"
            "*Course Topic: Classical Search & Heuristics*\n\n"
            "Finds the optimal Jebel Ali → Hamburg route using an admissible great-circle heuristic "
            "and dynamic edge costs driven directly by HMM risk `p` and MDP urgency `U`."
        )
