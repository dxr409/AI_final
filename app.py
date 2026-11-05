"""
app.py - Interactive Streamlit Demo Dashboard (Student Friendly)
================================================================
Simple, intuitive decision-support tool for classroom demonstration.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_loader import SupplyChainDataGenerator
from hmm_module import SimpleHMM
from mdp_planner import SimpleInventoryMDP
from graph_search import MultimodalNetwork

st.set_page_config(
    page_title="AI Supply Chain Advisor",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 AI Supply Chain & Tariff Advisory Tool")
st.markdown(r"""
**How 3 AI algorithms solve a global logistics crisis together:**
1. **HMM (Risk Guesser)**: Infers if shipping routes are Normal, Congested, or in Crisis.
2. **MDP (Order Planner)**: Decides how many boxes to order for the warehouse.
3. **A* (Route Finder)**: Picks the best way (Ship, Train, or Airplane) to deliver the goods.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎮 Simulation Controls")

scenario_choice = st.sidebar.selectbox(
    "Choose a World Scenario",
    options=["tariff_crisis", "canal_blockage", "baseline_normal"],
    format_func=lambda x: {
        "tariff_crisis": "Tariff Crisis (Customs & Surcharge Shock)",
        "canal_blockage": "Suez Canal Blockage (Heavy Delays)",
        "baseline_normal": "Peaceful Baseline (No Disruption)",
    }[x],
)

starting_boxes = st.sidebar.slider("Starting Warehouse Stock (Boxes)", min_value=0, max_value=8, value=4)
sim_days = st.sidebar.slider("Simulation Length (Days)", min_value=6, max_value=20, value=12)

# Initialize modules
hmm = SimpleHMM()
mdp = SimpleInventoryMDP(max_inventory=8)
network = MultimodalNetwork()
data_gen = SupplyChainDataGenerator(seed=42)

# Solve MDP policy
policy, V = mdp.solve_value_iteration()

# Run simulation
data = data_gen.generate_scenario(scenario_choice, T=sim_days)
hmm.reset()

sim_history = []
current_inv = starting_boxes

for day in range(sim_days):
    obs = data["observations"][day]
    obs_name = data_gen.OBS_NAMES[obs]

    # Algorithm 1: HMM
    belief = hmm.update_belief(obs)
    risk_prob = hmm.get_disruption_probability()
    inferred_state = hmm.get_most_likely_state()

    # Algorithm 2: MDP
    order_action = policy[current_inv, inferred_state]
    urgency = mdp.get_urgency_score(current_inv)

    # Algorithm 3: A*
    route = network.a_star_search(
        start_node="SHANGHAI",
        goal_node="HAMBURG",
        disruption_prob=risk_prob,
        urgency_score=urgency,
    )

    arrived = int(round(order_action * mdp.delivery_success[inferred_state]))
    next_inv = max(0, min(mdp.max_inventory, current_inv + arrived - mdp.daily_demand))

    sim_history.append({
        "Day": day + 1,
        "What We Saw": obs_name,
        "Inferred Situation": SimpleHMM.STATE_NAMES[inferred_state],
        "Disruption Risk": f"{risk_prob*100:.0f}%",
        "Stock in Warehouse": current_inv,
        "Boxes Ordered": order_action,
        "Delivery Urgency": f"{urgency*100:.0f}%",
        "Chosen Delivery Route": " ➔ ".join(route["path"]),
        "Transport Mode": route["modes"][0],
        "Shipping Cost": f"${route['total_base_cost']:.0f}",
        "Transit Days": f"{route['total_transit_days']:.0f} days",
    })

    current_inv = next_inv

df_sim = pd.DataFrame(sim_history)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Warehouse Stock", f"{df_sim['Stock in Warehouse'].iloc[-1]} boxes")
c2.metric("Latest Disruption Risk", df_sim["Disruption Risk"].iloc[-1])
c3.metric("Latest Chosen Mode", df_sim["Transport Mode"].iloc[-1])
c4.metric("Estimated Travel Time", df_sim["Transit Days"].iloc[-1])

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "🗺️ Interactive A* Map & Rerouting",
    "📦 Warehouse Order Decision Table (MDP)",
    "📊 Daily Simulation Log & HMM Belief",
])

# ================= TAB 1: INTERACTIVE A* MAP =================
with tab1:
    st.subheader("Interactive Multimodal Delivery Map")
    st.markdown(r"""
    Use the sliders below to see **A* immediately reroute** the shipment from Shanghai to Hamburg:
    """)

    col_test1, col_test2 = st.columns(2)
    with col_test1:
        test_risk = st.slider("Corridor Disruption Risk (%)", 0, 100, int(float(df_sim["Disruption Risk"].iloc[-1].replace("%", "")))) / 100.0
    with col_test2:
        test_urgency = st.slider("Warehouse Stockout Urgency (%)", 0, 100, 20) / 100.0

    test_route = network.a_star_search(
        start_node="SHANGHAI",
        goal_node="HAMBURG",
        disruption_prob=test_risk,
        urgency_score=test_urgency,
    )

    st.success(f"**A* Selected Route:** {' ➔ '.join(test_route['path'])} ({test_route['modes'][0]}) | **Cost:** ${test_route['total_dynamic_cost']:.1f} | **Time:** {test_route['total_transit_days']:.0f} days")

    # Plot simple map
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = network.node_positions
    active_path = set(zip(test_route["path"][:-1], test_route["path"][1:]))

    # Mode colors
    color_map = {"Sea": "#2980b9", "Rail": "#e67e22", "Air": "#27ae60"}

    # Draw routes
    for u, v, mode, base, days, risk in network.edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        is_active = (u, v) in active_path

        c = "#c0392b" if is_active else color_map[mode]
        lw = 4.0 if is_active else 1.5
        alpha = 1.0 if is_active else 0.4
        style = "-" if is_active else "--"

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=c, lw=lw, alpha=alpha, linestyle=style, shrinkA=15, shrinkB=15)
        )

    # Draw nodes
    display_names = {
        "SHANGHAI": "Shanghai",
        "SUEZ_CANAL": "Suez Canal",
        "CAPE_ROUTE": "Cape Route",
        "TRANS_ASIA_RAIL": "Trans-Asia Rail",
        "AIR_CARGO": "Air Cargo",
        "HAMBURG": "Hamburg",
    }
    for name, (x, y) in positions.items():
        is_in = name in test_route["path"]
        dot_color = "#c0392b" if is_in else "#34495e"
        ax.scatter(x, y, s=500 if is_in else 300, color=dot_color, zorder=5, edgecolor="black")
        clean_name = display_names.get(name, name.replace("_", " ").title())
        ax.text(x, y + 0.4, clean_name, ha="center", fontsize=9, fontweight="bold")

    legend_items = [
        patches.Patch(color=color_map["Sea"], label="Sea Route"),
        patches.Patch(color=color_map["Rail"], label="Rail Route"),
        patches.Patch(color=color_map["Air"], label="Air Cargo"),
        patches.Patch(color="#c0392b", label="A* Chosen Best Route"),
    ]
    ax.legend(handles=legend_items, loc="lower right")

    ax.set_xlim(-1, 12)
    ax.set_ylim(-3, 6.5)
    ax.axis("off")
    st.pyplot(fig)

    st.markdown("""
    **Route Options & Details:**
    - **Suez Canal (Sea)**: Cheapest ($100), but takes 20 days and passes through the vulnerable Suez Canal.
    - **Cape Route (Sea)**: Safe detour around Africa ($160), but takes 30 days.
    - **Trans-Asia Rail**: Train across Asia ($200), fast (10 days), bypasses sea canals completely.
    - **Air Cargo**: Airplane ($400), ultra-fast (2 days).
    """)

    with st.expander("📐 Calculations", expanded=False):
        st.markdown(r"""
        **Dynamic Edge Cost Formula:**
        $$\text{Total Cost} = \text{Base Cost} + \underbrace{(\text{Disruption Penalty from HMM})}_{\text{Base} \times \text{Risk} \times \text{Chokepoint Factor} \times 1.5} + \underbrace{(\text{Urgency Penalty from MDP})}_{\text{Urgency} \times 12 \times \text{Transit Days}}$$
        """)
        st.markdown(f"**Current Slider Inputs:** Disruption Risk $R = \\mathbf{{{test_risk:.2f}}}$ ({int(test_risk * 100)}%), Stockout Urgency $U = \\mathbf{{{test_urgency:.2f}}}$ ({int(test_urgency * 100)}%)")

        route_candidates = [
            {
                "name": "Suez Canal (Sea)",
                "base": 100.0,
                "days": 20.0,
                "disr_coef": 270.0,
                "urg_coef": 240.0,
            },
            {
                "name": "Cape Route (Sea)",
                "base": 160.0,
                "days": 30.0,
                "disr_coef": 0.0,
                "urg_coef": 360.0,
            },
            {
                "name": "Trans-Asia Rail",
                "base": 200.0,
                "days": 10.0,
                "disr_coef": 0.0,
                "urg_coef": 120.0,
            },
            {
                "name": "Air Cargo",
                "base": 400.0,
                "days": 2.0,
                "disr_coef": 0.0,
                "urg_coef": 24.0,
            },
        ]

        # Calculate all totals to identify the minimum
        totals = [rc["base"] + (rc["disr_coef"] * test_risk) + (rc["urg_coef"] * test_urgency) for rc in route_candidates]
        min_total = min(totals)

        calc_summary = []
        for rc in route_candidates:
            disr = rc["disr_coef"] * test_risk
            urg = rc["urg_coef"] * test_urgency
            total = rc["base"] + disr + urg
            is_best = (abs(total - min_total) < 1e-4)
            badge = " 🏆 **(Selected by A* - Lowest Cost)**" if is_best else ""

            st.markdown(f"##### {rc['name']}{badge}")
            st.latex(
                rf"\text{{Cost}} = \underbrace{{{rc['base']:.0f}}}_{{\text{{Base}}}} "
                rf"+ \underbrace{{({rc['disr_coef']:.0f} \times {test_risk:.2f})}}_{{{disr:.1f}\text{{ (Disruption)}}}} "
                rf"+ \underbrace{{({rc['urg_coef']:.0f} \times {test_urgency:.2f})}}_{{{urg:.1f}\text{{ (Urgency)}}}} "
                rf"= \mathbf{{{total:.1f}}}"
            )
            calc_summary.append({
                "Route": rc["name"],
                "Base Cost": f"${rc['base']:.0f}",
                "Disruption Penalty": f"+${disr:.1f}",
                "Urgency Penalty": f"+${urg:.1f}",
                "Total Dynamic Cost": f"${total:.1f}",
                "Status": "✅ Chosen by A*" if is_best else "—",
            })

        st.markdown("---")
        st.markdown("**Side-by-Side Cost Comparison:**")
        st.dataframe(pd.DataFrame(calc_summary), hide_index=True, use_container_width=True)

# ================= TAB 2: MDP POLICY TABLE =================
with tab2:
    st.subheader("MDP Optimal Inventory Reorder Table")
    st.markdown("""
    The table below shows what action the warehouse manager should take for every possible situation,
    calculated using **Bellman Value Iteration**:
    """)

    policy_display = []
    for inv in range(mdp.max_inventory + 1):
        policy_display.append({
            "Boxes in Warehouse": inv,
            "Normal Situation": f"Order {policy[inv, 0]} boxes",
            "Congestion Situation": f"Order {policy[inv, 1]} boxes",
            "Crisis Situation": f"Order {policy[inv, 2]} boxes",
        })
    st.table(pd.DataFrame(policy_display))

    st.markdown("""
    **Why does this make sense?**
    - When inventory is low (0 or 1 box), the algorithm orders boxes to prevent running out of stock.
    - When in **Congestion**, it orders **more** (e.g., 4 boxes) because 50% of shipments get delayed on the road.
    - When in **Crisis**, it orders **0 boxes** because tariffs are high and orders will just get stuck at the border!
    """)

# ================= TAB 3: DAILY SIMULATION LOG =================
with tab3:
    st.subheader("Day-by-Day Closed-Loop Simulation")
    st.dataframe(df_sim, use_container_width=True)

    st.markdown(r"""
    **Notice how the algorithms talk to each other:**
    1. The **Observation** tells the **HMM** what is happening.
    2. The **HMM** computes the **Disruption Risk**.
    3. The **MDP** checks current inventory and picks how many boxes to **Order**.
    4. The **A*** router picks the **Transport Mode** (Sea, Rail, Air) based on the risk and urgency!
    """)
