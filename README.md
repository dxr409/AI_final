# Supply Chain Vulnerability & Tariff Disruption Advisory System
**AI Fundamentals University Course Final Project**

An end-to-end, integrated decision-support tool that combines three core algorithm families from artificial intelligence:
1. **Hidden Markov Model (HMM)** — *Probabilistic Reasoning / Temporal Models*: Infers latent risk states of international transit corridors from noisy operational signals (freight rates and dwell delays).
2. **Markov Decision Process (MDP)** — *Sequential Decision Making under Uncertainty*: Optimizes warehouse replenishment policies via Bellman Value Iteration under disruption-dependent fulfillment rates and tariff shocks.
3. **Dynamic A\* Search** — *Classical Search & Path Planning*: Plans cost-minimal multimodal freight paths across a 6-node global transit network, dynamically penalizing chokepoints by HMM disruption beliefs and transit latency by MDP stockout urgency.

---

## 📁 Repository Structure
```
supply_chain_advisory/
├── data_loader.py            # Simple scenario generator
├── hmm_module.py             # Pure Python/NumPy Forward algorithm (Risk Detector)
├── mdp_planner.py            # Bellman Value Iteration solver (Order Planner)
├── graph_search.py           # Dynamic multimodal A* search (Route Finder)
├── simulation_runner.py      # Clean terminal pipeline runner for presentation
├── app.py                    # Interactive Streamlit dashboard with 2D network visualizer
├── test_pipeline.py          # Automated verification test suite
├── STUDENT_DEFENSE_GUIDE.md  # 30-sec pitch, teacher Q&A, and oral defense guide
├── MATH_DOCUMENTATION.md     # Clear, student-friendly mathematical formulas
└── requirements.txt          # Python dependency manifest
```

---

## 🚀 Quickstart Guide

### 1. Run the Headless Pipeline Simulation
```bash
python3 simulation_runner.py
```
This executes the end-to-end pipeline, prints the MDP optimal policy matrix, outputs A* route choices before and after shocks, and traces day-by-day closed-loop operations.

### 2. Run Verification Tests
```bash
python3 test_pipeline.py
```
Validates mathematical admissibility of the A* heuristic, HMM belief vector normalization, and MDP Value Iteration convergence.

### 3. Launch Interactive Streamlit Dashboard
```bash
./.venv/bin/streamlit run app.py
```
Navigate to `http://localhost:8501` to view live time-series plots, interactive policy tables, and dynamic path visualizations.

### 4. Preparing for Your Presentation & Defense
Read [`STUDENT_DEFENSE_GUIDE.md`](file:///Users/derlyight/Documents/antgrav%20files/supply_chain_advisory/STUDENT_DEFENSE_GUIDE.md) for the 30-second elevator pitch and answers to the top 5 questions your teacher will ask.
