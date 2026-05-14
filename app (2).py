import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import time
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GA + ML & RL Knapsack",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    border-right: 1px solid rgba(102, 126, 234, 0.3);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #a0aec0;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    padding: 10px 24px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
}

/* Metric boxes */
.metric-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.2), rgba(118,75,162,0.2));
    border: 1px solid rgba(102,126,234,0.4);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-val {
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Gene chips */
.gene-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0; }
.gene-on  { background: #667eea; color: white; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }
.gene-off { background: rgba(255,255,255,0.08); color: #64748b; border-radius: 6px; padding: 4px 10px; font-size: 12px; font-family: 'JetBrains Mono', monospace; }

/* Knapsack item cards */
.item-card {
    display: inline-block; width: 90px; border-radius: 10px;
    padding: 10px; margin: 4px; text-align: center;
    font-size: 12px; font-family: 'JetBrains Mono', monospace;
}
.item-in  { background: linear-gradient(135deg,#22c55e,#16a34a); color:white; border: 2px solid #4ade80; }
.item-out { background: rgba(255,255,255,0.06); color:#64748b; border: 1px solid rgba(255,255,255,0.1); }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(102,126,234,0.4) !important;
}

/* Sliders and selects */
.stSlider > div > div { background: #667eea !important; }
label { color: #cbd5e1 !important; font-family: 'Syne', sans-serif !important; }

h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# SHARED DATA
# ══════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [f"F{i+1}" for i in range(12)]
N_FEATURES = len(FEATURE_NAMES)

@st.cache_data
def get_dataset():
    X, y = make_classification(n_samples=300, n_features=N_FEATURES,
                               n_informative=6, n_redundant=3, random_state=42)
    return X, y

X_data, y_data = get_dataset()

KNAPSACK_ITEMS = [
    {"name": "Laptop",   "weight": 3, "value": 40, "emoji": "💻"},
    {"name": "Camera",   "weight": 2, "value": 30, "emoji": "📷"},
    {"name": "Books",    "weight": 4, "value": 20, "emoji": "📚"},
    {"name": "Clothes",  "weight": 2, "value": 15, "emoji": "👕"},
    {"name": "Charger",  "weight": 1, "value": 10, "emoji": "🔌"},
    {"name": "Tablet",   "weight": 2, "value": 25, "emoji": "📱"},
    {"name": "Shoes",    "weight": 3, "value": 18, "emoji": "👟"},
]
CAPACITY = 8

# ══════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 20px 0 10px 0;">
  <h1 style="font-size:2.8rem; font-weight:800; background:linear-gradient(90deg,#667eea,#a78bfa,#f472b6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    🧬 GA + ML &nbsp;&nbsp;|&nbsp;&nbsp; 🎒 RL Knapsack
  </h1>
  <p style="color:#94a3b8; font-size:1rem; margin-top:8px;">
    Genetic Algorithm · Feature Selection · Q-Learning · Reinforcement Learning
  </p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["🧬  GA + ML (Feature Selection)", "🎒  RL Knapsack (Q-Learning)"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — GA + ML
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    with st.sidebar:
        st.markdown("### 🧬 GA Settings")
        classifier = st.selectbox("Classifier", ["SVM", "KNN", "Decision Tree"])
        generations = st.slider("Generations", 5, 50, 20)
        pop_size    = st.slider("Population Size", 10, 60, 20, step=2)
        mut_rate    = st.slider("Mutation Rate", 0.01, 0.3, 0.1, step=0.01)
        crossover   = st.slider("Crossover Rate", 0.5, 1.0, 0.8, step=0.05)
        st.divider()
        st.markdown("### 🎒 RL Settings")
        alpha   = st.slider("Learning Rate α", 0.01, 1.0, 0.3, step=0.01)
        gamma   = st.slider("Discount Factor γ", 0.1, 1.0, 0.9, step=0.05)
        eps_dec = st.slider("ε Decay", 0.80, 0.999, 0.97, step=0.001)
        mode    = st.radio("Agent Mode", ["🔍 Explore", "🎯 Exploit"])

    # ── GA functions ─────────────────────────────────────────────────
    def make_clf(name):
        if name == "SVM":          return SVC(kernel="rbf", random_state=42)
        if name == "KNN":          return KNeighborsClassifier(n_neighbors=5)
        return DecisionTreeClassifier(max_depth=5, random_state=42)

    def fitness(chrom):
        idxs = [i for i, g in enumerate(chrom) if g == 1]
        if not idxs: return 0.0
        clf = make_clf(classifier)
        return cross_val_score(clf, X_data[:, idxs], y_data, cv=3, scoring="accuracy").mean()

    def tournament(pop, fits, k=3):
        candidates = random.sample(range(len(pop)), k)
        return pop[max(candidates, key=lambda i: fits[i])].copy()

    def crossover_fn(p1, p2):
        pt = random.randint(1, N_FEATURES - 1)
        return np.concatenate([p1[:pt], p2[pt:]])

    def mutate(chrom, rate):
        for i in range(len(chrom)):
            if random.random() < rate:
                chrom[i] = 1 - chrom[i]
        return chrom

    def run_ga(gens, pop_sz, mut, cross):
        pop = np.random.randint(0, 2, (pop_sz, N_FEATURES))
        best_hist, avg_hist, best_chrom = [], [], None
        best_fit = -1
        for g in range(gens):
            fits = [fitness(c) for c in pop]
            b = max(fits)
            if b > best_fit:
                best_fit = b
                best_chrom = pop[np.argmax(fits)].copy()
            best_hist.append(b)
            avg_hist.append(np.mean(fits))
            new_pop = []
            for _ in range(pop_sz):
                p1 = tournament(pop, fits)
                p2 = tournament(pop, fits)
                child = crossover_fn(p1, p2) if random.random() < cross else p1.copy()
                child = mutate(child, mut)
                new_pop.append(child)
            pop = np.array(new_pop)
        return best_hist, avg_hist, best_chrom, best_fit

    # ── UI ────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### How It Works")
    st.markdown("""
    The **Genetic Algorithm** evolves a population of binary chromosomes — each gene represents
    whether a feature is selected. Each generation selects the fittest individuals via tournament,
    crosses them over, and mutates offspring. Fitness = cross-validated accuracy of your chosen classifier
    trained only on selected features.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("▶  Run Evolution", key="run_ga"):
        progress = st.progress(0, text="Initialising population…")
        with st.spinner(""):
            best_hist, avg_hist, best_chrom, best_fit = run_ga(
                generations, pop_size, mut_rate, crossover
            )
            progress.progress(100, text="Evolution complete ✓")

        # KPIs
        n_sel = int(best_chrom.sum())
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in [
            (c1, f"{best_fit:.3f}", "Best Accuracy"),
            (c2, f"{n_sel}/{N_FEATURES}", "Features Selected"),
            (c3, f"{generations}", "Generations"),
            (c4, f"{pop_size}", "Population"),
        ]:
            col.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div>'
                         f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.write("")

        # Chromosome display
        st.markdown("**Best Chromosome (Blue = Selected)**")
        genes_html = '<div class="gene-row">'
        for i, g in enumerate(best_chrom):
            cls = "gene-on" if g else "gene-off"
            genes_html += f'<span class="{cls}">{FEATURE_NAMES[i]}</span>'
        genes_html += '</div>'
        st.markdown(genes_html, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=best_hist, name="Best Fitness", mode="lines+markers",
                line=dict(color="#667eea", width=2.5), marker=dict(size=5)))
            fig.add_trace(go.Scatter(y=avg_hist, name="Avg Fitness", mode="lines",
                line=dict(color="#a78bfa", width=1.5, dash="dash")))
            fig.update_layout(
                title="Fitness Over Generations",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", xaxis_title="Generation", yaxis_title="Accuracy",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=40, b=30)
            )
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            sel_feats = [FEATURE_NAMES[i] for i, g in enumerate(best_chrom) if g]
            all_colors = ["#667eea" if g else "rgba(255,255,255,0.1)" for g in best_chrom]
            fig2 = go.Figure(go.Bar(
                x=FEATURE_NAMES, y=best_chrom.tolist(),
                marker_color=all_colors,
                text=["✓" if g else "✗" for g in best_chrom],
                textposition="outside"
            ))
            fig2.update_layout(
                title="Selected Features",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", yaxis=dict(showticklabels=False),
                margin=dict(t=40, b=30)
            )
            fig2.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig2, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — RL Knapsack
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    # Session state init
    if "q_table" not in st.session_state:
        st.session_state.q_table = np.zeros((CAPACITY + 1, len(KNAPSACK_ITEMS), 2))
    if "epsilon" not in st.session_state:
        st.session_state.epsilon = 1.0
    if "episode_rewards" not in st.session_state:
        st.session_state.episode_rewards = []
    if "last_bag" not in st.session_state:
        st.session_state.last_bag = [False] * len(KNAPSACK_ITEMS)
    if "total_episodes" not in st.session_state:
        st.session_state.total_episodes = 0

    Q = st.session_state.q_table
    exploit = "Exploit" in mode

    def run_episode(eps):
        cap = CAPACITY
        taken = [False] * len(KNAPSACK_ITEMS)
        total_reward = 0
        for idx, item in enumerate(KNAPSACK_ITEMS):
            state = min(cap, CAPACITY)
            if exploit or random.random() > eps:
                action = int(np.argmax(Q[state, idx]))
            else:
                action = random.randint(0, 1)
            if action == 1 and cap >= item["weight"]:
                reward = item["value"]
                cap -= item["weight"]
                taken[idx] = True
            else:
                reward = 0
                action = 0
            new_state = min(cap, CAPACITY)
            future = np.max(Q[new_state, min(idx+1, len(KNAPSACK_ITEMS)-1)])
            Q[state, idx, action] += alpha * (reward + gamma * future - Q[state, idx, action])
            total_reward += reward
        return total_reward, taken

    # ── How it works card ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### How It Works")
    st.markdown("""
    The agent uses **Q-Learning** to solve the Knapsack problem. The state is the remaining
    knapsack capacity. For each item, the agent chooses **Take** or **Skip**. The Q-table
    is updated using the **Bellman Equation** after each decision. Run more episodes to watch
    ε decay and the agent shift from random exploration to a learned policy.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    ep_col, reset_col = st.columns([3, 1])
    with ep_col:
        n_eps = st.slider("Episodes to run", 1, 50, 5)
    with reset_col:
        st.write("")
        if st.button("🔄 Reset Agent"):
            st.session_state.q_table = np.zeros((CAPACITY + 1, len(KNAPSACK_ITEMS), 2))
            st.session_state.epsilon = 1.0
            st.session_state.episode_rewards = []
            st.session_state.last_bag = [False] * len(KNAPSACK_ITEMS)
            st.session_state.total_episodes = 0
            st.rerun()

    if st.button("▶  Run Episodes", key="run_rl"):
        for _ in range(n_eps):
            reward, bag = run_episode(st.session_state.epsilon)
            st.session_state.episode_rewards.append(reward)
            st.session_state.last_bag = bag
            st.session_state.epsilon = max(0.05, st.session_state.epsilon * eps_dec)
            st.session_state.total_episodes += 1
        st.rerun()

    # ── KPIs ──────────────────────────────────────────────────────────
    bag = st.session_state.last_bag
    total_val  = sum(KNAPSACK_ITEMS[i]["value"]  for i, t in enumerate(bag) if t)
    total_wt   = sum(KNAPSACK_ITEMS[i]["weight"] for i, t in enumerate(bag) if t)
    avg_reward = np.mean(st.session_state.episode_rewards) if st.session_state.episode_rewards else 0

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl in [
        (m1, f"{st.session_state.total_episodes}", "Episodes Run"),
        (m2, f"{st.session_state.epsilon:.3f}", "ε (Epsilon)"),
        (m3, f"{total_val}", "Bag Value"),
        (m4, f"{total_wt}/{CAPACITY}", "Weight Used"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.write("")

    # Capacity bar
    cap_pct = total_wt / CAPACITY
    bar_color = "#22c55e" if cap_pct < 0.8 else "#f59e0b"
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.06);border-radius:8px;padding:14px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;margin-bottom:8px;font-size:13px;color:#94a3b8;">
        <span>Knapsack Capacity</span><span>{total_wt} / {CAPACITY} kg</span>
      </div>
      <div style="background:rgba(255,255,255,0.1);border-radius:6px;height:14px;">
        <div style="background:{bar_color};width:{min(cap_pct,1)*100:.1f}%;height:14px;border-radius:6px;
          transition:width 0.4s ease;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Items grid ────────────────────────────────────────────────────
    st.markdown("**📦 Knapsack Contents**")
    items_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;">'
    for i, item in enumerate(KNAPSACK_ITEMS):
        cls = "item-in" if bag[i] else "item-out"
        status = "✓ IN" if bag[i] else "✗ OUT"
        items_html += (f'<div class="{cls}" style="min-width:90px;border-radius:10px;'
                       f'padding:12px 10px;text-align:center;">'
                       f'<div style="font-size:1.8rem">{item["emoji"]}</div>'
                       f'<div style="font-weight:700;font-size:13px;margin:4px 0">{item["name"]}</div>'
                       f'<div style="font-size:11px">{item["weight"]}kg · {item["value"]}pt</div>'
                       f'<div style="font-size:11px;margin-top:4px;font-weight:700">{status}</div>'
                       f'</div>')
    items_html += '</div>'
    st.markdown(items_html, unsafe_allow_html=True)

    col_r, col_q = st.columns(2)

    with col_r:
        if st.session_state.episode_rewards:
            fig_r = go.Figure(go.Bar(
                y=st.session_state.episode_rewards,
                marker_color=[
                    f"rgba(102,126,234,{0.4 + 0.6*i/max(len(st.session_state.episode_rewards)-1,1)})"
                    for i in range(len(st.session_state.episode_rewards))
                ],
                hovertemplate="Episode %{x}: %{y} pts<extra></extra>"
            ))
            fig_r.update_layout(
                title="Reward History per Episode",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", xaxis_title="Episode", yaxis_title="Total Reward",
                margin=dict(t=40, b=30)
            )
            fig_r.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
            fig_r.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Run some episodes to see reward history.")

    with col_q:
        # Q-table heatmap at current capacity state
        state_view = max(0, CAPACITY - total_wt)
        q_take = Q[state_view, :, 1]
        q_skip = Q[state_view, :, 0]
        pref   = ["TAKE" if qt > qs else "SKIP" for qt, qs in zip(q_take, q_skip)]

        df_q = pd.DataFrame({
            "Item":   [f'{item["emoji"]} {item["name"]}' for item in KNAPSACK_ITEMS],
            "Q(Take)": q_take.round(2),
            "Q(Skip)": q_skip.round(2),
            "Prefer":  pref,
        })

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(name="Q(Take)", x=df_q["Item"], y=df_q["Q(Take)"],
            marker_color="#22c55e", opacity=0.85))
        fig_q.add_trace(go.Bar(name="Q(Skip)", x=df_q["Item"], y=df_q["Q(Skip)"],
            marker_color="#ef4444", opacity=0.85))
        fig_q.update_layout(
            title=f"Q-Table (state = {state_view} kg remaining)",
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", margin=dict(t=40, b=30),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        fig_q.update_xaxes(gridcolor="rgba(255,255,255,0.05)", tickangle=-30)
        fig_q.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig_q, use_container_width=True)

    # Q-table raw view
    with st.expander("🔍 View Full Q-Table"):
        st.dataframe(df_q.set_index("Item"), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border-color:rgba(102,126,234,0.2);margin-top:32px;">
<p style="text-align:center;color:#475569;font-size:12px;">
  Built with Streamlit · Plotly · scikit-learn &nbsp;|&nbsp; 🧬 Genetic Algorithm + ML &nbsp;·&nbsp; 🎒 Q-Learning RL
</p>
""", unsafe_allow_html=True)
