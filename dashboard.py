"""
F1 Fantasy Dashboard — Australia GP 2026
Run with:  streamlit run dashboard.py
"""

import sys
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

# ── path setup ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "model"
sys.path.insert(0, str(MODEL_DIR))

from backtest    import (
    run_backtest,
    CURRENT_DRIVER_COSTS,
    CURRENT_CONSTRUCTOR_COSTS,
)
from pipeline    import (DEFAULT_TRAIN_SEASONS, build_weekend_summary_df,
                         run_optimisation_pipeline, run_predictive_backtest_pipeline,
                         run_predictive_pipeline)
from value_model import CIRCUIT_ALIAS

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_CIRCUIT   = "Albert Park Grand Prix Circuit"
CURRENT_SEASON   = 2026
BUDGET           = 100.0
AVAILABLE_SEASONS = [2022, 2023, 2024, 2025, 2026]
AVAILABLE_CIRCUITS = sorted({circuit for values in CIRCUIT_ALIAS.values() for circuit in values})
PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(128, 128, 128, 0.25)"
PREDICTIVE_MODE = "Predictive next-race points"
HEURISTIC_MODE = "Heuristic weighted signals"

DRIVER_TEAM_2026 = {
    "Max Verstappen"    : "Red Bull Racing",
    "Isack Hadjar"      : "Red Bull Racing",
    "George Russell"    : "Mercedes",
    "Kimi Antonelli"    : "Mercedes",
    "Lando Norris"      : "McLaren",
    "Oscar Piastri"     : "McLaren",
    "Charles Leclerc"   : "Ferrari",
    "Lewis Hamilton"    : "Ferrari",
    "Pierre Gasly"      : "Alpine",
    "Franco Colapinto"  : "Alpine",
    "Carlos Sainz"      : "Williams",
    "Alexander Albon"   : "Williams",
    "Fernando Alonso"   : "Aston Martin",
    "Lance Stroll"      : "Aston Martin",
    "Oliver Bearman"    : "Haas F1 Team",
    "Esteban Ocon"      : "Haas F1 Team",
    "Nico Hulkenberg"   : "Audi",
    "Gabriel Bortoleto" : "Audi",
    "Liam Lawson"       : "Racing Bulls",
    "Arvid Lindblad"    : "Racing Bulls",
    "Sergio Perez"      : "Cadillac",
    "Valtteri Bottas"   : "Cadillac",
}

TEAM_COLORS = {
    "Red Bull Racing" : "#3671C6",
    "Mercedes"        : "#27F4D2",
    "McLaren"         : "#FF8000",
    "Ferrari"         : "#E8002D",
    "Alpine"          : "#FF87BC",
    "Williams"        : "#64C4FF",
    "Aston Martin"    : "#229971",
    "Haas F1 Team"    : "#B6BABD",
    "Audi"            : "#C92D4B",
    "Racing Bulls"    : "#6692FF",
    "Cadillac"        : "#333333",
}


def expand_team_exclusions(selected_teams, driver_team_map):
    excluded_drivers = sorted(
        driver for driver, team in driver_team_map.items() if team in selected_teams
    )
    excluded_constructors = sorted(team for team in selected_teams if team in CURRENT_CONSTRUCTOR_COSTS)
    return excluded_drivers, excluded_constructors


def build_effective_exclusions(
    manual_excluded,
    news_excluded_drivers,
    news_excluded_constructors,
    news_excluded_teams,
    driver_team_map,
):
    team_drivers, team_constructors = expand_team_exclusions(news_excluded_teams, driver_team_map)
    effective = sorted(
        set(manual_excluded)
        | set(news_excluded_drivers)
        | set(news_excluded_constructors)
        | set(team_drivers)
        | set(team_constructors)
    )
    return effective, team_drivers, team_constructors


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Fantasy | Australia GP 2026",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 10px;
        padding: 10px 12px;
    }
    h1, h2, h3 { font-family: 'Formula1', sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── sidebar — model controls ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.markdown("---")
    st.subheader("Race Setup")
    model_mode = st.radio("Model type", [PREDICTIVE_MODE, HEURISTIC_MODE], index=0)
    current_season = st.selectbox("Current season", AVAILABLE_SEASONS,
                                  index=AVAILABLE_SEASONS.index(CURRENT_SEASON))
    target_circuit = st.selectbox("Target circuit", AVAILABLE_CIRCUITS,
                                  index=AVAILABLE_CIRCUITS.index(TARGET_CIRCUIT))
    train_seasons = st.multiselect("Training seasons", AVAILABLE_SEASONS,
                                   default=DEFAULT_TRAIN_SEASONS)
    include_current_season = st.toggle("Include current season races in training",
                                       value=True,
                                       help="Useful once the season is underway so season-to-date form and PPM can activate.")

    st.markdown("---")
    if model_mode == PREDICTIVE_MODE:
        st.subheader("Predictive Model Settings")
        ridge_alpha = st.slider("Regularisation strength", 0.1, 20.0, 5.0, 0.1,
                                help="Higher values shrink coefficients and reduce overfitting.")
        min_history_weekends = st.slider("Minimum history before training", 3, 12, 5, 1,
                                         help="Backtests and training examples only start after this many prior weekends.")
        current_season_boost = st.slider(
            "Current-season training boost",
            1.0, 6.0, 3.0, 0.1,
            help="Once this season has race data, those examples get extra influence in training."
        )
        season_decay = st.slider(
            "Older-season decay",
            0.3, 1.0, 0.8, 0.05,
            help="Lower values downweight older seasons more aggressively to reflect rule changes."
        )
        w_recent, w_track, w_ppm, w_quali = 0.35, 0.30, 0.15, 0.20
    else:
        st.subheader("Signal Weights")
        w_recent = st.slider("📈 Recent Form",    0.0, 1.0, 0.35, 0.05)
        w_track  = st.slider("🏁 Track History",  0.0, 1.0, 0.30, 0.05)
        w_ppm    = st.slider("💰 Points/Million", 0.0, 1.0, 0.15, 0.05)
        w_quali  = st.slider("⏱️ Quali Form",     0.0, 1.0, 0.20, 0.05)
        total_w  = w_recent + w_track + w_ppm + w_quali
        st.caption(f"Weight total: **{total_w:.2f}** {'✅' if abs(total_w - 1.0) < 0.01 else '⚠️ (auto-normalised)'}")
        ridge_alpha = 5.0
        min_history_weekends = 5
        current_season_boost = 3.0
        season_decay = 0.8

    st.markdown("---")
    st.subheader("Expert Inputs")
    use_bonus = st.toggle("Apply manual FP/testing adjustments", value=True)

    st.markdown("---")
    st.subheader("Team Constraints")
    locked = st.multiselect("🔒 Lock drivers",
                             options=list(CURRENT_DRIVER_COSTS.keys()))
    excluded = st.multiselect("🚫 Exclude drivers/constructors",
                               options=list(CURRENT_DRIVER_COSTS.keys()) +
                                       list(CURRENT_CONSTRUCTOR_COSTS.keys()),
                               help="Directly remove individual drivers or constructors from the optimizer pool.")
    news_excluded_teams = st.multiselect(
        "🚫 Exclude teams",
        options=sorted(CURRENT_CONSTRUCTOR_COSTS.keys()),
        help="Removes the constructor and all drivers mapped to that team from consideration.",
    )
    st.caption("Team exclusions are applied before optimization, just like driver exclusions.")
    st.markdown("**Optional news-driven filters**")
    news_excluded_drivers = st.multiselect(
        "Exclude extra drivers from news",
        options=list(CURRENT_DRIVER_COSTS.keys()),
        default=["Fernando Alonso", "Lance Stroll"],
        help="Useful for team issues, expected poor pace, penalties, injuries, or reliability concerns.",
    )
    news_excluded_constructors = st.multiselect(
        "Exclude extra constructors from news",
        options=list(CURRENT_CONSTRUCTOR_COSTS.keys()),
        help="Use this when the whole constructor should be removed from the pool.",
    )
    news_reason = st.text_area(
        "News note",
        value="Aston Martin look weak on current pace, so exclude Alonso and Stroll.",
        height=70,
        help="Optional context for why these exclusions are active.",
    )
    budget_cap = st.number_input("💵 Budget ($M)", value=100.0, min_value=50.0,
                                  max_value=200.0, step=1.0)

    st.markdown("---")
    st.subheader("Backtest")
    bt_seasons = st.multiselect("Seasons", AVAILABLE_SEASONS,
                                 default=DEFAULT_TRAIN_SEASONS)
    bt_circuit = st.selectbox("Circuit", ["all", "australia", "singapore",
                                           "las vegas", "qatar", "abu dhabi",
                                           "brazil", "mexico", "usa"])
    run_bt = st.button("▶ Run Backtest", use_container_width=True)


def apply_transparent_plot_layout(fig, *, height=None, margin=None,
                                  xaxis_title=None, yaxis_title=None,
                                  showlegend=None):
    layout = {
        "paper_bgcolor": PLOT_BG,
        "plot_bgcolor": PLOT_BG,
        "xaxis": dict(gridcolor=GRID_COLOR),
        "yaxis": dict(gridcolor=GRID_COLOR),
    }
    if height is not None:
        layout["height"] = height
    if margin is not None:
        layout["margin"] = margin
    if xaxis_title is not None:
        layout["xaxis_title"] = xaxis_title
    if yaxis_title is not None:
        layout["yaxis_title"] = yaxis_title
    if showlegend is not None:
        layout["showlegend"] = showlegend
    fig.update_layout(**layout)
    return fig


effective_excluded, team_excluded_drivers, team_excluded_constructors = build_effective_exclusions(
    manual_excluded=excluded,
    news_excluded_drivers=news_excluded_drivers,
    news_excluded_constructors=news_excluded_constructors,
    news_excluded_teams=news_excluded_teams,
    driver_team_map=DRIVER_TEAM_2026,
)
effective_locked = [driver for driver in locked if driver not in effective_excluded]


# ── data loading & model (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running model...", ttl=60)
def run_model(model_mode, train_seasons_tuple, current_season, target_circuit,
              w_r, w_t, w_p, w_q, use_bonus, include_current,
              locked_d, excl, bdgt, ridge_alpha, min_history_weekends,
              current_season_boost, season_decay):
    if model_mode == PREDICTIVE_MODE:
        output = run_predictive_pipeline(
            train_seasons=list(train_seasons_tuple),
            current_season=current_season,
            target_circuit=target_circuit,
            driver_team_map=DRIVER_TEAM_2026,
            budget_m=bdgt,
            use_testing_bonus=use_bonus,
            include_current_season=include_current,
            locked_drivers=list(locked_d),
            excluded=list(excl),
            alpha=ridge_alpha,
            min_history_weekends=min_history_weekends,
            current_season_boost=current_season_boost,
            season_decay=season_decay,
        )
        output["model_mode"] = "predictive"
        return output

    weights = {
        "recent_form": w_r,
        "track_history": w_t,
        "season_ppm": w_p,
        "quali_form": w_q,
    }
    output = run_optimisation_pipeline(
        train_seasons=list(train_seasons_tuple),
        current_season=current_season,
        target_circuit=target_circuit,
        weights=weights,
        driver_team_map=DRIVER_TEAM_2026,
        budget_m=bdgt,
        use_testing_bonus=use_bonus,
        include_current_season=include_current,
        locked_drivers=list(locked_d),
        excluded=list(excl),
    )
    output["model_mode"] = "heuristic"
    return output


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(model_mode, seasons_tuple, current_season, circuit, use_bonus,
                    budget_m, w_r, w_t, w_p, w_q, ridge_alpha, min_history_weekends,
                    current_season_boost, season_decay):
    if model_mode == PREDICTIVE_MODE:
        return run_predictive_backtest_pipeline(
            seasons=list(seasons_tuple),
            current_season=current_season,
            target_circuit=circuit,
            include_current_season=False,
            budget_m=budget_m,
            alpha=ridge_alpha,
            min_history_weekends=min_history_weekends,
            current_season_boost=current_season_boost,
            season_decay=season_decay,
        )

    bt_df = run_backtest(
        seasons=list(seasons_tuple),
        target_circuit=circuit,
        budget_m=budget_m,
        weights={
            "recent_form": w_r,
            "track_history": w_t,
            "season_ppm": w_p,
            "quali_form": w_q,
        },
        use_testing_bonus=use_bonus,
    )
    if not bt_df.empty:
        bt_df = bt_df.copy()
        bt_df["efficiency_pct"] = bt_df["efficiency"] * 100
    return bt_df, pd.DataFrame()


# ── run model ─────────────────────────────────────────────────────────────────
model_output = run_model(
    model_mode,
    tuple(sorted(train_seasons)),
    current_season,
    target_circuit,
    w_recent, w_track, w_ppm, w_quali,
    use_bonus, include_current_season,
    tuple(effective_locked), tuple(effective_excluded), budget_cap,
    ridge_alpha, min_history_weekends, current_season_boost, season_decay
)
d_scores = model_output["driver_scores_df"]
c_scores = model_output["constructor_scores_df"]
opt_result = model_output["result"]
history_df = model_output["history_df"]
training_weekends = model_output["weekends"]
source_df = model_output["source_df"]
model_weights = model_output.get("weights")
trained_model = model_output.get("model")
training_examples_df = model_output.get("training_examples_df", pd.DataFrame())
feature_columns = model_output.get("feature_columns", [])
diagnostics = model_output["diagnostics"]
weekend_summary_df = build_weekend_summary_df(training_weekends)
is_predictive = model_output["model_mode"] == "predictive"
primary_score_col = "predicted_points" if is_predictive else "norm_score"
primary_score_label = "Predicted Pts" if is_predictive else "Model Score"

picked_drivers = list(opt_result["drivers"]["driver"])
picked_constructors = list(opt_result["constructors"]["constructor"])

# ── header ────────────────────────────────────────────────────────────────────
st.title("🏎️  F1 Fantasy Model")
st.caption(
    f"{model_mode} · {target_circuit} · {current_season} season context · "
    f"training seasons: {', '.join(map(str, sorted(set(train_seasons))))}"
    + (" + current season" if include_current_season else "")
)

# ── top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Budget Used",  f"${opt_result['total_cost']:.1f}M",
           f"${budget_cap - opt_result['total_cost']:.1f}M remaining")
k2.metric(primary_score_label,  f"{opt_result['total_score']:.3f}")
k3.metric("Drivers",      f"{len(picked_drivers)} / 5")
k4.metric("Constructors", f"{len(picked_constructors)} / 2")
k5.metric("Training Races", f"{len(training_weekends)}")

st.markdown("---")

if not source_df.empty:
    source_bits = [
        f"{int(row['season'])}: {row['source']} ({int(row['weekends_loaded'])} races)"
        for _, row in source_df.iterrows()
    ]
    st.caption("Data sources loaded: " + " | ".join(source_bits))

if effective_excluded:
    st.info(
        "Excluded from optimizer: "
        + ", ".join(effective_excluded)
        + (f" | News note: {news_reason}" if news_reason.strip() else "")
    )

if len(effective_locked) < len(locked):
    removed_locks = sorted(set(locked) - set(effective_locked))
    st.warning(
        "These locked drivers were removed because they are also excluded: "
        + ", ".join(removed_locks)
    )

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Optimal Team", "📊 Driver Rankings", "🔁 Backtest",
    "🔬 Signal Explorer", "🗂 Data Tables", "📘 How It Works"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMAL TEAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Selected Drivers")
        if is_predictive:
            d_display = opt_result["drivers"][[
                "driver", "cost_m", "predicted_points", "recent_form",
                "track_history", "quali_form", "testing_bonus"
            ]].copy()
            d_display.columns = [
                "Driver", "Cost ($M)", "Predicted Pts", "Recent Form",
                "Track History", "Quali Form", "Expert Adj"
            ]
        else:
            d_display = opt_result["drivers"][["driver", "cost_m", "norm_score",
                                                "recent_form", "track_history",
                                                "testing_bonus"]].copy()
            d_display.columns = ["Driver", "Cost ($M)", "Score", "Recent Form",
                                   "Track History", "FP Bonus"]
        d_display["Team"] = d_display["Driver"].map(DRIVER_TEAM_2026)
        d_display["Cost ($M)"] = d_display["Cost ($M)"].map(lambda x: f"${x:.1f}M")
        score_col_name = "Predicted Pts" if is_predictive else "Score"
        d_display[score_col_name] = d_display[score_col_name].map(lambda x: f"{x:.3f}")
        d_display["Recent Form"] = d_display["Recent Form"].map(lambda x: f"{x:.1f}")
        d_display["Track History"] = d_display["Track History"].map(lambda x: f"{x:.1f}")
        if is_predictive:
            d_display["Quali Form"] = d_display["Quali Form"].map(lambda x: f"{x:.1f}")
            d_display["Expert Adj"] = d_display["Expert Adj"].map(lambda x: f"+{x:.1f}" if x > 0 else "—")
            display_cols = ["Driver", "Team", "Cost ($M)", "Predicted Pts",
                            "Recent Form", "Track History", "Quali Form", "Expert Adj"]
        else:
            d_display["FP Bonus"] = d_display["FP Bonus"].map(lambda x: f"+{x:.1f}" if x > 0 else "—")
            display_cols = ["Driver", "Team", "Cost ($M)", "Score",
                            "Recent Form", "Track History", "FP Bonus"]
        st.dataframe(d_display[display_cols], use_container_width=True, hide_index=True)

        st.subheader("Selected Constructors")
        constructor_score_col = "predicted_points" if is_predictive else "norm_score"
        c_display = opt_result["constructors"][["constructor", "cost_m", constructor_score_col]].copy()
        c_display.columns = ["Constructor", "Cost ($M)", primary_score_label]
        c_display["Cost ($M)"] = c_display["Cost ($M)"].map(lambda x: f"${x:.1f}M")
        c_display[primary_score_label] = c_display[primary_score_label].map(lambda x: f"{x:.3f}")
        st.dataframe(c_display, use_container_width=True, hide_index=True)

    with col_right:
        # Budget breakdown donut
        st.subheader("Budget Allocation")
        all_picked = (
            [{"name": r["driver"], "cost": r["cost_m"], "type": "Driver",
              "team": DRIVER_TEAM_2026.get(r["driver"], "Unknown")}
             for _, r in opt_result["drivers"].iterrows()] +
            [{"name": r["constructor"], "cost": r["cost_m"], "type": "Constructor",
              "team": r["constructor"]}
             for _, r in opt_result["constructors"].iterrows()]
        )
        remaining = budget_cap - opt_result["total_cost"]
        if remaining > 0:
            all_picked.append({
                "name": "Remaining Budget",
                "cost": remaining,
                "type": "Remaining",
                "team": "Remaining Budget",
            })

        budget_df = pd.DataFrame(all_picked)
        budget_df["color"] = budget_df["team"].map(TEAM_COLORS).fillna("#B0B7C3")

        fig_budget = go.Figure(go.Pie(
            labels=budget_df["name"],
            values=budget_df["cost"],
            hole=0.55,
            marker_colors=budget_df["color"],
            textinfo="label+value",
            texttemplate="%{label}<br>$%{value:.1f}M",
            hovertemplate="%{label}: $%{value:.1f}M (%{percent})<extra></extra>",
        ))
        fig_budget.update_traces(
            sort=False,
            textfont_size=11,
        )
        fig_budget.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=340,
            annotations=[dict(text=f"${opt_result['total_cost']:.1f}M<br>used",
                              x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        apply_transparent_plot_layout(fig_budget, showlegend=False)
        st.plotly_chart(fig_budget, use_container_width=True)

        # Score contribution bar
        st.subheader("Score Contributions")
        contrib_data = (
            [{"name": r["driver"], "score": r[primary_score_col], "type": "Driver",
              "team": DRIVER_TEAM_2026.get(r["driver"], "Unknown")}
             for _, r in opt_result["drivers"].iterrows()] +
            [{"name": r["constructor"], "score": r[constructor_score_col], "type": "Constructor",
              "team": r["constructor"]}
             for _, r in opt_result["constructors"].iterrows()]
        )
        contrib_df = pd.DataFrame(contrib_data).sort_values("score", ascending=True)
        contrib_df["color"] = contrib_df["team"].map(TEAM_COLORS).fillna("#888888")

        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["score"],
            y=contrib_df["name"],
            orientation="h",
            marker_color=contrib_df["color"],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        apply_transparent_plot_layout(
            fig_contrib,
            xaxis_title=primary_score_label,
            margin=dict(t=10, b=30, l=10, r=10),
            height=280,
        )
        st.plotly_chart(fig_contrib, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DRIVER RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.subheader("All Driver Scores")

        if is_predictive:
            show_df = d_scores[[
                "rank", "driver", "cost_m", "predicted_points", "recent_form",
                "track_history", "season_avg", "quali_form", "dnf_rate_5", "testing_bonus"
            ]].copy()
            show_df.columns = ["#", "Driver", "Cost", "Predicted Pts",
                               "Recent Form", "Track Hist", "Season Avg",
                               "Quali Form", "DNF Rate", "Expert Adj"]
        else:
            show_df = d_scores[["rank", "driver", "cost_m", "norm_score",
                                  "recent_form", "track_history",
                                  "season_ppm", "quali_form", "testing_bonus"]].copy()
            show_df.columns = ["#", "Driver", "Cost", "Score",
                                "Recent Form", "Track Hist", "PPM", "Quali Form", "FP Bonus"]
        show_df["Cost"] = show_df["Cost"].map(lambda x: f"${x:.1f}M")

        # Highlight picked drivers
        def highlight_picked(row):
            if row["Driver"] in picked_drivers:
                return ["background-color: rgba(46, 125, 50, 0.18)"] * len(row)
            return [""] * len(row)

        st.dataframe(
            show_df.style.apply(highlight_picked, axis=1).format({
                "Score"        : "{:.3f}",
                "Predicted Pts": "{:.3f}",
                "Recent Form"  : "{:.1f}",
                "Track Hist"   : "{:.1f}",
                "PPM"          : "{:.2f}",
                "Season Avg"   : "{:.2f}",
                "Quali Form"   : "{:.1f}",
                "DNF Rate"     : "{:.2f}",
                "FP Bonus"     : "{:.1f}",
                "Expert Adj"   : "{:.1f}",
            }),
            use_container_width=True, height=600, hide_index=True
        )
        st.caption("🟢 Green rows = in optimal team")

    with col_b:
        st.subheader("Score vs Cost")

        d_scores_plot = d_scores.copy()
        d_scores_plot["team"] = d_scores_plot["driver"].map(DRIVER_TEAM_2026).fillna("Unknown")
        d_scores_plot["color"] = d_scores_plot["team"].map(TEAM_COLORS).fillna("#888")
        d_scores_plot["in_team"] = d_scores_plot["driver"].isin(picked_drivers)
        d_scores_plot["marker_size"] = d_scores_plot["in_team"].map({True: 18, False: 10})
        d_scores_plot["marker_symbol"] = d_scores_plot["in_team"].map({True: "star", False: "circle"})

        fig_scatter = go.Figure()
        for team in d_scores_plot["team"].unique():
            sub = d_scores_plot[d_scores_plot["team"] == team]
            fig_scatter.add_trace(go.Scatter(
                x=sub["cost_m"],
                y=sub[primary_score_col],
                mode="markers+text",
                name=team,
                marker=dict(
                    color=TEAM_COLORS.get(team, "#888"),
                    size=sub["marker_size"],
                    symbol=sub["marker_symbol"],
                    line=dict(width=2, color="white"),
                ),
                text=sub["driver"].str.split().str[-1],
                textposition="top center",
                textfont=dict(size=9),
                hovertemplate=f"<b>%{{text}}</b><br>Cost: $%{{x:.1f}}M<br>{primary_score_label}: %{{y:.3f}}<extra></extra>",
            ))

        fig_scatter.add_vline(x=budget_cap / 7, line_dash="dash",
                               line_color="gray", annotation_text="avg budget/pick")
        apply_transparent_plot_layout(
            fig_scatter,
            xaxis_title="Cost ($M)",
            yaxis_title=primary_score_label,
            height=580,
        )
        fig_scatter.update_layout(legend=dict(orientation="v", x=1.02, y=1))
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("⭐ Stars = optimal team picks")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if run_bt and bt_seasons:
        bt_df, bt_source_df = cached_backtest(
            model_mode, tuple(sorted(bt_seasons)), current_season, bt_circuit, use_bonus,
            budget_cap, w_recent, w_track, w_ppm, w_quali, ridge_alpha, min_history_weekends,
            current_season_boost, season_decay
        )
        if not bt_df.empty:
            # KPIs
            bk1, bk2, bk3, bk4 = st.columns(4)
            bk1.metric("Races Evaluated", len(bt_df))
            bk2.metric("Avg Actual Pts",  f"{bt_df['actual_pts'].mean():.1f}")
            bk3.metric("Avg Oracle Pts",  f"{bt_df['oracle_pts'].mean():.1f}")
            bk4.metric("Avg Efficiency",  f"{bt_df['efficiency_pct'].mean():.1f}%")
            if is_predictive:
                bk5, bk6 = st.columns(2)
                bk5.metric("Avg Driver MAE", f"{bt_df['driver_mae'].mean():.2f}")
                bk6.metric("Avg Train Weight", f"{bt_df['avg_sample_weight'].mean():.2f}")

            col_bt1, col_bt2 = st.columns(2)

            with col_bt1:
                st.subheader("Efficiency Over Time")
                fig_eff = go.Figure()
                fig_eff.add_trace(go.Bar(
                    x=bt_df["race_name"],
                    y=bt_df["efficiency_pct"],
                    marker_color=[TEAM_COLORS.get("Ferrari") if e >= 50
                                  else TEAM_COLORS.get("McLaren") if e >= 35
                                  else "#666"
                                  for e in bt_df["efficiency_pct"]],
                    hovertemplate="%{x}<br>Efficiency: %{y:.1f}%<extra></extra>",
                ))
                fig_eff.add_hline(y=bt_df["efficiency_pct"].mean(), line_dash="dash",
                                   line_color="white",
                                   annotation_text=f"avg {bt_df['efficiency_pct'].mean():.1f}%")
                apply_transparent_plot_layout(
                    fig_eff,
                    yaxis_title="% of Oracle Score",
                    height=320,
                )
                fig_eff.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig_eff, use_container_width=True)

            with col_bt2:
                st.subheader("Actual vs Oracle Points")
                fig_pts = go.Figure()
                fig_pts.add_trace(go.Scatter(
                    x=bt_df["race_name"], y=bt_df["oracle_pts"],
                    mode="lines+markers", name="Oracle (perfect)",
                    line=dict(color="#aaaaaa", dash="dot"),
                    marker=dict(size=6),
                ))
                fig_pts.add_trace(go.Scatter(
                    x=bt_df["race_name"], y=bt_df["actual_pts"],
                    mode="lines+markers", name="Model picks",
                    line=dict(color=TEAM_COLORS["Ferrari"], width=2),
                    marker=dict(size=8),
                    fill="tonexty", fillcolor="rgba(232,0,45,0.1)",
                ))
                apply_transparent_plot_layout(
                    fig_pts,
                    yaxis_title="Fantasy Points",
                    height=320,
                )
                fig_pts.update_layout(xaxis_tickangle=-30,
                                      legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig_pts, use_container_width=True)

            st.subheader("Race-by-Race Results")
            bt_cols = ["season", "race_name", "picked_drivers",
                       "picked_constructors", "model_cost",
                       "actual_pts", "oracle_pts", "efficiency_pct"]
            bt_names = ["Season", "Race", "Drivers Picked",
                        "Constructors", "Cost ($M)",
                        "Actual Pts", "Oracle Pts", "Efficiency %"]
            if is_predictive:
                bt_cols.insert(5, "driver_mae")
                bt_names.insert(5, "Driver MAE")
            bt_display = bt_df[bt_cols].copy()
            bt_display.columns = bt_names
            st.dataframe(bt_display, use_container_width=True, hide_index=True)
            if not bt_source_df.empty:
                st.caption("Backtest source seasons: " + " | ".join(
                    f"{int(r['season'])}:{r['source']} ({int(r['weekends_loaded'])} races)"
                    for _, r in bt_source_df.iterrows()
                ))

            st.markdown("### Backtest Explained")
            if is_predictive:
                st.markdown("""
                For each historical race in this backtest, the model pretends it is standing **before** that race weekend:
                1. It trains only on races that happened earlier.
                2. It predicts driver fantasy points for the next race.
                3. It optimizes a valid lineup under the budget.
                4. It compares that lineup to what actually happened.
                5. It benchmarks the result against the best possible **budget-constrained** lineup in hindsight.
                """)
            else:
                st.markdown("""
                For each historical race in this backtest, the heuristic model:
                1. Uses only earlier races as training data.
                2. Scores drivers and constructors with the weighted signal model.
                3. Optimizes a valid lineup under the budget.
                4. Compares that lineup to the actual race outcome and an oracle benchmark.
                """)

            example_row = bt_df.sort_values(["season", "round"]).iloc[0]
            example_lines = [
                f"Example race: **{example_row['race_name']} ({int(example_row['season'])})**",
                f"Model lineup: `{example_row['picked_drivers']}`",
                f"Constructors: `{example_row['picked_constructors']}`",
                f"Lineup cost: **${example_row['model_cost']:.1f}M**",
                f"Actual score from that lineup: **{example_row['actual_pts']:.1f}**",
                f"Oracle score under the same budget: **{example_row['oracle_pts']:.1f}**",
                f"Efficiency: **{example_row['efficiency_pct']:.1f}%**",
            ]
            if is_predictive and "driver_mae" in example_row.index:
                example_lines.append(
                    f"Driver-level prediction MAE for that race: **{example_row['driver_mae']:.2f}**"
                )
            st.info("\n".join(example_lines))
        else:
            st.info("No matching races found for the selected circuit filter.")
    else:
        st.info("👈 Select seasons and circuit in the sidebar, then click **▶ Run Backtest**.")
        if is_predictive:
            st.markdown("""
            **What the predictive backtester does:**
            1. Builds training examples from prior race weekends only.
            2. Fits a next-race fantasy points model for each historical target race.
            3. Predicts driver and constructor points for that race.
            4. Optimises a team within the budget.
            5. Compares the selected team to a **budget-constrained oracle** built from actual points.

            **Efficiency** = `actual pts / oracle pts`.
            """)
        else:
            st.markdown("""
            **What the backtester does:**
            1. For each historical race at the selected circuit, it uses *only prior races* as training data
            2. Runs the value model to score all drivers/constructors
            3. Optimises a team within the $100M budget
            4. Scores that team against what actually happened
            5. Compares to the oracle (best possible team in hindsight)

            **Efficiency** = `actual pts / oracle pts` — useful for comparison, but note the current oracle is not budget-constrained.
            """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIGNAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    selected_driver = st.selectbox("Select driver to inspect", options=d_scores["driver"].tolist())
    row = d_scores[d_scores["driver"] == selected_driver].iloc[0]

    if is_predictive:
        st.subheader("Prediction Breakdown by Driver")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        col_r1.metric("Predicted Points", f"{row['predicted_points']:.2f}")
        col_r2.metric("Base Prediction", f"{row['base_predicted_points']:.2f}")
        col_r3.metric("Recent Form", f"{row['recent_form']:.1f}")
        col_r4.metric("Track History", f"{row['track_history']:.1f}")
        col_r5.metric("Expert Adj", f"{row['testing_bonus']:+.1f}")

        explore_cols = ["recent_form_3", "recent_form_5", "track_history", "season_avg", "quali_form_5"]
        explore_labels = ["Recent 3", "Recent 5", "Track", "Season Avg", "Quali"]
        explore_df = d_scores[["driver"] + explore_cols].copy()
        scaled_df = explore_df.set_index("driver")
        scaled_df = (scaled_df - scaled_df.mean()) / scaled_df.std(ddof=0).replace(0, 1)
        scaled_df = scaled_df.fillna(0.0)

        col_radar, col_compare = st.columns(2)
        with col_radar:
            st.subheader("Feature Radar")
            z_vals = [float(scaled_df.loc[selected_driver, col]) for col in explore_cols]
            fig_radar = go.Figure(go.Scatterpolar(
                r=z_vals + [z_vals[0]],
                theta=explore_labels + [explore_labels[0]],
                fill="toself",
                line_color=TEAM_COLORS.get(DRIVER_TEAM_2026.get(selected_driver, ""), "#ff0000"),
                name=selected_driver,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[-2, 2], gridcolor=GRID_COLOR),
                    angularaxis=dict(gridcolor=GRID_COLOR),
                    bgcolor=PLOT_BG,
                ),
                showlegend=False,
                height=350,
                paper_bgcolor=PLOT_BG,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_compare:
            st.subheader("Current Grid Feature Heatmap")
            heat_df = scaled_df.copy()
            heat_df.columns = explore_labels
            fig_heat = go.Figure(go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns.tolist(),
                y=heat_df.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                hovertemplate="%{y} — %{x}: %{z:.2f}<extra></extra>",
            ))
            fig_heat.update_layout(
                height=520,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Learned Feature Importance")
        coef_df = trained_model["coefficients_df"][["label", "coefficient"]].copy()
        fig_coef = go.Figure(go.Bar(
            x=coef_df["coefficient"],
            y=coef_df["label"],
            orientation="h",
            marker_color=["#2ca02c" if x >= 0 else "#d62728" for x in coef_df["coefficient"]],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
        apply_transparent_plot_layout(
            fig_coef,
            xaxis_title="Standardized coefficient",
            height=360,
            margin=dict(t=10, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        st.subheader("Signal Breakdown by Driver")
        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        col_r1.metric("Model Score",  f"{row['norm_score']:.3f}")
        col_r2.metric("Recent Form",  f"{row['recent_form']:.1f} pts")
        col_r3.metric("Track History",f"{row['track_history']:.1f} pts")
        col_r4.metric("Quali Form",   f"{row['quali_form']:.1f}")
        col_r5.metric("FP Bonus",     f"+{row['testing_bonus']:.1f}" if row['testing_bonus'] > 0 else "—")

        col_radar, col_compare = st.columns(2)

        with col_radar:
            st.subheader("Signal Radar")
            categories = ["Recent Form", "Track History", "PPM", "Quali Form"]
            z_vals = [
                float(row.get("recent_form_z", 0)),
                float(row.get("track_history_z", 0)),
                float(row.get("season_ppm_z", 0)),
                float(row.get("quali_form_z", 0)),
            ]
            fig_radar = go.Figure(go.Scatterpolar(
                r=z_vals + [z_vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                line_color=TEAM_COLORS.get(DRIVER_TEAM_2026.get(selected_driver, ""), "#ff0000"),
                name=selected_driver,
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[-2, 2], gridcolor=GRID_COLOR),
                    angularaxis=dict(gridcolor=GRID_COLOR),
                    bgcolor=PLOT_BG,
                ),
                showlegend=False,
                height=350,
                paper_bgcolor=PLOT_BG,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_compare:
            st.subheader("All Drivers — Signal Heatmap")
            heat_cols = ["recent_form_z", "track_history_z", "season_ppm_z", "quali_form_z"]
            heat_df = d_scores[["driver"] + heat_cols].set_index("driver")
            heat_df.columns = ["Recent", "Track", "PPM", "Quali"]

            fig_heat = go.Figure(go.Heatmap(
                z=heat_df.values,
                x=heat_df.columns.tolist(),
                y=heat_df.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                hovertemplate="%{y} — %{x}: %{z:.2f}<extra></extra>",
            ))
            fig_heat.update_layout(
                height=520,
                margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor=PLOT_BG,
                plot_bgcolor=PLOT_BG,
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Weight Sensitivity — How team changes with different weights")
        weight_combos = [
            {"label": "Track-heavy (current circuit focus)", "w_r": 0.15, "w_t": 0.55, "w_p": 0.10, "w_q": 0.20},
            {"label": "Form-heavy (hot streak)",             "w_r": 0.55, "w_t": 0.15, "w_p": 0.10, "w_q": 0.20},
            {"label": "Value-heavy (PPM focus)",             "w_r": 0.25, "w_t": 0.20, "w_p": 0.35, "w_q": 0.20},
            {"label": "Balanced (current)",                  "w_r": 0.35, "w_t": 0.30, "w_p": 0.15, "w_q": 0.20},
        ]
        sens_rows = []
        for combo in weight_combos:
            combo_output = run_model(
                HEURISTIC_MODE, tuple(sorted(train_seasons)), current_season, target_circuit,
                combo["w_r"], combo["w_t"], combo["w_p"], combo["w_q"],
                use_bonus, include_current_season, tuple(), tuple(), budget_cap,
                ridge_alpha, min_history_weekends
            )
            res = combo_output["result"]
            sens_rows.append({
                "Strategy"      : combo["label"],
                "Drivers"       : " · ".join(list(res["drivers"]["driver"])),
                "Constructors"  : " · ".join(list(res["constructors"]["constructor"])),
                "Cost"          : f"${res['total_cost']:.1f}M",
            })

        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA TABLES
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Training Data Snapshot")
    dt1, dt2, dt3, dt4 = st.columns(4)
    dt1.metric("Seasons Loaded", int(source_df["season"].nunique()) if not source_df.empty else 0)
    dt2.metric("Race Weekends", len(training_weekends))
    dt3.metric("Driver-Race Rows", len(history_df))
    dt4.metric("2025 Weekends", int((weekend_summary_df["season"] == 2025).sum()) if not weekend_summary_df.empty else 0)

    st.markdown("**Season Source Summary**")
    st.dataframe(source_df, use_container_width=True, hide_index=True)

    st.markdown("**Loaded Race Weekends**")
    st.dataframe(weekend_summary_df, use_container_width=True, hide_index=True, height=260)

    st.markdown("**Historical Driver-Level Training Data**")
    st.dataframe(history_df, use_container_width=True, hide_index=True, height=320)

    if is_predictive:
        st.markdown("**Predictive Training Examples**")
        st.dataframe(training_examples_df, use_container_width=True, hide_index=True, height=320)

        st.markdown("**Current Prediction Feature Table**")
        feature_cols = [
            "rank", "driver", "team", "cost_m", "predicted_points", "base_predicted_points",
            "recent_form_3", "recent_form_5", "track_history", "season_avg",
            "quali_form_5", "dnf_rate_5", "team_recent_form_3", "track_experience",
            "overall_experience", "testing_bonus",
        ]
    else:
        st.markdown("**Model Feature Table**")
        feature_cols = [
            "rank", "driver", "team", "cost_m", "recent_form", "track_history",
            "season_ppm", "quali_form", "testing_bonus", "recent_form_z",
            "track_history_z", "season_ppm_z", "quali_form_z", "norm_score",
        ]
    st.dataframe(d_scores[feature_cols], use_container_width=True, hide_index=True, height=420)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("Model Summary")
    if is_predictive:
        st.markdown(f"""
        This app is using a transparent **next-race fantasy points prediction model**.

        It works in five steps:
        1. Load historical race weekends for the selected training seasons.
        2. Convert every historical weekend into fantasy points using the scoring engine.
        3. Build one training row per driver per race using only information available **before** that race.
        4. Fit a ridge-style linear regression to predict next-race fantasy points.
        5. Predict the current grid, optionally apply expert adjustments, and optimise the best 5-driver / 2-constructor team under the budget cap.
        """)
    else:
        st.markdown(f"""
        This app is a transparent **heuristic ranking model** rather than a black-box ML model.

        It works in four steps:
        1. Load historical race weekends for the selected training seasons.
        2. Convert each weekend into fantasy points using the scoring engine.
        3. Build four driver signals: recent form, track history, season points-per-million, and qualifying form.
        4. Z-score those signals across the grid, apply the chosen weights, add optional testing bonuses, and optimise the best 5-driver / 2-constructor team under the budget cap.
        """)

    st.markdown("**Where the data comes from**")
    st.markdown("""
    - Historical race and qualifying data is pulled from the Jolpica Ergast-compatible API when available.
    - Downloaded seasons are cached locally and reused on future runs.
    - If the API is unavailable, the app falls back to embedded seed data included in this project.
    - The season source summary in the `Data Tables` tab shows exactly which seasons came from `api`, `cache`, or `seed`.
    - Current driver prices, constructor prices, and manual FP/testing bonuses come from the project files and constants in this repo.
    """)

    if is_predictive:
        st.markdown("**Model fit diagnostics**")
        metrics_df = pd.DataFrame([trained_model["train_metrics"]])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("**Season weighting rules**")
        st.markdown(
            f"- Current-season rows are boosted by **{diagnostics['current_season_boost']:.2f}x**.\n"
            f"- Older seasons decay by **{diagnostics['season_decay']:.2f}** per year of distance.\n"
            "- This means once a new 2026 race is loaded, its examples carry more influence than 2025, 2024, and earlier rows."
        )

        st.markdown("**Model formula**")
        st.code("predicted_points = intercept + Σ(standardized_feature_i * coefficient_i) + expert_adjustment")

        st.markdown("**Learned coefficients**")
        st.dataframe(trained_model["coefficients_df"][["label", "coefficient", "abs_coefficient"]],
                     use_container_width=True, hide_index=True)

        st.markdown("**Driver-level contribution breakdown**")
        explain_driver = st.selectbox("Inspect prediction decomposition", options=d_scores["driver"].tolist(),
                                      key="explain_driver")
        contribution_cols = [f"{col}_contrib" for col in feature_columns]
        explain_cols = ["driver", "team", "predicted_points", "base_predicted_points",
                        "testing_bonus"] + feature_columns + contribution_cols
        explain_row = d_scores[d_scores["driver"] == explain_driver][explain_cols]
        st.dataframe(explain_row, use_container_width=True, hide_index=True)

        st.markdown("**Notes**")
        st.markdown(
            "- The model is trained on historical driver-level examples rather than manually chosen weights.\n"
            "- Rule changes are handled by season-aware sample weighting, so newer seasons matter more than older ones.\n"
            "- Backtests use an expanding-window approach so each race is predicted only from earlier races.\n"
            "- The backtest oracle is budget-constrained, making the benchmark fairer than the earlier heuristic version."
        )
    else:
        st.markdown("**Current weights in use**")
        weight_df = pd.DataFrame([
            {"signal": "recent_form", "weight": model_weights["recent_form"]},
            {"signal": "track_history", "weight": model_weights["track_history"]},
            {"signal": "season_ppm", "weight": model_weights["season_ppm"]},
            {"signal": "quali_form", "weight": model_weights["quali_form"]},
        ])
        st.dataframe(weight_df.style.format({"weight": "{:.2%}"}),
                     use_container_width=True, hide_index=True)

        if diagnostics["season_ppm_active_drivers"] == 0:
            st.warning(
                "Season PPM is currently zero for every driver in this run. "
                "That usually means there are no completed races loaded yet for the selected current season."
            )
        else:
            st.success(
                f"Season PPM is active for {diagnostics['season_ppm_active_drivers']} drivers "
                f"based on loaded {current_season} race data."
            )

        st.markdown("**Score formula**")
        st.code(
            "norm_score = recent_w * recent_form_z + track_w * track_history_z + "
            "ppm_w * season_ppm_z + quali_w * quali_form_z + testing_bonus"
        )

        explain_df = d_scores.copy()
        explain_df["recent_contrib"] = explain_df["recent_form_z"] * model_weights["recent_form"]
        explain_df["track_contrib"] = explain_df["track_history_z"] * model_weights["track_history"]
        explain_df["ppm_contrib"] = explain_df["season_ppm_z"] * model_weights["season_ppm"]
        explain_df["quali_contrib"] = explain_df["quali_form_z"] * model_weights["quali_form"]

        st.markdown("**Driver-level contribution breakdown**")
        explain_driver = st.selectbox("Inspect driver scoring", options=explain_df["driver"].tolist(),
                                      key="explain_driver")
        explain_row = explain_df[explain_df["driver"] == explain_driver][[
            "driver", "team", "recent_form", "track_history", "season_ppm",
            "quali_form", "recent_form_z", "track_history_z", "season_ppm_z",
            "quali_form_z", "recent_contrib", "track_contrib", "ppm_contrib",
            "quali_contrib", "testing_bonus", "norm_score",
        ]]
        st.dataframe(explain_row, use_container_width=True, hide_index=True)

        st.markdown("**Notes**")
        st.markdown(
            "- `season_ppm` becomes more meaningful as current-season races are loaded.\n"
            "- `track_history` falls back toward recent form when a driver has no direct history at the target circuit.\n"
            "- The current backtest oracle is still an unconstrained hindsight benchmark, so treat it as directional rather than perfect truth."
        )
