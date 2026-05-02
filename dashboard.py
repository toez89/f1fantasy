"""
F1 Fantasy Dashboard — Miami GP 2026
Run with:  streamlit run dashboard.py
"""

import json
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
    CURRENT_BUDGET_CAP,
    CURRENT_COST_SNAPSHOT,
)
from pricing     import get_available_cost_snapshots, load_cost_snapshot
from pipeline    import (DEFAULT_TRAIN_SEASONS, build_constructor_season_progress_df,
                         build_driver_season_progress_df, build_weekend_summary_df,
                         run_predictive_model_comparison_pipeline,
                         run_optimisation_pipeline, run_predictive_backtest_pipeline,
                         run_predictive_pipeline)
from predictive_model import (DEFAULT_COMPARISON_MODELS, FEATURE_LABELS,
                              MODEL_LABELS, get_model_default_params)
from value_model import CIRCUIT_ALIAS

# ── constants ─────────────────────────────────────────────────────────────────
TARGET_CIRCUIT   = "Miami International Autodrome"
CURRENT_SEASON   = 2026
BUDGET           = CURRENT_BUDGET_CAP
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

BASELINE_FEATURE_OPTIONS = ["recent_form_5", "recent_form_3", "track_history", "season_avg"]
BLEND_MODEL_OPTIONS = [
    "blend_baseline_elastic_net",
    "blend_baseline_gradient_boosting",
]
TEAM_HISTORY_PATH = Path(__file__).parent / "team_history.csv"
TEAM_HISTORY_COLUMNS = [
    "season", "round", "race_name",
    "driver_1", "driver_2", "driver_3", "driver_4", "driver_5",
    "constructor_1", "constructor_2", "notes",
]


def build_predictive_model_configs(selected_model_type, selected_model_params):
    configs = {
        model_type: get_model_default_params(model_type)
        for model_type in MODEL_LABELS
    }
    configs[selected_model_type] = dict(selected_model_params)
    return configs


def format_model_params_text(params_json):
    params = json.loads(params_json)
    if not params:
        return "default"
    return ", ".join(f"{key}={value}" for key, value in params.items())


def build_blend_weight_grid(start, stop, step):
    weights = []
    current = float(start)
    while current <= float(stop) + 1e-9:
        weights.append(round(current, 2))
        current += float(step)
    return sorted(set(weights))


def load_team_history() -> pd.DataFrame:
    if not TEAM_HISTORY_PATH.exists():
        return pd.DataFrame(columns=TEAM_HISTORY_COLUMNS)

    team_history_df = pd.read_csv(TEAM_HISTORY_PATH)
    for column in TEAM_HISTORY_COLUMNS:
        if column not in team_history_df.columns:
            team_history_df[column] = ""
    return team_history_df[TEAM_HISTORY_COLUMNS].copy()


def snapshot_label(row: pd.Series) -> str:
    return f"{int(row['season'])} R{int(row['round'])} {row['race_name']}"


def build_snapshot_options(team_history_df: pd.DataFrame) -> list[str]:
    if team_history_df.empty:
        return []
    ordered = team_history_df.sort_values(["season", "round"]).reset_index(drop=True)
    return [snapshot_label(row) for _, row in ordered.iterrows()]


def get_snapshot_row(team_history_df: pd.DataFrame, label: str | None) -> pd.Series | None:
    if team_history_df.empty or not label:
        return None
    ordered = team_history_df.sort_values(["season", "round"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        if snapshot_label(row) == label:
            return row
    return None


def row_to_team_lists(row: pd.Series | None) -> tuple[list[str], list[str]]:
    if row is None:
        return [], []
    drivers = [
        str(row.get(f"driver_{idx}", "")).strip()
        for idx in range(1, 6)
        if str(row.get(f"driver_{idx}", "")).strip()
    ]
    constructors = [
        str(row.get(f"constructor_{idx}", "")).strip()
        for idx in range(1, 3)
        if str(row.get(f"constructor_{idx}", "")).strip()
    ]
    return drivers, constructors


def save_team_snapshot(
    season: int,
    round_number: int,
    race_name: str,
    drivers: list[str],
    constructors: list[str],
    notes: str = "",
) -> None:
    team_history_df = load_team_history()
    padded_drivers = list(drivers[:5]) + [""] * max(0, 5 - len(drivers))
    padded_constructors = list(constructors[:2]) + [""] * max(0, 2 - len(constructors))
    new_row = {
        "season": int(season),
        "round": int(round_number),
        "race_name": race_name.strip(),
        "driver_1": padded_drivers[0],
        "driver_2": padded_drivers[1],
        "driver_3": padded_drivers[2],
        "driver_4": padded_drivers[3],
        "driver_5": padded_drivers[4],
        "constructor_1": padded_constructors[0],
        "constructor_2": padded_constructors[1],
        "notes": notes.strip(),
    }

    if not team_history_df.empty:
        mask = (
            (team_history_df["season"].astype(int) == int(season))
            & (team_history_df["round"].astype(int) == int(round_number))
        )
        team_history_df = team_history_df.loc[~mask].copy()

    team_history_df = pd.concat([team_history_df, pd.DataFrame([new_row])], ignore_index=True)
    team_history_df = team_history_df.sort_values(["season", "round"]).reset_index(drop=True)
    team_history_df.to_csv(TEAM_HISTORY_PATH, index=False)


def expand_team_exclusions(selected_teams, driver_team_map, constructor_names):
    excluded_drivers = sorted(
        driver for driver, team in driver_team_map.items() if team in selected_teams
    )
    excluded_constructors = sorted(team for team in selected_teams if team in constructor_names)
    return excluded_drivers, excluded_constructors


def build_effective_exclusions(
    manual_excluded,
    news_excluded_drivers,
    news_excluded_constructors,
    news_excluded_teams,
    driver_team_map,
    constructor_names,
):
    team_drivers, team_constructors = expand_team_exclusions(
        news_excluded_teams, driver_team_map, constructor_names
    )
    effective = sorted(
        set(manual_excluded)
        | set(news_excluded_drivers)
        | set(news_excluded_constructors)
        | set(team_drivers)
        | set(team_constructors)
    )
    return effective, team_drivers, team_constructors


def filter_defaults(default_values, options):
    option_set = set(options)
    return [value for value in default_values if value in option_set]


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Fantasy | Miami GP 2026",
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
    team_history_df = load_team_history()
    team_snapshot_options = build_snapshot_options(team_history_df)
    default_snapshot_label = team_snapshot_options[-1] if team_snapshot_options else None
    selected_snapshot_label = st.selectbox(
        "Baseline team from CSV",
        options=team_snapshot_options,
        index=len(team_snapshot_options) - 1 if team_snapshot_options else None,
        help="Loads your current team baseline from team_history.csv",
    ) if team_snapshot_options else None
    selected_snapshot_row = get_snapshot_row(team_history_df, selected_snapshot_label)
    default_current_drivers, default_current_constructors = row_to_team_lists(selected_snapshot_row)

    st.title("⚙️ Model Controls")
    st.markdown("---")
    st.subheader("Race Setup")
    cost_snapshot_options = get_available_cost_snapshots()
    selected_cost_snapshot = st.selectbox(
        "Cost snapshot",
        options=cost_snapshot_options,
        index=cost_snapshot_options.index(CURRENT_COST_SNAPSHOT)
        if CURRENT_COST_SNAPSHOT in cost_snapshot_options else len(cost_snapshot_options) - 1,
        help="Choose which saved price snapshot the optimizer should use.",
    )
    selected_driver_costs, selected_constructor_costs, selected_budget_cap, selected_cost_snapshot = load_cost_snapshot(
        selected_cost_snapshot
    )
    previous_cost_snapshot = st.session_state.get("_selected_cost_snapshot")
    if "budget_cap_input" not in st.session_state or previous_cost_snapshot != selected_cost_snapshot:
        st.session_state["budget_cap_input"] = float(selected_budget_cap)
    st.session_state["_selected_cost_snapshot"] = selected_cost_snapshot
    driver_cost_options = list(selected_driver_costs.keys())
    constructor_cost_options = list(selected_constructor_costs.keys())
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
        predictive_model_type = st.selectbox(
            "Model family",
            options=list(MODEL_LABELS.keys()),
            index=list(MODEL_LABELS.keys()).index("ridge"),
            format_func=lambda key: MODEL_LABELS[key],
            help="Try different regression families on the same engineered driver-race dataset.",
        )
        predictive_model_params = get_model_default_params(predictive_model_type)
        if predictive_model_type == "ridge":
            ridge_alpha = st.slider("Regularisation strength", 0.1, 20.0, float(predictive_model_params["alpha"]), 0.1,
                                    help="Higher values shrink coefficients and reduce overfitting.")
            predictive_model_params["alpha"] = ridge_alpha
        elif predictive_model_type == "elastic_net":
            elastic_alpha = st.slider("Elastic Net alpha", 0.01, 5.0, float(predictive_model_params["alpha"]), 0.01,
                                      help="Overall regularisation strength.")
            elastic_l1_ratio = st.slider("Elastic Net l1 ratio", 0.0, 1.0, float(predictive_model_params["l1_ratio"]), 0.05,
                                         help="0 behaves more like ridge, 1 behaves more like lasso.")
            predictive_model_params["alpha"] = elastic_alpha
            predictive_model_params["l1_ratio"] = elastic_l1_ratio
        elif predictive_model_type == "random_forest":
            rf_estimators = st.slider("Random forest trees", 50, 500, int(predictive_model_params["n_estimators"]), 25)
            rf_max_depth_choice = st.selectbox(
                "Random forest max depth",
                options=["None", 2, 3, 4, 5, 6, 8, 10],
                index=["None", 2, 3, 4, 5, 6, 8, 10].index(
                    "None" if predictive_model_params["max_depth"] is None else predictive_model_params["max_depth"]
                ),
                help="Limit tree depth to reduce overfitting on a small dataset.",
            )
            rf_min_samples_leaf = st.slider("Random forest min samples per leaf", 1, 8, int(predictive_model_params["min_samples_leaf"]), 1)
            predictive_model_params["n_estimators"] = rf_estimators
            predictive_model_params["max_depth"] = None if rf_max_depth_choice == "None" else int(rf_max_depth_choice)
            predictive_model_params["min_samples_leaf"] = rf_min_samples_leaf
        elif predictive_model_type == "gradient_boosting":
            gb_estimators = st.slider("Boosting stages", 50, 500, int(predictive_model_params["n_estimators"]), 25)
            gb_learning_rate = st.slider("Learning rate", 0.01, 0.30, float(predictive_model_params["learning_rate"]), 0.01)
            gb_max_depth = st.slider("Boosting tree depth", 1, 5, int(predictive_model_params["max_depth"]), 1)
            gb_min_samples_leaf = st.slider("Boosting min samples per leaf", 1, 8, int(predictive_model_params["min_samples_leaf"]), 1)
            predictive_model_params["n_estimators"] = gb_estimators
            predictive_model_params["learning_rate"] = gb_learning_rate
            predictive_model_params["max_depth"] = gb_max_depth
            predictive_model_params["min_samples_leaf"] = gb_min_samples_leaf
        elif predictive_model_type == "blend_baseline_elastic_net":
            blend_weight = st.slider(
                "Elastic Net blend weight",
                0.0, 1.0, float(predictive_model_params["blend_weight"]), 0.05,
                help="0 = pure baseline, 1 = pure Elastic Net.",
            )
            baseline_feature = st.selectbox(
                "Baseline signal",
                options=BASELINE_FEATURE_OPTIONS,
                index=BASELINE_FEATURE_OPTIONS.index(predictive_model_params["baseline_feature"]),
                format_func=lambda key: FEATURE_LABELS.get(key, key),
            )
            elastic_alpha = st.slider("Elastic Net alpha", 0.01, 5.0, float(predictive_model_params["alpha"]), 0.01)
            elastic_l1_ratio = st.slider("Elastic Net l1 ratio", 0.0, 1.0, float(predictive_model_params["l1_ratio"]), 0.05)
            predictive_model_params["blend_weight"] = blend_weight
            predictive_model_params["baseline_feature"] = baseline_feature
            predictive_model_params["alpha"] = elastic_alpha
            predictive_model_params["l1_ratio"] = elastic_l1_ratio
        elif predictive_model_type == "blend_baseline_gradient_boosting":
            blend_weight = st.slider(
                "Boosting blend weight",
                0.0, 1.0, float(predictive_model_params["blend_weight"]), 0.05,
                help="0 = pure baseline, 1 = pure Gradient Boosting.",
            )
            baseline_feature = st.selectbox(
                "Baseline signal",
                options=BASELINE_FEATURE_OPTIONS,
                index=BASELINE_FEATURE_OPTIONS.index(predictive_model_params["baseline_feature"]),
                format_func=lambda key: FEATURE_LABELS.get(key, key),
            )
            gb_estimators = st.slider("Boosting stages", 50, 500, int(predictive_model_params["n_estimators"]), 25)
            gb_learning_rate = st.slider("Learning rate", 0.01, 0.30, float(predictive_model_params["learning_rate"]), 0.01)
            gb_max_depth = st.slider("Boosting tree depth", 1, 5, int(predictive_model_params["max_depth"]), 1)
            gb_min_samples_leaf = st.slider("Boosting min samples per leaf", 1, 8, int(predictive_model_params["min_samples_leaf"]), 1)
            predictive_model_params["blend_weight"] = blend_weight
            predictive_model_params["baseline_feature"] = baseline_feature
            predictive_model_params["n_estimators"] = gb_estimators
            predictive_model_params["learning_rate"] = gb_learning_rate
            predictive_model_params["max_depth"] = gb_max_depth
            predictive_model_params["min_samples_leaf"] = gb_min_samples_leaf
        else:
            baseline_feature = st.selectbox(
                "Baseline signal",
                options=BASELINE_FEATURE_OPTIONS,
                index=BASELINE_FEATURE_OPTIONS.index(predictive_model_params["baseline_feature"]),
                format_func=lambda key: FEATURE_LABELS.get(key, key),
                help="Simple benchmark that predicts points from a single existing signal.",
            )
            predictive_model_params["baseline_feature"] = baseline_feature

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
        comparison_model_types = st.multiselect(
            "Compare model families",
            options=list(MODEL_LABELS.keys()),
            default=list(DEFAULT_COMPARISON_MODELS),
            format_func=lambda key: MODEL_LABELS[key],
            help="Run the expanding-window backtest side by side for these model families.",
        )
        st.markdown("**Blend weight sweep**")
        run_blend_sweep = st.toggle(
            "Run automatic blend-weight sweep",
            value=False,
            help="Backtest several blend weights in one go for the selected blend family.",
        )
        blend_sweep_model_type = st.selectbox(
            "Blend family to sweep",
            options=BLEND_MODEL_OPTIONS,
            index=BLEND_MODEL_OPTIONS.index("blend_baseline_gradient_boosting"),
            format_func=lambda key: MODEL_LABELS[key],
            disabled=not run_blend_sweep,
        )
        blend_sweep_start = st.slider(
            "Sweep start weight",
            0.1, 0.9, 0.2, 0.05,
            disabled=not run_blend_sweep,
        )
        blend_sweep_end = st.slider(
            "Sweep end weight",
            0.1, 0.9, 0.8, 0.05,
            disabled=not run_blend_sweep,
        )
        blend_sweep_step = st.slider(
            "Sweep step",
            0.05, 0.25, 0.15, 0.05,
            disabled=not run_blend_sweep,
        )
        st.caption("Default predictive presets are tuned from a recent full-season backtest and can still be adjusted manually.")
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
        predictive_model_type = "ridge"
        predictive_model_params = get_model_default_params("ridge")
        comparison_model_types = list(DEFAULT_COMPARISON_MODELS)
        run_blend_sweep = False
        blend_sweep_model_type = "blend_baseline_gradient_boosting"
        blend_sweep_start = 0.2
        blend_sweep_end = 0.8
        blend_sweep_step = 0.15
        min_history_weekends = 5
        current_season_boost = 3.0
        season_decay = 0.8

    st.markdown("---")
    st.subheader("Expert Inputs")
    use_bonus = st.toggle("Apply manual FP/testing adjustments", value=True)

    st.markdown("---")
    st.subheader("Team Constraints")
    use_transfer_penalty = st.toggle(
        "Model transfer cost",
        value=True,
        help="Applies the F1 Fantasy transfer penalty to the optimizer objective.",
    )
    current_drivers = st.multiselect(
        "Current drivers",
        options=driver_cost_options,
        default=default_current_drivers,
        help="Your current five drivers before making this week's transfers.",
    )
    current_constructors = st.multiselect(
        "Current constructors",
        options=constructor_cost_options,
        default=default_current_constructors,
        help="Your current two constructors before making this week's transfers.",
    )
    free_transfers = st.selectbox(
        "Free transfers available",
        options=[0, 1, 2, 3],
        index=2,
        help="Official game default is 2, with one unused transfer able to roll over.",
        disabled=not use_transfer_penalty,
    )
    locked = st.multiselect("🔒 Lock drivers",
                             options=driver_cost_options)
    excluded = st.multiselect("🚫 Exclude drivers/constructors",
                               options=driver_cost_options + constructor_cost_options,
                               help="Directly remove individual drivers or constructors from the optimizer pool.")
    news_excluded_teams = st.multiselect(
        "🚫 Exclude teams",
        options=sorted(constructor_cost_options),
        help="Removes the constructor and all drivers mapped to that team from consideration.",
    )
    st.caption("Team exclusions are applied before optimization, just like driver exclusions.")
    st.markdown("**Optional news-driven filters**")
    news_excluded_drivers = st.multiselect(
        "Exclude extra drivers from news",
        options=driver_cost_options,
        default=[],
        help="Useful for team issues, expected poor pace, penalties, injuries, or reliability concerns.",
    )
    news_excluded_constructors = st.multiselect(
        "Exclude extra constructors from news",
        options=constructor_cost_options,
        help="Use this when the whole constructor should be removed from the pool.",
    )
    news_reason = st.text_area(
        "News note",
        value="",
        height=70,
        help="Optional context for why these exclusions are active.",
    )
    budget_cap = st.number_input("💵 Budget ($M)", key="budget_cap_input", min_value=50.0,
                                  max_value=200.0, step=0.1,
                                  help="Defaults to the selected cost snapshot's saved budget cap.")
    transfer_penalty_points = 10.0 if use_transfer_penalty else 0.0

    st.markdown("---")
    st.subheader("Save Team History")
    save_team_season = st.selectbox(
        "Save season",
        options=AVAILABLE_SEASONS,
        index=AVAILABLE_SEASONS.index(current_season),
        help="Season to write into team_history.csv",
    )
    default_save_round = int(selected_snapshot_row["round"]) if selected_snapshot_row is not None else 1
    save_team_round = st.number_input(
        "Save round",
        min_value=1,
        max_value=30,
        value=default_save_round,
        step=1,
        help="Round number for this saved team snapshot.",
    )
    default_race_name = str(selected_snapshot_row["race_name"]) if selected_snapshot_row is not None else ""
    save_team_race_name = st.text_input(
        "Save race name",
        value=default_race_name,
        help="Race name stored alongside the lineup in the CSV.",
    )
    save_team_notes = st.text_input(
        "Save note",
        value="",
        help="Optional note for later reference.",
    )
    save_team_clicked = st.button("Save current team to CSV", use_container_width=True)

    st.markdown("---")
    st.subheader("Backtest")
    bt_seasons = st.multiselect("Seasons", AVAILABLE_SEASONS,
                                 default=DEFAULT_TRAIN_SEASONS)
    bt_circuit = st.selectbox("Circuit", ["all", "australia", "singapore",
                                           "china", "japan", "miami",
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
    constructor_names=constructor_cost_options,
)
effective_locked = [driver for driver in locked if driver not in effective_excluded]
predictive_model_params_json = json.dumps(predictive_model_params, sort_keys=True)
selected_driver_costs_json = json.dumps(selected_driver_costs, sort_keys=True)
selected_constructor_costs_json = json.dumps(selected_constructor_costs, sort_keys=True)
comparison_model_configs_json = json.dumps(
    build_predictive_model_configs(predictive_model_type, predictive_model_params),
    sort_keys=True,
)
blend_sweep_weights = build_blend_weight_grid(
    min(blend_sweep_start, blend_sweep_end),
    max(blend_sweep_start, blend_sweep_end),
    blend_sweep_step,
)

if save_team_clicked:
    if len(current_drivers) != 5 or len(current_constructors) != 2 or not save_team_race_name.strip():
        st.error(
            "To save a team snapshot, select exactly 5 drivers, 2 constructors, and enter a race name."
        )
    else:
        save_team_snapshot(
            season=save_team_season,
            round_number=int(save_team_round),
            race_name=save_team_race_name,
            drivers=current_drivers,
            constructors=current_constructors,
            notes=save_team_notes,
        )
        st.success(
            f"Saved team snapshot to {TEAM_HISTORY_PATH.name}: "
            f"{save_team_season} R{int(save_team_round)} {save_team_race_name}."
        )


# ── data loading & model (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Running model...", ttl=60)
def run_model(model_mode, train_seasons_tuple, current_season, target_circuit,
              w_r, w_t, w_p, w_q, use_bonus, include_current,
              locked_d, excl, bdgt, predictive_model_type, predictive_model_params_json,
              min_history_weekends, current_season_boost, season_decay,
              current_drivers_tuple, current_constructors_tuple,
              free_transfers, transfer_penalty_points,
              driver_costs_json, constructor_costs_json):
    resolved_driver_costs = json.loads(driver_costs_json)
    resolved_constructor_costs = json.loads(constructor_costs_json)
    if model_mode == PREDICTIVE_MODE:
        predictive_model_params = json.loads(predictive_model_params_json)
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
            current_drivers=list(current_drivers_tuple),
            current_constructors=list(current_constructors_tuple),
            free_transfers=free_transfers,
            transfer_penalty=transfer_penalty_points,
            driver_costs=resolved_driver_costs,
            constructor_costs=resolved_constructor_costs,
            model_type=predictive_model_type,
            model_params=predictive_model_params,
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
        current_drivers=list(current_drivers_tuple),
        current_constructors=list(current_constructors_tuple),
        free_transfers=free_transfers,
        transfer_penalty=transfer_penalty_points,
        driver_costs=resolved_driver_costs,
        constructor_costs=resolved_constructor_costs,
    )
    output["model_mode"] = "heuristic"
    return output


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(model_mode, seasons_tuple, current_season, circuit, use_bonus,
                    budget_m, w_r, w_t, w_p, w_q, predictive_model_type, predictive_model_params_json,
                    min_history_weekends,
                    current_season_boost, season_decay):
    if model_mode == PREDICTIVE_MODE:
        predictive_model_params = json.loads(predictive_model_params_json)
        return run_predictive_backtest_pipeline(
            seasons=list(seasons_tuple),
            current_season=current_season,
            target_circuit=circuit,
            include_current_season=False,
            budget_m=budget_m,
            model_type=predictive_model_type,
            model_params=predictive_model_params,
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


@st.cache_data(show_spinner="Comparing predictive models...")
def cached_model_comparison(seasons_tuple, current_season, circuit, comparison_model_types_tuple,
                            comparison_model_configs_json, budget_m, min_history_weekends,
                            current_season_boost, season_decay):
    comparison_model_configs = json.loads(comparison_model_configs_json)
    return run_predictive_model_comparison_pipeline(
        seasons=list(seasons_tuple),
        current_season=current_season,
        target_circuit=circuit,
        model_types=list(comparison_model_types_tuple),
        model_param_overrides=comparison_model_configs,
        include_current_season=False,
        budget_m=budget_m,
        min_history_weekends=min_history_weekends,
        current_season_boost=current_season_boost,
        season_decay=season_decay,
    )


@st.cache_data(show_spinner="Sweeping blend weights...")
def cached_blend_weight_sweep(seasons_tuple, current_season, circuit, blend_model_type,
                              blend_weights_tuple, blend_base_params_json, budget_m,
                              min_history_weekends, current_season_boost, season_decay):
    base_params = json.loads(blend_base_params_json)
    rows = []
    detail_frames = []

    for blend_weight in blend_weights_tuple:
        model_params = dict(base_params)
        model_params["blend_weight"] = float(blend_weight)
        bt_df, _ = run_predictive_backtest_pipeline(
            seasons=list(seasons_tuple),
            current_season=current_season,
            target_circuit=circuit,
            include_current_season=False,
            budget_m=budget_m,
            model_type=blend_model_type,
            model_params=model_params,
            min_history_weekends=min_history_weekends,
            current_season_boost=current_season_boost,
            season_decay=season_decay,
        )
        if not bt_df.empty:
            bt_df = bt_df.copy()
            bt_df["blend_weight"] = float(blend_weight)
            detail_frames.append(bt_df)

        rows.append(
            {
                "blend_weight": float(blend_weight),
                "races_evaluated": int(len(bt_df)),
                "avg_efficiency_pct": round(float(bt_df["efficiency_pct"].mean()), 2)
                if not bt_df.empty else None,
                "median_efficiency_pct": round(float(bt_df["efficiency_pct"].median()), 2)
                if not bt_df.empty else None,
                "avg_driver_mae": round(float(bt_df["driver_mae"].mean()), 3)
                if not bt_df.empty else None,
                "median_driver_mae": round(float(bt_df["driver_mae"].median()), 3)
                if not bt_df.empty else None,
            }
        )

    sweep_df = pd.DataFrame(rows)
    if not sweep_df.empty:
        sweep_df = sweep_df.sort_values("blend_weight").reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return sweep_df, detail_df


# ── run model ─────────────────────────────────────────────────────────────────
model_output = run_model(
    model_mode,
    tuple(sorted(train_seasons)),
    current_season,
    target_circuit,
    w_recent, w_track, w_ppm, w_quali,
    use_bonus, include_current_season,
    tuple(effective_locked), tuple(effective_excluded), budget_cap,
    predictive_model_type, predictive_model_params_json,
    min_history_weekends, current_season_boost, season_decay,
    tuple(current_drivers), tuple(current_constructors),
    free_transfers, transfer_penalty_points,
    selected_driver_costs_json, selected_constructor_costs_json,
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
selected_model_type = model_output.get("model_type", predictive_model_type)
selected_model_label = model_output.get("model_label", MODEL_LABELS.get(selected_model_type, selected_model_type))
selected_model_params = model_output.get("model_params", predictive_model_params)
weekend_summary_df = build_weekend_summary_df(training_weekends)
driver_season_progress_df = build_driver_season_progress_df(history_df, current_season)
constructor_season_progress_df = build_constructor_season_progress_df(training_weekends, current_season)
is_predictive = model_output["model_mode"] == "predictive"
primary_score_col = "predicted_points" if is_predictive else "norm_score"
primary_score_label = "Predicted Pts" if is_predictive else "Model Score"

picked_drivers = list(opt_result["drivers"]["driver"])
picked_constructors = list(opt_result["constructors"]["constructor"])

# ── header ────────────────────────────────────────────────────────────────────
st.title("🏎️  F1 Fantasy Model")
st.caption(
    f"{model_mode}"
    + (f" ({selected_model_label})" if is_predictive else "")
    + f" · {target_circuit} · {current_season} season context · "
    f"training seasons: {', '.join(map(str, sorted(set(train_seasons))))}"
    + (" + current season" if include_current_season else "")
    + f" · costs: {selected_cost_snapshot}"
)

# ── top KPI row ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Budget Used",  f"${opt_result['total_cost']:.1f}M",
           f"${budget_cap - opt_result['total_cost']:.1f}M remaining")
k2.metric(primary_score_label,  f"{opt_result['base_score']:.3f}")
k3.metric("Drivers",      f"{len(picked_drivers)} / 5")
k4.metric("Constructors", f"{len(picked_constructors)} / 2")
k5.metric("Transfers", f"{opt_result['transfer_count']}",
          f"{opt_result['paid_transfers']} paid")
k6.metric("After Transfer Cost", f"{opt_result['total_score']:.3f}",
          f"-{opt_result['transfer_cost']:.1f}" if opt_result["transfer_cost"] > 0 else "no penalty")

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

if use_transfer_penalty and (len(current_drivers) != 5 or len(current_constructors) != 2):
    st.warning(
        "Transfer modelling expects 5 current drivers and 2 current constructors. "
        "The optimizer will still run, but the transfer count will be based on the selections above."
    )

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Optimal Team", "📊 Driver Rankings", "📈 Season Tracker", "🔁 Backtest",
    "🔬 Signal Explorer", "🗂 Data Tables", "📘 How It Works"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OPTIMAL TEAM
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Transfer Plan")
        transfer_rows = []
        for name in opt_result["incoming_drivers"]:
            transfer_rows.append({"Move": "IN", "Type": "Driver", "Name": name})
        for name in opt_result["outgoing_drivers"]:
            transfer_rows.append({"Move": "OUT", "Type": "Driver", "Name": name})
        for name in opt_result["incoming_constructors"]:
            transfer_rows.append({"Move": "IN", "Type": "Constructor", "Name": name})
        for name in opt_result["outgoing_constructors"]:
            transfer_rows.append({"Move": "OUT", "Type": "Constructor", "Name": name})

        transfer_summary = (
            f"{opt_result['transfer_count']} total transfers, "
            f"{opt_result['free_transfers']} free, "
            f"{opt_result['paid_transfers']} paid, "
            f"{opt_result['transfer_cost']:.1f} point cost."
        )
        st.caption(transfer_summary)
        if transfer_rows:
            st.dataframe(pd.DataFrame(transfer_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No changes needed from your current team.")

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
# TAB 3 — SEASON TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"{current_season} Fantasy Points Tracker")

    if driver_season_progress_df.empty:
        st.info("No current-season fantasy points are loaded yet.")
    else:
        latest_driver_totals = (
            driver_season_progress_df.sort_values(["driver_name", "round"])
            .groupby("driver_name", as_index=False)
            .tail(1)
            .sort_values("cumulative_fantasy_pts", ascending=False)
            .reset_index(drop=True)
        )
        default_driver_tracker = picked_drivers or latest_driver_totals["driver_name"].head(8).tolist()
        driver_tracker_options = latest_driver_totals["driver_name"].tolist()
        selected_driver_tracker = st.multiselect(
            "Drivers to track",
            options=driver_tracker_options,
            default=filter_defaults(default_driver_tracker, driver_tracker_options),
            key="season_tracker_drivers",
        )
        driver_view_mode = st.radio(
            "Driver chart view",
            ["Cumulative season points", "Per-race points"],
            horizontal=True,
            key="season_tracker_driver_view",
        )
        driver_value_col = (
            "cumulative_fantasy_pts"
            if driver_view_mode == "Cumulative season points"
            else "fantasy_pts"
        )
        driver_yaxis_title = (
            "Cumulative Fantasy Points"
            if driver_view_mode == "Cumulative season points"
            else "Fantasy Points"
        )

        driver_chart_df = driver_season_progress_df[
            driver_season_progress_df["driver_name"].isin(selected_driver_tracker)
        ].copy()
        tracker_col1, tracker_col2 = st.columns([1.25, 0.75])

        with tracker_col1:
            fig_driver_tracker = go.Figure()
            for driver_name in selected_driver_tracker:
                driver_rows = driver_chart_df[driver_chart_df["driver_name"] == driver_name]
                if driver_rows.empty:
                    continue
                team_name = driver_rows["team"].iloc[0]
                fig_driver_tracker.add_trace(go.Scatter(
                    x=driver_rows["race_label"],
                    y=driver_rows[driver_value_col],
                    mode="lines+markers",
                    name=driver_name,
                    line=dict(color=TEAM_COLORS.get(team_name, "#B0B7C3"), width=3),
                    marker=dict(size=8),
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
                ))
            apply_transparent_plot_layout(
                fig_driver_tracker,
                yaxis_title=driver_yaxis_title,
                height=380,
            )
            fig_driver_tracker.update_layout(legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_driver_tracker, use_container_width=True)

        with tracker_col2:
            st.markdown("**Current Driver Standings**")
            driver_totals_display = latest_driver_totals[[
                "driver_name", "team", "fantasy_pts", "cumulative_fantasy_pts"
            ]].copy()
            driver_totals_display.columns = ["Driver", "Team", "Latest Race", "Season Total"]
            st.dataframe(driver_totals_display, use_container_width=True, hide_index=True, height=380)

        st.markdown("**Race-by-Race Driver Points**")
        driver_race_display = driver_season_progress_df[[
            "round", "race_name", "driver_name", "team", "fantasy_pts", "cumulative_fantasy_pts"
        ]].copy()
        driver_race_display.columns = ["Round", "Race", "Driver", "Team", "Race Pts", "Season Total"]
        st.dataframe(driver_race_display, use_container_width=True, hide_index=True, height=280)

    st.markdown("---")
    st.subheader(f"{current_season} Constructor Fantasy Points")

    if constructor_season_progress_df.empty:
        st.info("No current-season constructor points are loaded yet.")
    else:
        latest_constructor_totals = (
            constructor_season_progress_df.sort_values(["constructor", "round"])
            .groupby("constructor", as_index=False)
            .tail(1)
            .sort_values("cumulative_fantasy_pts", ascending=False)
            .reset_index(drop=True)
        )
        default_constructor_tracker = picked_constructors or latest_constructor_totals["constructor"].head(6).tolist()
        constructor_tracker_options = latest_constructor_totals["constructor"].tolist()
        selected_constructor_tracker = st.multiselect(
            "Constructors to track",
            options=constructor_tracker_options,
            default=filter_defaults(default_constructor_tracker, constructor_tracker_options),
            key="season_tracker_constructors",
        )
        constructor_chart_df = constructor_season_progress_df[
            constructor_season_progress_df["constructor"].isin(selected_constructor_tracker)
        ].copy()

        constructor_col1, constructor_col2 = st.columns([1.25, 0.75])
        with constructor_col1:
            fig_constructor_tracker = go.Figure()
            for constructor_name in selected_constructor_tracker:
                constructor_rows = constructor_chart_df[
                    constructor_chart_df["constructor"] == constructor_name
                ]
                if constructor_rows.empty:
                    continue
                fig_constructor_tracker.add_trace(go.Scatter(
                    x=constructor_rows["race_label"],
                    y=constructor_rows["cumulative_fantasy_pts"],
                    mode="lines+markers",
                    name=constructor_name,
                    line=dict(color=TEAM_COLORS.get(constructor_name, "#B0B7C3"), width=3),
                    marker=dict(size=8),
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.1f}<extra></extra>",
                ))
            apply_transparent_plot_layout(
                fig_constructor_tracker,
                yaxis_title="Cumulative Fantasy Points",
                height=320,
            )
            fig_constructor_tracker.update_layout(legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_constructor_tracker, use_container_width=True)

        with constructor_col2:
            constructor_totals_display = latest_constructor_totals[[
                "constructor", "fantasy_pts", "cumulative_fantasy_pts"
            ]].copy()
            constructor_totals_display.columns = ["Constructor", "Latest Race", "Season Total"]
            st.dataframe(constructor_totals_display, use_container_width=True, hide_index=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    if run_bt and bt_seasons:
        if is_predictive and comparison_model_types:
            comparison_df, comparison_detail_df, comparison_source_df = cached_model_comparison(
                tuple(sorted(bt_seasons)),
                current_season,
                bt_circuit,
                tuple(comparison_model_types),
                comparison_model_configs_json,
                budget_cap,
                min_history_weekends,
                current_season_boost,
                season_decay,
            )
        else:
            comparison_df, comparison_detail_df, comparison_source_df = (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        if is_predictive and run_blend_sweep and blend_sweep_weights:
            blend_sweep_base_params = get_model_default_params(blend_sweep_model_type)
            for key, value in predictive_model_params.items():
                if key in blend_sweep_base_params:
                    blend_sweep_base_params[key] = value
            blend_sweep_df, blend_sweep_detail_df = cached_blend_weight_sweep(
                tuple(sorted(bt_seasons)),
                current_season,
                bt_circuit,
                blend_sweep_model_type,
                tuple(blend_sweep_weights),
                json.dumps(blend_sweep_base_params, sort_keys=True),
                budget_cap,
                min_history_weekends,
                current_season_boost,
                season_decay,
            )
        else:
            blend_sweep_df, blend_sweep_detail_df = pd.DataFrame(), pd.DataFrame()

        bt_df, bt_source_df = cached_backtest(
            model_mode, tuple(sorted(bt_seasons)), current_season, bt_circuit, use_bonus,
            budget_cap, w_recent, w_track, w_ppm, w_quali,
            predictive_model_type, predictive_model_params_json, min_history_weekends,
            current_season_boost, season_decay
        )
        if not bt_df.empty:
            if is_predictive and not comparison_df.empty:
                st.subheader("Model Comparison")
                best_efficiency_row = comparison_df.dropna(subset=["avg_efficiency_pct"]).sort_values(
                    "avg_efficiency_pct", ascending=False
                ).iloc[0]
                best_mae_row = comparison_df.dropna(subset=["avg_driver_mae"]).sort_values(
                    "avg_driver_mae", ascending=True
                ).iloc[0]
                st.info(
                    "Best lineup efficiency: "
                    f"{best_efficiency_row['model_label']} ({best_efficiency_row['avg_efficiency_pct']:.1f}%)"
                    + " | Best driver MAE: "
                    f"{best_mae_row['model_label']} ({best_mae_row['avg_driver_mae']:.2f})"
                )

                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    fig_comp_eff = go.Figure(go.Bar(
                        x=comparison_df["model_label"],
                        y=comparison_df["avg_efficiency_pct"],
                        marker_color=TEAM_COLORS["Ferrari"],
                        hovertemplate="%{x}<br>Avg efficiency: %{y:.1f}%<extra></extra>",
                    ))
                    apply_transparent_plot_layout(
                        fig_comp_eff,
                        yaxis_title="Avg Efficiency %",
                        height=300,
                    )
                    st.plotly_chart(fig_comp_eff, use_container_width=True)

                with comp_col2:
                    fig_comp_mae = go.Figure(go.Bar(
                        x=comparison_df["model_label"],
                        y=comparison_df["avg_driver_mae"],
                        marker_color=TEAM_COLORS["Mercedes"],
                        hovertemplate="%{x}<br>Avg driver MAE: %{y:.2f}<extra></extra>",
                    ))
                    apply_transparent_plot_layout(
                        fig_comp_mae,
                        yaxis_title="Avg Driver MAE",
                        height=300,
                    )
                    st.plotly_chart(fig_comp_mae, use_container_width=True)

                comparison_display = comparison_df[[
                    "model_label", "races_evaluated", "avg_driver_mae", "avg_actual_pts",
                    "avg_oracle_pts", "avg_efficiency_pct", "avg_train_rows",
                    "avg_sample_weight", "model_params",
                ]].copy()
                comparison_display.columns = [
                    "Model", "Races", "Avg Driver MAE", "Avg Actual Pts",
                    "Avg Oracle Pts", "Avg Efficiency %", "Avg Train Rows",
                    "Avg Train Weight", "Params",
                ]
                comparison_display["Params"] = comparison_display["Params"].map(format_model_params_text)
                st.dataframe(comparison_display, use_container_width=True, hide_index=True)

                if not comparison_detail_df.empty:
                    last_season = int(comparison_detail_df["season"].max())
                    last_season_df = comparison_detail_df[
                        comparison_detail_df["season"] == last_season
                    ].copy().sort_values(["round", "race_name", "model_label"])
                    last_season_df["race_label"] = last_season_df.apply(
                        lambda row: f"R{int(row['round'])} {row['race_name']}",
                        axis=1,
                    )

                    st.subheader(f"Last Season Race-by-Race ({last_season})")
                    race_col1, race_col2 = st.columns(2)

                    with race_col1:
                        fig_last_eff = go.Figure()
                        for model_label in comparison_display["Model"]:
                            model_rows = last_season_df[last_season_df["model_label"] == model_label]
                            if model_rows.empty:
                                continue
                            fig_last_eff.add_trace(go.Scatter(
                                x=model_rows["race_label"],
                                y=model_rows["efficiency_pct"],
                                mode="lines+markers",
                                name=model_label,
                                hovertemplate="%{x}<br>%{fullData.name} efficiency: %{y:.1f}%<extra></extra>",
                            ))
                        apply_transparent_plot_layout(
                            fig_last_eff,
                            yaxis_title="Efficiency %",
                            height=360,
                        )
                        fig_last_eff.update_layout(xaxis_tickangle=-35, legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig_last_eff, use_container_width=True)

                    with race_col2:
                        fig_last_mae = go.Figure()
                        for model_label in comparison_display["Model"]:
                            model_rows = last_season_df[last_season_df["model_label"] == model_label]
                            if model_rows.empty:
                                continue
                            fig_last_mae.add_trace(go.Scatter(
                                x=model_rows["race_label"],
                                y=model_rows["driver_mae"],
                                mode="lines+markers",
                                name=model_label,
                                hovertemplate="%{x}<br>%{fullData.name} driver MAE: %{y:.2f}<extra></extra>",
                            ))
                        apply_transparent_plot_layout(
                            fig_last_mae,
                            yaxis_title="Driver MAE",
                            height=360,
                        )
                        fig_last_mae.update_layout(xaxis_tickangle=-35, legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig_last_mae, use_container_width=True)

                    season_winner_df = last_season_df.loc[
                        last_season_df.groupby("race_label")["efficiency_pct"].idxmax()
                    ][["race_label", "model_label", "efficiency_pct"]].copy()
                    season_winner_df.columns = ["Race", "Best Model", "Best Efficiency %"]
                    st.dataframe(season_winner_df, use_container_width=True, hide_index=True)

                    if "baseline" in comparison_detail_df["model_type"].unique():
                        baseline_last_season_df = last_season_df[
                            last_season_df["model_type"] == "baseline"
                        ][["race_label", "efficiency_pct", "driver_mae"]].rename(
                            columns={
                                "efficiency_pct": "baseline_efficiency_pct",
                                "driver_mae": "baseline_driver_mae",
                            }
                        )
                        vs_baseline_df = last_season_df.merge(
                            baseline_last_season_df,
                            on="race_label",
                            how="left",
                        )
                        vs_baseline_df = vs_baseline_df[
                            vs_baseline_df["model_type"] != "baseline"
                        ].copy()
                        vs_baseline_df["efficiency_delta_pct"] = (
                            vs_baseline_df["efficiency_pct"] - vs_baseline_df["baseline_efficiency_pct"]
                        )
                        vs_baseline_df["driver_mae_delta"] = (
                            vs_baseline_df["driver_mae"] - vs_baseline_df["baseline_driver_mae"]
                        )

                        if not vs_baseline_df.empty:
                            st.subheader("Against Baseline")
                            delta_col1, delta_col2 = st.columns(2)

                            with delta_col1:
                                fig_vs_base_eff = go.Figure()
                                for model_label in sorted(vs_baseline_df["model_label"].unique()):
                                    model_rows = vs_baseline_df[vs_baseline_df["model_label"] == model_label]
                                    fig_vs_base_eff.add_trace(go.Scatter(
                                        x=model_rows["race_label"],
                                        y=model_rows["efficiency_delta_pct"],
                                        mode="lines+markers",
                                        name=model_label,
                                        hovertemplate="%{x}<br>%{fullData.name} vs baseline: %{y:.1f} pts<extra></extra>",
                                    ))
                                fig_vs_base_eff.add_hline(y=0.0, line_dash="dash", line_color="gray")
                                apply_transparent_plot_layout(
                                    fig_vs_base_eff,
                                    yaxis_title="Efficiency Delta vs Baseline",
                                    height=340,
                                )
                                fig_vs_base_eff.update_layout(xaxis_tickangle=-35, legend=dict(orientation="h", y=1.15))
                                st.plotly_chart(fig_vs_base_eff, use_container_width=True)

                            with delta_col2:
                                fig_vs_base_mae = go.Figure()
                                for model_label in sorted(vs_baseline_df["model_label"].unique()):
                                    model_rows = vs_baseline_df[vs_baseline_df["model_label"] == model_label]
                                    fig_vs_base_mae.add_trace(go.Scatter(
                                        x=model_rows["race_label"],
                                        y=model_rows["driver_mae_delta"],
                                        mode="lines+markers",
                                        name=model_label,
                                        hovertemplate="%{x}<br>%{fullData.name} MAE delta: %{y:.2f}<extra></extra>",
                                    ))
                                fig_vs_base_mae.add_hline(y=0.0, line_dash="dash", line_color="gray")
                                apply_transparent_plot_layout(
                                    fig_vs_base_mae,
                                    yaxis_title="Driver MAE Delta vs Baseline",
                                    height=340,
                                )
                                fig_vs_base_mae.update_layout(xaxis_tickangle=-35, legend=dict(orientation="h", y=1.15))
                                st.plotly_chart(fig_vs_base_mae, use_container_width=True)

                            baseline_summary_df = (
                                vs_baseline_df.groupby("model_label")
                                .agg(
                                    avg_efficiency_delta_pct=("efficiency_delta_pct", "mean"),
                                    avg_driver_mae_delta=("driver_mae_delta", "mean"),
                                    races_beating_baseline_eff=("efficiency_delta_pct", lambda s: int((s > 0).sum())),
                                    races_beating_baseline_mae=("driver_mae_delta", lambda s: int((s < 0).sum())),
                                )
                                .reset_index()
                                .sort_values("avg_efficiency_delta_pct", ascending=False)
                            )
                            baseline_summary_df.columns = [
                                "Model",
                                "Avg Efficiency Delta",
                                "Avg MAE Delta",
                                "Eff Wins vs Baseline",
                                "MAE Wins vs Baseline",
                            ]
                            st.dataframe(baseline_summary_df, use_container_width=True, hide_index=True)

                if not comparison_source_df.empty:
                    st.caption("Comparison source seasons: " + " | ".join(
                        f"{int(r['season'])}:{r['source']} ({int(r['weekends_loaded'])} races)"
                        for _, r in comparison_source_df.iterrows()
                    ))

                if not blend_sweep_df.empty:
                    st.subheader("Blend Weight Sweep")
                    best_efficiency_weight = blend_sweep_df.sort_values(
                        ["avg_efficiency_pct", "avg_driver_mae"],
                        ascending=[False, True],
                    ).iloc[0]
                    best_mae_weight = blend_sweep_df.sort_values(
                        ["avg_driver_mae", "avg_efficiency_pct"],
                        ascending=[True, False],
                    ).iloc[0]
                    st.info(
                        f"{MODEL_LABELS[blend_sweep_model_type]} best avg efficiency at "
                        f"`{best_efficiency_weight['blend_weight']:.2f}`"
                        f" ({best_efficiency_weight['avg_efficiency_pct']:.2f}%)"
                        + " | best avg MAE at "
                        f"`{best_mae_weight['blend_weight']:.2f}`"
                        f" ({best_mae_weight['avg_driver_mae']:.3f})"
                    )

                    sweep_col1, sweep_col2 = st.columns(2)
                    with sweep_col1:
                        fig_sweep_eff = go.Figure(go.Scatter(
                            x=blend_sweep_df["blend_weight"],
                            y=blend_sweep_df["avg_efficiency_pct"],
                            mode="lines+markers",
                            name="Avg Efficiency %",
                            hovertemplate="weight=%{x:.2f}<br>avg efficiency=%{y:.2f}%<extra></extra>",
                        ))
                        apply_transparent_plot_layout(
                            fig_sweep_eff,
                            xaxis_title="Blend Weight",
                            yaxis_title="Avg Efficiency %",
                            height=320,
                        )
                        st.plotly_chart(fig_sweep_eff, use_container_width=True)

                    with sweep_col2:
                        fig_sweep_mae = go.Figure(go.Scatter(
                            x=blend_sweep_df["blend_weight"],
                            y=blend_sweep_df["avg_driver_mae"],
                            mode="lines+markers",
                            name="Avg Driver MAE",
                            hovertemplate="weight=%{x:.2f}<br>avg driver MAE=%{y:.3f}<extra></extra>",
                        ))
                        apply_transparent_plot_layout(
                            fig_sweep_mae,
                            xaxis_title="Blend Weight",
                            yaxis_title="Avg Driver MAE",
                            height=320,
                        )
                        st.plotly_chart(fig_sweep_mae, use_container_width=True)

                    sweep_display = blend_sweep_df.copy()
                    sweep_display.columns = [
                        "Blend Weight",
                        "Races",
                        "Avg Efficiency %",
                        "Median Efficiency %",
                        "Avg Driver MAE",
                        "Median Driver MAE",
                    ]
                    st.dataframe(sweep_display, use_container_width=True, hide_index=True)

                    if not blend_sweep_detail_df.empty:
                        last_sweep_season = int(blend_sweep_detail_df["season"].max())
                        last_sweep_df = blend_sweep_detail_df[
                            blend_sweep_detail_df["season"] == last_sweep_season
                        ].copy().sort_values(["round", "blend_weight"])
                        last_sweep_df["race_label"] = last_sweep_df.apply(
                            lambda row: f"R{int(row['round'])} {row['race_name']}",
                            axis=1,
                        )
                        fig_sweep_race = go.Figure()
                        for blend_weight in sorted(last_sweep_df["blend_weight"].unique()):
                            weight_rows = last_sweep_df[last_sweep_df["blend_weight"] == blend_weight]
                            fig_sweep_race.add_trace(go.Scatter(
                                x=weight_rows["race_label"],
                                y=weight_rows["efficiency_pct"],
                                mode="lines+markers",
                                name=f"w={blend_weight:.2f}",
                                hovertemplate="%{x}<br>%{fullData.name} efficiency=%{y:.1f}%<extra></extra>",
                            ))
                        apply_transparent_plot_layout(
                            fig_sweep_race,
                            yaxis_title=f"{last_sweep_season} Efficiency %",
                            height=360,
                        )
                        fig_sweep_race.update_layout(xaxis_tickangle=-35, legend=dict(orientation="h", y=1.15))
                        st.plotly_chart(fig_sweep_race, use_container_width=True)

                st.markdown("---")

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
                st.caption(
                    f"Selected predictive model: {selected_model_label} "
                    f"({format_model_params_text(json.dumps(selected_model_params, sort_keys=True))})"
                )

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
# TAB 5 — SIGNAL EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
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

        explore_cols = [
            "recent_form_5",
            "era_recent_form",
            "era_track_history",
            "season_recent_form",
            "season_quali_form",
        ]
        explore_labels = ["Recent 5", "Era Form", "Era Track", "2026 Form", "2026 Quali"]
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
        if trained_model["importance_display"] == "coefficients":
            coef_df = trained_model["coefficients_df"][["label", "coefficient"]].copy()
            fig_coef = go.Figure(go.Bar(
                x=coef_df["coefficient"],
                y=coef_df["label"],
                orientation="h",
                marker_color=["#2ca02c" if x >= 0 else "#d62728" for x in coef_df["coefficient"]],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            importance_xaxis_title = "Standardized coefficient"
        else:
            coef_df = trained_model["feature_importances_df"][["label", "importance"]].copy()
            fig_coef = go.Figure(go.Bar(
                x=coef_df["importance"],
                y=coef_df["label"],
                orientation="h",
                marker_color=TEAM_COLORS["McLaren"],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ))
            importance_xaxis_title = "Feature importance"
        apply_transparent_plot_layout(
            fig_coef,
            xaxis_title=importance_xaxis_title,
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
                predictive_model_type, predictive_model_params_json,
                min_history_weekends, current_season_boost, season_decay
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
# TAB 6 — DATA TABLES
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
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
            "recent_form_3", "recent_form_5", "era_recent_form",
            "track_history", "era_track_history", "season_avg", "season_recent_form",
            "season_quali_form", "quali_form_5", "dnf_rate_5", "team_recent_form_3",
            "team_season_avg", "current_season_starts", "track_experience",
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
# TAB 7 — HOW IT WORKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader("Model Summary")
    if is_predictive:
        st.markdown(f"""
        This app is using a **{selected_model_label}** next-race fantasy points prediction model.

        It works in five steps:
        1. Load historical race weekends for the selected training seasons.
        2. Convert every historical weekend into fantasy points using the scoring engine.
        3. Build one training row per driver per race using only information available **before** that race.
        4. Fit the selected predictive model family on the engineered driver-race feature set.
        5. Predict the current grid, optionally apply expert adjustments, and optimise the best 5-driver / 2-constructor team under the budget cap.
        6. If transfer modelling is enabled, apply the free-transfer allowance and subtract 10 points per extra move from the optimizer objective.
        """)
    else:
        st.markdown(f"""
        This app is a transparent **heuristic ranking model** rather than a black-box ML model.

        It works in four steps:
        1. Load historical race weekends for the selected training seasons.
        2. Convert each weekend into fantasy points using the scoring engine.
        3. Build four driver signals: recent form, track history, season points-per-million, and qualifying form.
        4. Z-score those signals across the grid, apply the chosen weights, add optional testing bonuses, and optimise the best 5-driver / 2-constructor team under the budget cap.
        5. If transfer modelling is enabled, apply the free-transfer allowance and subtract 10 points per extra move from the optimizer objective.
        """)

    st.markdown("**Where the data comes from**")
    st.markdown("""
    - Historical race and qualifying data is pulled from the Jolpica Ergast-compatible API when available.
    - Sprint results and fastest-lap flags are also pulled when the upstream endpoint provides them.
    - Downloaded seasons are cached locally and reused on future runs.
    - If the API is unavailable, the app falls back to embedded seed data included in this project.
    - The season source summary in the `Data Tables` tab shows exactly which seasons came from `api`, `cache`, or `seed`.
    - Current driver prices, constructor prices, and manual FP/testing bonuses come from the project files and constants in this repo.
    """)

    st.markdown("**2026 scoring coverage**")
    st.markdown("""
    - The scoring engine now applies 2026 qualifying, sprint, race, and constructor rules for any fields present in the weekend data.
    - This includes Q2/Q3 constructor bonuses, qualifying no-time penalties, sprint scoring, race fastest lap, and constructor disqualification penalties.
    - Overtakes, Driver of the Day, and pit-stop bonuses are supported as optional fields in the data model, but they are usually unavailable from the cached Jolpica weekend data in this repo.
    - When those optional fields are missing, the model currently treats them as zero rather than inventing estimates.
    """)

    if is_predictive:
        st.markdown("**Model fit diagnostics**")
        metrics_df = pd.DataFrame([trained_model["train_metrics"]])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.markdown("**Active model settings**")
        st.dataframe(
            pd.DataFrame(
                [{"model": selected_model_label, "params": format_model_params_text(json.dumps(selected_model_params, sort_keys=True))}]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Season weighting rules**")
        st.markdown(
            f"- Current-season rows are boosted by **{diagnostics['current_season_boost']:.2f}x**.\n"
            f"- Older seasons decay by **{diagnostics['season_decay']:.2f}** per year of distance.\n"
            "- This means once a new 2026 race is loaded, its examples carry more influence than 2025, 2024, and earlier rows."
        )

        st.markdown("**Model formula**")
        if trained_model["importance_display"] == "coefficients":
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
        else:
            st.code("predicted_points = model(feature_vector) + expert_adjustment")

            if trained_model.get("component_summary_df") is not None:
                st.markdown("**Blend components**")
                st.dataframe(
                    trained_model["component_summary_df"],
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("**Learned feature importance**")
            st.dataframe(trained_model["feature_importances_df"][["label", "importance", "abs_importance"]],
                         use_container_width=True, hide_index=True)

            st.markdown("**Driver-level feature snapshot**")
            explain_driver = st.selectbox("Inspect prediction inputs", options=d_scores["driver"].tolist(),
                                          key="explain_driver")
            explain_cols = ["driver", "team", "predicted_points", "base_predicted_points",
                            "testing_bonus"] + feature_columns
            explain_row = d_scores[d_scores["driver"] == explain_driver][explain_cols]
            st.dataframe(explain_row, use_container_width=True, hide_index=True)
            st.caption("Per-feature prediction contributions are only available for the linear model families.")

        st.markdown("**Notes**")
        st.markdown(
            "- The model is trained on historical driver-level examples rather than manually chosen weights.\n"
            "- Rule changes are handled by season-aware sample weighting, so newer seasons matter more than older ones.\n"
            "- Backtests use an expanding-window approach so each race is predicted only from earlier races.\n"
            "- Tree and baseline models show global feature importance instead of signed linear coefficients.\n"
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
