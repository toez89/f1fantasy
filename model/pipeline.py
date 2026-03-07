"""
Shared pipeline helpers for the CLI and Streamlit dashboard.

These helpers keep data loading and model orchestration in one place so the
dashboard can stay focused on presentation and inspection.
"""

import json
from pathlib import Path

import pandas as pd

from backtest import (
    CURRENT_CONSTRUCTOR_COSTS,
    CURRENT_DRIVER_COSTS,
    TESTING_BONUSES,
    compute_all_weekends,
)
from fetch_data import build_weekend_records
from optimizer import optimise_team
from predictive_model import (
    FEATURE_COLUMNS,
    add_training_sample_weights,
    build_constructor_prediction_df,
    build_current_feature_frame,
    build_training_examples,
    fit_ridge_model,
    predict_points,
    run_predictive_backtest,
)
from seed_data import get_seeded_weekends
from value_model import build_history_df, score_all


CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

DEFAULT_TRAIN_SEASONS = [2022, 2023, 2024, 2025]

HISTORY_COLUMNS = [
    "season",
    "round",
    "race_name",
    "circuit",
    "driver_name",
    "team",
    "quali_pos",
    "race_pos",
    "dnf",
    "fantasy_pts",
]


def _empty_history_df() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_or_fetch_season(season: int) -> tuple[list[dict], str]:
    """
    Load a season from cache when available, otherwise fetch from the API.
    Falls back to embedded seed data when offline.
    """
    cache = CACHE_DIR / f"weekends_{season}.json"

    if cache.exists():
        weekends = json.loads(cache.read_text())
        return compute_all_weekends(weekends), "cache"

    try:
        weekends = build_weekend_records(season)
        cache.write_text(json.dumps(weekends))
        return compute_all_weekends(weekends), "api"
    except Exception:
        weekends = [w for w in get_seeded_weekends() if w["season"] == season]
        return compute_all_weekends(weekends), "seed"


def load_training_dataset(
    train_seasons: list[int],
    current_season: int | None = None,
    include_current_season: bool = True,
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """
    Load all requested race weekends and return:
      1. flattened driver-race history,
      2. raw weekend records,
      3. a per-season source summary.
    """
    requested_seasons = sorted(set(train_seasons))
    if include_current_season and current_season is not None:
        requested_seasons = sorted(set(requested_seasons + [current_season]))

    all_weekends: list[dict] = []
    source_rows: list[dict] = []

    for season in requested_seasons:
        weekends, source = load_or_fetch_season(season)
        all_weekends.extend(weekends)
        source_rows.append(
            {
                "season": season,
                "source": source,
                "weekends_loaded": len(weekends),
            }
        )

    history_df = build_history_df(all_weekends) if all_weekends else _empty_history_df()
    source_df = pd.DataFrame(source_rows)
    return history_df, all_weekends, source_df


def build_constructor_scores(
    driver_scores_df: pd.DataFrame,
    driver_team_map: dict[str, str],
    constructor_costs: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate constructor scores from the scored drivers."""
    costs = constructor_costs or CURRENT_CONSTRUCTOR_COSTS
    rows = []

    for constructor, cost_m in costs.items():
        team_drivers = driver_scores_df[
            driver_scores_df["driver"].map(driver_team_map) == constructor
        ]
        rows.append(
            {
                "constructor": constructor,
                "norm_score": float(team_drivers["norm_score"].sum()),
                "cost_m": cost_m,
            }
        )

    return pd.DataFrame(rows)


def run_optimisation_pipeline(
    train_seasons: list[int],
    current_season: int,
    target_circuit: str,
    weights: dict[str, float],
    driver_team_map: dict[str, str],
    budget_m: float = 100.0,
    use_testing_bonus: bool = True,
    include_current_season: bool = True,
    locked_drivers: list[str] | None = None,
    excluded: list[str] | None = None,
    driver_costs: dict[str, float] | None = None,
    constructor_costs: dict[str, float] | None = None,
) -> dict:
    """
    Execute the full model pipeline and return everything the dashboard needs
    to explain the result.
    """
    resolved_driver_costs = driver_costs or CURRENT_DRIVER_COSTS
    resolved_constructor_costs = constructor_costs or CURRENT_CONSTRUCTOR_COSTS

    total_weight = sum(weights.values())
    if total_weight <= 0:
        normalised_weights = {
            "recent_form": 0.35,
            "track_history": 0.30,
            "season_ppm": 0.15,
            "quali_form": 0.20,
        }
    else:
        normalised_weights = {k: v / total_weight for k, v in weights.items()}

    history_df, weekends, source_df = load_training_dataset(
        train_seasons=train_seasons,
        current_season=current_season,
        include_current_season=include_current_season,
    )
    bonuses = TESTING_BONUSES if use_testing_bonus else {}

    driver_scores_df = score_all(
        df=history_df,
        driver_costs=resolved_driver_costs,
        constructor_costs=resolved_constructor_costs,
        circuit=target_circuit,
        current_season=current_season,
        weights=normalised_weights,
        testing_bonuses=bonuses,
    )
    driver_scores_df["team"] = driver_scores_df["driver"].map(driver_team_map)

    constructor_scores_df = build_constructor_scores(
        driver_scores_df=driver_scores_df,
        driver_team_map=driver_team_map,
        constructor_costs=resolved_constructor_costs,
    )

    result = optimise_team(
        driver_scores_df,
        constructor_scores_df,
        budget_m=budget_m,
        locked_drivers=list(locked_drivers or []),
        excluded=list(excluded or []),
    )

    diagnostics = {
        "season_ppm_active_drivers": int((driver_scores_df["season_ppm"] > 0).sum()),
        "track_history_non_null_drivers": int(
            driver_scores_df["track_history"].notna().sum()
        ),
        "training_records": int(len(history_df)),
        "training_weekends": int(len(weekends)),
    }

    return {
        "history_df": history_df,
        "weekends": weekends,
        "source_df": source_df,
        "driver_scores_df": driver_scores_df,
        "constructor_scores_df": constructor_scores_df,
        "result": result,
        "weights": normalised_weights,
        "bonuses": bonuses,
        "diagnostics": diagnostics,
    }


def run_predictive_pipeline(
    train_seasons: list[int],
    current_season: int,
    target_circuit: str,
    driver_team_map: dict[str, str],
    budget_m: float = 100.0,
    use_testing_bonus: bool = True,
    include_current_season: bool = True,
    locked_drivers: list[str] | None = None,
    excluded: list[str] | None = None,
    driver_costs: dict[str, float] | None = None,
    constructor_costs: dict[str, float] | None = None,
    alpha: float = 5.0,
    min_history_weekends: int = 5,
    current_season_boost: float = 3.0,
    season_decay: float = 0.8,
) -> dict:
    """
    Train a next-race fantasy points model and predict the current grid.
    """
    resolved_driver_costs = driver_costs or CURRENT_DRIVER_COSTS
    resolved_constructor_costs = constructor_costs or CURRENT_CONSTRUCTOR_COSTS
    history_df, weekends, source_df = load_training_dataset(
        train_seasons=train_seasons,
        current_season=current_season,
        include_current_season=include_current_season,
    )

    training_examples_df = build_training_examples(
        weekends,
        min_history_weekends=min_history_weekends,
    )
    training_examples_df = add_training_sample_weights(
        training_examples_df,
        target_season=current_season,
        current_season_boost=current_season_boost,
        season_decay=season_decay,
    )
    model = fit_ridge_model(training_examples_df, alpha=alpha)

    current_feature_df = build_current_feature_frame(
        history_df=history_df,
        current_season=current_season,
        target_circuit=target_circuit,
        driver_team_map=driver_team_map,
        driver_costs=resolved_driver_costs,
    )
    expert_adjustments = TESTING_BONUSES if use_testing_bonus else {}
    driver_scores_df = predict_points(
        model,
        current_feature_df,
        expert_adjustments=expert_adjustments,
    )
    constructor_scores_df = build_constructor_prediction_df(
        driver_scores_df,
        resolved_constructor_costs,
    )

    result = optimise_team(
        driver_scores_df,
        constructor_scores_df,
        budget_m=budget_m,
        locked_drivers=list(locked_drivers or []),
        excluded=list(excluded or []),
    )

    diagnostics = {
        "training_records": int(len(history_df)),
        "training_weekends": int(len(weekends)),
        "training_examples": int(len(training_examples_df)),
        "feature_count": len(FEATURE_COLUMNS),
        "current_season_loaded_rows": int((history_df["season"] == current_season).sum())
        if not history_df.empty else 0,
        "current_season_boost": current_season_boost,
        "season_decay": season_decay,
    }

    return {
        "history_df": history_df,
        "weekends": weekends,
        "source_df": source_df,
        "training_examples_df": training_examples_df,
        "driver_scores_df": driver_scores_df,
        "constructor_scores_df": constructor_scores_df,
        "result": result,
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "expert_adjustments": expert_adjustments,
        "diagnostics": diagnostics,
    }


def run_predictive_backtest_pipeline(
    seasons: list[int],
    current_season: int,
    target_circuit: str,
    include_current_season: bool = False,
    budget_m: float = 100.0,
    alpha: float = 5.0,
    min_history_weekends: int = 5,
    current_season_boost: float = 3.0,
    season_decay: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load seasons for predictive backtesting and return:
      1. backtest results
      2. source summary
    """
    _, weekends, source_df = load_training_dataset(
        train_seasons=seasons,
        current_season=current_season,
        include_current_season=include_current_season,
    )
    bt_df = run_predictive_backtest(
        weekends=weekends,
        target_circuit=target_circuit,
        driver_costs=CURRENT_DRIVER_COSTS,
        constructor_costs=CURRENT_CONSTRUCTOR_COSTS,
        budget_m=budget_m,
        alpha=alpha,
        min_history_weekends=min_history_weekends,
        current_season_boost=current_season_boost,
        season_decay=season_decay,
    )
    if not bt_df.empty:
        bt_df = bt_df.copy()
        bt_df["efficiency_pct"] = bt_df["efficiency"] * 100
    return bt_df, source_df


def build_weekend_summary_df(weekends: list[dict]) -> pd.DataFrame:
    """Create a compact race-weekend summary table for dashboard inspection."""
    if not weekends:
        return pd.DataFrame(
            columns=["season", "round", "race_name", "circuit", "drivers", "scored"]
        )

    rows = []
    for weekend in weekends:
        driver_count = len(weekend.get("drivers", []))
        rows.append(
            {
                "season": weekend["season"],
                "round": weekend["round"],
                "race_name": weekend["race_name"],
                "circuit": weekend["circuit"],
                "drivers": driver_count,
                "scored": "scores" in weekend,
            }
        )

    return pd.DataFrame(rows).sort_values(["season", "round"]).reset_index(drop=True)
