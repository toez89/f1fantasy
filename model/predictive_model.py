"""
Predictive next-race fantasy points model.

This module builds race-by-race training examples, fits a simple ridge-style
linear regression, and predicts fantasy points for the next race weekend.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from optimizer import optimise_team
from value_model import build_history_df


FEATURE_COLUMNS = [
    "recent_form_3",
    "recent_form_5",
    "track_history",
    "season_avg",
    "quali_form_5",
    "dnf_rate_5",
    "team_recent_form_3",
    "track_experience",
    "overall_experience",
]

FEATURE_LABELS = {
    "recent_form_3": "Recent Form (3)",
    "recent_form_5": "Recent Form (5)",
    "track_history": "Track History",
    "season_avg": "Season Avg",
    "quali_form_5": "Qualifying Form",
    "dnf_rate_5": "DNF Rate",
    "team_recent_form_3": "Team Recent Form",
    "track_experience": "Track Experience",
    "overall_experience": "Overall Experience",
}


def _sorted_history(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["season", "round"], ascending=False)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.average(values, axis=0, weights=weights)


def _weighted_std(values: np.ndarray, weights: np.ndarray, means: np.ndarray) -> np.ndarray:
    variance = np.average((values - means) ** 2, axis=0, weights=weights)
    return np.sqrt(variance)


def _mean(series: pd.Series, default: float = 0.0) -> float:
    if series.empty:
        return default
    return float(series.mean())


def _safe_quali_form(df: pd.DataFrame, default: float = 10.0) -> float:
    quali = df["quali_pos"].dropna()
    if quali.empty:
        return default
    return float(max(0.0, 21.0 - quali.mean()))


def _feature_row(
    history_df: pd.DataFrame,
    driver: str,
    team: str,
    circuit: str,
    season: int,
) -> dict:
    driver_rows = _sorted_history(history_df[history_df["driver_name"] == driver])
    recent_3 = driver_rows.head(3)
    recent_5 = driver_rows.head(5)
    track_rows = _sorted_history(driver_rows[driver_rows["circuit"] == circuit])
    season_rows = _sorted_history(driver_rows[driver_rows["season"] == season])
    team_rows = _sorted_history(history_df[history_df["team"] == team]).head(6)

    recent_form_3 = _mean(recent_3["fantasy_pts"])
    recent_form_5 = _mean(recent_5["fantasy_pts"])
    track_history = _mean(track_rows["fantasy_pts"], default=recent_form_5)
    season_avg = _mean(season_rows["fantasy_pts"])
    quali_form_5 = _safe_quali_form(recent_5)
    dnf_rate_5 = _mean(recent_5["dnf"].astype(float))
    team_recent_form_3 = _mean(team_rows["fantasy_pts"], default=recent_form_5)
    track_experience = float(min(len(track_rows), 5))
    overall_experience = float(min(len(driver_rows), 20))

    return {
        "recent_form": round(recent_form_5, 3),
        "recent_form_3": round(recent_form_3, 3),
        "recent_form_5": round(recent_form_5, 3),
        "track_history": round(track_history, 3),
        "quali_form": round(quali_form_5, 3),
        "season_avg": round(season_avg, 3),
        "quali_form_5": round(quali_form_5, 3),
        "dnf_rate_5": round(dnf_rate_5, 3),
        "team_recent_form_3": round(team_recent_form_3, 3),
        "track_experience": round(track_experience, 3),
        "overall_experience": round(overall_experience, 3),
    }


def build_training_examples(
    weekends: list[dict],
    min_history_weekends: int = 5,
) -> pd.DataFrame:
    """Build point-prediction examples using only prior-weekend information."""
    rows = []
    ordered_weekends = sorted(weekends, key=lambda w: (w["season"], w["round"]))

    for idx, target_weekend in enumerate(ordered_weekends):
        if idx < min_history_weekends:
            continue

        prior_history = build_history_df(ordered_weekends[:idx])
        if prior_history.empty:
            continue

        actual_scores = target_weekend.get("scores", {})
        for driver in target_weekend["drivers"]:
            row = _feature_row(
                prior_history,
                driver=driver["driver_name"],
                team=driver["team"],
                circuit=target_weekend["circuit"],
                season=target_weekend["season"],
            )
            row.update(
                {
                    "season": target_weekend["season"],
                    "round": target_weekend["round"],
                    "race_name": target_weekend["race_name"],
                    "circuit": target_weekend["circuit"],
                    "driver": driver["driver_name"],
                    "team": driver["team"],
                    "target_fantasy_points": float(
                        actual_scores.get(driver["driver_name"], 0.0)
                    ),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def add_training_sample_weights(
    training_df: pd.DataFrame,
    target_season: int,
    current_season_boost: float = 3.0,
    season_decay: float = 0.8,
) -> pd.DataFrame:
    """
    Weight newer-rule seasons more heavily than older seasons.

    - current season rows get `current_season_boost`
    - older seasons decay geometrically with `season_decay`
    """
    weighted_df = training_df.copy()
    season_gap = (target_season - weighted_df["season"]).clip(lower=0)
    base_weight = np.where(
        season_gap == 0,
        current_season_boost,
        np.power(season_decay, season_gap),
    )

    # Later rounds in the same season are slightly more relevant than early rounds.
    round_scale = 1.0 + ((weighted_df["round"] - 1) / 100.0)

    weighted_df["season_gap"] = season_gap.astype(int)
    weighted_df["sample_weight"] = np.round(base_weight * round_scale, 4)
    return weighted_df


def fit_ridge_model(
    training_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    alpha: float = 5.0,
    sample_weight_column: str = "sample_weight",
) -> dict:
    """Fit a small ridge-style linear regression with closed-form solution."""
    features = feature_columns or FEATURE_COLUMNS
    if training_df.empty:
        raise ValueError("Training data is empty; cannot fit predictive model.")

    x = training_df[features].fillna(0.0).to_numpy(dtype=float)
    y = training_df["target_fantasy_points"].to_numpy(dtype=float)
    if sample_weight_column in training_df.columns:
        sample_weights = training_df[sample_weight_column].fillna(1.0).to_numpy(dtype=float)
    else:
        sample_weights = np.ones(len(training_df), dtype=float)

    means = _weighted_mean(x, sample_weights)
    stds = _weighted_std(x, sample_weights, means)
    stds = np.where(stds == 0, 1.0, stds)

    x_scaled = (x - means) / stds
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    sqrt_w = np.sqrt(sample_weights)
    weighted_design = design * sqrt_w[:, None]
    weighted_y = y * sqrt_w

    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0

    beta = (
        np.linalg.pinv(weighted_design.T @ weighted_design + alpha * penalty)
        @ weighted_design.T
        @ weighted_y
    )
    intercept = float(beta[0])
    coefficients = beta[1:]
    fitted = design @ beta

    residuals = y - fitted
    mae = float(np.average(np.abs(residuals), weights=sample_weights))
    rmse = float(math.sqrt(np.average(np.square(residuals), weights=sample_weights)))
    y_mean = float(np.average(y, weights=sample_weights))
    total_var = float(np.sum(sample_weights * np.square(y - y_mean)))
    weighted_resid_var = float(np.sum(sample_weights * np.square(residuals)))
    r2 = 0.0 if total_var == 0 else float(1.0 - (weighted_resid_var / total_var))

    coefficients_df = pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)

    return {
        "feature_columns": features,
        "means": means,
        "stds": stds,
        "intercept": intercept,
        "coefficients": coefficients,
        "coefficients_df": coefficients_df,
        "alpha": alpha,
        "train_metrics": {
            "rows": int(len(training_df)),
            "avg_sample_weight": round(float(sample_weights.mean()), 3),
            "max_sample_weight": round(float(sample_weights.max()), 3),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
        },
    }


def predict_points(
    model: dict,
    feature_df: pd.DataFrame,
    expert_adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict driver fantasy points and expose per-feature contributions."""
    features = model["feature_columns"]
    x = feature_df[features].fillna(0.0).to_numpy(dtype=float)
    x_scaled = (x - model["means"]) / model["stds"]

    base_predictions = model["intercept"] + x_scaled @ model["coefficients"]
    adjustment_map = expert_adjustments or {}

    predicted_df = feature_df.copy()
    predicted_df["base_predicted_points"] = np.round(base_predictions, 3)
    predicted_df["expert_adjustment"] = predicted_df["driver"].map(adjustment_map).fillna(0.0)
    predicted_df["testing_bonus"] = predicted_df["expert_adjustment"]
    predicted_df["predicted_points"] = np.round(
        predicted_df["base_predicted_points"] + predicted_df["expert_adjustment"],
        3,
    )
    predicted_df["norm_score"] = predicted_df["predicted_points"]
    predicted_df["value_score"] = predicted_df["predicted_points"] / predicted_df["cost_m"]

    for idx, feature in enumerate(features):
        predicted_df[f"{feature}_contrib"] = np.round(
            x_scaled[:, idx] * model["coefficients"][idx], 3
        )

    predicted_df["rank"] = (
        predicted_df["predicted_points"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    predicted_df = predicted_df.sort_values("predicted_points", ascending=False).reset_index(drop=True)
    predicted_df["rank"] = predicted_df.index + 1
    return predicted_df


def build_current_feature_frame(
    history_df: pd.DataFrame,
    current_season: int,
    target_circuit: str,
    driver_team_map: dict[str, str],
    driver_costs: dict[str, float],
) -> pd.DataFrame:
    """Build one row of predictive features per current driver."""
    rows = []
    for driver, cost_m in driver_costs.items():
        team = driver_team_map.get(driver, "Unknown")
        row = _feature_row(
            history_df,
            driver=driver,
            team=team,
            circuit=target_circuit,
            season=current_season,
        )
        row.update(
            {
                "driver": driver,
                "team": team,
                "cost_m": float(cost_m),
                "target_circuit": target_circuit,
                "current_season": current_season,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_constructor_prediction_df(
    driver_predictions_df: pd.DataFrame,
    constructor_costs: dict[str, float],
) -> pd.DataFrame:
    """Convert driver point predictions into constructor point predictions."""
    rows = []
    for constructor, cost_m in constructor_costs.items():
        team_drivers = driver_predictions_df[driver_predictions_df["team"] == constructor]
        predicted_points = float(team_drivers["predicted_points"].sum())
        rows.append(
            {
                "constructor": constructor,
                "predicted_points": round(predicted_points, 3),
                "norm_score": round(predicted_points, 3),
                "cost_m": float(cost_m),
            }
        )
    return pd.DataFrame(rows)


def run_predictive_backtest(
    weekends: list[dict],
    target_circuit: str,
    driver_costs: dict[str, float],
    constructor_costs: dict[str, float],
    budget_m: float = 100.0,
    alpha: float = 5.0,
    min_history_weekends: int = 5,
    current_season_boost: float = 3.0,
    season_decay: float = 0.8,
) -> pd.DataFrame:
    """Run an expanding-window backtest for the predictive points model."""
    rows = []
    ordered_weekends = sorted(weekends, key=lambda w: (w["season"], w["round"]))

    for idx, target_weekend in enumerate(ordered_weekends):
        circuit_match = (
            target_circuit.lower() == "all"
            or target_circuit.lower() in target_weekend["circuit"].lower()
            or target_circuit.lower() in target_weekend["race_name"].lower()
        )
        if not circuit_match or idx < min_history_weekends:
            continue

        train_weekends = ordered_weekends[:idx]
        training_examples = build_training_examples(
            train_weekends,
            min_history_weekends=min_history_weekends,
        )
        if training_examples.empty:
            continue

        training_examples = add_training_sample_weights(
            training_examples,
            target_season=target_weekend["season"],
            current_season_boost=current_season_boost,
            season_decay=season_decay,
        )
        model = fit_ridge_model(training_examples, alpha=alpha)
        history_df = build_history_df(train_weekends)

        race_driver_costs = {
            d["driver_name"]: driver_costs.get(d["driver_name"], 8.0)
            for d in target_weekend["drivers"]
        }
        race_constructor_costs = {}
        team_map = {}
        for driver in target_weekend["drivers"]:
            team_map[driver["driver_name"]] = driver["team"]
            race_constructor_costs.setdefault(
                driver["team"], constructor_costs.get(driver["team"], 10.0)
            )

        feature_df = build_current_feature_frame(
            history_df=history_df,
            current_season=target_weekend["season"],
            target_circuit=target_weekend["circuit"],
            driver_team_map=team_map,
            driver_costs=race_driver_costs,
        )
        predicted_driver_df = predict_points(model, feature_df)
        predicted_constructor_df = build_constructor_prediction_df(
            predicted_driver_df,
            race_constructor_costs,
        )

        selected_team = optimise_team(
            predicted_driver_df,
            predicted_constructor_df,
            budget_m=budget_m,
        )

        actual_scores = target_weekend.get("scores", {})
        actual_constructor_scores = target_weekend.get("constructor_scores", {})
        actual_driver_df = pd.DataFrame(
            [
                {
                    "driver": driver_name,
                    "norm_score": float(actual_scores.get(driver_name, 0.0)),
                    "cost_m": race_driver_costs.get(driver_name, 8.0),
                }
                for driver_name in race_driver_costs
            ]
        )
        actual_constructor_df = pd.DataFrame(
            [
                {
                    "constructor": constructor,
                    "norm_score": float(actual_constructor_scores.get(constructor, 0.0)),
                    "cost_m": cost_m,
                }
                for constructor, cost_m in race_constructor_costs.items()
            ]
        )
        oracle_team = optimise_team(
            actual_driver_df,
            actual_constructor_df,
            budget_m=budget_m,
        )

        picked_drivers = list(selected_team["drivers"]["driver"])
        picked_constructors = list(selected_team["constructors"]["constructor"])
        actual_team_points = (
            sum(actual_scores.get(name, 0.0) for name in picked_drivers)
            + sum(actual_constructor_scores.get(name, 0.0) for name in picked_constructors)
        )
        oracle_points = float(oracle_team["total_score"])

        compare_df = predicted_driver_df[["driver", "predicted_points"]].copy()
        compare_df["actual_points"] = compare_df["driver"].map(actual_scores).fillna(0.0)
        driver_mae = float(np.mean(np.abs(compare_df["predicted_points"] - compare_df["actual_points"])))

        rows.append(
            {
                "season": target_weekend["season"],
                "round": target_weekend["round"],
                "race_name": target_weekend["race_name"],
                "circuit": target_weekend["circuit"],
                "train_rows": int(len(training_examples)),
                "avg_sample_weight": round(float(training_examples["sample_weight"].mean()), 3),
                "driver_mae": round(driver_mae, 3),
                "model_cost": selected_team["total_cost"],
                "picked_drivers": ", ".join(picked_drivers),
                "picked_constructors": ", ".join(picked_constructors),
                "actual_pts": round(actual_team_points, 1),
                "oracle_pts": round(oracle_points, 1),
                "efficiency": round(actual_team_points / oracle_points, 3) if oracle_points > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)
