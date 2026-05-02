"""
Predictive next-race fantasy points model.

This module builds race-by-race training examples, fits a simple ridge-style
linear regression, and predicts fantasy points for the next race weekend.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet

from optimizer import optimise_team
from value_model import build_history_df


FEATURE_COLUMNS = [
    "recent_form_3",
    "recent_form_5",
    "era_recent_form",
    "track_history",
    "era_track_history",
    "season_avg",
    "season_recent_form",
    "season_quali_form",
    "quali_form_5",
    "dnf_rate_5",
    "team_recent_form_3",
    "team_season_avg",
    "current_season_starts",
    "track_experience",
    "overall_experience",
]

FEATURE_LABELS = {
    "recent_form_3": "Recent Form (3)",
    "recent_form_5": "Recent Form (5)",
    "era_recent_form": "Era Recent Form",
    "track_history": "Track History",
    "era_track_history": "Era Track History",
    "season_avg": "Season Avg",
    "season_recent_form": "Current Season Form",
    "season_quali_form": "Current Season Qualifying",
    "quali_form_5": "Qualifying Form",
    "dnf_rate_5": "DNF Rate",
    "team_recent_form_3": "Team Recent Form",
    "team_season_avg": "Team Season Avg",
    "current_season_starts": "Current Season Starts",
    "track_experience": "Track Experience",
    "overall_experience": "Overall Experience",
}

MODEL_LABELS = {
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "baseline": "Naive Baseline",
    "blend_baseline_elastic_net": "Blend: Baseline + Elastic Net",
    "blend_baseline_gradient_boosting": "Blend: Baseline + Gradient Boosting",
}

LINEAR_MODEL_TYPES = {"ridge", "elastic_net"}
TREE_MODEL_TYPES = {"random_forest", "gradient_boosting"}
BLEND_MODEL_TYPES = {
    "blend_baseline_elastic_net",
    "blend_baseline_gradient_boosting",
}
SUPPORTED_MODEL_TYPES = tuple(MODEL_LABELS.keys())
DEFAULT_COMPARISON_MODELS = (
    "baseline",
    "elastic_net",
    "gradient_boosting",
    "blend_baseline_elastic_net",
    "blend_baseline_gradient_boosting",
)
DEFAULT_MODEL_PARAMS = {
    "ridge": {"alpha": 7.5},
    "elastic_net": {"alpha": 0.1, "l1_ratio": 0.2, "max_iter": 10000},
    "random_forest": {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 2},
    "gradient_boosting": {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 1,
        "min_samples_leaf": 2,
    },
    "baseline": {"baseline_feature": "season_avg"},
    "blend_baseline_elastic_net": {
        "blend_weight": 0.5,
        "baseline_feature": "season_avg",
        "alpha": 0.1,
        "l1_ratio": 0.2,
        "max_iter": 10000,
    },
    "blend_baseline_gradient_boosting": {
        "blend_weight": 0.5,
        "baseline_feature": "season_avg",
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 1,
        "min_samples_leaf": 2,
    },
}

CURRENT_SEASON_FEATURE_WEIGHT = 3.0
ERA_SEASON_DECAY = 0.45


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


def _season_weights(
    df: pd.DataFrame,
    target_season: int,
    current_season_weight: float = CURRENT_SEASON_FEATURE_WEIGHT,
    season_decay: float = ERA_SEASON_DECAY,
) -> np.ndarray:
    if df.empty:
        return np.array([], dtype=float)
    season_gap = (target_season - df["season"]).clip(lower=0).to_numpy(dtype=float)
    base_weight = np.where(
        season_gap == 0.0,
        current_season_weight,
        np.power(season_decay, season_gap),
    )
    round_scale = 1.0 + ((df["round"].to_numpy(dtype=float) - 1.0) / 100.0)
    return base_weight * round_scale


def _weighted_mean_by_season(
    df: pd.DataFrame,
    value_column: str,
    target_season: int,
    default: float = 0.0,
) -> float:
    if df.empty:
        return default
    values = df[value_column].to_numpy(dtype=float)
    weights = _season_weights(df, target_season)
    if len(weights) == 0 or np.isclose(weights.sum(), 0.0):
        return default
    return float(np.average(values, weights=weights))


def _weighted_quali_form(
    df: pd.DataFrame,
    target_season: int,
    default: float = 10.0,
) -> float:
    quali_rows = df[df["quali_pos"].notna()]
    if quali_rows.empty:
        return default
    inverted_quali = 21.0 - quali_rows["quali_pos"].to_numpy(dtype=float)
    weights = _season_weights(quali_rows, target_season)
    if len(weights) == 0 or np.isclose(weights.sum(), 0.0):
        return default
    return float(max(0.0, np.average(inverted_quali, weights=weights)))


def _resolve_model_type(model_type: str) -> str:
    resolved = (model_type or "ridge").strip().lower()
    if resolved not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Expected one of {SUPPORTED_MODEL_TYPES}."
        )
    return resolved


def get_model_default_params(model_type: str) -> dict[str, Any]:
    resolved_model_type = _resolve_model_type(model_type)
    return dict(DEFAULT_MODEL_PARAMS.get(resolved_model_type, {}))


def _compute_regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    sample_weights: np.ndarray,
) -> dict[str, float]:
    residuals = actual - predicted
    mae = float(np.average(np.abs(residuals), weights=sample_weights))
    rmse = float(math.sqrt(np.average(np.square(residuals), weights=sample_weights)))
    y_mean = float(np.average(actual, weights=sample_weights))
    total_var = float(np.sum(sample_weights * np.square(actual - y_mean)))
    weighted_resid_var = float(np.sum(sample_weights * np.square(residuals)))
    r2 = 0.0 if total_var == 0 else float(1.0 - (weighted_resid_var / total_var))
    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 3),
    }


def _build_linear_summary_df(features: list[str], coefficients: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)


def _build_importance_df(features: list[str], importances: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": features,
            "label": [FEATURE_LABELS.get(f, f) for f in features],
            "importance": importances,
            "abs_importance": np.abs(importances),
        }
    ).sort_values("abs_importance", ascending=False)


def _extract_feature_importance_array(model: dict, features: list[str]) -> np.ndarray:
    if model.get("coefficients_df") is not None:
        coefficients_df = model["coefficients_df"].set_index("feature")
        return coefficients_df.reindex(features)["abs_coefficient"].fillna(0.0).to_numpy(dtype=float)
    if model.get("feature_importances_df") is not None:
        importances_df = model["feature_importances_df"].set_index("feature")
        value_column = "abs_importance" if "abs_importance" in importances_df.columns else "importance"
        return importances_df.reindex(features)[value_column].fillna(0.0).to_numpy(dtype=float)
    return np.zeros(len(features), dtype=float)


def _resolve_blend_components(model_type: str) -> tuple[str, str]:
    if model_type == "blend_baseline_elastic_net":
        return "baseline", "elastic_net"
    if model_type == "blend_baseline_gradient_boosting":
        return "baseline", "gradient_boosting"
    raise ValueError(f"Unsupported blend model_type '{model_type}'.")


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
    season_recent_rows = season_rows.head(3)
    era_recent_rows = driver_rows.head(8)
    era_track_rows = track_rows.head(6)
    team_season_rows = _sorted_history(
        history_df[(history_df["team"] == team) & (history_df["season"] == season)]
    ).head(6)

    recent_form_3 = _mean(recent_3["fantasy_pts"])
    recent_form_5 = _mean(recent_5["fantasy_pts"])
    track_history = _mean(track_rows["fantasy_pts"], default=recent_form_5)
    season_avg = _mean(season_rows["fantasy_pts"])
    era_recent_form = _weighted_mean_by_season(
        era_recent_rows,
        "fantasy_pts",
        target_season=season,
        default=recent_form_5,
    )
    era_track_history = _weighted_mean_by_season(
        era_track_rows,
        "fantasy_pts",
        target_season=season,
        default=recent_form_5,
    )
    season_recent_form = _mean(
        season_recent_rows["fantasy_pts"],
        default=era_recent_form,
    )
    quali_form_5 = _safe_quali_form(recent_5)
    season_quali_form = _weighted_quali_form(
        season_rows.head(5),
        target_season=season,
        default=quali_form_5,
    )
    dnf_rate_5 = _mean(recent_5["dnf"].astype(float))
    team_recent_form_3 = _mean(team_rows["fantasy_pts"], default=recent_form_5)
    team_season_avg = _mean(team_season_rows["fantasy_pts"], default=team_recent_form_3)
    current_season_starts = float(min(len(season_rows), 5))
    track_experience = float(min(len(track_rows), 5))
    overall_experience = float(min(len(driver_rows), 20))

    return {
        "recent_form": round(recent_form_5, 3),
        "recent_form_3": round(recent_form_3, 3),
        "recent_form_5": round(recent_form_5, 3),
        "era_recent_form": round(era_recent_form, 3),
        "track_history": round(track_history, 3),
        "era_track_history": round(era_track_history, 3),
        "quali_form": round(quali_form_5, 3),
        "season_avg": round(season_avg, 3),
        "season_recent_form": round(season_recent_form, 3),
        "season_quali_form": round(season_quali_form, 3),
        "quali_form_5": round(quali_form_5, 3),
        "dnf_rate_5": round(dnf_rate_5, 3),
        "team_recent_form_3": round(team_recent_form_3, 3),
        "team_season_avg": round(team_season_avg, 3),
        "current_season_starts": round(current_season_starts, 3),
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


def fit_predictive_model(
    training_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_type: str = "ridge",
    model_params: dict[str, Any] | None = None,
    sample_weight_column: str = "sample_weight",
) -> dict:
    """Fit one of the supported predictive models on the engineered feature set."""
    features = feature_columns or FEATURE_COLUMNS
    resolved_model_type = _resolve_model_type(model_type)
    params = dict(model_params or {})
    if training_df.empty:
        raise ValueError("Training data is empty; cannot fit predictive model.")

    x = training_df[features].fillna(0.0).to_numpy(dtype=float)
    y = training_df["target_fantasy_points"].to_numpy(dtype=float)
    if sample_weight_column in training_df.columns:
        sample_weights = (
            training_df[sample_weight_column].fillna(1.0).to_numpy(dtype=float)
        )
    else:
        sample_weights = np.ones(len(training_df), dtype=float)

    train_metrics = {
        "rows": int(len(training_df)),
        "avg_sample_weight": round(float(sample_weights.mean()), 3),
        "max_sample_weight": round(float(sample_weights.max()), 3),
    }
    model_label = MODEL_LABELS[resolved_model_type]

    if resolved_model_type in BLEND_MODEL_TYPES:
        baseline_type, secondary_type = _resolve_blend_components(resolved_model_type)
        blend_weight = float(params.get("blend_weight", 0.5))
        blend_weight = min(max(blend_weight, 0.0), 1.0)

        baseline_params = {"baseline_feature": params.get("baseline_feature", "season_avg")}
        secondary_params = {
            key: value
            for key, value in params.items()
            if key not in {"blend_weight", "baseline_feature"}
        }
        baseline_model = fit_predictive_model(
            training_df=training_df,
            feature_columns=features,
            model_type=baseline_type,
            model_params=baseline_params,
            sample_weight_column=sample_weight_column,
        )
        secondary_model = fit_predictive_model(
            training_df=training_df,
            feature_columns=features,
            model_type=secondary_type,
            model_params=secondary_params,
            sample_weight_column=sample_weight_column,
        )
        baseline_fitted, _ = _predict_base_points(baseline_model, training_df[features])
        secondary_fitted, _ = _predict_base_points(secondary_model, training_df[features])
        fitted = ((1.0 - blend_weight) * baseline_fitted) + (blend_weight * secondary_fitted)
        train_metrics.update(_compute_regression_metrics(y, fitted, sample_weights))
        combined_importance = (
            (1.0 - blend_weight) * _extract_feature_importance_array(baseline_model, features)
            + blend_weight * _extract_feature_importance_array(secondary_model, features)
        )
        component_summary_df = pd.DataFrame(
            [
                {
                    "component": baseline_model["model_label"],
                    "weight": round(1.0 - blend_weight, 3),
                    "params": str(baseline_params),
                },
                {
                    "component": secondary_model["model_label"],
                    "weight": round(blend_weight, 3),
                    "params": str(secondary_params),
                },
            ]
        )
        return {
            "model_type": resolved_model_type,
            "model_label": model_label,
            "feature_columns": features,
            "feature_labels": [FEATURE_LABELS.get(f, f) for f in features],
            "coefficients_df": None,
            "feature_importances_df": _build_importance_df(features, combined_importance),
            "importance_display": "feature_importance",
            "explainability": "blend",
            "component_models": [baseline_model, secondary_model],
            "component_summary_df": component_summary_df,
            "blend_weight": blend_weight,
            "model_params": params,
            "train_metrics": train_metrics,
        }

    if resolved_model_type in LINEAR_MODEL_TYPES:
        means = _weighted_mean(x, sample_weights)
        stds = _weighted_std(x, sample_weights, means)
        stds = np.where(stds == 0, 1.0, stds)
        x_scaled = (x - means) / stds

        if resolved_model_type == "ridge":
            alpha = float(params.get("alpha", 5.0))
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
            estimator = None
        else:
            alpha = float(params.get("alpha", 0.5))
            l1_ratio = float(params.get("l1_ratio", 0.5))
            max_iter = int(params.get("max_iter", 10000))
            estimator = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=max_iter,
                random_state=42,
            )
            estimator.fit(x_scaled, y, sample_weight=sample_weights)
            intercept = float(estimator.intercept_)
            coefficients = estimator.coef_.astype(float)
            fitted = estimator.predict(x_scaled)

        train_metrics.update(_compute_regression_metrics(y, fitted, sample_weights))
        coefficients_df = _build_linear_summary_df(features, coefficients)
        model = {
            "model_type": resolved_model_type,
            "model_label": model_label,
            "feature_columns": features,
            "feature_labels": [FEATURE_LABELS.get(f, f) for f in features],
            "means": means,
            "stds": stds,
            "intercept": intercept,
            "coefficients": coefficients,
            "coefficients_df": coefficients_df,
            "feature_importances_df": None,
            "importance_display": "coefficients",
            "explainability": "linear",
            "estimator": estimator,
            "model_params": params,
            "train_metrics": train_metrics,
        }
        if resolved_model_type == "ridge":
            model["alpha"] = alpha
        else:
            model["alpha"] = alpha
            model["l1_ratio"] = l1_ratio
        return model

    if resolved_model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=42,
        )
        estimator.fit(x, y, sample_weight=sample_weights)
        fitted = estimator.predict(x)
        importances = estimator.feature_importances_.astype(float)
        train_metrics.update(_compute_regression_metrics(y, fitted, sample_weights))
        return {
            "model_type": resolved_model_type,
            "model_label": model_label,
            "feature_columns": features,
            "feature_labels": [FEATURE_LABELS.get(f, f) for f in features],
            "coefficients_df": None,
            "feature_importances_df": _build_importance_df(features, importances),
            "importance_display": "feature_importance",
            "explainability": "tree",
            "estimator": estimator,
            "model_params": params,
            "train_metrics": train_metrics,
        }

    if resolved_model_type == "gradient_boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 200)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            random_state=42,
        )
        estimator.fit(x, y, sample_weight=sample_weights)
        fitted = estimator.predict(x)
        importances = estimator.feature_importances_.astype(float)
        train_metrics.update(_compute_regression_metrics(y, fitted, sample_weights))
        return {
            "model_type": resolved_model_type,
            "model_label": model_label,
            "feature_columns": features,
            "feature_labels": [FEATURE_LABELS.get(f, f) for f in features],
            "coefficients_df": None,
            "feature_importances_df": _build_importance_df(features, importances),
            "importance_display": "feature_importance",
            "explainability": "tree",
            "estimator": estimator,
            "model_params": params,
            "train_metrics": train_metrics,
        }

    baseline_feature = params.get("baseline_feature", "recent_form_5")
    if baseline_feature not in features:
        raise ValueError(
            f"Unsupported baseline_feature '{baseline_feature}'. Expected one of {features}."
        )
    fitted = training_df[baseline_feature].fillna(0.0).to_numpy(dtype=float)
    train_metrics.update(_compute_regression_metrics(y, fitted, sample_weights))
    return {
        "model_type": resolved_model_type,
        "model_label": model_label,
        "feature_columns": features,
        "feature_labels": [FEATURE_LABELS.get(f, f) for f in features],
        "coefficients_df": None,
        "feature_importances_df": _build_importance_df(
            features,
            np.array(
                [1.0 if feature == baseline_feature else 0.0 for feature in features],
                dtype=float,
            ),
        ),
        "importance_display": "feature_importance",
        "explainability": "baseline",
        "estimator": None,
        "baseline_feature": baseline_feature,
        "model_params": params,
        "train_metrics": train_metrics,
    }


def fit_ridge_model(
    training_df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    alpha: float = 5.0,
    sample_weight_column: str = "sample_weight",
) -> dict:
    """Fit a small ridge-style linear regression with closed-form solution."""
    return fit_predictive_model(
        training_df=training_df,
        feature_columns=feature_columns,
        model_type="ridge",
        model_params={"alpha": alpha},
        sample_weight_column=sample_weight_column,
    )


def _predict_base_points(
    model: dict,
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    features = model["feature_columns"]
    x = feature_df[features].fillna(0.0).to_numpy(dtype=float)

    if model["explainability"] == "linear":
        x_scaled = (x - model["means"]) / model["stds"]
        base_predictions = model["intercept"] + x_scaled @ model["coefficients"]
        contributions = {
            f"{feature}_contrib": x_scaled[:, idx] * model["coefficients"][idx]
            for idx, feature in enumerate(features)
        }
        return base_predictions, contributions

    if model["explainability"] == "tree":
        return model["estimator"].predict(x), {}

    if model["explainability"] == "blend":
        baseline_predictions, _ = _predict_base_points(model["component_models"][0], feature_df)
        secondary_predictions, _ = _predict_base_points(model["component_models"][1], feature_df)
        blend_weight = float(model.get("blend_weight", 0.5))
        return ((1.0 - blend_weight) * baseline_predictions) + (blend_weight * secondary_predictions), {}

    baseline_feature = model.get("baseline_feature", "recent_form_5")
    return (
        feature_df[baseline_feature].fillna(0.0).to_numpy(dtype=float),
        {},
    )


def predict_points(
    model: dict,
    feature_df: pd.DataFrame,
    expert_adjustments: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Predict driver fantasy points and expose per-feature contributions."""
    base_predictions, contributions = _predict_base_points(model, feature_df)
    adjustment_map = expert_adjustments or {}

    predicted_df = feature_df.copy()
    predicted_df["model_type"] = model["model_type"]
    predicted_df["model_label"] = model["model_label"]
    predicted_df["base_predicted_points"] = np.round(base_predictions, 3)
    predicted_df["expert_adjustment"] = predicted_df["driver"].map(adjustment_map).fillna(0.0)
    predicted_df["testing_bonus"] = predicted_df["expert_adjustment"]
    predicted_df["predicted_points"] = np.round(
        predicted_df["base_predicted_points"] + predicted_df["expert_adjustment"],
        3,
    )
    predicted_df["norm_score"] = predicted_df["predicted_points"]
    predicted_df["value_score"] = predicted_df["predicted_points"] / predicted_df["cost_m"]

    for column, values in contributions.items():
        predicted_df[column] = np.round(values, 3)

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
    model_type: str = "ridge",
    model_params: dict[str, Any] | None = None,
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
        model = fit_predictive_model(
            training_examples,
            model_type=model_type,
            model_params=model_params,
        )
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
                "model_type": model["model_type"],
                "model_label": model["model_label"],
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
