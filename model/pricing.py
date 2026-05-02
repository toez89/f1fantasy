"""
CSV-backed price snapshot helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).parent.parent
DRIVER_COST_HISTORY_PATH = ROOT_DIR / "driver_cost_history.csv"
CONSTRUCTOR_COST_HISTORY_PATH = ROOT_DIR / "constructor_cost_history.csv"
BUDGET_CAP_HISTORY_PATH = ROOT_DIR / "budget_cap_history.csv"
DEFAULT_SNAPSHOT_LABEL = "Miami GP costs"


def _load_cost_history(path: Path, name_column: str) -> pd.DataFrame:
    if not path.exists():
        columns = ["snapshot_label", name_column, "cost_m"]
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(path)
    required_columns = ["snapshot_label", name_column, "cost_m"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in {path.name}")
    df["snapshot_label"] = df["snapshot_label"].astype(str).str.strip()
    df[name_column] = df[name_column].astype(str).str.strip()
    df["cost_m"] = df["cost_m"].astype(float)
    return df[required_columns].copy()


def load_driver_cost_history() -> pd.DataFrame:
    return _load_cost_history(DRIVER_COST_HISTORY_PATH, "driver")


def load_constructor_cost_history() -> pd.DataFrame:
    return _load_cost_history(CONSTRUCTOR_COST_HISTORY_PATH, "constructor")


def load_budget_cap_history() -> pd.DataFrame:
    if not BUDGET_CAP_HISTORY_PATH.exists():
        return pd.DataFrame(columns=["snapshot_label", "budget_cap_m"])

    df = pd.read_csv(BUDGET_CAP_HISTORY_PATH)
    required_columns = ["snapshot_label", "budget_cap_m"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in {BUDGET_CAP_HISTORY_PATH.name}")
    df["snapshot_label"] = df["snapshot_label"].astype(str).str.strip()
    df["budget_cap_m"] = df["budget_cap_m"].astype(float)
    return df[required_columns].copy()


def get_available_cost_snapshots() -> list[str]:
    driver_df = load_driver_cost_history()
    constructor_df = load_constructor_cost_history()
    budget_df = load_budget_cap_history()
    driver_labels = set(driver_df["snapshot_label"].tolist())
    constructor_labels = set(constructor_df["snapshot_label"].tolist())
    budget_labels = set(budget_df["snapshot_label"].tolist())
    shared_labels = driver_labels & constructor_labels & budget_labels

    ordered_labels: list[str] = []
    for label in (
        driver_df["snapshot_label"].tolist()
        + constructor_df["snapshot_label"].tolist()
        + budget_df["snapshot_label"].tolist()
    ):
        if label in shared_labels and label not in ordered_labels:
            ordered_labels.append(label)

    if ordered_labels:
        return ordered_labels

    fallback_seen: list[str] = []
    for label in (
        driver_df["snapshot_label"].tolist()
        + constructor_df["snapshot_label"].tolist()
        + budget_df["snapshot_label"].tolist()
    ):
        if label not in fallback_seen:
            fallback_seen.append(label)
    return fallback_seen or [DEFAULT_SNAPSHOT_LABEL]


def get_default_cost_snapshot_label() -> str:
    snapshots = get_available_cost_snapshots()
    return snapshots[-1]


def load_cost_snapshot(snapshot_label: str | None = None) -> tuple[dict[str, float], dict[str, float], float, str]:
    resolved_label = snapshot_label or get_default_cost_snapshot_label()
    driver_df = load_driver_cost_history()
    constructor_df = load_constructor_cost_history()
    budget_df = load_budget_cap_history()

    driver_rows = driver_df[driver_df["snapshot_label"] == resolved_label]
    constructor_rows = constructor_df[constructor_df["snapshot_label"] == resolved_label]
    budget_rows = budget_df[budget_df["snapshot_label"] == resolved_label]

    if driver_rows.empty or constructor_rows.empty or budget_rows.empty:
        fallback_label = get_default_cost_snapshot_label()
        driver_rows = driver_df[driver_df["snapshot_label"] == fallback_label]
        constructor_rows = constructor_df[constructor_df["snapshot_label"] == fallback_label]
        budget_rows = budget_df[budget_df["snapshot_label"] == fallback_label]
        resolved_label = fallback_label

    driver_costs = dict(zip(driver_rows["driver"], driver_rows["cost_m"]))
    constructor_costs = dict(zip(constructor_rows["constructor"], constructor_rows["cost_m"]))
    budget_cap = float(budget_rows["budget_cap_m"].iloc[-1]) if not budget_rows.empty else 100.0
    return driver_costs, constructor_costs, budget_cap, resolved_label
