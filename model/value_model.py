"""
F1 Fantasy Value Model
======================
Produces a predicted fantasy-points score for each driver/constructor
for an upcoming race weekend.

Scoring factors (all configurable via weights dict):
  1. recent_form        – average fantasy pts over last N races (season-level)
  2. track_history      – average fantasy pts at THIS circuit historically
  3. season_ppm         – season-to-date points-per-million (value metric)
  4. quali_form         – average qualifying position (recent races, inverted)
  5. testing_bonus      – manual adjustment for 2026 pre-season / FP data

The model blends these signals using configurable weights, then runs
a linear-programming optimizer to select the best team within the budget.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Default model weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = {
    "recent_form"   : 0.35,   # last-N-races average pts
    "track_history" : 0.30,   # avg pts at this specific circuit
    "season_ppm"    : 0.15,   # season pts-per-million
    "quali_form"    : 0.20,   # average qualifying position (inverted)
}

RECENT_N = 5          # number of recent races to use for form
CIRCUIT_ALIAS = {
    # Map common short names → ergast circuit names
    "australia" : ["Albert Park Grand Prix Circuit"],
    "bahrain"   : ["Bahrain International Circuit"],
    "saudi"     : ["Jeddah Street Circuit"],
    "imola"     : ["Autodromo Enzo e Dino Ferrari"],
    "monaco"    : ["Circuit de Monaco"],
    "spain"     : ["Circuit de Barcelona-Catalunya"],
    "canada"    : ["Circuit Gilles Villeneuve"],
    "austria"   : ["Red Bull Ring"],
    "britain"   : ["Silverstone Circuit"],
    "hungary"   : ["Hungaroring"],
    "belgium"   : ["Circuit de Spa-Francorchamps"],
    "netherlands": ["Circuit Zandvoort"],
    "italy"     : ["Autodromo Nazionale di Monza"],
    "singapore" : ["Marina Bay Street Circuit"],
    "japan"     : ["Suzuka International Racing Course"],
    "qatar"     : ["Losail International Circuit"],
    "usa"       : ["Circuit of the Americas"],
    "mexico"    : ["Autodromo Hermanos Rodriguez"],
    "brazil"    : ["Autodromo Jose Carlos Pace"],
    "vegas"     : ["Las Vegas Strip Street Circuit"],
    "abu_dhabi" : ["Yas Marina Circuit"],
}


# ---------------------------------------------------------------------------
# Build historical dataframe from weekend records
# ---------------------------------------------------------------------------

def build_history_df(weekend_records: list[dict]) -> pd.DataFrame:
    """
    Flatten a list of build_weekend_records() dicts + pre-computed
    fantasy scores into a tidy DataFrame.

    Parameters
    ----------
    weekend_records : output of backtest.compute_all_weekends()
                      Each record has a 'scores' key added by the backtester.

    Returns
    -------
    DataFrame with columns:
      season, round, race_name, circuit, driver_name, team,
      quali_pos, race_pos, dnf, fantasy_pts
    """
    rows = []
    for w in weekend_records:
        scores = w.get("scores", {})
        for d in w["drivers"]:
            rows.append({
                "season"      : w["season"],
                "round"       : w["round"],
                "race_name"   : w["race_name"],
                "circuit"     : w["circuit"],
                "driver_name" : d["driver_name"],
                "team"        : d["team"],
                "quali_pos"   : d.get("quali_pos"),
                "race_pos"    : d.get("race_pos"),
                "dnf"         : d.get("dnf", False),
                "fantasy_pts" : scores.get(d["driver_name"], 0.0),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Individual signal calculators
# ---------------------------------------------------------------------------

def calc_recent_form(df: pd.DataFrame, driver: str, n: int = RECENT_N) -> float:
    """Average fantasy points over the last n races for this driver."""
    sub = df[df["driver_name"] == driver].sort_values(
        ["season", "round"], ascending=False
    ).head(n)
    return float(sub["fantasy_pts"].mean()) if len(sub) > 0 else 0.0


def calc_track_history(df: pd.DataFrame, driver: str, circuit: str) -> float:
    """Average fantasy points at this specific circuit."""
    sub = df[(df["driver_name"] == driver) & (df["circuit"] == circuit)]
    return float(sub["fantasy_pts"].mean()) if len(sub) > 0 else np.nan


def calc_season_ppm(df: pd.DataFrame, driver: str,
                    cost_m: float, current_season: int) -> float:
    """Season-to-date points per million for the current season."""
    sub = df[(df["driver_name"] == driver) & (df["season"] == current_season)]
    total_pts = float(sub["fantasy_pts"].sum())
    return total_pts / cost_m if cost_m > 0 else 0.0


def calc_quali_form(df: pd.DataFrame, driver: str, n: int = RECENT_N) -> float:
    """
    Average qualifying position over last n races, inverted so that
    P1 = 20, P20 = 1 (higher = better).
    """
    sub = df[(df["driver_name"] == driver) & df["quali_pos"].notna()].sort_values(
        ["season", "round"], ascending=False
    ).head(n)
    if len(sub) == 0:
        return 10.0   # neutral midfield assumption
    avg_pos = float(sub["quali_pos"].mean())
    return max(0.0, 21.0 - avg_pos)   # invert: P1 → 20, P20 → 1


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_driver(
    df           : pd.DataFrame,
    driver       : str,
    cost_m       : float,
    circuit      : str,
    current_season: int,
    weights      : dict | None = None,
    testing_bonus: float = 0.0,
) -> dict:
    """
    Compute a composite value score for one driver.

    Returns a dict with individual signal values and the final composite score.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    recent     = calc_recent_form(df, driver)
    track      = calc_track_history(df, driver, circuit)
    ppm        = calc_season_ppm(df, driver, cost_m, current_season)
    quali      = calc_quali_form(df, driver)

    # If no track history, fall back to recent form
    if np.isnan(track):
        track = recent

    # Normalise signals to a common scale (z-score across all drivers done
    # outside this function; here we just collect raw values)
    composite = (
        w["recent_form"]   * recent  +
        w["track_history"] * track   +
        w["season_ppm"]    * ppm     +
        w["quali_form"]    * quali   +
        testing_bonus
    )

    return {
        "driver"        : driver,
        "cost_m"        : cost_m,
        "recent_form"   : round(recent, 2),
        "track_history" : round(track, 2),
        "season_ppm"    : round(ppm, 2),
        "quali_form"    : round(quali, 2),
        "testing_bonus" : testing_bonus,
        "raw_score"     : round(composite, 4),
    }


def score_all(
    df             : pd.DataFrame,
    driver_costs   : dict[str, float],
    constructor_costs: dict[str, float],
    circuit        : str,
    current_season : int,
    weights        : dict | None = None,
    testing_bonuses: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Score all drivers and constructors.

    Returns a DataFrame sorted by normalised value score (score / cost).
    """
    bonuses = testing_bonuses or {}
    rows = []
    for driver, cost in driver_costs.items():
        r = score_driver(df, driver, cost, circuit, current_season,
                         weights=weights, testing_bonus=bonuses.get(driver, 0.0))
        rows.append(r)

    scores_df = pd.DataFrame(rows)

    # Z-score normalise each signal across all drivers for fair comparison
    for col in ["recent_form", "track_history", "season_ppm", "quali_form"]:
        mu  = scores_df[col].mean()
        std = scores_df[col].std()
        scores_df[f"{col}_z"] = (scores_df[col] - mu) / std if std > 0 else 0.0

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    scores_df["norm_score"] = (
        w["recent_form"]   * scores_df["recent_form_z"]   +
        w["track_history"] * scores_df["track_history_z"] +
        w["season_ppm"]    * scores_df["season_ppm_z"]    +
        w["quali_form"]    * scores_df["quali_form_z"]    +
        scores_df["testing_bonus"]
    )

    scores_df["value_score"] = scores_df["norm_score"] / scores_df["cost_m"]
    scores_df = scores_df.sort_values("norm_score", ascending=False).reset_index(drop=True)
    scores_df["rank"] = scores_df.index + 1

    # ---- Constructors ----
    # Constructor score = sum of driver scores for that team
    # Build a team → driver mapping from current driver list
    team_scores: dict[str, float] = {}
    team_map_sample = {}   # not available here; constructors scored separately
    # Simple approach: use average of available driver norms per team
    # (optimizer handles this separately)

    return scores_df
