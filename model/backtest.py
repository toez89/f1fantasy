"""
F1 Fantasy Backtester
=====================
Runs the value model + optimizer on every race in historical seasons
and measures how the model's picks would have actually performed.

Usage:
    python backtest.py --seasons 2023 2024 --circuit australia
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fetch_data   import build_weekend_records
from seed_data    import get_seeded_weekends
from scoring      import compute_race_weekend, constructor_fantasy_points
from value_model  import build_history_df, score_all
from optimizer    import optimise_team, print_team


CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

CURRENT_DRIVER_COSTS = {
    "Max Verstappen"    : 27.7,
    "George Russell"    : 27.4,
    "Lando Norris"      : 27.2,
    "Oscar Piastri"     : 25.5,
    "Kimi Antonelli"    : 23.2,
    "Charles Leclerc"   : 22.8,
    "Lewis Hamilton"    : 22.5,
    "Isack Hadjar"      : 15.1,
    "Pierre Gasly"      : 12.0,
    "Carlos Sainz"      : 11.8,
    "Alexander Albon"   : 11.6,
    "Fernando Alonso"   : 10.0,
    "Lance Stroll"      :  8.0,
    "Oliver Bearman"    :  7.4,
    "Esteban Ocon"      :  7.3,
    "Nico Hulkenberg"   :  6.8,
    "Liam Lawson"       :  6.5,
    "Gabriel Bortoleto" :  6.4,
    "Arvid Lindblad"    :  6.2,
    "Franco Colapinto"  :  6.2,
    "Sergio Perez"      :  6.0,
    "Valtteri Bottas"   :  5.9,
}

CURRENT_CONSTRUCTOR_COSTS = {
    "Mercedes"     : 29.3,
    "McLaren"      : 28.9,
    "Red Bull Racing": 28.2,
    "Ferrari"      : 23.3,
    "Alpine"       : 12.5,
    "Williams"     : 12.0,
    "Aston Martin" : 10.3,
    "Haas F1 Team" : 7.4,
    "Audi"         : 6.6,
    "Racing Bulls" : 6.3,
    "Cadillac"     : 6.0,
}

# 2026 FP1/FP2 testing bonuses (manual adjustment based on practice sessions)
TESTING_BONUSES = {
    "Oscar Piastri"     :  1.5,   # FP2 P1 at home race
    "Kimi Antonelli"    :  1.2,   # FP2 P2, only 0.005s behind Russell in FP1
    "Charles Leclerc"   :  1.0,   # FP1 P1
    "George Russell"    :  0.8,   # Pre-season favourite, FP1 P3
    "Arvid Lindblad"    :  0.7,   # FP2 P8 as rookie, impressive pace
    "Nico Hulkenberg"   :  0.5,   # P4 FP2, 9 straight top-11 at Albert Park
    "Max Verstappen"    :  0.3,   # FP1 P3, consistent
    "Lando Norris"      :  0.3,   # FP2 P7
    "Oliver Bearman"    :  0.2,   # Haas solid pre-season testing
    "Lewis Hamilton"    :  0.2,   # FP1 P2
}


# ---------------------------------------------------------------------------
# Core backtest pipeline
# ---------------------------------------------------------------------------

def compute_all_weekends(weekend_records: list[dict]) -> list[dict]:
    """Add computed fantasy scores to each weekend record."""
    for w in weekend_records:
        scores = compute_race_weekend(w["drivers"])
        w["scores"] = scores
        # Constructor scores
        team_map = {d["driver_name"]: d["team"] for d in w["drivers"]}
        w["constructor_scores"] = constructor_fantasy_points(scores, team_map)
    return weekend_records


def run_backtest(
    seasons          : list[int],
    target_circuit   : str,
    budget_m         : float = 100.0,
    weights          : dict | None = None,
    use_testing_bonus: bool = True,
) -> pd.DataFrame:
    """
    For each race in `seasons`, use all prior data as training,
    optimise a team, then score it against the actual results.

    Returns a DataFrame of per-race backtest results.
    """
    print(f"\n{'='*60}")
    print(f"  BACKTEST: {seasons}  |  Circuit: {target_circuit}")
    print(f"{'='*60}")

    # --- Fetch all historical data ---
    all_weekends = []
    for s in seasons:
        cache = CACHE_DIR / f"weekends_{s}.json"
        if cache.exists():
            wkds = json.loads(cache.read_text())
            print(f"  [cache] Season {s}: {len(wkds)} races")
        else:
            try:
                print(f"  Fetching season {s} from API...")
                wkds = build_weekend_records(s)
                cache.write_text(json.dumps(wkds))
                print(f"  Got {len(wkds)} races")
            except Exception as e:
                print(f"  API unavailable, using seeded data for {s}")
                wkds = [w for w in get_seeded_weekends() if w["season"] == s]
                print(f"  Loaded {len(wkds)} seeded races for {s}")
        wkds = compute_all_weekends(wkds)
        all_weekends.extend(wkds)

    backtest_rows = []

    for idx, target_w in enumerate(all_weekends):
        # Only test at the target circuit (or all circuits if "all")
        circuit_match = (
            target_circuit.lower() == "all" or
            target_circuit.lower() in target_w["circuit"].lower() or
            target_circuit.lower() in target_w["race_name"].lower()
        )
        if not circuit_match:
            continue

        # Training data = everything BEFORE this race
        train_weekends = all_weekends[:idx]
        if len(train_weekends) < 1:
            continue    # not enough history

        df_train = build_history_df(train_weekends)

        # Build costs from actual drivers in this race
        race_driver_costs = {
            d["driver_name"]: CURRENT_DRIVER_COSTS.get(d["driver_name"], 8.0)
            for d in target_w["drivers"]
        }
        race_constructor_costs = {}
        for d in target_w["drivers"]:
            t = d["team"]
            if t not in race_constructor_costs:
                race_constructor_costs[t] = CURRENT_CONSTRUCTOR_COSTS.get(t, 10.0)

        bonuses = TESTING_BONUSES if use_testing_bonus else {}

        # Score drivers
        driver_scores_df = score_all(
            df            = df_train,
            driver_costs  = race_driver_costs,
            constructor_costs = race_constructor_costs,
            circuit       = target_w["circuit"],
            current_season= target_w["season"],
            weights       = weights,
            testing_bonuses = bonuses,
        )

        # Constructor scores (sum of driver norm scores per team)
        c_rows = []
        team_score_map: dict[str, float] = {}
        team_cost_map: dict[str, float]  = {}
        for _, row in driver_scores_df.iterrows():
            # Find team for this driver in this race
            d_info = next((d for d in target_w["drivers"]
                           if d["driver_name"] == row["driver"]), None)
            if d_info:
                t = d_info["team"]
                team_score_map[t] = team_score_map.get(t, 0) + row["norm_score"]
                team_cost_map[t]  = race_constructor_costs.get(t, 10.0)
        for team, score in team_score_map.items():
            c_rows.append({
                "constructor": team,
                "norm_score" : score,
                "cost_m"     : team_cost_map.get(team, 10.0),
            })
        constructor_scores_df = pd.DataFrame(c_rows)
        if constructor_scores_df.empty:
            continue

        # Optimise team
        result = optimise_team(driver_scores_df, constructor_scores_df,
                                budget_m=budget_m)

        # Actual fantasy points for this race
        actual_scores = target_w.get("scores", {})
        actual_c_scores = target_w.get("constructor_scores", {})

        picked_drivers = list(result["drivers"]["driver"])
        picked_constructors = list(result["constructors"]["constructor"])

        actual_pts = (
            sum(actual_scores.get(d, 0) for d in picked_drivers) +
            sum(actual_c_scores.get(c, 0) for c in picked_constructors)
        )

        # Theoretical best (oracle)
        all_d_scored = sorted(actual_scores.items(), key=lambda x: x[1], reverse=True)
        all_c_scored = sorted(actual_c_scores.items(), key=lambda x: x[1], reverse=True)
        oracle_pts = (
            sum(v for _, v in all_d_scored[:5]) +
            sum(v for _, v in all_c_scored[:2])
        )

        backtest_rows.append({
            "season"            : target_w["season"],
            "round"             : target_w["round"],
            "race_name"         : target_w["race_name"],
            "circuit"           : target_w["circuit"],
            "picked_drivers"    : ", ".join(picked_drivers),
            "picked_constructors": ", ".join(picked_constructors),
            "model_cost"        : result["total_cost"],
            "actual_pts"        : round(actual_pts, 1),
            "oracle_pts"        : round(oracle_pts, 1),
            "efficiency"        : round(actual_pts / oracle_pts, 3) if oracle_pts > 0 else 0,
        })

        print(f"\n  [{target_w['season']} R{target_w['round']:02d}] {target_w['race_name']}")
        print(f"    Picked: {', '.join(picked_drivers)}")
        print(f"    Constructors: {', '.join(picked_constructors)}")
        print(f"    Actual pts: {actual_pts:.1f}  |  Oracle: {oracle_pts:.1f}  "
              f"|  Efficiency: {actual_pts/oracle_pts:.1%}" if oracle_pts > 0 else "")

    return pd.DataFrame(backtest_rows)


def summarise_backtest(bt_df: pd.DataFrame):
    if bt_df.empty:
        print("\n  No backtest results.")
        return

    print(f"\n{'='*60}")
    print("  BACKTEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Races evaluated   : {len(bt_df)}")
    print(f"  Avg actual pts    : {bt_df['actual_pts'].mean():.1f}")
    print(f"  Avg oracle pts    : {bt_df['oracle_pts'].mean():.1f}")
    print(f"  Avg efficiency    : {bt_df['efficiency'].mean():.1%}")
    print(f"  Best race         : {bt_df.loc[bt_df['actual_pts'].idxmax(), 'race_name']} "
          f"({bt_df['actual_pts'].max():.1f} pts)")
    print(f"  Worst race        : {bt_df.loc[bt_df['actual_pts'].idxmin(), 'race_name']} "
          f"({bt_df['actual_pts'].min():.1f} pts)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Fantasy Backtester")
    parser.add_argument("--seasons",  nargs="+", type=int,
                        default=[2023, 2024],
                        help="Seasons to backtest over (e.g. 2023 2024)")
    parser.add_argument("--circuit",  type=str, default="australia",
                        help="Circuit to test (or 'all' for every race)")
    parser.add_argument("--budget",   type=float, default=100.0,
                        help="Budget cap in millions (default 100)")
    parser.add_argument("--no-bonus", action="store_true",
                        help="Disable testing/FP bonuses")
    args = parser.parse_args()

    bt = run_backtest(
        seasons           = args.seasons,
        target_circuit    = args.circuit,
        budget_m          = args.budget,
        use_testing_bonus = not args.no_bonus,
    )
    summarise_backtest(bt)

    if not bt.empty:
        out = Path(__file__).parent.parent / "backtest_results.csv"
        bt.to_csv(out, index=False)
        print(f"  Results saved to: {out}\n")
