"""
F1 Fantasy Backtester
=====================
Runs the value model + optimizer on every race in historical seasons
and measures how the model's picks would have actually performed.

Usage:
    python backtest.py --seasons 2023 2024 --circuit miami
"""

import argparse
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fetch_data   import build_weekend_records
from pricing      import load_cost_snapshot
from seed_data    import get_seeded_weekends
from scoring      import compute_race_weekend, constructor_fantasy_points
from value_model  import build_history_df, score_all
from optimizer    import optimise_team, print_team


CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
CURRENT_SEASON_CACHE_TTL_SECONDS = 60 * 60 * 12

CURRENT_DRIVER_COSTS, CURRENT_CONSTRUCTOR_COSTS, CURRENT_BUDGET_CAP, CURRENT_COST_SNAPSHOT = load_cost_snapshot()

# Manual adjustments are intentionally empty until fresh Miami practice/news is added.
TESTING_BONUSES = {}


# ---------------------------------------------------------------------------
# Core backtest pipeline
# ---------------------------------------------------------------------------

def _load_cached_weekends(cache_path: Path, max_age_seconds: int | None = None) -> list[dict] | None:
    """Return cached weekends when the cache contains usable race data."""
    if not cache_path.exists():
        return None
    if max_age_seconds is not None:
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > max_age_seconds:
            return None
    wkds = json.loads(cache_path.read_text())
    if not wkds:
        return None

    first_driver = (((wkds or [{}])[0]).get("drivers") or [{}])[0]
    required_fields = {
        "quali_reached_q2",
        "quali_reached_q3",
        "quali_no_time",
        "quali_dsq",
        "sprint_pos",
        "race_fastest_lap",
        "race_dsq",
    }
    return wkds if required_fields.issubset(first_driver.keys()) else None


def compute_all_weekends(weekend_records: list[dict]) -> list[dict]:
    """Add computed fantasy scores to each weekend record."""
    for w in weekend_records:
        scores = w.get("scores")
        if not scores:
            scores = compute_race_weekend(w["drivers"], weekend_meta=w)
            w["scores"] = scores

        if not w.get("constructor_scores"):
            team_map = {d["driver_name"]: d["team"] for d in w["drivers"]}
            w["constructor_scores"] = constructor_fantasy_points(
                scores,
                team_map,
                driver_results=w["drivers"],
                weekend_meta=w,
            )
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
        cache_ttl = CURRENT_SEASON_CACHE_TTL_SECONDS if s == 2026 else None
        cached_weekends = _load_cached_weekends(cache, max_age_seconds=cache_ttl)
        if cached_weekends is not None:
            wkds = cached_weekends
            print(f"  [cache] Season {s}: {len(wkds)} races")
        else:
            try:
                print(f"  Fetching season {s} from API...")
                wkds = build_weekend_records(s, use_cache=cache_ttl is None)
                if wkds:
                    cache.write_text(json.dumps(wkds))
                    print(f"  Got {len(wkds)} races")
                else:
                    print(f"  API returned no completed races for {s}, checking seed data")
            except Exception as e:
                print(f"  API unavailable, using seeded data for {s}")
                wkds = []
            if not wkds:
                stale_weekends = _load_cached_weekends(cache)
                if stale_weekends is not None:
                    wkds = stale_weekends
                    print(f"  [stale cache] Season {s}: {len(wkds)} races")
            if not wkds:
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
        boost_driver = result.get("boost_driver")

        actual_pts = (
            sum(actual_scores.get(d, 0) for d in picked_drivers) +
            sum(actual_c_scores.get(c, 0) for c in picked_constructors) +
            actual_scores.get(boost_driver, 0)
        )

        # Theoretical best (oracle)
        all_d_scored = sorted(actual_scores.items(), key=lambda x: x[1], reverse=True)
        all_c_scored = sorted(actual_c_scores.items(), key=lambda x: x[1], reverse=True)
        oracle_pts = (
            sum(v for _, v in all_d_scored[:5]) +
            sum(v for _, v in all_c_scored[:2]) +
            (all_d_scored[0][1] if all_d_scored else 0)
        )

        backtest_rows.append({
            "season"            : target_w["season"],
            "round"             : target_w["round"],
            "race_name"         : target_w["race_name"],
            "circuit"           : target_w["circuit"],
            "picked_drivers"    : ", ".join(picked_drivers),
            "picked_constructors": ", ".join(picked_constructors),
            "boost_driver"      : boost_driver,
            "model_cost"        : result["total_cost"],
            "actual_pts"        : round(actual_pts, 1),
            "oracle_pts"        : round(oracle_pts, 1),
            "efficiency"        : round(actual_pts / oracle_pts, 3) if oracle_pts > 0 else 0,
        })

        print(f"\n  [{target_w['season']} R{target_w['round']:02d}] {target_w['race_name']}")
        print(f"    Picked: {', '.join(picked_drivers)}")
        print(f"    Constructors: {', '.join(picked_constructors)}")
        print(f"    2x Boost: {boost_driver}")
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
    parser.add_argument("--circuit",  type=str, default="miami",
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
