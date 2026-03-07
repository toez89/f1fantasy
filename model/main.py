"""
F1 Fantasy Model — Main Entry Point
====================================
Run this script to:
  1. Score all drivers/constructors for an upcoming race
  2. Output the optimised team
  3. Optionally run the backtester

Usage examples:
  # Optimise team for Australia 2026 (default):
  python main.py

  # Backtest over all 2023+2024 Australia races:
  python main.py --backtest --seasons 2023 2024 --circuit australia

  # Backtest ALL races in 2024:
  python main.py --backtest --seasons 2024 --circuit all

  # Optimise with custom weights:
  python main.py --w-recent 0.5 --w-track 0.2 --w-ppm 0.1 --w-quali 0.2

  # Lock in Leclerc, exclude Stroll:
  python main.py --lock "Charles Leclerc" --exclude "Lance Stroll"
"""

import argparse
import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fetch_data   import build_weekend_records
from scoring      import compute_race_weekend, constructor_fantasy_points
from value_model  import build_history_df, score_all
from optimizer    import optimise_team, print_team
from backtest     import (compute_all_weekends, run_backtest, summarise_backtest,
                           CURRENT_DRIVER_COSTS, CURRENT_CONSTRUCTOR_COSTS,
                           TESTING_BONUSES)
from seed_data    import get_seeded_weekends

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

TARGET_CIRCUIT = "Albert Park Grand Prix Circuit"
CURRENT_SEASON = 2026

TRAIN_SEASONS  = [2022, 2023, 2024]    # seasons used to train the model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_or_fetch(season: int) -> list[dict]:
    cache = CACHE_DIR / f"weekends_{season}.json"
    if cache.exists():
        wkds = json.loads(cache.read_text())
        print(f"  [cache] Season {season}: {len(wkds)} races loaded")
        return compute_all_weekends(wkds)
    # Try live API first, fall back to seeded data
    try:
        print(f"  Fetching season {season} from Jolpica API...")
        wkds = build_weekend_records(season)
        cache.write_text(json.dumps(wkds))
        print(f"  Fetched {len(wkds)} races from API")
    except Exception as e:
        print(f"  API unavailable ({type(e).__name__}), using seeded data")
        wkds = [w for w in get_seeded_weekends() if w["season"] == season]
        print(f"  Loaded {len(wkds)} seeded races for {season}")
    return compute_all_weekends(wkds)


# ---------------------------------------------------------------------------
# Main optimise workflow
# ---------------------------------------------------------------------------

def run_optimise(args):
    print("\n" + "=" * 62)
    print("  F1 FANTASY MODEL — AUSTRALIA GP 2026")
    print("=" * 62)

    # Step 1: Load training data
    print("\n[1/4] Loading historical data...")
    all_weekends = []
    for s in TRAIN_SEASONS:
        all_weekends.extend(load_or_fetch(s))

    df_train = build_history_df(all_weekends)
    print(f"  Training set: {len(df_train)} driver-race records "
          f"across {df_train['season'].nunique()} seasons")

    # Step 2: Score all drivers
    print("\n[2/4] Scoring drivers & constructors...")
    weights = {
        "recent_form"  : args.w_recent,
        "track_history": args.w_track,
        "season_ppm"   : args.w_ppm,
        "quali_form"   : args.w_quali,
    }
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}   # normalise

    bonuses = TESTING_BONUSES if not args.no_bonus else {}

    driver_scores_df = score_all(
        df              = df_train,
        driver_costs    = CURRENT_DRIVER_COSTS,
        constructor_costs = CURRENT_CONSTRUCTOR_COSTS,
        circuit         = TARGET_CIRCUIT,
        current_season  = CURRENT_SEASON,
        weights         = weights,
        testing_bonuses = bonuses,
    )

    # Constructor scores: sum of drivers' norm scores per team
    # Map driver → team via 2026 grid
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

    c_agg: dict[str, float] = {}
    for _, row in driver_scores_df.iterrows():
        t = DRIVER_TEAM_2026.get(row["driver"], "Unknown")
        c_agg[t] = c_agg.get(t, 0) + row["norm_score"]

    constructor_scores_df = pd.DataFrame([
        {"constructor": t, "norm_score": s,
         "cost_m": CURRENT_CONSTRUCTOR_COSTS.get(t, 10.0)}
        for t, s in c_agg.items()
    ])

    # Step 3: Print driver rankings
    print("\n[3/4] Driver value rankings (top 15):")
    print(f"\n  {'#':<4} {'Driver':<26} {'Cost':>7}  {'Score':>7}  "
          f"{'Recent':>7}  {'Track':>7}  {'QPM':>7}")
    print("  " + "-" * 68)
    for _, row in driver_scores_df.head(15).iterrows():
        print(f"  {row['rank']:<4} {row['driver']:<26} "
              f"${row['cost_m']:>5.1f}M  "
              f"{row['norm_score']:>7.3f}  "
              f"{row['recent_form']:>7.1f}  "
              f"{row['track_history']:>7.1f}  "
              f"{row['quali_form']:>7.1f}")

    # Step 4: Optimise
    print("\n[4/4] Optimising team...")
    locked_d = list(args.lock)       if args.lock    else []
    excl     = list(args.exclude)    if args.exclude else []

    result = optimise_team(
        driver_scores_df,
        constructor_scores_df,
        budget_m          = args.budget,
        locked_drivers    = locked_d,
        excluded          = excl,
    )
    print_team(result)

    # Save outputs
    out_dir = Path(__file__).parent.parent
    driver_scores_df.to_csv(out_dir / "driver_scores.csv", index=False)
    constructor_scores_df.to_csv(out_dir / "constructor_scores.csv", index=False)

    team_out = {
        "drivers"      : list(result["drivers"]["driver"]),
        "constructors" : list(result["constructors"]["constructor"]),
        "total_cost"   : result["total_cost"],
        "total_score"  : result["total_score"],
    }
    (out_dir / "optimal_team.json").write_text(json.dumps(team_out, indent=2))
    print(f"  Scores saved to: driver_scores.csv, constructor_scores.csv")
    print(f"  Team saved to  : optimal_team.json\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="F1 Fantasy optimiser & backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--backtest",  action="store_true",
                        help="Run backtester instead of optimiser")
    parser.add_argument("--seasons",   nargs="+", type=int, default=[2023, 2024],
                        help="Seasons for backtest (default: 2023 2024)")
    parser.add_argument("--circuit",   type=str, default="australia",
                        help="Circuit for backtest (default: australia, or 'all')")
    parser.add_argument("--budget",    type=float, default=100.0,
                        help="Budget cap in millions (default: 100)")
    parser.add_argument("--lock",      nargs="+", default=[],
                        help="Driver names to lock into team")
    parser.add_argument("--exclude",   nargs="+", default=[],
                        help="Driver/constructor names to exclude")
    parser.add_argument("--no-bonus",  action="store_true",
                        help="Disable FP/testing bonuses")
    # Weight overrides
    parser.add_argument("--w-recent",  type=float, default=0.35)
    parser.add_argument("--w-track",   type=float, default=0.30)
    parser.add_argument("--w-ppm",     type=float, default=0.15)
    parser.add_argument("--w-quali",   type=float, default=0.20)

    args = parser.parse_args()

    if args.backtest:
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
            print(f"  Saved to: {out}\n")
    else:
        run_optimise(args)


if __name__ == "__main__":
    main()
