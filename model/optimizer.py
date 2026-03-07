"""
F1 Fantasy Team Optimizer
=========================
Uses linear programming (PuLP) to find the highest-scoring team
within the budget and selection constraints.

Constraints:
  - Exactly 5 drivers
  - Exactly 2 constructors
  - Total cost ≤ budget (default $100M)
  - Objective: maximise sum of predicted fantasy scores
"""

import pandas as pd
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

from itertools import combinations


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

def optimise_team(
    driver_scores    : pd.DataFrame,   # must have: driver, norm_score, cost_m
    constructor_scores: pd.DataFrame,  # must have: constructor, norm_score, cost_m
    budget_m         : float = 100.0,
    n_drivers        : int = 5,
    n_constructors   : int = 2,
    locked_drivers   : list[str] | None = None,
    locked_constructors: list[str] | None = None,
    excluded         : list[str] | None = None,
) -> dict:
    """
    Find the optimal fantasy team.

    Parameters
    ----------
    driver_scores       : DataFrame with columns [driver, norm_score, cost_m]
    constructor_scores  : DataFrame with columns [constructor, norm_score, cost_m]
    budget_m            : total budget in millions
    n_drivers           : number of drivers to pick
    n_constructors      : number of constructors to pick
    locked_drivers      : drivers that MUST be in the team
    locked_constructors : constructors that MUST be in the team
    excluded            : drivers/constructors to exclude entirely

    Returns
    -------
    dict with keys: drivers, constructors, total_cost, total_score
    """
    locked_d = set(locked_drivers or [])
    locked_c = set(locked_constructors or [])
    excl     = set(excluded or [])

    d_df = driver_scores[~driver_scores["driver"].isin(excl)].copy()
    c_df = constructor_scores[~constructor_scores["constructor"].isin(excl)].copy()

    if HAS_PULP:
        return _optimise_pulp(d_df, c_df, budget_m, n_drivers, n_constructors,
                               locked_d, locked_c)
    else:
        return _optimise_brute(d_df, c_df, budget_m, n_drivers, n_constructors,
                                locked_d, locked_c)


# ---------------------------------------------------------------------------
# PuLP (integer linear programming) solver  — fast, exact
# ---------------------------------------------------------------------------

def _optimise_pulp(d_df, c_df, budget_m, n_drivers, n_constructors,
                   locked_d, locked_c):
    prob = pulp.LpProblem("F1_Fantasy", pulp.LpMaximize)

    d_vars = {row.driver: pulp.LpVariable(f"d_{i}", cat="Binary")
              for i, row in d_df.iterrows()}
    c_vars = {row.constructor: pulp.LpVariable(f"c_{i}", cat="Binary")
              for i, row in c_df.iterrows()}

    # Objective
    prob += (
        pulp.lpSum(d_vars[r.driver]       * r.norm_score for _, r in d_df.iterrows()) +
        pulp.lpSum(c_vars[r.constructor]  * r.norm_score for _, r in c_df.iterrows())
    )

    # Budget
    prob += (
        pulp.lpSum(d_vars[r.driver]       * r.cost_m for _, r in d_df.iterrows()) +
        pulp.lpSum(c_vars[r.constructor]  * r.cost_m for _, r in c_df.iterrows())
    ) <= budget_m

    # Count constraints
    prob += pulp.lpSum(d_vars.values()) == n_drivers
    prob += pulp.lpSum(c_vars.values()) == n_constructors

    # Locked selections
    for name in locked_d:
        if name in d_vars:
            prob += d_vars[name] == 1
    for name in locked_c:
        if name in c_vars:
            prob += c_vars[name] == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected_d = [n for n, v in d_vars.items() if pulp.value(v) == 1]
    selected_c = [n for n, v in c_vars.items() if pulp.value(v) == 1]

    return _format_result(d_df, c_df, selected_d, selected_c)


# ---------------------------------------------------------------------------
# Brute-force fallback (used if PuLP unavailable)
# ---------------------------------------------------------------------------

def _optimise_brute(d_df, c_df, budget_m, n_drivers, n_constructors,
                    locked_d, locked_c):
    free_d = [r.driver for _, r in d_df.iterrows() if r.driver not in locked_d]
    free_c = [r.constructor for _, r in c_df.iterrows()
               if r.constructor not in locked_c]

    need_d = n_drivers      - len(locked_d)
    need_c = n_constructors - len(locked_c)

    d_score_map = dict(zip(d_df["driver"], d_df["norm_score"]))
    d_cost_map  = dict(zip(d_df["driver"], d_df["cost_m"]))
    c_score_map = dict(zip(c_df["constructor"], c_df["norm_score"]))
    c_cost_map  = dict(zip(c_df["constructor"], c_df["cost_m"]))

    locked_d_cost  = sum(d_cost_map.get(n, 0) for n in locked_d)
    locked_c_cost  = sum(c_cost_map.get(n, 0) for n in locked_c)
    locked_d_score = sum(d_score_map.get(n, 0) for n in locked_d)
    locked_c_score = sum(c_score_map.get(n, 0) for n in locked_c)

    remaining_budget = budget_m - locked_d_cost - locked_c_cost

    best_score, best_d, best_c = -999, [], []

    for dc in combinations(free_d, need_d):
        dc_cost  = sum(d_cost_map[n] for n in dc)
        dc_score = sum(d_score_map[n] for n in dc)
        for cc in combinations(free_c, need_c):
            cc_cost = sum(c_cost_map[n] for n in cc)
            if dc_cost + cc_cost > remaining_budget:
                continue
            total = dc_score + locked_d_score + sum(c_score_map[n] for n in cc) + locked_c_score
            if total > best_score:
                best_score = total
                best_d = list(dc) + list(locked_d)
                best_c = list(cc) + list(locked_c)

    return _format_result(d_df, c_df, best_d, best_c)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_result(d_df, c_df, selected_d, selected_c) -> dict:
    d_rows = d_df[d_df["driver"].isin(selected_d)].copy()
    c_rows = c_df[c_df["constructor"].isin(selected_c)].copy()

    total_cost  = float(d_rows["cost_m"].sum() + c_rows["cost_m"].sum())
    total_score = float(d_rows["norm_score"].sum() + c_rows["norm_score"].sum())

    return {
        "drivers"      : d_rows.sort_values("norm_score", ascending=False),
        "constructors" : c_rows.sort_values("norm_score", ascending=False),
        "total_cost"   : round(total_cost, 1),
        "total_score"  : round(total_score, 4),
        "budget_used"  : f"${total_cost}M / $100.0M",
    }


def print_team(result: dict):
    """Pretty-print the optimised team."""
    print("\n" + "=" * 62)
    print("  OPTIMISED F1 FANTASY TEAM")
    print("=" * 62)
    print(f"\n{'DRIVERS (5)':}")
    d = result["drivers"]
    for _, row in d.iterrows():
        print(f"  {row['driver']:<26} ${row['cost_m']:.1f}M  score={row['norm_score']:.3f}")
    print(f"\n{'CONSTRUCTORS (2)':}")
    c = result["constructors"]
    for _, row in c.iterrows():
        print(f"  {row['constructor']:<26} ${row['cost_m']:.1f}M  score={row['norm_score']:.3f}")
    print(f"\n  Budget used : {result['budget_used']}")
    print(f"  Total score : {result['total_score']:.4f}")
    print("=" * 62 + "\n")
