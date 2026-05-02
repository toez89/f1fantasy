"""
F1 Fantasy Team Optimizer
=========================
Uses linear programming (PuLP) to find the highest-scoring team
within the budget and selection constraints.

Constraints:
  - Exactly 5 drivers
  - Exactly 2 constructors
  - Total cost ≤ budget (default $100M)
  - Objective: maximise sum of predicted fantasy scores, including one free 2x driver boost
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
    current_drivers  : list[str] | None = None,
    current_constructors: list[str] | None = None,
    free_transfers   : int = 2,
    transfer_penalty : float = 10.0,
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
    dict with keys: drivers, constructors, boost_driver, total_cost, total_score
    """
    locked_d = set(locked_drivers or [])
    locked_c = set(locked_constructors or [])
    excl     = set(excluded or [])
    current_d = list(current_drivers or [])
    current_c = list(current_constructors or [])

    d_df = driver_scores[~driver_scores["driver"].isin(excl)].copy()
    c_df = constructor_scores[~constructor_scores["constructor"].isin(excl)].copy()

    if HAS_PULP:
        return _optimise_pulp(d_df, c_df, budget_m, n_drivers, n_constructors,
                               locked_d, locked_c, current_d, current_c,
                               int(free_transfers), float(transfer_penalty))
    else:
        return _optimise_brute(d_df, c_df, budget_m, n_drivers, n_constructors,
                                locked_d, locked_c, current_d, current_c,
                                int(free_transfers), float(transfer_penalty))


# ---------------------------------------------------------------------------
# PuLP (integer linear programming) solver  — fast, exact
# ---------------------------------------------------------------------------

def _optimise_pulp(d_df, c_df, budget_m, n_drivers, n_constructors,
                   locked_d, locked_c, current_d, current_c,
                   free_transfers, transfer_penalty):
    prob = pulp.LpProblem("F1_Fantasy", pulp.LpMaximize)

    d_vars = {row.driver: pulp.LpVariable(f"d_{i}", cat="Binary")
              for i, row in d_df.iterrows()}
    c_vars = {row.constructor: pulp.LpVariable(f"c_{i}", cat="Binary")
              for i, row in c_df.iterrows()}
    boost_vars = {row.driver: pulp.LpVariable(f"boost_{i}", cat="Binary")
                  for i, row in d_df.iterrows()}

    kept_driver_expr = pulp.lpSum(
        d_vars[name] for name in current_d if name in d_vars
    )
    kept_constructor_expr = pulp.lpSum(
        c_vars[name] for name in current_c if name in c_vars
    )
    transfer_count_expr = (
        len(current_d) - kept_driver_expr + len(current_c) - kept_constructor_expr
    )
    excess_transfers = pulp.LpVariable("excess_transfers", lowBound=0)

    # Objective
    prob += (
        pulp.lpSum(d_vars[r.driver]       * r.norm_score for _, r in d_df.iterrows()) +
        pulp.lpSum(c_vars[r.constructor]  * r.norm_score for _, r in c_df.iterrows()) -
        (transfer_penalty * excess_transfers) +
        pulp.lpSum(boost_vars[r.driver]    * r.norm_score for _, r in d_df.iterrows())
    )

    # Budget
    prob += (
        pulp.lpSum(d_vars[r.driver]       * r.cost_m for _, r in d_df.iterrows()) +
        pulp.lpSum(c_vars[r.constructor]  * r.cost_m for _, r in c_df.iterrows())
    ) <= budget_m

    # Count constraints
    prob += pulp.lpSum(d_vars.values()) == n_drivers
    prob += pulp.lpSum(c_vars.values()) == n_constructors
    prob += pulp.lpSum(boost_vars.values()) == 1

    # Locked selections
    for name in locked_d:
        if name in d_vars:
            prob += d_vars[name] == 1
    for name in locked_c:
        if name in c_vars:
            prob += c_vars[name] == 1

    for name in boost_vars:
        prob += boost_vars[name] <= d_vars[name]

    prob += excess_transfers >= transfer_count_expr - free_transfers

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    selected_d = [n for n, v in d_vars.items() if pulp.value(v) == 1]
    selected_c = [n for n, v in c_vars.items() if pulp.value(v) == 1]
    selected_boost = next((n for n, v in boost_vars.items() if pulp.value(v) == 1), None)

    return _format_result(
        d_df, c_df, selected_d, selected_c,
        current_d, current_c, free_transfers, transfer_penalty, budget_m,
        boost_driver=selected_boost,
    )


# ---------------------------------------------------------------------------
# Brute-force fallback (used if PuLP unavailable)
# ---------------------------------------------------------------------------

def _optimise_brute(d_df, c_df, budget_m, n_drivers, n_constructors,
                    locked_d, locked_c, current_d, current_c,
                    free_transfers, transfer_penalty):
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

    best_score, best_d, best_c = -999999, [], []

    for dc in combinations(free_d, need_d):
        dc_cost  = sum(d_cost_map[n] for n in dc)
        dc_score = sum(d_score_map[n] for n in dc)
        for cc in combinations(free_c, need_c):
            cc_cost = sum(c_cost_map[n] for n in cc)
            if dc_cost + cc_cost > remaining_budget:
                continue
            selected_d = list(dc) + list(locked_d)
            selected_c = list(cc) + list(locked_c)
            transfer_count = _count_transfers(selected_d, selected_c, current_d, current_c)
            paid_transfers = max(0, transfer_count - free_transfers)
            boost_score = max(d_score_map[n] for n in selected_d) if selected_d else 0.0
            total = (
                dc_score + locked_d_score + sum(c_score_map[n] for n in cc) + locked_c_score
                + boost_score
                - (paid_transfers * transfer_penalty)
            )
            if total > best_score:
                best_score = total
                best_d = selected_d
                best_c = selected_c

    return _format_result(
        d_df, c_df, best_d, best_c,
        current_d, current_c, free_transfers, transfer_penalty, budget_m,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_transfers(selected_d, selected_c, current_d, current_c) -> int:
    if not current_d and not current_c:
        return 0
    kept_drivers = len(set(selected_d) & set(current_d))
    kept_constructors = len(set(selected_c) & set(current_c))
    return (len(current_d) - kept_drivers) + (len(current_c) - kept_constructors)


def _format_result(
    d_df, c_df, selected_d, selected_c,
    current_d, current_c, free_transfers, transfer_penalty, budget_m,
    boost_driver=None,
) -> dict:
    d_rows = d_df[d_df["driver"].isin(selected_d)].copy()
    c_rows = c_df[c_df["constructor"].isin(selected_c)].copy()

    total_cost  = float(d_rows["cost_m"].sum() + c_rows["cost_m"].sum())
    unboosted_score = float(d_rows["norm_score"].sum() + c_rows["norm_score"].sum())
    transfer_count = _count_transfers(selected_d, selected_c, current_d, current_c)
    paid_transfers = max(0, int(transfer_count) - int(free_transfers))
    transfer_cost = float(paid_transfers * transfer_penalty)
    boost_score = None
    if not d_rows.empty:
        if boost_driver is not None and boost_driver in set(d_rows["driver"]):
            boost_row = d_rows[d_rows["driver"] == boost_driver].iloc[0]
        else:
            boost_row = d_rows.sort_values("norm_score", ascending=False).iloc[0]
            boost_driver = boost_row["driver"]
        boost_score = float(boost_row["norm_score"])
    boost_bonus = float(boost_score or 0.0)
    boosted_score = float(unboosted_score + boost_bonus)
    total_score = float(boosted_score - transfer_cost)

    current_d_set = set(current_d)
    current_c_set = set(current_c)
    selected_d_set = set(selected_d)
    selected_c_set = set(selected_c)

    kept_drivers = sorted(selected_d_set & current_d_set)
    incoming_drivers = sorted(selected_d_set - current_d_set)
    outgoing_drivers = sorted(current_d_set - selected_d_set)
    kept_constructors = sorted(selected_c_set & current_c_set)
    incoming_constructors = sorted(selected_c_set - current_c_set)
    outgoing_constructors = sorted(current_c_set - selected_c_set)

    return {
        "drivers"      : d_rows.sort_values("norm_score", ascending=False),
        "constructors" : c_rows.sort_values("norm_score", ascending=False),
        "total_cost"   : round(total_cost, 1),
        "total_score"  : round(total_score, 4),
        "base_score"   : round(boosted_score, 4),
        "unboosted_score": round(unboosted_score, 4),
        "transfer_count": int(transfer_count),
        "free_transfers": int(free_transfers),
        "paid_transfers": int(paid_transfers),
        "transfer_cost": round(transfer_cost, 4),
        "boost_driver": boost_driver,
        "boost_score": round(boost_score, 4) if boost_score is not None else None,
        "boost_bonus": round(boost_bonus, 4),
        "kept_drivers": kept_drivers,
        "incoming_drivers": incoming_drivers,
        "outgoing_drivers": outgoing_drivers,
        "kept_constructors": kept_constructors,
        "incoming_constructors": incoming_constructors,
        "outgoing_constructors": outgoing_constructors,
        "budget_used"  : f"${total_cost:.1f}M / ${float(budget_m):.1f}M",
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
    if result.get("boost_driver"):
        print(
            f"\n  2x Boost    : {result['boost_driver']} "
            f"(adds {result['boost_bonus']:.3f})"
        )
    print(f"\n  Budget used : {result['budget_used']}")
    print(f"  Total score : {result['total_score']:.4f}")
    print("=" * 62 + "\n")
