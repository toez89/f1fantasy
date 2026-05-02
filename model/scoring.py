"""
F1 Fantasy 2026 scoring engine.

This module computes driver and constructor weekend totals using the 2026
rule set supplied for this project. It supports the rule items that can be
derived from current weekend data and accepts optional manual fields for the
rest (for example overtakes, pit-stop bonuses, or Driver of the Day).
"""

from __future__ import annotations


QUALI_RESULT_POINTS = {
    1: 10, 2: 9, 3: 8, 4: 7, 5: 6,
    6: 5, 7: 4, 8: 3, 9: 2, 10: 1,
}
SPRINT_RESULT_POINTS = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}
RACE_RESULT_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}

QUALI_NO_TIME_OR_DSQ_PENALTY = -5
QUALI_CONSTRUCTOR_DSQ_PENALTY = -5

SPRINT_POSITION_DELTA_BONUS = 1
SPRINT_POSITION_DELTA_PENALTY = -1
SPRINT_OVERTAKE_POINTS = 1
SPRINT_FASTEST_LAP_POINTS = 5
SPRINT_DNF_OR_DSQ_PENALTY = -10
SPRINT_CONSTRUCTOR_DSQ_PENALTY = -10

RACE_POSITION_DELTA_BONUS = 1
RACE_POSITION_DELTA_PENALTY = -1
RACE_OVERTAKE_POINTS = 1
RACE_FASTEST_LAP_POINTS = 10
RACE_DRIVER_OF_DAY_POINTS = 10
RACE_DNF_OR_DSQ_PENALTY = -20
RACE_CONSTRUCTOR_DSQ_PENALTY = -20

PITSTOP_TIME_POINTS = [
    (2.0, 20),
    (2.2, 10),
    (2.5, 5),
    (3.0, 2),
]
FASTEST_PITSTOP_BONUS = 5
PITSTOP_WORLD_RECORD_BONUS = 15


def _normalise_name(driver: dict) -> str:
    return driver.get("driver_name") or driver.get("name", "Unknown")


def _as_bool(value) -> bool:
    return bool(value)


def _positions_delta_points(
    start_pos: int | None,
    finish_pos: int | None,
    non_classified: bool,
    gain_points: int,
    loss_points: int,
) -> float:
    if non_classified or start_pos is None or finish_pos is None:
        return 0.0

    delta = start_pos - finish_pos
    if delta > 0:
        return float(delta * gain_points)
    if delta < 0:
        return float(abs(delta) * loss_points)
    return 0.0


def _pitstop_time_points(best_time: float | None) -> float:
    if best_time is None:
        return 0.0
    for threshold, points in PITSTOP_TIME_POINTS:
        if best_time < threshold:
            return float(points)
    return 0.0


def qualifying_driver_points(driver: dict) -> float:
    """Compute 2026 qualifying points for one driver."""
    if _as_bool(driver.get("quali_no_time")) or _as_bool(driver.get("quali_dsq")):
        return float(QUALI_NO_TIME_OR_DSQ_PENALTY)
    return float(QUALI_RESULT_POINTS.get(driver.get("quali_pos"), 0))


def sprint_driver_points(driver: dict) -> float:
    """Compute 2026 sprint points for one driver."""
    pts = 0.0
    sprint_pos = driver.get("sprint_pos")
    sprint_dnf = _as_bool(driver.get("sprint_dnf"))
    sprint_dsq = _as_bool(driver.get("sprint_dsq"))

    if not sprint_dnf and not sprint_dsq and sprint_pos is not None:
        pts += SPRINT_RESULT_POINTS.get(sprint_pos, 0)

    pts += _positions_delta_points(
        start_pos=driver.get("sprint_grid_pos"),
        finish_pos=sprint_pos,
        non_classified=sprint_dnf or sprint_dsq,
        gain_points=SPRINT_POSITION_DELTA_BONUS,
        loss_points=SPRINT_POSITION_DELTA_PENALTY,
    )
    pts += float(driver.get("sprint_overtakes", 0) or 0) * SPRINT_OVERTAKE_POINTS

    if _as_bool(driver.get("sprint_fastest_lap")):
        pts += SPRINT_FASTEST_LAP_POINTS
    if sprint_dnf or sprint_dsq:
        pts += SPRINT_DNF_OR_DSQ_PENALTY
    return float(pts)


def race_driver_points(driver: dict) -> float:
    """Compute 2026 race points for one driver."""
    pts = 0.0
    race_pos = driver.get("race_pos")
    race_dnf = _as_bool(driver.get("dnf"))
    race_dsq = _as_bool(driver.get("race_dsq"))

    if not race_dnf and not race_dsq and race_pos is not None:
        pts += RACE_RESULT_POINTS.get(race_pos, 0)

    pts += _positions_delta_points(
        start_pos=driver.get("grid_pos"),
        finish_pos=race_pos,
        non_classified=race_dnf or race_dsq,
        gain_points=RACE_POSITION_DELTA_BONUS,
        loss_points=RACE_POSITION_DELTA_PENALTY,
    )
    pts += float(driver.get("race_overtakes", 0) or 0) * RACE_OVERTAKE_POINTS

    if _as_bool(driver.get("race_fastest_lap")):
        pts += RACE_FASTEST_LAP_POINTS
    if _as_bool(driver.get("driver_of_day")):
        pts += RACE_DRIVER_OF_DAY_POINTS
    if race_dnf or race_dsq:
        pts += RACE_DNF_OR_DSQ_PENALTY
    return float(pts)


def driver_fantasy_points(driver: dict) -> float:
    """Compute total weekend points for one driver."""
    return float(
        qualifying_driver_points(driver)
        + sprint_driver_points(driver)
        + race_driver_points(driver)
    )


def _qualifying_constructor_bonus(team_drivers: list[dict]) -> float:
    q2_count = sum(1 for driver in team_drivers if _as_bool(driver.get("quali_reached_q2")))
    q3_count = sum(1 for driver in team_drivers if _as_bool(driver.get("quali_reached_q3")))

    if q3_count >= 2:
        return 10.0
    if q3_count == 1:
        return 5.0
    if q2_count >= 2:
        return 3.0
    if q2_count == 1:
        return 1.0
    return -1.0


def _constructor_pitstop_bonus(
    constructor: str,
    weekend_meta: dict | None,
) -> float:
    if not weekend_meta:
        return 0.0

    pts = 0.0
    pitstop_times = weekend_meta.get("constructor_pitstop_times", {}) or {}
    best_time = pitstop_times.get(constructor)
    if isinstance(best_time, list) and best_time:
        best_time = min(best_time)
    pts += _pitstop_time_points(best_time)

    fastest_pitstop = weekend_meta.get("fastest_pitstop")
    if fastest_pitstop == constructor or constructor in set(weekend_meta.get("fastest_pitstop_teams", []) or []):
        pts += FASTEST_PITSTOP_BONUS

    world_record = weekend_meta.get("pitstop_world_record")
    if world_record == constructor or constructor in set(weekend_meta.get("pitstop_world_record_teams", []) or []):
        pts += PITSTOP_WORLD_RECORD_BONUS
    return float(pts)


def constructor_fantasy_points(
    driver_scores: dict[str, float],
    team_map: dict[str, str],
    driver_results: list[dict] | None = None,
    weekend_meta: dict | None = None,
) -> dict[str, float]:
    """
    Compute constructor totals.

    When `driver_results` is omitted, this falls back to a simple sum of driver
    totals for backwards compatibility.
    """
    if not driver_results:
        totals: dict[str, float] = {}
        for driver, pts in driver_scores.items():
            team = team_map.get(driver, "Unknown")
            totals[team] = totals.get(team, 0.0) + float(pts)
        return totals

    teams: dict[str, list[dict]] = {}
    for driver in driver_results:
        teams.setdefault(driver["team"], []).append(driver)

    totals: dict[str, float] = {}
    for constructor, team_drivers in teams.items():
        quali_total = sum(qualifying_driver_points(driver) for driver in team_drivers)
        sprint_total = sum(sprint_driver_points(driver) for driver in team_drivers)
        race_total = 0.0
        for driver in team_drivers:
            race_points = race_driver_points(driver)
            if _as_bool(driver.get("driver_of_day")):
                race_points -= RACE_DRIVER_OF_DAY_POINTS
            race_total += race_points

        quali_dsq_penalty = QUALI_CONSTRUCTOR_DSQ_PENALTY * sum(
            1 for driver in team_drivers if _as_bool(driver.get("quali_dsq"))
        )
        sprint_dsq_penalty = SPRINT_CONSTRUCTOR_DSQ_PENALTY * sum(
            1 for driver in team_drivers if _as_bool(driver.get("sprint_dsq"))
        )
        race_dsq_penalty = RACE_CONSTRUCTOR_DSQ_PENALTY * sum(
            1 for driver in team_drivers if _as_bool(driver.get("race_dsq"))
        )

        totals[constructor] = float(
            quali_total
            + _qualifying_constructor_bonus(team_drivers)
            + quali_dsq_penalty
            + sprint_total
            + sprint_dsq_penalty
            + race_total
            + race_dsq_penalty
            + _constructor_pitstop_bonus(constructor, weekend_meta)
        )

    return totals


def compute_race_weekend(
    driver_results: list[dict],
    weekend_meta: dict | None = None,
) -> dict[str, float]:
    """Compute total weekend points for every driver in a weekend."""
    return {
        _normalise_name(driver): driver_fantasy_points(driver)
        for driver in driver_results
    }
