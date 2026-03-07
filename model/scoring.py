"""
F1 Fantasy Scoring Engine
Computes official F1 Fantasy points from race/qualifying results.

Official 2026 scoring rules:
  - Race finish:     P1=25, P2=18, P3=15, P4=12, P5=10, P6=8, P7=6, P8=4, P9=2, P10=1
  - Qualifying:      P1=10, P2=9, P3=8, P4=7, P5=6, P6=5, P7=4, P8=3, P9=2, P10=1
  - Places gained (race): +2 per position moved forward
  - Places lost   (race): -1 per position moved back
  - Beat teammate in qualifying: +3
  - Beat teammate in race:       +3
  - DNF / DSQ / NC (race):      -20
  - DNF (sprint):                -10
  - Fastest lap:    removed in 2025/2026
  Constructors score the SUM of both drivers' points.
"""

RACE_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
               6: 8,  7: 6,  8: 4,  9: 2,  10: 1}

QUALI_POINTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6,
                6: 5,  7: 4, 8: 3, 9: 2, 10: 1}

PLACES_GAINED_BONUS   = 2   # per position forward
PLACES_LOST_PENALTY   = -1  # per position backward
BEAT_TEAMMATE_BONUS   = 3   # qualifying + race (each)
DNF_PENALTY           = -20
DNF_SPRINT_PENALTY    = -10


def driver_fantasy_points(
    quali_pos: int | None,
    race_pos:  int | None,
    grid_pos:  int | None,       # starting grid position (after penalties etc.)
    beat_tm_quali: bool = False,
    beat_tm_race:  bool = False,
    dnf: bool = False,
) -> float:
    """
    Calculate total F1 Fantasy points for a single driver for one race weekend.

    Parameters
    ----------
    quali_pos      : final qualifying position (1-20). None if no time set.
    race_pos       : race finish position (1-20). None if DNF/DSQ.
    grid_pos       : grid position at race start (used for places-gained calc).
                     Defaults to quali_pos if not supplied.
    beat_tm_quali  : True if driver beat their teammate in qualifying.
    beat_tm_race   : True if driver beat their teammate in race classification.
    dnf            : True if the driver did not finish / was disqualified.

    Returns
    -------
    total points (float)
    """
    pts = 0.0

    # --- Qualifying points ---
    if quali_pos is not None:
        pts += QUALI_POINTS.get(quali_pos, 0)

    # --- Race finish points ---
    if not dnf and race_pos is not None:
        pts += RACE_POINTS.get(race_pos, 0)

    # --- Places gained / lost ---
    start = grid_pos if grid_pos is not None else quali_pos
    if start is not None and race_pos is not None and not dnf:
        delta = start - race_pos          # positive = moved forward
        if delta > 0:
            pts += delta * PLACES_GAINED_BONUS
        elif delta < 0:
            pts += delta * abs(PLACES_LOST_PENALTY)  # delta is negative

    # --- Teammate battles ---
    if beat_tm_quali:
        pts += BEAT_TEAMMATE_BONUS
    if beat_tm_race:
        pts += BEAT_TEAMMATE_BONUS

    # --- DNF penalty ---
    if dnf:
        pts += DNF_PENALTY

    return pts


def compute_race_weekend(driver_results: list[dict]) -> dict[str, float]:
    """
    Compute fantasy points for every driver in a race weekend.

    Parameters
    ----------
    driver_results : list of dicts, one per driver, each containing:
        {
          'name'       : str,
          'team'       : str,
          'quali_pos'  : int | None,
          'race_pos'   : int | None,
          'grid_pos'   : int | None,   # optional
          'dnf'        : bool,
        }

    Returns
    -------
    dict mapping driver name -> fantasy points
    """
    # Group by team to resolve teammate battles
    teams: dict[str, list[dict]] = {}
    for d in driver_results:
        teams.setdefault(d['team'], []).append(d)

    scores = {}
    for team_drivers in teams.values():
        # Support both 'name' and 'driver_name' keys
        def _name(d): return d.get('driver_name') or d.get('name', 'Unknown')

        if len(team_drivers) == 2:
            a, b = team_drivers
            # Qualifying battle
            aq, bq = a.get('quali_pos'), b.get('quali_pos')
            beat_tm_quali_a = (aq is not None and bq is not None and aq < bq)
            beat_tm_quali_b = (aq is not None and bq is not None and bq < aq)
            # Race battle (only for finishers)
            ar, br = (None if a.get('dnf') else a.get('race_pos')), \
                     (None if b.get('dnf') else b.get('race_pos'))
            beat_tm_race_a = (ar is not None and br is not None and ar < br)
            beat_tm_race_b = (ar is not None and br is not None and br < ar)
        else:
            beat_tm_quali_a = beat_tm_race_a = False
            beat_tm_quali_b = beat_tm_race_b = False
            a = team_drivers[0]; b = None

        scores[_name(a)] = driver_fantasy_points(
            quali_pos=a.get('quali_pos'),
            race_pos=a.get('race_pos'),
            grid_pos=a.get('grid_pos'),
            beat_tm_quali=beat_tm_quali_a,
            beat_tm_race=beat_tm_race_a,
            dnf=a.get('dnf', False),
        )
        if b:
            scores[_name(b)] = driver_fantasy_points(
                quali_pos=b.get('quali_pos'),
                race_pos=b.get('race_pos'),
                grid_pos=b.get('grid_pos'),
                beat_tm_quali=beat_tm_quali_b,
                beat_tm_race=beat_tm_race_b,
                dnf=b.get('dnf', False),
            )

    return scores


def constructor_fantasy_points(driver_scores: dict[str, float],
                                team_map: dict[str, str]) -> dict[str, float]:
    """
    Sum driver points per constructor.

    Parameters
    ----------
    driver_scores : {driver_name: points}
    team_map      : {driver_name: team_name}

    Returns
    -------
    {team_name: total_points}
    """
    totals: dict[str, float] = {}
    for driver, pts in driver_scores.items():
        team = team_map.get(driver, 'Unknown')
        totals[team] = totals.get(team, 0) + pts
    return totals
