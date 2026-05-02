"""
F1 Data Fetcher
Uses the Jolpica API (Ergast-compatible) to pull historical race and qualifying data.
API base: https://api.jolpi.ca/ergast/f1/
"""

import requests
import time
import json
from pathlib import Path

BASE_URL  = "https://api.jolpi.ca/ergast/f1"
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Canonical name overrides: API full name → model name used in seed data / team maps
DRIVER_NAME_OVERRIDES: dict[str, str] = {
    "Andrea Kimi Antonelli": "Kimi Antonelli",
    "Nico Hülkenberg": "Nico Hulkenberg",
    "Sergio Pérez": "Sergio Perez",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("?", "_")
    return CACHE_DIR / f"{safe}.json"


def _get(url: str, params: dict | None = None, use_cache: bool = True) -> dict:
    cache_key = url + str(params or "")
    cp = _cache_path(cache_key)
    if use_cache and cp.exists():
        return json.loads(cp.read_text())

    time.sleep(0.25)          # be polite to the API
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    cp.write_text(json.dumps(data))
    return data


def _get_all_pages(url: str, limit: int = 100, use_cache: bool = True) -> list:
    """Paginate through all results from an Ergast-style endpoint."""
    results, offset = [], 0
    while True:
        data = _get(url, params={"limit": limit, "offset": offset}, use_cache=use_cache)
        mr   = data.get("MRData", {})
        # figure out which table key holds the rows
        table = mr.get("RaceTable") or mr.get("QualifyingTable") or mr.get("DriverTable") or {}
        rows  = (table.get("Races") or table.get("Drivers") or [])
        if not rows:
            break
        results.extend(rows)
        total = int(mr.get("total", 0))
        offset += limit
        if offset >= total:
            break
    return results


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------

def fetch_race_results(season: int, use_cache: bool = True) -> list[dict]:
    """Return all race results for a season as a list of Race dicts."""
    url = f"{BASE_URL}/{season}/results.json"
    return _get_all_pages(url, use_cache=use_cache)


def fetch_qualifying_results(season: int, use_cache: bool = True) -> list[dict]:
    """Return all qualifying results for a season."""
    url = f"{BASE_URL}/{season}/qualifying.json"
    return _get_all_pages(url, use_cache=use_cache)


def fetch_sprint_results(season: int, use_cache: bool = True) -> list[dict]:
    """Return all sprint results for a season (empty list if none)."""
    try:
        url = f"{BASE_URL}/{season}/sprint.json"
        return _get_all_pages(url, use_cache=use_cache)
    except Exception:
        return []


def fetch_drivers(season: int) -> list[dict]:
    """Return driver list for a season."""
    url = f"{BASE_URL}/{season}/drivers.json"
    data = _get(url)
    return data.get("MRData", {}).get("DriverTable", {}).get("Drivers", [])


# ---------------------------------------------------------------------------
# Transform to per-race weekend records
# ---------------------------------------------------------------------------

def build_weekend_records(season: int, use_cache: bool = True) -> list[dict]:
    """
    Pull race + qualifying data for a season and return a clean list of
    per-race weekend records.

    Each record:
    {
      'season'    : int,
      'round'     : int,
      'race_name' : str,
      'circuit'   : str,
      'drivers'   : [
          {
            'driver_id'  : str,
            'driver_name': str,
            'team'       : str,
            'quali_pos'  : int | None,
            'race_pos'   : int | None,
            'grid_pos'   : int | None,
            'dnf'        : bool,
          },
          ...
      ]
    }
    """
    races    = fetch_race_results(season, use_cache=use_cache)
    qualis   = fetch_qualifying_results(season, use_cache=use_cache)
    sprints  = fetch_sprint_results(season, use_cache=use_cache)

    # Index qualifying by round number
    quali_by_round: dict[int, dict] = {}
    for q in qualis:
        rnd = int(q["round"])
        quali_by_round[rnd] = {
            qr["Driver"]["driverId"]: {
                "quali_pos": _safe_int(qr.get("position")),
                "quali_q1_time": qr.get("Q1") or None,
                "quali_q2_time": qr.get("Q2") or None,
                "quali_q3_time": qr.get("Q3") or None,
                "quali_no_time": not bool(qr.get("Q1")),
                "quali_dsq": _is_dsq(qr.get("positionText", "")) or _is_dsq(qr.get("status", "")),
                "quali_reached_q2": "Q2" in qr,
                "quali_reached_q3": "Q3" in qr,
            }
            for qr in q.get("QualifyingResults", [])
        }

    sprint_by_round: dict[int, dict] = {}
    for sprint in sprints:
        rnd = int(sprint["round"])
        sprint_by_round[rnd] = {
            sr["Driver"]["driverId"]: {
                "sprint_pos": _safe_int(sr.get("position")),
                "sprint_grid_pos": _safe_int(sr.get("grid")),
                "sprint_dnf": _is_non_classified(sr.get("status", "")),
                "sprint_dsq": _is_dsq(sr.get("status", "")) or sr.get("positionText") == "D",
                "sprint_fastest_lap": (sr.get("FastestLap", {}) or {}).get("rank") == "1",
            }
            for sr in sprint.get("SprintResults", [])
        }

    weekends = []
    for race in races:
        rnd       = int(race["round"])
        race_name = race.get("raceName", "")
        circuit   = race.get("Circuit", {}).get("circuitName", "")
        results   = race.get("Results", [])

        quali_pos_map = quali_by_round.get(rnd, {})
        sprint_result_map = sprint_by_round.get(rnd, {})

        drivers = []
        for r in results:
            did      = r["Driver"]["driverId"]
            raw_name = r["Driver"]["givenName"] + " " + r["Driver"]["familyName"]
            name     = DRIVER_NAME_OVERRIDES.get(raw_name, raw_name)
            team     = r["Constructor"]["name"]
            rpos_str = r.get("position")
            status   = r.get("status", "")
            grid_str = r.get("grid")

            quali_meta = quali_pos_map.get(did, {})
            sprint_meta = sprint_result_map.get(did, {})

            race_dsq = _is_dsq(status) or r.get("positionText") == "D"
            dnf = _is_non_classified(status)
            r_pos  = int(rpos_str) if (rpos_str and not dnf and not race_dsq) else None
            g_pos  = int(grid_str) if grid_str and grid_str != "0" else None
            q_pos  = quali_meta.get("quali_pos")
            race_fastest_lap = (r.get("FastestLap", {}) or {}).get("rank") == "1"

            drivers.append({
                "driver_id"  : did,
                "driver_name": name,
                "team"       : team,
                "quali_pos"  : q_pos,
                "quali_q1_time": quali_meta.get("quali_q1_time"),
                "quali_q2_time": quali_meta.get("quali_q2_time"),
                "quali_q3_time": quali_meta.get("quali_q3_time"),
                "quali_no_time": bool(quali_meta.get("quali_no_time", q_pos is None)),
                "quali_dsq": bool(quali_meta.get("quali_dsq", False)),
                "quali_reached_q2": bool(quali_meta.get("quali_reached_q2", False)),
                "quali_reached_q3": bool(quali_meta.get("quali_reached_q3", False)),
                "sprint_pos": sprint_meta.get("sprint_pos"),
                "sprint_grid_pos": sprint_meta.get("sprint_grid_pos"),
                "sprint_dnf": bool(sprint_meta.get("sprint_dnf", False)),
                "sprint_dsq": bool(sprint_meta.get("sprint_dsq", False)),
                "sprint_fastest_lap": bool(sprint_meta.get("sprint_fastest_lap", False)),
                "sprint_overtakes": sprint_meta.get("sprint_overtakes", 0),
                "race_pos"   : r_pos,
                "grid_pos"   : g_pos,
                "race_fastest_lap": race_fastest_lap,
                "race_overtakes": 0,
                "driver_of_day": False,
                "race_dsq": race_dsq,
                "dnf"        : dnf,
            })

        weekends.append({
            "season"   : season,
            "round"    : rnd,
            "race_name": race_name,
            "circuit"  : circuit,
            "drivers"  : drivers,
            "sprint_present": bool(sprint_result_map),
            "constructor_pitstop_times": {},
            "fastest_pitstop": None,
            "pitstop_world_record": None,
        })

    weekends.sort(key=lambda x: x["round"])
    return weekends


def _is_classified(status: str) -> bool:
    """Return True if the status counts as a classified finish in F1."""
    classified_keywords = ["+", "lap", "laps", "finished"]
    s = status.lower()
    return any(k in s for k in classified_keywords)


def _is_dsq(status: str) -> bool:
    s = (status or "").lower()
    return "disqual" in s or s == "d"


def _is_non_classified(status: str) -> bool:
    s = (status or "").lower()
    return not _is_classified(s) and not _is_dsq(s)


def _safe_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Quick driver name normaliser (maps ergast → fantasy display name)
# ---------------------------------------------------------------------------

DRIVER_NAME_MAP = {
    "max_verstappen"    : "Max Verstappen",
    "george_russell"    : "George Russell",
    "lando_norris"      : "Lando Norris",
    "oscar_piastri"     : "Oscar Piastri",
    "kimi_antonelli"    : "Kimi Antonelli",
    "charles_leclerc"   : "Charles Leclerc",
    "lewis_hamilton"    : "Lewis Hamilton",
    "isack_hadjar"      : "Isack Hadjar",
    "pierre_gasly"      : "Pierre Gasly",
    "carlos_sainz"      : "Carlos Sainz",
    "alexander_albon"   : "Alexander Albon",
    "fernando_alonso"   : "Fernando Alonso",
    "lance_stroll"      : "Lance Stroll",
    "oliver_bearman"    : "Oliver Bearman",
    "esteban_ocon"      : "Esteban Ocon",
    "nico_hulkenberg"   : "Nico Hulkenberg",
    "liam_lawson"       : "Liam Lawson",
    "gabriel_bortoleto" : "Gabriel Bortoleto",
    "arvid_lindblad"    : "Arvid Lindblad",
    "franco_colapinto"  : "Franco Colapinto",
    "sergio_perez"      : "Sergio Perez",
    "valtteri_bottas"   : "Valtteri Bottas",
}

TEAM_NAME_MAP = {
    "Red Bull"         : "Red Bull Racing",
    "Mercedes"         : "Mercedes",
    "Ferrari"          : "Ferrari",
    "McLaren"          : "McLaren",
    "Alpine F1 Team"   : "Alpine",
    "Williams"         : "Williams",
    "Aston Martin"     : "Aston Martin",
    "Haas F1 Team"     : "Haas F1 Team",
    "Alfa Romeo"       : "Audi",           # rebranded
    "AlphaTauri"       : "Racing Bulls",   # rebranded
    "RB F1 Team"       : "Racing Bulls",
    "Sauber"           : "Audi",
    "Kick Sauber"      : "Audi",
}


if __name__ == "__main__":
    import pprint
    print("Fetching 2024 season data...")
    weekends = build_weekend_records(2024)
    print(f"Got {len(weekends)} race weekends")
    pprint.pprint(weekends[0])
