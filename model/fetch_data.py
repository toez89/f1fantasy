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


def _get_all_pages(url: str, limit: int = 100) -> list:
    """Paginate through all results from an Ergast-style endpoint."""
    results, offset = [], 0
    while True:
        data = _get(url, params={"limit": limit, "offset": offset})
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

def fetch_race_results(season: int) -> list[dict]:
    """Return all race results for a season as a list of Race dicts."""
    url = f"{BASE_URL}/{season}/results.json"
    return _get_all_pages(url)


def fetch_qualifying_results(season: int) -> list[dict]:
    """Return all qualifying results for a season."""
    url = f"{BASE_URL}/{season}/qualifying.json"
    return _get_all_pages(url)


def fetch_sprint_results(season: int) -> list[dict]:
    """Return all sprint results for a season (empty list if none)."""
    try:
        url = f"{BASE_URL}/{season}/sprint.json"
        return _get_all_pages(url)
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

def build_weekend_records(season: int) -> list[dict]:
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
    races   = fetch_race_results(season)
    qualis  = fetch_qualifying_results(season)

    # Index qualifying by round number
    quali_by_round: dict[int, dict] = {}
    for q in qualis:
        rnd = int(q["round"])
        quali_by_round[rnd] = {
            qr["Driver"]["driverId"]: int(qr["position"])
            for qr in q.get("QualifyingResults", [])
        }

    weekends = []
    for race in races:
        rnd       = int(race["round"])
        race_name = race.get("raceName", "")
        circuit   = race.get("Circuit", {}).get("circuitName", "")
        results   = race.get("Results", [])

        quali_pos_map = quali_by_round.get(rnd, {})

        drivers = []
        for r in results:
            did      = r["Driver"]["driverId"]
            name     = r["Driver"]["givenName"] + " " + r["Driver"]["familyName"]
            team     = r["Constructor"]["name"]
            rpos_str = r.get("position")
            status   = r.get("status", "")
            grid_str = r.get("grid")

            # Race position – only set if driver actually finished
            dnf    = "finished" not in status.lower() and not _is_classified(status)
            r_pos  = int(rpos_str) if (rpos_str and not dnf) else None
            g_pos  = int(grid_str) if grid_str and grid_str != "0" else None
            q_pos  = quali_pos_map.get(did)

            drivers.append({
                "driver_id"  : did,
                "driver_name": name,
                "team"       : team,
                "quali_pos"  : q_pos,
                "race_pos"   : r_pos,
                "grid_pos"   : g_pos,
                "dnf"        : dnf,
            })

        weekends.append({
            "season"   : season,
            "round"    : rnd,
            "race_name": race_name,
            "circuit"  : circuit,
            "drivers"  : drivers,
        })

    weekends.sort(key=lambda x: x["round"])
    return weekends


def _is_classified(status: str) -> bool:
    """Return True if the status counts as a classified finish in F1."""
    classified_keywords = ["+", "lap", "laps", "finished"]
    s = status.lower()
    return any(k in s for k in classified_keywords)


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
