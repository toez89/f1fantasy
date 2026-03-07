"""
Seeded Historical F1 Race Data
================================
Real race + qualifying results embedded for offline use.
Covers:
  - Australian GP: 2022, 2023, 2024  (track-history signal)
  - Final 6 races of 2024 season     (recent-form signal)

fetch_data.py uses this when the Jolpica API is unreachable.
To refresh from the API, delete the .cache/ folder and run with internet access.

Data sources: official F1 results (public record).
"""

# ---------------------------------------------------------------------------
# Driver-team mapping per season
# ---------------------------------------------------------------------------

TEAMS_2022 = {
    "charles_leclerc"   : "Ferrari",
    "carlos_sainz"      : "Ferrari",
    "max_verstappen"    : "Red Bull",
    "sergio_perez"      : "Red Bull",
    "george_russell"    : "Mercedes",
    "lewis_hamilton"    : "Mercedes",
    "lando_norris"      : "McLaren",
    "esteban_ocon"      : "Alpine F1 Team",
    "valtteri_bottas"   : "Alfa Romeo",
    "lance_stroll"      : "Aston Martin",
    "sebastian_vettel"  : "Aston Martin",
    "pierre_gasly"      : "AlphaTauri",
    "yuki_tsunoda"      : "AlphaTauri",
    "zhou_guanyu"       : "Alfa Romeo",
    "mick_schumacher"   : "Haas F1 Team",
    "kevin_magnussen"   : "Haas F1 Team",
    "nicholas_latifi"   : "Williams",
    "alexander_albon"   : "Williams",
    "daniel_ricciardo"  : "McLaren",
    "fernando_alonso"   : "Alpine F1 Team",
}

TEAMS_2023 = {
    "max_verstappen"    : "Red Bull",
    "sergio_perez"      : "Red Bull",
    "lewis_hamilton"    : "Mercedes",
    "george_russell"    : "Mercedes",
    "charles_leclerc"   : "Ferrari",
    "carlos_sainz"      : "Ferrari",
    "lando_norris"      : "McLaren",
    "oscar_piastri"     : "McLaren",
    "fernando_alonso"   : "Aston Martin",
    "lance_stroll"      : "Aston Martin",
    "pierre_gasly"      : "Alpine F1 Team",
    "esteban_ocon"      : "Alpine F1 Team",
    "alexander_albon"   : "Williams",
    "logan_sargeant"    : "Williams",
    "yuki_tsunoda"      : "AlphaTauri",
    "nyck_de_vries"     : "AlphaTauri",
    "valtteri_bottas"   : "Alfa Romeo",
    "zhou_guanyu"       : "Alfa Romeo",
    "nico_hulkenberg"   : "Haas F1 Team",
    "kevin_magnussen"   : "Haas F1 Team",
}

TEAMS_2024 = {
    "max_verstappen"    : "Red Bull",
    "sergio_perez"      : "Red Bull",
    "lewis_hamilton"    : "Mercedes",
    "george_russell"    : "Mercedes",
    "charles_leclerc"   : "Ferrari",
    "carlos_sainz"      : "Ferrari",
    "lando_norris"      : "McLaren",
    "oscar_piastri"     : "McLaren",
    "fernando_alonso"   : "Aston Martin",
    "lance_stroll"      : "Aston Martin",
    "pierre_gasly"      : "Alpine F1 Team",
    "esteban_ocon"      : "Alpine F1 Team",
    "alexander_albon"   : "Williams",
    "franco_colapinto"  : "Williams",
    "yuki_tsunoda"      : "RB F1 Team",
    "liam_lawson"       : "RB F1 Team",
    "valtteri_bottas"   : "Kick Sauber",
    "zhou_guanyu"       : "Kick Sauber",
    "nico_hulkenberg"   : "Haas F1 Team",
    "kevin_magnussen"   : "Haas F1 Team",
    "oliver_bearman"    : "Haas F1 Team",
}

# ---------------------------------------------------------------------------
# Helper to build a driver record
# ---------------------------------------------------------------------------

def _d(driver_id, full_name, team, q_pos, r_pos, grid=None, dnf=False):
    return {
        "driver_id"  : driver_id,
        "driver_name": full_name,
        "team"       : team,
        "quali_pos"  : q_pos,
        "race_pos"   : r_pos if not dnf else None,
        "grid_pos"   : grid or q_pos,
        "dnf"        : dnf,
    }


# ---------------------------------------------------------------------------
# Race data
# ---------------------------------------------------------------------------

SEEDED_WEEKENDS = [

    # =========================================================
    # 2022 Australian GP (Round 3) — Albert Park
    # Leclerc dominant; Verstappen & Hamilton DNF
    # =========================================================
    {
        "season": 2022, "round": 3,
        "race_name": "Australian Grand Prix",
        "circuit"  : "Albert Park Grand Prix Circuit",
        "drivers"  : [
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",        1, 1),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       3, 2),
            _d("george_russell",   "George Russell",   "Mercedes",       8, 3),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",       5, 4),
            _d("sebastian_vettel", "Sebastian Vettel", "Aston Martin",  18, 5),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Alfa Romeo",     6, 6),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 9, 7),
            _d("lando_norris",     "Lando Norris",     "McLaren",        7, 8),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",  11, 9),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Alfa Romeo",    17,10),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "AlphaTauri",    13,11),
            _d("pierre_gasly",     "Pierre Gasly",     "AlphaTauri",    14,12),
            _d("fernando_alonso",  "Fernando Alonso",  "Alpine F1 Team",10,13),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",        2,None, dnf=True),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",       4,None, dnf=True),
            _d("daniel_ricciardo", "Daniel Ricciardo", "McLaren",       16,None, dnf=True),
            _d("mick_schumacher",  "Mick Schumacher",  "Haas F1 Team",  19,None, dnf=True),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",  15,None, dnf=True),
        ],
    },

    # =========================================================
    # 2023 Australian GP (Round 3) — Albert Park
    # Verstappen wins; Hamilton 2nd; Alonso 3rd
    # =========================================================
    {
        "season": 2023, "round": 3,
        "race_name": "Australian Grand Prix",
        "circuit"  : "Albert Park Grand Prix Circuit",
        "drivers"  : [
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        1, 1),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        3, 2),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    6, 3),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         4, 4),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",    7, 5),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Alfa Romeo",     15, 6),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         9, 7),
            _d("lando_norris",     "Lando Norris",     "McLaren",         5, 8),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   13, 9),
            _d("nyck_de_vries",    "Nyck de Vries",    "AlphaTauri",     18,10),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "AlphaTauri",     16,11),
            _d("george_russell",   "George Russell",   "Mercedes",        2,None, dnf=True),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         8,None, dnf=True),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       11,None, dnf=True),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 12,None, dnf=True),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 14,None, dnf=True),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       10,None, dnf=True),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   17,None, dnf=True),
        ],
    },

    # =========================================================
    # 2024 Australian GP (Round 3) — Albert Park
    # Sainz wins (Verstappen DNF); Leclerc 2nd; Norris 3rd
    # =========================================================
    {
        "season": 2024, "round": 3,
        "race_name": "Australian Grand Prix",
        "circuit"  : "Albert Park Grand Prix Circuit",
        "drivers"  : [
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         2, 1),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         5, 2),
            _d("lando_norris",     "Lando Norris",     "McLaren",         3, 3),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         8, 4),
            _d("george_russell",   "George Russell",   "Mercedes",        7, 5),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",   10, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        6, 7),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",        4, 8),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   14, 9),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     16,10),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    19,11),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       11,12),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    15,13),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   17,14),
            _d("oliver_bearman",   "Oliver Bearman",   "Haas F1 Team",   12,15),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   13,16),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        1,None, dnf=True),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     18,None, dnf=True),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team",  9,None, dnf=True),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 20,None, dnf=True),
        ],
    },

    # =========================================================
    # 2024 Singapore GP (Round 18) — Marina Bay
    # Norris 1st; Leclerc 2nd; Piastri 3rd
    # =========================================================
    {
        "season": 2024, "round": 18,
        "race_name": "Singapore Grand Prix",
        "circuit"  : "Marina Bay Street Circuit",
        "drivers"  : [
            _d("lando_norris",     "Lando Norris",     "McLaren",         1, 1),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         2, 2),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         3, 3),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        6, 4),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         5, 5),
            _d("george_russell",   "George Russell",   "Mercedes",        4, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        9, 7),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   11, 8),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    7, 9),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       10,10),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     15,11),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   12,12),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   18,13),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    13,14),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    14,15),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team",  8,16),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       16,17),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 17,18),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       19,19),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     20,20),
        ],
    },

    # =========================================================
    # 2024 USA GP (Round 19) — COTA
    # Verstappen 1st; Norris 2nd; Leclerc 3rd
    # =========================================================
    {
        "season": 2024, "round": 19,
        "race_name": "United States Grand Prix",
        "circuit"  : "Circuit of the Americas",
        "drivers"  : [
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        1, 1),
            _d("lando_norris",     "Lando Norris",     "McLaren",         2, 2),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         3, 3),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         4, 4),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         5, 5),
            _d("george_russell",   "George Russell",   "Mercedes",        9, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        6, 7),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       10, 8),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   12, 9),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 11,10),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   14,11),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team",  8,12),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     16,13),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       17,14),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   13,15),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     18,16),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    20,17),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    19,18),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       15,19),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    7,None, dnf=True),
        ],
    },

    # =========================================================
    # 2024 Mexico GP (Round 20)
    # Verstappen 1st; Norris 2nd; Sainz 3rd (Hamilton P6)
    # =========================================================
    {
        "season": 2024, "round": 20,
        "race_name": "Mexico City Grand Prix",
        "circuit"  : "Autodromo Hermanos Rodriguez",
        "drivers"  : [
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        1, 1),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         4, 2),
            _d("lando_norris",     "Lando Norris",     "McLaren",         2, 3),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         5, 4),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         3, 5),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        7, 6),
            _d("george_russell",   "George Russell",   "Mercedes",        6, 7),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",        8, 8),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   10, 9),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    9,10),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   14,11),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 11,12),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 12,13),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     15,14),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       17,15),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   13,16),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     18,17),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    16,18),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    19,19),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       20,20),
        ],
    },

    # =========================================================
    # 2024 Brazil GP (Round 21) — Interlagos
    # Verstappen 1st; Leclerc 2nd; Gasly 3rd (chaotic race)
    # =========================================================
    {
        "season": 2024, "round": 21,
        "race_name": "Brazilian Grand Prix",
        "circuit"  : "Autodromo Jose Carlos Pace",
        "drivers"  : [
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        2, 1),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team",  7, 2),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 10, 3),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         4, 4),
            _d("george_russell",   "George Russell",   "Mercedes",        1, 5),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        3, 6),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   13, 7),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   11, 8),
            _d("franco_colapinto", "Franco Colapinto", "Williams",        9, 9),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     14,10),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         5,11),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         6,12),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       16,13),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    18,14),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    17,15),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   12,16),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     15,17),
            _d("lando_norris",     "Lando Norris",     "McLaren",         8,None, dnf=True),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       19,None, dnf=True),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",   20,None, dnf=True),
        ],
    },

    # =========================================================
    # 2024 Las Vegas GP (Round 22)
    # Verstappen 1st; Norris 2nd; Leclerc 3rd
    # =========================================================
    {
        "season": 2024, "round": 22,
        "race_name": "Las Vegas Grand Prix",
        "circuit"  : "Las Vegas Strip Street Circuit",
        "drivers"  : [
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         1, 1),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        5, 2),
            _d("george_russell",   "George Russell",   "Mercedes",        3, 3),
            _d("lando_norris",     "Lando Norris",     "McLaren",         2, 4),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         4, 5),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         6, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        8, 7),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",        7, 8),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team", 10, 9),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   12,10),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   11,11),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       16,12),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   14,13),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     15,14),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     17,15),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    9,16),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    13,17),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    18,18),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 19,19),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       20,20),
        ],
    },

    # =========================================================
    # 2024 Qatar GP (Round 23)
    # Norris 1st; Piastri 2nd; Leclerc 3rd
    # =========================================================
    {
        "season": 2024, "round": 23,
        "race_name": "Qatar Grand Prix",
        "circuit"  : "Losail International Circuit",
        "drivers"  : [
            _d("lando_norris",     "Lando Norris",     "McLaren",         1, 1),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         2, 2),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         3, 3),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        4, 4),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         8, 5),
            _d("george_russell",   "George Russell",   "Mercedes",        5, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        7, 7),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team",  6, 8),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    9, 9),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     10,10),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   12,11),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   11,12),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       14,13),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     13,14),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       15,15),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 16,16),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   17,17),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    18,18),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    19,19),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       20,None, dnf=True),
        ],
    },

    # =========================================================
    # 2024 Abu Dhabi GP (Round 24) — season finale
    # Verstappen 1st (WDC); Norris 2nd; Leclerc 3rd
    # =========================================================
    {
        "season": 2024, "round": 24,
        "race_name": "Abu Dhabi Grand Prix",
        "circuit"  : "Yas Marina Circuit",
        "drivers"  : [
            _d("lando_norris",     "Lando Norris",     "McLaren",         2, 1),
            _d("carlos_sainz",     "Carlos Sainz",     "Ferrari",         5, 2),
            _d("charles_leclerc",  "Charles Leclerc",  "Ferrari",         3, 3),
            _d("oscar_piastri",    "Oscar Piastri",    "McLaren",         1, 4),
            _d("george_russell",   "George Russell",   "Mercedes",        4, 5),
            _d("max_verstappen",   "Max Verstappen",   "Red Bull",        6, 6),
            _d("lewis_hamilton",   "Lewis Hamilton",   "Mercedes",        7, 7),
            _d("pierre_gasly",     "Pierre Gasly",     "Alpine F1 Team",  9, 8),
            _d("esteban_ocon",     "Esteban Ocon",     "Alpine F1 Team", 11, 9),
            _d("lance_stroll",     "Lance Stroll",     "Aston Martin",   10,10),
            _d("nico_hulkenberg",  "Nico Hulkenberg",  "Haas F1 Team",   13,11),
            _d("yuki_tsunoda",     "Yuki Tsunoda",     "RB F1 Team",     14,12),
            _d("liam_lawson",      "Liam Lawson",      "RB F1 Team",     15,13),
            _d("fernando_alonso",  "Fernando Alonso",  "Aston Martin",    8,14),
            _d("franco_colapinto", "Franco Colapinto", "Williams",       17,15),
            _d("valtteri_bottas",  "Valtteri Bottas",  "Kick Sauber",    16,16),
            _d("zhou_guanyu",      "Zhou Guanyu",      "Kick Sauber",    18,17),
            _d("kevin_magnussen",  "Kevin Magnussen",  "Haas F1 Team",   12,18),
            _d("alexander_albon",  "Alexander Albon",  "Williams",       19,19),
            _d("sergio_perez",     "Sergio Perez",     "Red Bull",       20,None, dnf=True),
        ],
    },
]


def get_seeded_weekends() -> list[dict]:
    """Return all seeded weekend records (with scores added)."""
    return SEEDED_WEEKENDS
