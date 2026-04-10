"""
Generate synthetic corporate employee travel profiles for a company in Yverdon-les-Bains.

Employees commute Renens VD -> Yverdon-les-Bains in the morning,
and return Yverdon-les-Bains -> Renens VD in the evening.

Assumptions:
  - Work starts at 9AM at the latest  → morning departure from Renens 07:00-08:25
  - Work ends at 5PM                  → evening departure from Yverdon 17:00-19:00
  - Remote days: Monday and Friday (not all employees, see REMOTE_PROBABILITY)
  - ~100 employees travel on a typical Tue/Wed/Thu

Output:
  data/synthetic/employees.csv         one row per employee
  data/synthetic/corporate_load.csv    one row per (date, trip) in dataset.csv
                                       with corporate_load feature added

Usage:
  python data/generate_synthetic.py
"""

import csv
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- Config -----------------------------------------------------------
RANDOM_SEED = 42

# ~143 employees so that after removing ~30% on Mon/Fri we get ~100
N_EMPLOYEES = 143
REMOTE_PROBABILITY = 0.70  # probability an employee works remotely on Mon/Fri
REMOTE_DAYS = {0, 4}  # Monday=0, Friday=4

HOME_STATION = "Renens VD"
WORK_STATION = "Yverdon-les-Bains"

# Morning: must arrive Yverdon by 09:00, S30 takes ~35min
MORNING_DEPARTURE_START = (7, 0)  # 07:00
MORNING_DEPARTURE_END = (8, 25)  # 08:25

# Evening: after 17:00
EVENING_DEPARTURE_START = (17, 0)
EVENING_DEPARTURE_END = (19, 0)

# Match employee to trip if within this window (minutes)
MATCH_WINDOW_MINUTES = 20

DATASET_PATH = Path("data/processed/dataset.csv")
EMPLOYEES_PATH = Path("data/synthetic/employees.csv")
LOAD_PATH = Path("data/synthetic/corporate_load.csv")

EMPLOYEE_COLUMNS = [
    "employee_id",
    "home_station",
    "work_station",
    "morning_departure",  # HH:MM from home station
    "evening_departure",  # HH:MM from work station
    "is_remote_mon_fri",  # 1 = works remotely on Mon and Fri
    "travel_class",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Helper functions -------------------------------------------------


def random_time(start: tuple[int, int], end: tuple[int, int]) -> str:
    """Return a random HH:MM string between start and end times."""
    start_min = start[0] * 60 + start[1]
    end_min = end[0] * 60 + end[1]
    t = random.randint(start_min, end_min)
    return f"{t // 60:02d}:{t % 60:02d}"


def time_to_minutes(hhmm: str) -> int:
    """Convert HH:MM string to total minutes since midnight."""
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def is_remote(employee: dict, day_of_week: int) -> bool:
    """Return True if employee works remotely on this day."""
    if day_of_week in REMOTE_DAYS:
        return employee["is_remote_mon_fri"] == "1"
    return False


def employee_on_trip(employee: dict, trip: dict) -> bool:
    """
    Return True if this employee is expected on this trip.
    Matches direction, station, day, and departure time window.
    """
    dow = int(trip["day_of_week"])
    if is_remote(employee, dow):
        return False

    direction = int(trip["direction"])  # 1 = towards Yverdon, 0 = towards Renens

    # morning trip: Renens -> Yverdon (direction=1)
    if direction == 1:
        if employee["home_station"] != HOME_STATION:
            return False
        emp_min = time_to_minutes(employee["morning_departure"])
        trip_min = int(trip["hour"]) * 60 + int(trip["minute"])
        return abs(emp_min - trip_min) <= MATCH_WINDOW_MINUTES

    # evening trip: Yverdon -> Renens (direction=0)
    else:
        if employee["work_station"] != WORK_STATION:
            return False
        emp_min = time_to_minutes(employee["evening_departure"])
        trip_min = int(trip["hour"]) * 60 + int(trip["minute"])
        return abs(emp_min - trip_min) <= MATCH_WINDOW_MINUTES


# --- Generation -------------------------------------------------------


def generate_employees() -> list[dict]:
    """Generate N_EMPLOYEES synthetic employee records."""
    employees = []
    for i in range(N_EMPLOYEES):
        employees.append(
            {
                "employee_id": f"E{i:04d}",
                "home_station": HOME_STATION,
                "work_station": WORK_STATION,
                "morning_departure": random_time(MORNING_DEPARTURE_START, MORNING_DEPARTURE_END),
                "evening_departure": random_time(EVENING_DEPARTURE_START, EVENING_DEPARTURE_END),
                "is_remote_mon_fri": "1" if random.random() < REMOTE_PROBABILITY else "0",
                "travel_class": random.choice(["1st", "2nd", "2nd", "2nd"]),
            }
        )
    return employees


def compute_corporate_load(employees: list[dict], trips: list[dict]) -> list[dict]:
    """
    For each trip in the dataset, count how many employees are expected on it.
    Returns the full dataset with a corporate_load column added.
    """
    results = []
    for trip in trips:
        load = sum(1 for e in employees if employee_on_trip(e, trip))
        results.append({**trip, "corporate_load": load})
    return results


# --- Main -------------------------------------------------------------


def main():
    random.seed(RANDOM_SEED)

    EMPLOYEES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: generate employees
    log.info(f"Generating {N_EMPLOYEES} employees ...")
    employees = generate_employees()

    with open(EMPLOYEES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EMPLOYEE_COLUMNS)
        writer.writeheader()
        writer.writerows(employees)
    log.info(f"  Written to {EMPLOYEES_PATH}")

    # Step 2: load dataset
    log.info(f"Loading {DATASET_PATH} ...")
    with open(DATASET_PATH, encoding="utf-8") as f:
        trips = list(csv.DictReader(f))
    log.info(f"  {len(trips)} trips loaded")

    # Step 3: compute corporate_load per trip
    log.info("Computing corporate load per trip ...")
    enriched = compute_corporate_load(employees, trips)

    # Step 4: write enriched dataset back
    log.info(f"Writing {LOAD_PATH} ...")
    fieldnames = list(trips[0].keys()) + ["corporate_load"]
    with open(LOAD_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    # Step 5: summary
    morning_loads = [r["corporate_load"] for r in enriched if int(r["direction"]) == 1]
    evening_loads = [r["corporate_load"] for r in enriched if int(r["direction"]) == 0]

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0

    log.info("--- Summary ---")
    log.info(f"  Employees generated:       {len(employees)}")
    log.info(
        f"  Remote on Mon/Fri:         {sum(1 for e in employees if e['is_remote_mon_fri']=='1')}"
    )
    log.info(f"  Avg morning corporate load:{avg(morning_loads)} employees/trip")
    log.info(f"  Avg evening corporate load:{avg(evening_loads)} employees/trip")
    log.info(f"  Done. Written to {LOAD_PATH}")


if __name__ == "__main__":
    main()
