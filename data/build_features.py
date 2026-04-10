"""
Build training dataset from filtered IstDaten for the Renens VD <-> Yverdon corridor.

Input:  data/raw/renens_yverdon_filtered.csv
Output: data/processed/dataset.csv

Logic per trip (FAHRT_BEZEICHNER):
  - Each trip has 2 rows: one for Renens VD, one for Yverdon-les-Bains
  - Determine direction from scheduled departure times
  - Use the ORIGIN stop (first departure) as the prediction target
  - Skip trips where origin has no scheduled departure time (terminus stops)
  - Compute delay_minutes = AB_PROGNOSE - ABFAHRTSZEIT
  - Label: is_delayed = 1 if delay_minutes > 5 else 0

Usage:
  python data/build_features.py
"""

import csv
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# --- Config -----------------------------------------------------------
INPUT_PATH = Path("data/raw/renens_yverdon_filtered.csv")
OUTPUT_PATH = Path("data/processed/dataset.csv")

STATION_RENENS = "Renens VD"
STATION_YVERDON = "Yverdon-les-Bains"
DELAY_THRESHOLD_MINUTES = 5
DATETIME_FORMATS = ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M")

OUTPUT_COLUMNS = [
    "date",
    "trip_id",
    "line",
    "direction",  # 0 = towards Renens, 1 = towards Yverdon
    "hour",
    "minute",
    "day_of_week",  # 0 = Monday, 6 = Sunday
    "is_cancelled",
    "delay_minutes",
    "is_delayed",  # label
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Helper functions -------------------------------------------------


def parse_datetime(dt_str: str) -> datetime | None:
    """Parse a datetime string, return None if empty or malformed."""
    if not dt_str.strip():
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def compute_delay(scheduled: str, actual: str) -> float | None:
    """Compute delay in minutes between scheduled and actual departure."""
    sched = parse_datetime(scheduled)
    act = parse_datetime(actual)
    if sched is None or act is None:
        return None
    return (act - sched).total_seconds() / 60


def determine_origin(row_a: dict, row_b: dict) -> dict | None:
    """
    Given two stop rows for a trip, return the origin row (first departure).
    Returns None if neither row has a valid scheduled departure time.
    """
    time_a = parse_datetime(row_a["ABFAHRTSZEIT"])
    time_b = parse_datetime(row_b["ABFAHRTSZEIT"])

    if time_a is None and time_b is None:
        return None  # skip: no departure times available
    if time_a is None:
        return row_b
    if time_b is None:
        return row_a
    return row_a if time_a < time_b else row_b


def build_features(origin_row: dict) -> dict | None:
    """
    Extract features from the origin stop row.
    Returns None if delay cannot be computed.
    """
    delay = compute_delay(
        origin_row["ABFAHRTSZEIT"],
        origin_row["AB_PROGNOSE"],
    )
    if delay is None:
        return None

    sched = parse_datetime(origin_row["ABFAHRTSZEIT"])
    direction = 1 if origin_row["HALTESTELLEN_NAME"] == STATION_YVERDON else 0

    return {
        "date": origin_row["BETRIEBSTAG"],
        "trip_id": origin_row["FAHRT_BEZEICHNER"],
        "line": origin_row["LINIEN_TEXT"],
        "direction": direction,
        "hour": sched.hour,
        "minute": sched.minute,
        "day_of_week": sched.weekday(),
        "is_cancelled": 1 if origin_row["FAELLT_AUS_TF"].lower() == "true" else 0,
        "delay_minutes": round(delay, 2),
        "is_delayed": 1 if delay > DELAY_THRESHOLD_MINUTES else 0,
    }


# --- Main -------------------------------------------------------------


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: group rows by trip ID
    log.info(f"Reading {INPUT_PATH} ...")
    trips: dict[tuple[str, str], list[dict]] = defaultdict(list)
    total_input_rows = 0

    with open(INPUT_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            trips[(row["FAHRT_BEZEICHNER"], row["BETRIEBSTAG"])].append(row)
            total_input_rows += 1

    log.info(f"  {total_input_rows} rows, {len(trips)} trips")

    # Step 2: build one feature row per trip
    log.info("Building features ...")
    skipped_no_departure = 0
    skipped_no_delay = 0
    output_rows = []

    for trip_id, rows in trips.items():
        if len(rows) != 2:
            # should not happen after corridor filter, but guard anyway
            continue

        origin = determine_origin(rows[0], rows[1])
        if origin is None:
            skipped_no_departure += 1
            continue

        features = build_features(origin)
        if features is None:
            skipped_no_delay += 1
            continue

        output_rows.append(features)

    # Step 3: write output
    log.info(f"Writing {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    # Step 4: summary
    total = len(output_rows)
    n_delayed = sum(r["is_delayed"] for r in output_rows)
    n_cancelled = sum(r["is_cancelled"] for r in output_rows)

    log.info("--- Summary ---")
    log.info(f"  Output rows:        {total}")
    log.info(f"  Delayed (>5 min):   {n_delayed} ({100*n_delayed/total:.1f}%)")
    log.info(f"  Cancelled:          {n_cancelled} ({100*n_cancelled/total:.1f}%)")
    log.info(f"  Skipped (no dept):  {skipped_no_departure}")
    log.info(f"  Skipped (no delay): {skipped_no_delay}")
    log.info(f"Done. Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
