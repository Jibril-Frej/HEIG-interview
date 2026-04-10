"""
Build GRU training sequences from filtered IstDaten for Renens VD <-> Yverdon corridor.

Input:  data/raw/renens_yverdon_filtered.csv
Output:
  data/processed/sequences.npz  — padded numpy arrays (X, y, mask) for GRU training
  data/processed/dataset.csv    — flat CSV for inspection

Sequence definition:
  One sequence = all trips on a given (date, direction), ordered by scheduled departure.
  Each step = one trip with features observable at or before departure.

Features per step (X):
  hour, minute, day_of_week, direction, is_cancelled, departure_delay_minutes

Label per step (y):
  is_delayed = 1 if arrival delay at destination > DELAY_THRESHOLD_MINUTES

Usage:
  python data/build_features.py
"""

import csv
import logging
import numpy as np
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# --- Config -----------------------------------------------------------
INPUT_PATH = Path("data/raw/renens_yverdon_filtered.csv")
OUTPUT_CSV = Path("data/processed/dataset.csv")
OUTPUT_NPZ = Path("data/processed/sequences.npz")

STATION_RENENS = "Renens VD"
STATION_YVERDON = "Yverdon-les-Bains"
DELAY_THRESHOLD_MINUTES = 3
DATETIME_FORMATS = ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M")

FEATURE_COLUMNS = [
    "hour",
    "minute",
    "day_of_week",
    "direction",
    "is_cancelled",
    "departure_delay_minutes",
]
CSV_COLUMNS = ["date", "line", "direction", "seq_pos", "seq_len"] + FEATURE_COLUMNS + ["is_delayed"]

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
    """Compute delay in minutes between two datetime strings."""
    sched = parse_datetime(scheduled)
    act = parse_datetime(actual)
    if sched is None or act is None:
        return None
    return (act - sched).total_seconds() / 60


def determine_origin(row_a: dict, row_b: dict) -> tuple[dict, dict] | tuple[None, None]:
    """
    Given two stop rows for a trip, return (origin_row, dest_row).
    Origin is the stop with the earlier scheduled departure.
    Returns (None, None) if neither row has a valid scheduled departure time.
    """
    time_a = parse_datetime(row_a["ABFAHRTSZEIT"])
    time_b = parse_datetime(row_b["ABFAHRTSZEIT"])

    if time_a is None and time_b is None:
        return None, None
    if time_a is None:
        return row_b, row_a
    if time_b is None:
        return row_a, row_b
    return (row_a, row_b) if time_a <= time_b else (row_b, row_a)


def build_trip(origin_row: dict, dest_row: dict) -> dict | None:
    """
    Build feature dict for a single trip.
    Returns None if departure or arrival delay cannot be computed.
    """
    dep_delay = compute_delay(origin_row["ABFAHRTSZEIT"], origin_row["AB_PROGNOSE"])
    arr_delay = compute_delay(dest_row["ANKUNFTSZEIT"], dest_row["AN_PROGNOSE"])
    if dep_delay is None or arr_delay is None:
        return None

    sched = parse_datetime(origin_row["ABFAHRTSZEIT"])
    direction = 1 if origin_row["HALTESTELLEN_NAME"] == STATION_YVERDON else 0

    return {
        "date": origin_row["BETRIEBSTAG"],
        "line": origin_row["LINIEN_TEXT"],
        "direction": direction,
        "hour": sched.hour,
        "minute": sched.minute,
        "day_of_week": sched.weekday(),
        "is_cancelled": 1 if origin_row["FAELLT_AUS_TF"].lower() == "true" else 0,
        "departure_delay_minutes": round(dep_delay, 2),
        "is_delayed": 1 if arr_delay > DELAY_THRESHOLD_MINUTES else 0,
        "_sched_dt": sched,  # for sorting only, not written to output
    }


def build_sequences(
    trips_by_day_dir: dict[tuple, list[dict]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[dict]]]:
    """
    Pad per-(date, direction) trip lists into numpy arrays for GRU training.
    Returns X (n_seq, max_len, n_feat), y (n_seq, max_len), mask (n_seq, max_len),
    and the sorted sequences for CSV output.
    """
    sequences = [
        sorted(trips, key=lambda t: t["_sched_dt"])
        for _, trips in sorted(trips_by_day_dir.items())
    ]

    max_len = max(len(s) for s in sequences)
    n_feat = len(FEATURE_COLUMNS)

    X = np.zeros((len(sequences), max_len, n_feat), dtype=np.float32)
    y = np.zeros((len(sequences), max_len), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for t, trip in enumerate(seq):
            X[i, t] = [trip[f] for f in FEATURE_COLUMNS]
            y[i, t] = trip["is_delayed"]
            mask[i, t] = 1.0

    return X, y, mask, sequences


# --- Main -------------------------------------------------------------


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: group rows by (trip_id, date)
    log.info(f"Reading {INPUT_PATH} ...")
    raw_trips: dict[tuple[str, str], list[dict]] = defaultdict(list)
    total_input_rows = 0

    with open(INPUT_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            raw_trips[(row["FAHRT_BEZEICHNER"], row["BETRIEBSTAG"])].append(row)
            total_input_rows += 1

    log.info(f"  {total_input_rows} rows, {len(raw_trips)} trips")

    # Step 2: build one feature dict per trip, group by (date, direction)
    log.info("Building features ...")
    trips_by_day_dir: dict[tuple[str, int], list[dict]] = defaultdict(list)
    skipped = 0

    for rows in raw_trips.values():
        if len(rows) != 2:
            skipped += 1
            continue
        origin, dest = determine_origin(rows[0], rows[1])
        if origin is None or dest is None:
            skipped += 1
            continue
        trip = build_trip(origin, dest)
        if trip is None:
            skipped += 1
            continue
        trips_by_day_dir[(trip["date"], trip["direction"])].append(trip)

    n_trips = sum(len(v) for v in trips_by_day_dir.values())
    log.info(f"  {n_trips} valid trips across {len(trips_by_day_dir)} (date, direction) sequences")
    log.info(f"  {skipped} trips skipped")

    # Step 3: build padded numpy sequences
    log.info("Building sequences ...")
    X, y, mask, sequences = build_sequences(trips_by_day_dir)
    seq_lengths = [int(mask[i].sum()) for i in range(len(sequences))]
    log.info(f"  {len(sequences)} sequences, max length {X.shape[1]}, {X.shape[2]} features")

    # Step 4: save numpy arrays
    log.info(f"Writing {OUTPUT_NPZ} ...")
    np.savez(OUTPUT_NPZ, X=X, y=y, mask=mask, feature_columns=FEATURE_COLUMNS)

    # Step 5: save flat CSV for inspection
    log.info(f"Writing {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for seq in sequences:
            seq_len = len(seq)
            for pos, trip in enumerate(seq):
                row = {k: trip.get(k) for k in CSV_COLUMNS if k not in ("seq_pos", "seq_len")}
                row["seq_pos"] = pos
                row["seq_len"] = seq_len
                writer.writerow(row)

    # Step 6: summary
    total_steps = int(mask.sum())
    n_delayed = int((y * mask).sum())
    log.info("--- Summary ---")
    log.info(f"  Sequences:          {len(sequences)}")
    log.info(f"  Total steps:        {total_steps}")
    log.info(f"  Delayed (>3 min):   {n_delayed} ({100*n_delayed/total_steps:.1f}%)")
    log.info(f"  Avg seq length:     {sum(seq_lengths)/len(seq_lengths):.1f}")
    log.info(f"  Max seq length:     {max(seq_lengths)}")
    log.info(f"Done. Written to {OUTPUT_NPZ} and {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
