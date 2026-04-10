"""
Download and filter SBB IstDaten (actual data) for the Renens VD <-> Yverdon-les-Bains corridor.

Strategy:
  - Stream each daily CSV from the archive (one ZIP per month)
  - Keep only rows where HALTESTELLEN_NAME is one of the two target stations
  - Then keep only trip IDs (FAHRT_BEZEICHNER) that appear at BOTH stations
  - This captures all lines serving the corridor (S30, IC51, etc.)

Output:
  data/raw/renens_yverdon_filtered.csv  (one row per stop, all matching trips)

Usage:
  python data/download_sbb.py --months 2025-01 2025-02 2025-03
"""

import argparse
import csv
import io
import logging
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

# --- Config -----------------------------------------------------------
STATIONS = {"Renens VD", "Yverdon-les-Bains"}
VALID_STATUSES = {"REAL", "GESCHAETZT"}
OPERATOR_FILTER = "SBB"

ARCHIVE_URL_TEMPLATE = (
    "https://archive.opentransportdata.swiss/istdaten/{year}/ist-daten-v2-{year}-{month:02d}.zip"
)
OUTPUT_PATH = Path("data/raw/renens_yverdon_filtered.csv")

COLUMNS = [
    "BETRIEBSTAG",
    "FAHRT_BEZEICHNER",
    "BETREIBER_ABK",
    "LINIEN_TEXT",
    "VERKEHRSMITTEL_TEXT",
    "FAELLT_AUS_TF",
    "BPUIC",
    "HALTESTELLEN_NAME",
    "ANKUNFTSZEIT",
    "AN_PROGNOSE",
    "AN_PROGNOSE_STATUS",
    "ABFAHRTSZEIT",
    "AB_PROGNOSE",
    "AB_PROGNOSE_STATUS",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


# --- Core functions ---------------------------------------------------


def parse_month(month_str: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' string into (year, month) integers."""
    year, month = month_str.split("-")
    return int(year), int(month)


def build_zip_url(year: int, month: int) -> str:
    """Build the archive ZIP URL for a given year and month."""
    return ARCHIVE_URL_TEMPLATE.format(year=year, month=month)


def stream_zip(url: str) -> zipfile.ZipFile:
    """Download a ZIP file into memory and return a ZipFile object."""
    log.info(f"Downloading {url} ...")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    log.info(f"Downloaded {len(response.content) / 1e6:.1f} MB")
    return zipfile.ZipFile(io.BytesIO(response.content))


def filter_csv_member(zf: zipfile.ZipFile, member_name: str) -> list[dict]:
    """
    Read one CSV member from a ZIP and return rows for target stations only.
    Only keeps rows where operator is SBB and status is REAL or GESCHAETZT.
    """
    rows = []
    with zf.open(member_name) as f:
        reader = csv.DictReader(
            io.TextIOWrapper(f, encoding="utf-8"),
            delimiter=";",
        )
        for row in reader:
            if row.get("BETREIBER_ABK") != OPERATOR_FILTER:
                continue
            if row.get("HALTESTELLEN_NAME") not in STATIONS:
                continue
            # keep row if either arrival or departure status is valid
            an_status = row.get("AN_PROGNOSE_STATUS", "")
            ab_status = row.get("AB_PROGNOSE_STATUS", "")
            if an_status not in VALID_STATUSES and ab_status not in VALID_STATUSES:
                continue
            rows.append({col: row.get(col, "") for col in COLUMNS})
    return rows


def keep_corridor_trips(rows: list[dict]) -> list[dict]:
    """
    From rows covering both stations, keep only trips that
    appear at BOTH Renens VD and Yverdon-les-Bains.
    """
    # build set of stations seen per trip
    trip_stations: dict[str, set] = defaultdict(set)
    for row in rows:
        trip_stations[row["FAHRT_BEZEICHNER"]].add(row["HALTESTELLEN_NAME"])

    # keep only trips present at both endpoints
    corridor_trips = {
        trip_id for trip_id, stations in trip_stations.items() if stations == STATIONS
    }

    log.info(
        f"  {len(trip_stations)} trips touched target stations, "
        f"{len(corridor_trips)} serve full corridor"
    )

    return [r for r in rows if r["FAHRT_BEZEICHNER"] in corridor_trips]


def process_month(year: int, month: int, writer: csv.DictWriter) -> int:
    """Download, filter, and write one month of data. Returns row count written."""
    url = build_zip_url(year, month)
    zf = stream_zip(url)

    csv_members = [n for n in zf.namelist() if n.endswith(".csv")]
    log.info(f"  ZIP contains {len(csv_members)} daily CSV files")

    total = 0
    for member in sorted(csv_members):
        log.info(f"  Processing {member} ...")
        rows = filter_csv_member(zf, member)
        corridor_rows = keep_corridor_trips(rows)
        for row in corridor_rows:
            writer.writerow(row)
        total += len(corridor_rows)
        log.info(f"    → {len(corridor_rows)} corridor rows kept")

    return total


# --- Entry point ------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download SBB IstDaten for Renens↔Yverdon corridor"
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2025-01", "2025-02", "2025-03"],
        help="Months to download in YYYY-MM format (default: 2025-01 2025-02 2025-03)",
    )
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()

        grand_total = 0
        for month_str in args.months:
            year, month = parse_month(month_str)
            log.info(f"=== Processing {year}-{month:02d} ===")
            count = process_month(year, month, writer)
            grand_total += count
            log.info(f"  Month total: {count} rows")

    log.info(f"Done. {grand_total} rows written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
