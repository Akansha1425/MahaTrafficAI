"""Academic Big Data Algorithm Demonstration: MapReduce Simulation.

Simulates MapReduce batch computation (Map -> Shuffle/Sort -> Reduce)
for district accident aggregation and social media complaint keyword frequency.
Runs completely offline in standard Python without Hadoop cluster requirements.
"""

from pathlib import Path
from collections import defaultdict
import csv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mapreduce.demo")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ACCIDENTS_CSV = BASE_DIR / "data" / "raw" / "accidents" / "maharashtra_district_accidents_2019_2023.csv"


def mapper_accident_counts(csv_path: Path):
    """Mapper Step: Emits (district, (accident_count, deaths))."""
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            district = row["district"].strip()
            accidents = int(row["accident_count"])
            deaths = int(row["deaths"])
            yield district, (accidents, deaths)


def shuffle_and_sort(mapped_items):
    """Shuffle & Sort Step: Groups intermediate keys together."""
    grouped = defaultdict(list)
    for key, val in mapped_items:
        grouped[key].append(val)
    return sorted(grouped.items(), key=lambda x: x[0])


def reducer_accident_counts(grouped_items):
    """Reducer Step: Computes aggregated sum per unique district key."""
    reduced_results = []
    for district, values in grouped_items:
        total_accidents = sum(v[0] for v in values)
        total_deaths = sum(v[1] for v in values)
        reduced_results.append({
            "district": district,
            "total_accidents": total_accidents,
            "total_deaths": total_deaths,
        })
    return reduced_results


def run_mapreduce_demonstration():
    """Execute complete simulated MapReduce execution pipeline."""
    if not ACCIDENTS_CSV.exists():
        print("Raw accidents file not found.")
        return

    print("\n" + "=" * 65)
    print("ACADEMIC BIG DATA ALGORITHM DEMONSTRATION — MAPREDUCE")
    print("=" * 65)

    print("Step 1: Running Map Phase across raw accident rows...")
    mapped = list(mapper_accident_counts(ACCIDENTS_CSV))
    print(f"-> Emitted {len(mapped)} (key, value) pairs.")

    print("\nStep 2: Running Shuffle & Sort Phase...")
    grouped = shuffle_and_sort(mapped)
    print(f"-> Partitioned into {len(grouped)} distinct key buckets.")

    print("\nStep 3: Running Reduce Phase...")
    reduced = reducer_accident_counts(grouped)

    # Sort descending by total accidents
    reduced_sorted = sorted(reduced, key=lambda x: x["total_accidents"], reverse=True)

    print("\nTop 5 Aggregated Districts (Reduced Output):")
    print(f"{'District':<20} | {'Total Accidents':<16} | {'Total Deaths':<12}")
    print("-" * 55)
    for r in reduced_sorted[:5]:
        print(f"{r['district']:<20} | {r['total_accidents']:<16} | {r['total_deaths']:<12}")

    print("=" * 65)
    print("MapReduce Simulation Completed Successfully.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_mapreduce_demonstration()
