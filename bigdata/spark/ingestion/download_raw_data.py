"""Raw dataset fetcher and archival ingest script.

Downloads official MoRTH open data tables from OpenCity CKAN repository
and archives them in data/raw/accidents/ without modification.
Runs locally using Python and HTTPX without external infrastructure.
"""

from pathlib import Path
import httpx
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingestion.download")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_ACCIDENTS_DIR = BASE_DIR / "data" / "raw" / "accidents"
RAW_ACCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

OFFICIAL_RESOURCES = {
    "morth_major_cities_accidents_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "ac75369a-d065-463a-941e-702c6943e796/download/road-accidents-2024-cities-accidents-fatalities.csv"
    ),
    "morth_major_cities_accidents_2022.csv": (
        "https://data.opencity.in/dataset/b4f04d8f-737e-49de-974f-b023619f6baa/resource/"
        "a512c246-b8e3-46d4-8777-8f1dfc5c276f/download/a650b430-0f88-43b0-a1bd-68fb0d01dcf2.csv"
    ),
    "morth_cities_traffic_violations_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "39ffbadc-aebd-4e54-8f00-1e5ae00acff6/download/road-accidents-2024-cities-fatalities-traffic-violation.csv"
    ),
    "morth_cities_fatalities_by_mode_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "3cfe0b01-a4dd-4928-963a-d17e3f5d528d/download/road-accidents-2024-cities-fatalities-mode.csv"
    ),
    "morth_statewise_accidents_2020_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "115647dd-4e46-4a22-8638-bd230edbcca4/download/road-accidents-2024-states-road-accidents.csv"
    ),
    "morth_statewise_fatalities_2020_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "b834e6cb-52f9-4ae5-8b00-ff1416141df0/download/road-accidents-2024-states-fatalities.csv"
    ),
    "morth_collision_types_2024.csv": (
        "https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8/resource/"
        "f081a83c-a857-4a2d-82b3-c28103707df2/download/road-accidents-2024-type-of-collision.csv"
    ),
}


def download_official_accidents_data() -> None:
    """Download official MoRTH tables via HTTPX and write to data/raw/accidents/."""
    client = httpx.Client(timeout=30.0, follow_redirects=True, verify=False)
    for filename, url in OFFICIAL_RESOURCES.items():
        dest = RAW_ACCIDENTS_DIR / filename
        logger.info("Fetching: %s -> %s", filename, dest)
        try:
            resp = client.get(url)
            resp.raise_for_status()
            # Save raw bytes without modifying
            dest.write_bytes(resp.content)
            logger.info("Saved %s (Size: %d bytes)", filename, len(resp.content))
        except Exception as exc:
            logger.error("Failed to download %s: %s", filename, exc)


if __name__ == "__main__":
    download_official_accidents_data()
