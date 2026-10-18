"""
Download the causRCA industrial dataset from Zenodo.

Source:
https://zenodo.org/records/15876410

Raw files are stored in:
data/raw/causRCA/

Existing files are preserved.
"""

import json
import logging
import urllib.request
from pathlib import Path


ZENODO_RECORD_ID = "15876410"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "causRCA"

API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def fetch_metadata() -> dict:
    """Retrieve dataset metadata from Zenodo."""

    logger.info("Fetching causRCA metadata...")

    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "causal-rca-thesis/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def download_dataset() -> None:
    """Download all files associated with the Zenodo record."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    metadata = fetch_metadata()

    files = metadata.get("files", [])

    logger.info("Found %d files.", len(files))

    if not files:
        raise RuntimeError("No downloadable files found in Zenodo record.")

    for file in files:
        filename = file["key"]
        download_url = file["links"]["self"]

        destination = DATA_DIR / filename

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            logger.info("Skipping existing file: %s", filename)
            continue

        logger.info("Downloading: %s", filename)

        urllib.request.urlretrieve(
            download_url,
            destination,
        )

        logger.info("Saved: %s", destination)

    logger.info("causRCA dataset download completed.")


if __name__ == "__main__":
    download_dataset()