"""
Extract the causRCA dataset from its ZIP archive.

The original ZIP file is preserved.
Extracted files are stored under data/raw/causRCA/extracted/.
"""

import logging
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "causRCA"

ZIP_PATH = DATA_DIR / "causRCA.zip"

EXTRACT_DIR = DATA_DIR


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def extract_dataset() -> None:
    """Extract causRCA files while preserving the original archive."""

    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Dataset archive not found: {ZIP_PATH}"
        )

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        members = archive.infolist()

        logger.info("Found %d files/directories in archive.", len(members))

        for member in members:
            destination = (EXTRACT_DIR / member.filename).resolve()

            # Prevent ZIP path traversal.
            if not destination.is_relative_to(EXTRACT_DIR.resolve()):
                raise ValueError(
                    f"Unsafe archive path: {member.filename}"
                )

        archive.extractall(EXTRACT_DIR)

    logger.info("Extraction completed: %s", EXTRACT_DIR)


if __name__ == "__main__":
    extract_dataset()