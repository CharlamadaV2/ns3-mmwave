"""
Filesystem locations and per-file conventions for ARPO-style datasets.

Defaults target the bundled Spring Lake zip under ``data/``. Callers that
analyze a different bundle should construct their own ``DatasetPaths``
instead of relying on the module-level aliases at the bottom of this file.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scratch/mesh-sim/
DATA_DIR = ROOT / "data"

DEFAULT_ZIP_PATH = DATA_DIR / "arpo_spring_lake_data.zip"
DEFAULT_EXTRACT_DIR = DATA_DIR / "arpo_extracted"

# Scenarios whose CSVs are known to be contaminated or partial -- e.g.
# the collection process crashed and wrote malformed jsonl
KNOWN_BAD_SCENARIOS: tuple[str, ...] = (
    "1-1_static_baseline_1_04162026",
)


@dataclass(frozen=True)
class DatasetPaths:
    """Locations + filename conventions for one ARPO-style dataset."""

    zip_path: Path = DEFAULT_ZIP_PATH
    extract_dir: Path = DEFAULT_EXTRACT_DIR

    @property
    def csv_root(self) -> Path:
        return self.extract_dir / "csv"

    @property
    def plots_dir(self) -> Path:
        return self.extract_dir / "_plots"

    @property
    def per_day_dir(self) -> Path:
        return self.plots_dir / "per_day"

    @property
    def multi_day_dir(self) -> Path:
        return self.plots_dir / "multi_day"


DEFAULT = DatasetPaths()

# Module-level aliases preserved for existing callers
ZIP_PATH = DEFAULT.zip_path
EXTRACT_DIR = DEFAULT.extract_dir
CSV_ROOT = DEFAULT.csv_root
PLOTS_DIR = DEFAULT.plots_dir
PER_DAY_DIR = DEFAULT.per_day_dir
MULTI_DAY_DIR = DEFAULT.multi_day_dir
