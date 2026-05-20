## @file paths.py
# @brief Filesystem locations and per-file conventions for ARPO-style datasets.
#
# Defaults target the bundled Spring Lake zip under ``data/``. Callers that
# analyse a different bundle should construct their own @ref DatasetPaths
# instead of relying on the module-level aliases at the bottom of this file.
#
# Directory layout expected after extraction:
# @code
# data/
#   arpo_spring_lake_data.zip
#   arpo_extracted/
#     csv/
#       <scenario>/
#         <node>/
#           bh2.csv
#           gps.csv | geotak_gps.csv
#     _plots/
#       per_day/
#       multi_day/
# @endcode

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  ##< Repository root (scratch/mesh-sim/).
DATA_DIR = ROOT / "data"                    ##< Top-level data directory.

DEFAULT_ZIP_PATH    = DATA_DIR / "arpo_spring_lake_data.zip"  ##< Expected zip location.
DEFAULT_EXTRACT_DIR = DATA_DIR / "arpo_extracted"             ##< Extraction target.

## @brief Scenario names whose CSVs are known to be contaminated or partial.
#
# For example, the collection process crashed mid-run and wrote malformed JSONL.
# These are silently skipped during multi-day aggregation.
KNOWN_BAD_SCENARIOS: tuple[str, ...] = (
    "1-1_static_baseline_1_04162026",
)


## @brief Locations and filename conventions for one ARPO-style dataset.
#
# Frozen so it can be used as a dict key or placed in a set. Construct a
# custom instance to point at a different zip/extract tree without touching
# any module-level state.
#
# @code
# custom = DatasetPaths(
#     zip_path=Path("/mnt/data/other_bundle.zip"),
#     extract_dir=Path("/mnt/data/other_extracted"),
# )
# @endcode
@dataclass(frozen=True)
class DatasetPaths:

    zip_path: Path = DEFAULT_ZIP_PATH    ##< Path to the source zip bundle.
    extract_dir: Path = DEFAULT_EXTRACT_DIR  ##< Root directory for extracted content.

    ## @brief Root of all per-scenario CSV directories.
    # @return ``<extract_dir>/csv``
    @property
    def csv_root(self) -> Path:
        return self.extract_dir / "csv"

    ## @brief Root of all generated plot output.
    # @return ``<extract_dir>/_plots``
    @property
    def plots_dir(self) -> Path:
        return self.extract_dir / "_plots"

    ## @brief Per-scenario PNG and trace CSV output directory.
    # @return ``<plots_dir>/per_day``
    @property
    def per_day_dir(self) -> Path:
        return self.plots_dir / "per_day"

    ## @brief Multi-day ECDF and K-S output directory.
    # @return ``<plots_dir>/multi_day``
    @property
    def multi_day_dir(self) -> Path:
        return self.plots_dir / "multi_day"


## @brief Default dataset instance targeting the Spring Lake bundle.
DEFAULT = DatasetPaths()

# Module-level aliases preserved for existing callers that import these names
# directly. Prefer importing DEFAULT and accessing its properties for new code.
ZIP_PATH      = DEFAULT.zip_path       ##< Alias for DEFAULT.zip_path.
EXTRACT_DIR   = DEFAULT.extract_dir    ##< Alias for DEFAULT.extract_dir.
CSV_ROOT      = DEFAULT.csv_root       ##< Alias for DEFAULT.csv_root.
PLOTS_DIR     = DEFAULT.plots_dir      ##< Alias for DEFAULT.plots_dir.
PER_DAY_DIR   = DEFAULT.per_day_dir    ##< Alias for DEFAULT.per_day_dir.
MULTI_DAY_DIR = DEFAULT.multi_day_dir  ##< Alias for DEFAULT.multi_day_dir.