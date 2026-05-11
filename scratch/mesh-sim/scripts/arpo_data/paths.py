"""Filesystem locations and per-file conventions for the ARPO dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scratch/mesh-sim/
ZIP_PATH = ROOT / "data" / "arpo_spring_lake_data.zip"
EXTRACT_DIR = ROOT / "data" / "arpo_extracted"
CSV_ROOT = EXTRACT_DIR / "csv"
PLOTS_DIR = EXTRACT_DIR / "_plots"

PRIORITY_FILES = (
    "bh2.csv", "gps.csv", "geotak_gps.csv",
    "ping.csv", "mesh.csv", "system.csv",
    "chrony.csv", "mcm.csv",
)
NODE_DIRS = ("rab1", "rab2", "rab3", "sdwan")
