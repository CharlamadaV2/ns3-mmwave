"""Unzip the ARPO Spring Lake bundle into ``data/arpo_extracted/``."""

import sys
import zipfile

from .paths import CSV_ROOT, EXTRACT_DIR, ZIP_PATH


def extract() -> int:
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found", file=sys.stderr)
        return 1

    if EXTRACT_DIR.exists() and EXTRACT_DIR.stat().st_mtime >= ZIP_PATH.stat().st_mtime:
        n_csv = sum(1 for _ in EXTRACT_DIR.rglob("*.csv"))
        print(f"Already extracted ({n_csv} CSVs in {EXTRACT_DIR})")
        return 0

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store") or name.endswith("/"):
                continue
            zf.extract(info, EXTRACT_DIR)
            n_files += 1
            total_bytes += info.file_size

    EXTRACT_DIR.touch()
    scenarios = sorted(p.name for p in CSV_ROOT.iterdir() if p.is_dir())
    print(f"Extracted {n_files} files ({total_bytes / 1e6:.1f} MB) to {EXTRACT_DIR}")
    print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
    return 0
