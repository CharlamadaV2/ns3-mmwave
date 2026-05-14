"""Unzip an ARPO-style data bundle into ``paths.extract_dir``."""

import fnmatch
import sys
import zipfile

from .paths import DEFAULT, DatasetPaths

# OS-specific metadata files that bundlers commonly leave inside zips.
DEFAULT_JUNK_PATTERNS: tuple[str, ...] = (
    "__MACOSX/*",  # macOS Finder resource-fork sidecar tree
    ".DS_Store", "*/.DS_Store",
    "Thumbs.db", "*/Thumbs.db",  # Windows Explorer thumbnail cache
    "desktop.ini", "*/desktop.ini",  # Windows folder config
)


def _is_junk(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def extract(
        paths: DatasetPaths = DEFAULT,
        junk_patterns: tuple[str, ...] = DEFAULT_JUNK_PATTERNS,
) -> int:
    if not paths.zip_path.exists():
        print(f"ERROR: {paths.zip_path} not found", file=sys.stderr)
        return 1

    if (
            paths.extract_dir.exists()
            and paths.extract_dir.stat().st_mtime >= paths.zip_path.stat().st_mtime
    ):
        n_csv = sum(1 for _ in paths.extract_dir.rglob("*.csv"))
        print(f"Already extracted ({n_csv} CSVs in {paths.extract_dir})")
        return 0

    paths.extract_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(paths.zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.endswith("/") or _is_junk(name, junk_patterns):
                continue
            zf.extract(info, paths.extract_dir)
            n_files += 1
            total_bytes += info.file_size

    paths.extract_dir.touch()
    print(f"Extracted {n_files} files ({total_bytes / 1e6:.1f} MB) to {paths.extract_dir}")
    if paths.csv_root.is_dir():
        scenarios = sorted(p.name for p in paths.csv_root.iterdir() if p.is_dir())
        print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
    return 0
