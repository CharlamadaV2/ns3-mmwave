## @package extract
# @brief Unzips ARPO data bundle into the configured extraction directory.
#
# The extractor skips OS-generated junk files (macOS resource forks, Windows
# thumbnail caches, etc.) and is idempotent — re-running when the zip is older
# than the already-extracted tree is a no-op.

import fnmatch
import sys
import zipfile

from .paths import DEFAULT, DatasetPaths

## @brief OS-specific metadata files that bundlers commonly leave inside zips.
#
# Each entry is a glob pattern matched against the full in-zip path.
# Extend this tuple if your bundler adds other noise files.
DEFAULT_JUNK_PATTERNS: tuple[str, ...] = (
    "__MACOSX/*",          # macOS Finder resource-fork sidecar tree
    ".DS_Store",
    "*/.DS_Store",
    "Thumbs.db",           # Windows Explorer thumbnail cache
    "*/Thumbs.db",
    "desktop.ini",         # Windows folder configuration file
    "*/desktop.ini",
)


## @brief Test whether a zip entry path matches any junk pattern.
#
# @param name     Full in-zip entry path (e.g. ``"__MACOSX/._file.csv"``).
# @param patterns Tuple of glob patterns to match against.
# @return ``True`` if the entry should be skipped, ``False`` otherwise.
def _is_junk(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


## @brief Extract the ARPO data bundle to the configured directory.
#
# The function is idempotent: if the extraction directory already exists and
# its mtime is newer than the zip, extraction is skipped and a summary of the
# existing CSVs is printed instead.
#
# After extraction the directory's mtime is updated so subsequent calls can
# detect that extraction is current.
#
# @param paths         Dataset path configuration; defaults to @ref DEFAULT
#                      (the Spring Lake bundle).
# @param junk_patterns Glob patterns for entries to skip; defaults to
#                      @ref DEFAULT_JUNK_PATTERNS.
# @return 0 on success, 1 if the zip file is not found.
def extract(
        paths: DatasetPaths = DEFAULT,
        junk_patterns: tuple[str, ...] = DEFAULT_JUNK_PATTERNS,
) -> int:
    if not paths.zip_path.exists():
        print(f"ERROR: {paths.zip_path} not found", file=sys.stderr)
        return 1

    # Skip if already extracted and the zip hasn't been updated since.
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
            # Skip directory entries and known junk files.
            if name.endswith("/") or _is_junk(name, junk_patterns):
                continue
            zf.extract(info, paths.extract_dir)
            n_files += 1
            total_bytes += info.file_size

    # Touch the directory so the mtime check above works on the next call.
    paths.extract_dir.touch()
    print(f"Extracted {n_files} files ({total_bytes / 1e6:.1f} MB) to {paths.extract_dir}")
    if paths.csv_root.is_dir():
        scenarios = sorted(p.name for p in paths.csv_root.iterdir() if p.is_dir())
        print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
    return 0