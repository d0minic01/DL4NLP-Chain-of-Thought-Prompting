"""Archive run artifacts (results/, plots/, run.log) into a timestamped zip."""

import argparse
import re
import zipfile
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")
ARCHIVE_DIR = Path("archived_runs")
LOG_FILE = Path("run.log")

_COMPOUND_EXTS = (".tar.gz", ".tar.bz2", ".tar.xz")
_TIMESTAMP_RE = re.compile(r"\d{8}_\d{6}")


def archive_run(
    results_dir: Path = RESULTS_DIR,
    plots_dir: Path = PLOTS_DIR,
    archive_dir: Path = ARCHIVE_DIR,
    timestamp: str | None = None,
    label: str | None = None,
) -> Path | None:
    """Zip results/ and plots/ into a timestamped archive in archived_runs/."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name_parts = ["run", timestamp]
    if label:
        name_parts.append(label)
    archive_name = "_".join(name_parts) + ".zip"

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / archive_name

    files_added = 0
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src_dir in (results_dir, plots_dir):
            if src_dir.exists():
                for file in sorted(src_dir.rglob("*")):
                    if file.is_file():
                        zf.write(file, file)
                        files_added += 1
        if LOG_FILE.exists():
            zf.write(LOG_FILE, LOG_FILE)
            files_added += 1

    if files_added == 0:
        archive_path.unlink()
        print("Nothing to archive (results/ and plots/ are empty).")
        return None

    size_mb = archive_path.stat().st_size / 1024**2
    print(f"Archived {files_added} files -> {archive_path} ({size_mb:.1f} MB)")
    return archive_path


def add_timestamps_to_existing(
    archive_dir: Path = ARCHIVE_DIR,
    dry_run: bool = False,
) -> None:
    """Rename archives in archived_runs/ that lack a timestamp by inserting their mtime."""
    if not archive_dir.exists():
        print(f"{archive_dir} does not exist.")
        return

    renamed = 0
    for path in sorted(archive_dir.iterdir()):
        if not path.is_file():
            continue
        if _TIMESTAMP_RE.search(path.name):
            continue

        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        ts = mtime.strftime("%Y%m%d_%H%M%S")

        compound = next((ext for ext in _COMPOUND_EXTS if path.name.endswith(ext)), None)
        if compound:
            base = path.name[: -len(compound)]
            new_name = f"{base}_{ts}{compound}"
        else:
            new_name = f"{path.stem}_{ts}{path.suffix}"

        new_path = archive_dir / new_name
        action = "would rename" if dry_run else "renaming"
        print(f"  {action}: {path.name} -> {new_name}")
        if not dry_run:
            path.rename(new_path)
        renamed += 1

    if renamed == 0:
        print("All archives already have timestamps.")
    elif dry_run:
        print(f"(dry-run) {renamed} file(s) would be renamed.")
    else:
        print(f"Renamed {renamed} file(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive run artifacts into archived_runs/")
    parser.add_argument(
        "--rename-existing",
        action="store_true",
        help="Add timestamps to archives in archived_runs/ that lack one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --rename-existing: show what would be renamed without doing it",
    )
    parser.add_argument(
        "--label",
        help="Optional suffix appended to the archive name (e.g. 'rtx5090')",
    )
    args = parser.parse_args()

    if args.rename_existing:
        add_timestamps_to_existing(dry_run=args.dry_run)
    else:
        archive_run(label=args.label)


if __name__ == "__main__":
    main()
