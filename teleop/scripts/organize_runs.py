from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from widowxai_quest_teleop.config import resolve_project_path


RUN_NAME = re.compile(
    r"^(?P<date>\d{8})-\d{6}(?:-\d{9})?_(?P<label>.+)$"
)
KNOWN_ARTIFACTS = (
    "telemetry.csv",
    "cad_telemetry.csv",
    "diagnostic.csv",
    "config_snapshot.json",
    "metadata.json",
    "summary.json",
)
MILESTONE_RUNS = {
    "20260722-143605_calibrated-adaptive-30pct-arm-test": "milestone-1",
    "20260722-153618_no-catchup-30ms-arm-test-v2": "milestone-2-smooth-30ms",
    "20260722-155851_step1-send-barrier-8ms-arm-test": "milestone-3-step-1-smooth",
    "20260722-160240_step2-25ms-arm-test": "milestone-3-step-2",
    "20260722-162633_step3-velocity-feedforward-arm-test": "milestone-3-step-3-best",
    "20260722-165441_milestone3-step3-60s-acceptance": "best-60s-acceptance",
    "20260723-165313_four-mode-50pct-continuous": "right-mirror-50pct-accepted",
    "20260723-171126_four-mode-50pct-continuous": "current-right-mirror-50pct",
}
MANIFEST_COLUMNS = (
    "date",
    "run",
    "label",
    "category",
    "milestone",
    "status",
    "rows",
    "size_bytes",
    "git_commit",
    "artifacts",
    "path",
)


def date_directory(raw: str) -> str:
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def classify(label: str) -> str:
    value = label.lower()
    if "cad-" in value:
        return "cad-simulation"
    if "diagnostic" in value or "six-axis" in value:
        return "arm-diagnostic"
    if "replay" in value or "recording" in value:
        return "replay"
    if "mujoco" in value or value.endswith("-sim"):
        return "simulation"
    if (
        "arm-test" in value
        or "hardware-demo" in value
        or "acceptance" in value
        or value.endswith("-arm")
    ):
        return "physical-arm"
    if any(
        token in value
        for token in ("dry", "offline", "synthetic", "headless", "smoke")
    ):
        return "offline-validation"
    return "experiment"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def run_rows(run_dir: Path, summary: dict[str, Any]) -> str:
    rows = summary.get("rows")
    if isinstance(rows, int):
        return str(rows)
    return ""


def run_status(run_dir: Path, summary: dict[str, Any]) -> str:
    if summary:
        return "complete"
    for name in ("telemetry.csv", "cad_telemetry.csv", "diagnostic.csv"):
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            return "partial-or-unsummarized"
    return "metadata-only"


def rewrite_summary_paths(run_dir: Path) -> None:
    path = run_dir / "summary.json"
    summary = load_json(path)
    if not summary:
        return
    changed = False
    for key, value in tuple(summary.items()):
        if key.endswith("_csv") and isinstance(value, str):
            resolved = str((run_dir / Path(value).name).resolve())
            if value != resolved:
                summary[key] = resolved
                changed = True
    if changed:
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def immediate_runs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and RUN_NAME.match(path.name)
    )


def indexed_runs(root: Path) -> list[Path]:
    runs: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir() or not RUN_NAME.match(path.name):
            continue
        if any((path / artifact).exists() for artifact in KNOWN_ARTIFACTS):
            runs.append(path)
    return sorted(runs, key=lambda path: path.name)


def migrate(root: Path, *, dry_run: bool) -> int:
    moved = 0
    for source in immediate_runs(root):
        match = RUN_NAME.match(source.name)
        assert match is not None
        destination = root / date_directory(match.group("date")) / source.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        print(f"{'would move' if dry_run else 'move'}: {source} -> {destination}")
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        moved += 1
    return moved


def build_manifest(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for run_dir in indexed_runs(root):
        match = RUN_NAME.match(run_dir.name)
        assert match is not None
        summary = load_json(run_dir / "summary.json")
        snapshot = load_json(run_dir / "config_snapshot.json")
        artifacts = sorted(
            path.name for path in run_dir.iterdir() if path.is_file()
        )
        records.append(
            {
                "date": date_directory(match.group("date")),
                "run": run_dir.name,
                "label": match.group("label"),
                "category": classify(match.group("label")),
                "milestone": MILESTONE_RUNS.get(run_dir.name, ""),
                "status": run_status(run_dir, summary),
                "rows": run_rows(run_dir, summary),
                "size_bytes": str(
                    sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
                ),
                "git_commit": str(snapshot.get("git_commit", "")),
                "artifacts": ";".join(artifacts),
                "path": str(run_dir.relative_to(root)),
            }
        )
    return records


def write_manifest(root: Path, records: list[dict[str, str]]) -> Path:
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move run artifacts into dated folders and regenerate runs/manifest.csv"
    )
    parser.add_argument("--root", default="runs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show pending moves without changing the run archive or manifest",
    )
    args = parser.parse_args()

    root = resolve_project_path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    moved = migrate(root, dry_run=args.dry_run)
    if args.dry_run:
        print(f"dry run complete: {moved} run directories would move")
        return

    runs = indexed_runs(root)
    for run_dir in runs:
        rewrite_summary_paths(run_dir)
    manifest = write_manifest(root, build_manifest(root))
    print(f"organized {len(runs)} runs; moved {moved}; manifest: {manifest}")


if __name__ == "__main__":
    main()
