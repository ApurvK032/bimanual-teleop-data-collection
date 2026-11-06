"""List episodes with their QC verdict, and accept or reject them (only accepted episodes are exported).

    python -m bimanual_collect review                                  # every episode in the data folder
    python -m bimanual_collect review data/raw/<date>/<session>        # one session
    python -m bimanual_collect review <episode> --accept --task "unfold the handkerchief"
    python -m bimanual_collect review <episode> --reject --note "arm hit the table"

Writes <episode>/review.json and refreshes <data_raw>/index.csv. A task given here replaces the one typed when
recording (the recording itself is never changed).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import catalog
from .lerobot_align import episode_dirs


def write_review(episode: Path, status: str | None = None, task: str | None = None, outcome: str | None = None,
                 note: str | None = None) -> dict:
    """Create or update <episode>/review.json. None keeps the previous value."""
    if status not in (None, "accepted", "rejected", "unreviewed"):
        raise ValueError("status must be accepted, rejected or unreviewed")
    if outcome not in (None, "", "success", "failure", "unsure"):
        raise ValueError("outcome must be success, failure or unsure")
    previous = catalog.review(episode) or {}
    record = {
        "status": status or previous.get("status", "unreviewed"),
        "task": task if task is not None else previous.get("task") or catalog.task_of(episode),
        "outcome": outcome if outcome is not None else previous.get("outcome") or catalog.label_of(episode, "outcome"),
        "note": note if note is not None else previous.get("note", ""),
        "reviewed": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (Path(episode) / "review.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def show(episodes: list[Path]) -> None:
    rows = [catalog.index_row(episode) for episode in episodes]
    header = ("episode", "task", "length", "mode", "drops", "sync", "QC", "review")
    table = [header] + [(
        row["episode"], (row["task"] or "-")[:28], f"{row['duration_s']} s", f"{row['mode']}@{row['fps']}",
        f"{row['camera_drop_pct']}%", f"{row['sync_ms']:+.1f} ms" if row["sync_ms"] != "" else "-",
        row["qc"], row["review"]) for row in rows]
    widths = [max(len(str(line[i])) for line in table) for i in range(len(header))]
    for n, line in enumerate(table):
        print("  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)))
        if n == 0:
            print("  ".join("-" * width for width in widths))
    for row in rows:
        if row["reasons"] and row["review"] == "unreviewed":
            print(f"\n{row['episode']} needs review:\n  " + "\n  ".join(row["reasons"].split("; ")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="session or episode folders (default: all)")
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--accept", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--task", help="task label for the dataset")
    parser.add_argument("--outcome", choices=["success", "failure", "unsure"], help="how the attempt went")
    parser.add_argument("--note", default="", help="why (kept in review.json)")
    args = parser.parse_args()

    episodes = episode_dirs(args.paths) if args.paths else catalog.all_episodes()
    if args.accept or args.reject or args.task or args.outcome:
        if not args.paths:
            parser.error("name the episode(s) to accept or reject")
        for episode in episodes:
            status = "accepted" if args.accept else "rejected" if args.reject else None
            record = write_review(episode, status, args.task, args.outcome, args.note or None)
            print(f"{catalog.relative(episode)}: {record['status']}, task {record['task']!r}")
        catalog.write_index()
    show(episodes)


if __name__ == "__main__":
    main()
