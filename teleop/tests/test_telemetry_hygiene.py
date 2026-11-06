from __future__ import annotations

import json
from pathlib import Path

from scripts.organize_runs import (
    build_manifest,
    migrate,
    rewrite_summary_paths,
    write_manifest,
)
from widowxai_quest_teleop import telemetry
from widowxai_quest_teleop.telemetry import TelemetryLogger


class _FixedClock:
    @staticmethod
    def strftime(pattern: str) -> str:
        values = {
            "%Y%m%d-%H%M%S": "20260723-101112",
            "%Y-%m-%d": "2026-07-23",
        }
        return values[pattern]


def test_telemetry_logger_uses_dated_directory(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(telemetry, "time", _FixedClock())

    with TelemetryLogger("dated run", {"project": {"name": "test"}}, tmp_path) as logger:
        logger.log(ik_status="ok")
        run_dir = logger.run_dir

    assert run_dir == tmp_path / "2026-07-23" / "20260723-101112_dated-run"
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 1
    assert summary["ik_failures"] == 0
    assert summary["telemetry_csv"] == str(run_dir / "telemetry.csv")


def test_organizer_moves_indexes_and_repairs_summary_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    source = root / "20260722-165441_milestone3-step3-60s-acceptance"
    source.mkdir(parents=True)
    (source / "telemetry.csv").write_text("header\nrow\n", encoding="utf-8")
    (source / "config_snapshot.json").write_text(
        json.dumps({"git_commit": "95e04a1"}), encoding="utf-8"
    )
    (source / "summary.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "ik_failures": 0,
                "telemetry_csv": f"/obsolete/runs/{source.name}/telemetry.csv",
            }
        ),
        encoding="utf-8",
    )

    assert migrate(root, dry_run=False) == 1
    destination = root / "2026-07-22" / source.name
    rewrite_summary_paths(destination)
    records = build_manifest(root)
    manifest = write_manifest(root, records)

    assert destination.is_dir()
    assert len(records) == 1
    assert records[0]["milestone"] == "best-60s-acceptance"
    assert records[0]["status"] == "complete"
    assert records[0]["rows"] == "1"
    assert records[0]["path"] == f"2026-07-22/{source.name}"
    summary = json.loads(
        (destination / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["telemetry_csv"] == str(
        (destination / "telemetry.csv").resolve()
    )
    assert manifest.read_text(encoding="utf-8").count("\n") == 2
