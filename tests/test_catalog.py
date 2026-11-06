"""QC verdicts, review records and the index (no camera, no arm log needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bimanual_collect import catalog


def make_episode(root: Path, name: str = "episode_0000", **meta) -> Path:
    episode = root / "2026-11-01" / "session_120000" / name
    episode.mkdir(parents=True)
    data = {"task": "fold the towel", "duration_s": 30.0, "frames_recorded": 900, "camera_frames_missing": 0,
            "recorder_queue_drops": 0, "camera": {"mode": "720"}, "video": {"nominal_fps": 30, "size": [2560, 720]}}
    data.update(meta)
    (episode / "episode.json").write_text(json.dumps(data))
    return episode


GOOD_ALIGN = {"ok": True, "error": None, "steps": 890}
GOOD_SYNC = {"ok": True, "error": None, "shift_ms": -2.5, "r": 0.85}


def test_clean_episode_passes(tmp_path) -> None:
    episode = make_episode(tmp_path)
    record = catalog.evaluate(episode, json.loads((episode / "episode.json").read_text()), [], GOOD_ALIGN, GOOD_SYNC)
    assert record["verdict"] == "pass" and record["reasons"] == []


def test_each_problem_is_named(tmp_path) -> None:
    episode = make_episode(tmp_path, task="", camera_frames_missing=60, duration_s=3.0)
    meta = json.loads((episode / "episode.json").read_text())
    record = catalog.evaluate(episode, meta, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], {"ok": False, "error": "no arm log"},
                              {"ok": True, "shift_ms": 40.0, "r": 0.3})
    assert record["verdict"] == "needs_review"
    names = {reason.split(":")[0] for reason in record["reasons"]}
    assert names == {"camera drops", "torn frames", "duration", "alignment", "sync", "task label"}


def test_review_sets_task_and_export_selection(tmp_path, monkeypatch) -> None:
    accepted = make_episode(tmp_path, "episode_0000", task="")
    other = make_episode(tmp_path, "episode_0001")
    monkeypatch.setattr(catalog, "relative", lambda path: Path(path).name)
    monkeypatch.setattr(catalog, "write_index", lambda root=None: tmp_path / "index.csv")
    from bimanual_collect import review

    monkeypatch.setattr(sys, "argv", ["review", str(accepted), "--accept", "--task", "unfold the handkerchief"])
    review.main()
    assert catalog.is_accepted(accepted) and not catalog.is_accepted(other)
    assert catalog.task_of(accepted) == "unfold the handkerchief"  # review label wins over the empty recorded one
    assert catalog.task_of(other) == "fold the towel"


def test_index_lists_every_episode(tmp_path) -> None:
    make_episode(tmp_path, "episode_0000")
    make_episode(tmp_path, "episode_0001", camera_frames_missing=90)
    path = catalog.write_index(tmp_path)
    lines = path.read_text().splitlines()
    assert lines[0].startswith("episode,task") and len(lines) == 3
    assert ",not processed,unreviewed," in lines[1] and ",9.09," in lines[2]
