"""Tests never depend on the machine they run on:
  - every test uses the committed example rig config unless it sets its own (not this PC's configs/rig.yaml, which the
    app edits when a bench is saved);
  - the disk looks 500 GB free (the recorder refuses to start below the rig's limit, and CI runners have little disk);
    a test that checks the disk limits sets its own value."""
from __future__ import annotations

import os

import pytest

from bimanual_collect import rig


@pytest.fixture(autouse=True)
def example_rig(monkeypatch):
    if "BIMANUAL_RIG" not in os.environ:
        monkeypatch.setenv("BIMANUAL_RIG", str(rig.CANDIDATES[1]))
    rig.load.cache_clear()
    yield
    rig.load.cache_clear()


@pytest.fixture(autouse=True)
def plenty_of_disk(monkeypatch):
    import shutil

    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: real(path)._replace(free=int(500e9)))
