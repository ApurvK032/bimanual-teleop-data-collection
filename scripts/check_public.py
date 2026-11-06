"""Fail if anything that would be published looks private. Run before every commit and push.

    python scripts/check_public.py

Checks every file git would publish (tracked + untracked, not ignored; submodules are their own repos): private
IPv4 addresses (192.0.2.x documentation addresses and the arm network 192.168.1.x are fine), home paths, MAC
addresses, e-mail addresses (GitHub noreply is fine), access tokens, files over 5 MB, and media outside
docs/media/. Bench-specific words (hostname, serial numbers, lab names, ...) go one per line in
configs/local/private_terms.txt, which is git-ignored so the list itself is never published.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5_000_000
MEDIA_MAX_BYTES = 10_000_000  # demo GIFs under docs/media/
MEDIA = {".mp4", ".mkv", ".gif", ".png", ".jpg", ".jpeg", ".webm", ".npy", ".npz"}
PATTERNS = {
    # 192.168.1.x is allowed: the arm network (Trossen's default subnet; the PC takes 192.168.1.10), which the arm
    # profiles and docs need. Every other private address (campus, home networks) still fails.
    "private IP": re.compile(
        r"\b(?!192\.168\.1\.\d{1,3}\b)(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    "home path": re.compile(r"(?<![\w/.-])(?:/home/|/Users/)[A-Za-z0-9._-]+"),  # a path start, not "rest/home/rest"
    "MAC address": re.compile(r"\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b|\benx[0-9a-f]{12}\b", re.I),
    "e-mail": re.compile(r"\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[a-z]{2,}\b"),
    "token": re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
}


def published_files() -> list[Path]:
    """Tracked + untracked-not-ignored files. Submodules are their own repositories and are skipped."""
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                         cwd=ROOT, capture_output=True, check=True).stdout.decode()
    stage = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    submodules = [entry.split("\t", 1)[1] for entry in stage.split("\0") if entry.startswith("160000 ")]
    files = [ROOT / name for name in out.split("\0")
             if name and not any(name == s or name.startswith(s + "/") for s in submodules)]
    return [path for path in files if path.is_file()]


def main() -> int:
    terms_file = ROOT / "configs" / "local" / "private_terms.txt"
    terms = [t.strip() for t in terms_file.read_text().splitlines() if t.strip() and not t.startswith("#")] \
        if terms_file.exists() else []
    problems = []
    for path in published_files():
        name = path.relative_to(ROOT)
        size = path.stat().st_size
        limit = MEDIA_MAX_BYTES if str(name).startswith("docs/media/") else MAX_BYTES
        if size > limit:
            problems.append(f"{name}: {size / 1e6:.1f} MB (limit {limit / 1e6:.0f} MB)")
        if path.suffix.lower() in MEDIA and not str(name).startswith("docs/media/"):
            problems.append(f"{name}: media file outside docs/media/")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    problems.append(f"{name}:{number}: {label} {match.group(0)!r}")
            lowered = line.lower()
            for term in terms:
                if term.lower() in lowered:
                    problems.append(f"{name}:{number}: private term {term!r}")
    if not terms_file.exists():
        print(f"note: {terms_file.relative_to(ROOT)} not found; only the generic patterns were checked")
    for problem in problems:
        print(problem)
    print(f"{len(published_files())} files checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
