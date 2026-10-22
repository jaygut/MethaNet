#!/usr/bin/env python3
"""Reject commits that break the repository's single identity or leak machine provenance.

Used by .githooks/pre-commit, .githooks/pre-push and the provenance CI job.

Modes:
  staged             identity of the pending commit and the staged additions
  range <rev-list args>  every listed commit: identities, messages, additions
  tree [<rev>]       every tracked file at <rev> (default HEAD)

The patterns are written with character classes so this file never matches
itself. Retired identities and machine details belong only in the sealed
original-history record kept for counsel, never in tracked content.
"""

from __future__ import annotations

import re
import subprocess
import sys

IDENTITY = ("Jay Gutierrez", "jay.guti40@gmail.com")
# GitHub automation that may appear in fetched history: web-flow merges and Pages deploys.
ALLOWED_AUTOMATION = {
    ("GitHub", "noreply@github.com"),
    ("github-actions[bot]", "41898282+github-actions[bot]@users.noreply.github.com"),
}
PATTERNS = {
    "retired affiliation domain": r"new[\s_-]*atlanti[s]",
    "HPC account name": r"jcorre3[8]",
    "HPC account owner": r"javier[\s_-]+corre[a]",
    "HPC institution": r"eafi[t]",
    "HPC login host": r"apolo-[3]",
    "placeholder email": r"jaygut@exampl[e]\.com",
    "retired email domain": r"graphofli[f]e",
}
COMBINED = re.compile("|".join(f"(?P<p{i}>{rx})" for i, rx in enumerate(PATTERNS.values())), re.I)
LABELS = list(PATTERNS)
MAX_SCAN_BYTES = 50_000_000


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, errors="replace", check=check).stdout


def label(match: re.Match[str]) -> str:
    return LABELS[int(match.lastgroup[1:])]


def scan_text(text: str, where: str) -> list[str]:
    return [f"{where}: {label(m)}" for m in COMBINED.finditer(text)]


def scan_added_lines(diff: str, where: str) -> list[str]:
    problems: list[str] = []
    path = "?"
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else line[4:]
            problems += scan_text(path, f"{where} path")
        elif line.startswith("+") and not line.startswith("+++"):
            problems += scan_text(line[1:], f"{where} {path}")
    return problems


def check_identity(name: str, email: str, role: str, where: str, allow_automation: bool) -> list[str]:
    if (name, email) == IDENTITY:
        return []
    if allow_automation and (name, email) in ALLOWED_AUTOMATION:
        return []
    return [f"{where}: {role} is {name} <{email}>; expected {IDENTITY[0]} <{IDENTITY[1]}>"]


def ident_of(var: str) -> tuple[str, str]:
    raw = git("var", var).strip()
    m = re.match(r"(.*) <(.*)> \d+ [+-]\d{4}$", raw)
    return (m.group(1), m.group(2)) if m else (raw, "")


def staged() -> list[str]:
    problems: list[str] = []
    for var, role in (("GIT_AUTHOR_IDENT", "author"), ("GIT_COMMITTER_IDENT", "committer")):
        problems += check_identity(*ident_of(var), role, "pending commit", allow_automation=False)
    problems += scan_added_lines(git("diff", "--cached", "-U0", "--text", "--no-color", "--no-ext-diff"), "staged")
    return problems


def commit_range(rev_args: list[str]) -> list[str]:
    problems: list[str] = []
    for sha in git("rev-list", *rev_args).split():
        fields = git("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", sha).split("\x00", 4)
        an, ae, cn, ce, body = fields
        short = sha[:12]
        problems += check_identity(an, ae, "author", short, allow_automation=True)
        problems += check_identity(cn, ce, "committer", short, allow_automation=True)
        problems += scan_text(body, f"{short} message")
        diff = git("diff-tree", "-p", "-U0", "--text", "--no-color", "--no-ext-diff", "--root", "-m",
                   "--first-parent", "--no-commit-id", sha)
        problems += scan_added_lines(diff[:MAX_SCAN_BYTES], short)
    return problems


def tree(rev: str) -> list[str]:
    problems: list[str] = []
    for name in git("ls-tree", "-r", "--name-only", rev).splitlines():
        problems += scan_text(name, f"{rev} path")
    # POSIX ERE form of the same patterns, so the scan does not depend on git's PCRE support.
    ere = "|".join(rx.replace(r"\s", "[:space:]") for rx in PATTERNS.values())
    result = subprocess.run(["git", "grep", "-I", "-i", "-n", "-z", "-E", ere, rev, "--"],
                            capture_output=True, text=True, errors="replace")
    if result.returncode > 1:
        raise SystemExit(f"provenance: git grep failed: {result.stderr.strip()}")
    for record in result.stdout.splitlines():
        parts = record.split("\0", 2)
        if len(parts) == 3:
            problems += scan_text(parts[2], f"{parts[0]}:{parts[1]}")
    return problems


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in {"staged", "range", "tree"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "staged":
        problems = staged()
    elif argv[0] == "range":
        if len(argv) < 2:
            print("usage: check_provenance.py range <rev-list args>", file=sys.stderr)
            return 2
        problems = commit_range(argv[1:])
    else:
        problems = tree(argv[1] if len(argv) > 1 else "HEAD")
    for problem in problems[:200]:
        print(f"provenance: {problem}", file=sys.stderr)
    if len(problems) > 200:
        print(f"provenance: ... {len(problems) - 200} more", file=sys.stderr)
    if problems:
        print("provenance: blocked. See scripts/check_provenance.py for the identity and pattern rules.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
