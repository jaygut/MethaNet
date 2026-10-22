"""The provenance guard blocks foreign identities and machine details, and passes clean commits."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts/check_provenance.py"
NAME, EMAIL = "Jay Gutierrez", "jay.guti40@gmail.com"
# Assembled at run time so this file never contains the blocked strings itself.
HOST = "apolo" + "-3" + ".example"
ACCOUNT = "/home/rsg-" + "jcorre" + "38/data"


def run(repo: Path, *args: str, name: str = NAME, email: str = EMAIL, check: bool = True):
    env = dict(os.environ, GIT_AUTHOR_NAME=name, GIT_AUTHOR_EMAIL=email,
               GIT_COMMITTER_NAME=name, GIT_COMMITTER_EMAIL=email)
    return subprocess.run(args, cwd=repo, env=env, capture_output=True, text=True, check=check)


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(repo, "git", "init", "-q")
    (repo / "notes.md").write_text("clean\n")
    run(repo, "git", "add", "notes.md")
    run(repo, "git", "commit", "-q", "-m", "Start")
    return repo


def checker(repo: Path, *args: str, name: str = NAME, email: str = EMAIL):
    return run(repo, sys.executable, str(CHECKER), *args, name=name, email=email, check=False)


def test_clean_commit_passes(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "notes.md").write_text("clean\nstill clean\n")
    run(repo, "git", "add", "notes.md")
    assert checker(repo, "staged").returncode == 0
    assert checker(repo, "range", "HEAD").returncode == 0
    assert checker(repo, "tree", "HEAD").returncode == 0


def test_foreign_identity_is_blocked(tmp_path):
    repo = make_repo(tmp_path)
    result = checker(repo, "staged", name="Someone Else", email="someone@example.org")
    assert result.returncode == 1
    assert "author is Someone Else" in result.stderr
    run(repo, "git", "commit", "-q", "--allow-empty", "-m", "Other", name="Someone Else",
        email="someone@example.org")
    assert checker(repo, "range", "HEAD").returncode == 1


def test_machine_details_are_blocked(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "run.sh").write_text(f"ssh {HOST}\ncd {ACCOUNT}\n")
    run(repo, "git", "add", "run.sh")
    staged = checker(repo, "staged")
    assert staged.returncode == 1
    assert "HPC login host" in staged.stderr and "HPC account name" in staged.stderr
    run(repo, "git", "commit", "-q", "-m", "Add run script")
    assert checker(repo, "range", "HEAD").returncode == 1
    assert checker(repo, "tree", "HEAD").returncode == 1


def test_this_repository_passes_its_own_tree_check():
    assert checker(REPO_ROOT, "tree", "HEAD").returncode == 0
