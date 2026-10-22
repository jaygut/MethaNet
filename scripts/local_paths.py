"""Machine-local roots for MethaNet scripts.

Tracked manifests record repository paths as ``$METHANET_ROOT/...`` rather
than as account-specific absolute paths. Local values come from the
environment first, then from the git-ignored ``.methanet.env`` at the
repository root (copy ``.methanet.env.example``), then from defaults.

Keys: METHANET_ROOT (repository checkout), METHANET_SCRATCH (large caches,
default ``~/scratch``), HPC_HOST and HPC_USER (cluster login).
"""

from __future__ import annotations

import getpass
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_FILE = ".methanet.env"
PATH_TOKENS = ("METHANET_ROOT", "METHANET_SCRATCH", "HOME", "HPC_USER")
TOKEN_RE = re.compile(r"\$(?:\{(%s)\}|(%s)\b)" % ("|".join(PATH_TOKENS), "|".join(PATH_TOKENS)))


def read_local_env(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Return KEY=VALUE pairs from the git-ignored .methanet.env, if present."""
    path = Path(repo_root) / LOCAL_ENV_FILE
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        values[key] = value.strip().strip("'\"")
    return values


def local_setting(key: str, repo_root: Path = REPO_ROOT, default: str | None = None) -> str | None:
    """Environment first, then .methanet.env, then the default."""
    return os.environ.get(key) or read_local_env(repo_root).get(key) or default


def methanet_root(default: Path | None = None) -> Path:
    """METHANET_ROOT from the environment or .methanet.env, else the default, else this checkout."""
    value = local_setting("METHANET_ROOT")
    if value:
        return Path(value).expanduser()
    return Path(default) if default is not None else REPO_ROOT


def methanet_scratch(repo_root: Path = REPO_ROOT) -> Path:
    value = local_setting("METHANET_SCRATCH", repo_root)
    return Path(value).expanduser() if value else Path.home() / "scratch"


def expand_local_path(value: str | Path, repo_root: Path) -> Path:
    """Expand path tokens in a manifest value; relative values resolve under repo_root.

    ``$METHANET_ROOT`` always means the repository the caller is working on.
    """
    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2)
        if key == "METHANET_ROOT":
            return str(repo_root)
        if key == "METHANET_SCRATCH":
            return str(methanet_scratch(repo_root))
        if key == "HOME":
            return str(Path.home())
        return local_setting("HPC_USER", repo_root) or getpass.getuser()

    path = Path(TOKEN_RE.sub(replace, str(value))).expanduser()
    return path if path.is_absolute() else Path(repo_root) / path


def portable_path(path: str | Path, repo_root: Path) -> str:
    """Write-side inverse of expand_local_path: never record an account-specific root."""
    resolved = Path(path).expanduser().resolve()
    for token, base in (
        ("$METHANET_ROOT", Path(repo_root)),
        ("$METHANET_SCRATCH", methanet_scratch(repo_root)),
        ("$HOME", Path.home()),
    ):
        try:
            relative = resolved.relative_to(base.resolve())
        except ValueError:
            continue
        return token if relative == Path(".") else f"{token}/{relative.as_posix()}"
    return str(path)
