"""Explicitly managed, isolated Neo4j development runtime. Never prints credentials."""
from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

from .model import ROOT, digest
from .projection import connect, load_neo4j

CACHE = ROOT / ".cache"
LOCK = json.loads((ROOT / "config/runtime-lock.json").read_text())
RUNTIME = CACHE / ("neo4j-community-" + LOCK["neo4j"]["version"])
AUTH = CACHE / "neo4j-local-auth.json"


def install():
    CACHE.mkdir(exist_ok=True)
    for name, spec in LOCK.items():
        filename = "robot-" + spec["version"] + ".jar" if name == "robot" else "neo4j-community-" + spec["version"] + "-unix.tar.gz"
        path = CACHE / filename
        if not path.exists():
            with urllib.request.urlopen(spec["url"], timeout=30) as response, path.open("xb") as f:
                shutil.copyfileobj(response, f)
        if digest(path) != spec["sha256"]:
            raise ValueError(f"Runtime checksum mismatch: {filename}; refusing to execute")
        if name == "neo4j" and not RUNTIME.exists():
            with tarfile.open(path, "r:gz") as archive:
                archive.extractall(CACHE, filter="data")
    return dict(status="installed", neo4j_version=LOCK["neo4j"]["version"], directory=str(RUNTIME))


def credentials():
    if not AUTH.is_file() or AUTH.stat().st_mode & 0o077:
        raise ValueError("Local credentials missing or permissions exceed 0600")
    return json.loads(AUTH.read_text())


def run(command, timeout=60):
    result = subprocess.run(command, cwd=RUNTIME, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout.strip()


def start():
    if not RUNTIME.is_dir():
        raise ValueError("Run local-neo4j install first")
    for port in (17687, 17474):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise ValueError(f"Port {port} is already in use; refusing to alter an existing service")
    shutil.copyfile(ROOT / "config/neo4j.conf", RUNTIME / "conf/neo4j.conf")
    if not AUTH.exists():
        auth = dict(user="neo4j", password=secrets.token_urlsafe(32), uri="bolt://127.0.0.1:17687")
        with os.fdopen(os.open(AUTH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
            json.dump(auth, f)
        run([str(RUNTIME / "bin/neo4j-admin"), "dbms", "set-initial-password", auth["password"]])
    auth = credentials()
    run([str(RUNTIME / "bin/neo4j"), "start"])
    deadline = time.monotonic() + 55
    error = None
    while time.monotonic() < deadline:
        try:
            with connect(auth["uri"], auth["user"], auth["password"]):
                return dict(status="running", browser="http://127.0.0.1:17474", credentials_file=str(AUTH), authentication=True)
        except Exception as exc:
            error = type(exc).__name__
            time.sleep(1)
    raise RuntimeError(f"Local runtime did not become ready: {error}; inspect {RUNTIME / 'logs'}")


def stop():
    return dict(status=run([str(RUNTIME / "bin/neo4j"), "stop"]))


def load(path: Path):
    auth = credentials()
    return load_neo4j(path, auth["uri"], auth["user"], auth["password"])
