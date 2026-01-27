#!/usr/bin/env python3
"""List or download files from a Zenodo record."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def list_files(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(url) as resp:
        payload = json.load(resp)
    return payload.get("files", [])


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as handle:
        handle.write(resp.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", default="14532347")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--download-key")
    parser.add_argument("--out-dir", default="data/mucc")
    args = parser.parse_args()

    files = list_files(args.record)
    if args.list or not args.download_key:
        for f in files:
            print(f"{f['key']}\t{f['size']}\t{f['links']['self']}")
        return

    match = None
    for f in files:
        if f["key"] == args.download_key:
            match = f
            break

    if not match:
        keys = ", ".join([f["key"] for f in files])
        raise SystemExit(f"Key not found. Available keys: {keys}")

    out_path = Path(args.out_dir) / match["key"]
    download(match["links"]["self"], out_path)
    print(f"Downloaded to {out_path}")


if __name__ == "__main__":
    main()
