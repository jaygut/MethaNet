#!/usr/bin/env python3
"""Generate the Blue Catalyst FG batch runbook notebook.

This avoids brittle hand-editing of ipynb JSON in shell scripts.
"""

from __future__ import annotations

import json
from pathlib import Path


def build_notebook_payload() -> dict:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Blue Catalyst FG Batch Runbook\n",
            "\n",
            "Stages:\n",
            "- plan: build frozen embedding index + batch manifests\n",
            "- merge: reconcile FG batch outputs with frozen embeddings\n",
        ],
    }

    code_lines = [
        "import os\n",
        "import subprocess\n",
        "from pathlib import Path\n",
        "\n",
        "root_env = os.getenv('METHANET_ROOT', '').strip() or os.getenv('BC_MROOT', '').strip()\n",
        "if root_env:\n",
        "    root = Path(root_env).expanduser().resolve()\n",
        "else:\n",
        "    candidate = Path.cwd().resolve()\n",
        "    # nbconvert often executes from notebooks/, so recover repo root.\n",
        "    root = candidate.parent if candidate.name == 'notebooks' else candidate\n",
        "script = root / 'scripts' / 'blue_catalyst_fg_batch_pipeline.py'\n",
        "if not script.exists():\n",
        "    raise RuntimeError(f'Missing pipeline script: {script}')\n",
        "stage = os.getenv('BC_FG_STAGE', 'plan').strip().lower()\n",
        "source_run = os.getenv('BC_FG_SOURCE_EMBED_RUN_ID', 'unknown_embed')\n",
        "art_default = root / 'results' / 'blue_catalyst_poc' / 'runs'\n",
        "art_default = art_default / 'fg_runbook' / 'fg_artifacts'\n",
        "artifacts = Path(os.getenv('BC_FG_ARTIFACTS_DIR', str(art_default)))\n",
        "artifacts = artifacts.expanduser().resolve()\n",
        "batch_size = int(os.getenv('BC_FG_BATCH_SIZE', '25'))\n",
        "min_join = float(os.getenv('BC_FG_MIN_JOIN_COVERAGE', '0.95'))\n",
        "hash_prot = os.getenv('BC_FG_HASH_PROTEOMES', '0') == '1'\n",
        "embed_meta = os.getenv('BC_FG_EMBED_METADATA', '').strip()\n",
        "embed_npz = os.getenv('BC_FG_EMBED_NPZ', '').strip()\n",
        "artifacts.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "if stage == 'plan':\n",
        "    if not embed_meta or not embed_npz:\n",
        "        raise RuntimeError('Missing BC_FG_EMBED_METADATA/BC_FG_EMBED_NPZ')\n",
        "    cmd = ['python', str(script), 'plan']\n",
        "    cmd += ['--embedding-metadata', embed_meta]\n",
        "    cmd += ['--embedding-npz', embed_npz]\n",
        "    cmd += ['--embedding-run-id', source_run]\n",
        "    cmd += ['--output-dir', str(artifacts)]\n",
        "    cmd += ['--batch-size', str(batch_size)]\n",
        "    if hash_prot:\n",
        "        cmd.append('--hash-proteomes')\n",
        "elif stage == 'merge':\n",
        "    cmd = ['python', str(script), 'merge']\n",
        "    cmd += ['--fg-plan-dir', str(artifacts)]\n",
        "    cmd += ['--batch-results-dir', str(artifacts / 'batch_results')]\n",
        "    cmd += ['--output-dir', str(artifacts)]\n",
        "    cmd += ['--min-join-coverage', str(min_join)]\n",
        "else:\n",
        "    raise RuntimeError(f'Unsupported BC_FG_STAGE={stage}')\n",
        "\n",
        "print('Executing:', ' '.join(cmd))\n",
        "subprocess.run(cmd, check=True)\n",
    ]

    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines,
    }

    return {
        "cells": [markdown_cell, code_cell],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    notebook_path = repo_root / "notebooks" / "blue_catalyst_fgintel_batch_runbook.ipynb"
    payload = build_notebook_payload()
    notebook_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[OK] Wrote runbook notebook: {notebook_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
