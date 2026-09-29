"""Hash-bound release gate for ESM-2 geometry. Never infer pooling from dimension."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_embedding_contract(root: Path, path: Path, inputs: list[dict]) -> dict:
    """Reject missing, changed, extra or incompatible geometry inputs before loading."""
    if not path.is_file():
        raise ValueError(f'Embedding configuration contract required: {path}')
    doc = json.loads(path.read_text())
    if doc.get('status') != 'verified' or doc.get('pooling_layers') != [33]:
        raise ValueError('Embedding contract must verify final-layer [33] pooling')
    if doc.get('model_name') != 'facebook/esm2_t33_650M_UR50D':
        raise ValueError('Unexpected embedding model')
    expected = {}
    for row in doc['artifacts']:
        p = (root / row['path']).resolve()
        if p in expected:
            raise ValueError(f'Duplicate contract artifact: {p}')
        if row.get('pooling_layers') != doc['pooling_layers']:
            raise ValueError(f'Mixed pooling configuration: {p}')
        if row.get('model_name') != doc['model_name'] or not row.get('configuration_evidence'):
            raise ValueError(f'Missing model/configuration evidence: {p}')
        expected[p] = row
    observed = {(Path(x['path']) / 'genome_embeddings.npz').resolve() for x in inputs
                if (Path(x['path']) / 'genome_embeddings.npz').exists()}
    if observed != set(expected):
        raise ValueError(f'Embedding input set differs from contract: {observed ^ set(expected)}')
    for p, row in expected.items():
        if file_sha256(p) != row['sha256']:
            raise ValueError(f'Embedding artifact changed: {p}')
    for evidence in doc.get('verification_files', []):
        p = root / evidence['path']
        if not p.is_file() or file_sha256(p) != evidence['sha256']:
            raise ValueError(f'Embedding verification evidence changed: {p}')
    return {
        'status': 'verified', 'geometry_date': doc['geometry_date'],
        'model_name': doc['model_name'], 'pooling_layers': doc['pooling_layers'],
        'contract_sha256': file_sha256(path), 'artifact_count': len(expected),
        'scope': 'Configuration-reconciled exploratory genome geometry; biological transfer and flux prediction remain unvalidated.',
        'remaining_limits': doc.get('remaining_limits', []),
    }
