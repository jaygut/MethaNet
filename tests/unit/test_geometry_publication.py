"""Landing-only publication must preserve the exact reconciled report pair."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'geometry_sync', ROOT/'web/emergentbiome-methanet/tools/sync_geometry_release.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


@pytest.fixture
def publication(tmp_path, monkeypatch):
    report = tmp_path/'report.html'
    report.write_text('verified reconciled report')
    geometry = {
        'dimension_zscore_reciprocal_pair_counts': {'rumen↔wetland': 3},
        'random_pair_similarity_median': .99, 'raw_cross_domain_directed_edges': 10,
    }
    nearest = dict(zip([
        'wetland_outside_core_nearest_rumen_units', 'wetland_outside_core_units',
        'mangrove_nearest_rumen_units', 'outside_core_nearest_similarity_median',
        'target_candidate_nearest_rumen_cards', 'target_candidate_cards',
    ], [2, 8, 1, .995, 1, 2]))
    atlas = {'bridges': [{'cs': True}, {'cs': False}], 'meta': {
        'geometry_audit': geometry, 'nearest_core_audit': nearest, 'n_bridge_nodes': 4,
        'embedding_configuration': {'status': 'verified', 'pooling_layers': [33],
            'geometry_date': '2026-09-29',
            'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest()},
    }}
    (tmp_path/'data').mkdir()
    data = tmp_path/'data/atlas.json'
    data.write_text(json.dumps(atlas))
    config = tmp_path/'config.js'
    config.write_text('    geometryDate: "2026-09-29",\n'+
        '\n'.join(f'    {k}: {v},' for k,v in sync.expected(atlas).items()))
    monkeypatch.setattr(sync, 'WEB', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['sync', '--published-report', str(report)])
    return report, config, data, atlas


def test_paired_release_passes(publication):
    sync.main()


def test_changed_report_blocks_landing(publication):
    publication[0].write_text('historical mixed-pooling report')
    with pytest.raises(SystemExit, match='does not match landing geometry'):
        sync.main()


def test_stale_geometry_quantity_blocks_landing(publication):
    config = publication[1]
    config.write_text(config.read_text().replace('bridgeEdges: 2,', 'bridgeEdges: 2226,'))
    with pytest.raises(SystemExit, match='Stale geometry quantity: bridgeEdges'):
        sync.main()


def test_mixed_configuration_blocks_landing(publication):
    _, _, data, atlas = publication
    atlas['meta']['embedding_configuration']['pooling_layers'] = list(range(20,34))
    data.write_text(json.dumps(atlas))
    with pytest.raises(SystemExit, match='configuration mismatch'):
        sync.main()


def test_stale_geometry_date_blocks_landing(publication):
    config = publication[1]
    config.write_text(config.read_text().replace('2026-09-29', '2026-08-10'))
    with pytest.raises(SystemExit, match='Stale landing geometry date'):
        sync.main()
