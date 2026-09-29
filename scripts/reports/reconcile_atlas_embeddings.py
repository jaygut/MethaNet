#!/usr/bin/env python3
"""Promote verified pilot vectors into a new atlas registry; preserve old releases."""
from pathlib import Path
import csv
import json
import shutil
import numpy as np
from atlas_embedding_contract import file_sha256, validate_embedding_contract

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/blue_catalyst_poc/reembed_single_configuration_20260928'
OUT = ROOT / 'results/blue_catalyst_poc/reconciled_layer33_20260929/artifacts'
CONTRACT = ROOT / 'configs/atlas_embedding_contract_20260929.json'
REGISTRY = ROOT / 'configs/methanet_atlas_lanes_20260929.tsv'


def main():
    verification = RUN / 'verify_and_collate_summary.json'
    rec = json.loads(verification.read_text())
    if rec['status'] != 'pass' or rec['records_ok'] != 662 or not all(g['pass'] for g in rec['gates'].values()):
        raise SystemExit('Pilot reproduction/configuration gate did not pass')
    src = RUN / 'pilot_final_layer_vectors.npz'
    if file_sha256(src) != rec['final_layer_vectors']['sha256']:
        raise SystemExit('Verified pilot hash changed')
    z = np.load(src, allow_pickle=False)
    assert z['pooling_layers'].tolist() == [33] and z['vectors'].shape == (662, 1280)
    assert np.isfinite(z['vectors']).all() and bool(z['pilot_reproduction_pass'].all())
    assert len(set(z['proteome_id'])) == 662
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep all 662 records. The authoritative freeze selects the 625 MAG/bin core;
    # no old projections, bridge ranks or candidate scores are copied here.
    data = {k:z[k] for k in z.files}
    data.update(embeddings=z['vectors'], n_proteins_used=z['n_proteins_embedded'],
                n_valid_proteins_seen=z['n_valid_sequences'])
    np.savez_compressed(OUT / 'genome_embeddings.npz', **data)
    rows = list(csv.DictReader((ROOT / 'configs/methanet_atlas_lanes.tsv').open(), delimiter='\t'))
    rows[0]['esm2_artifacts_dirs'] = str(OUT.relative_to(ROOT))
    rows[0]['notes'] += ' Pilot recomputed with ESM-2 layer 33 on retained FASTAs; configuration verified 2026-09-29. Old geometry superseded.'
    with REGISTRY.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n'); w.writeheader(); w.writerows(rows)
    # Freeze only the configuration evidence needed here into this dated run.
    # Subsequent manuscript edits must not change a published geometry contract.
    evidence_dir=RUN/'configuration_evidence';evidence_dir.mkdir(exist_ok=True)
    source_dir=ROOT/'docs/white-paper/v18/analysis_v16/configuration'
    for name in ['run_fingerprint.tsv','same_dna_46_reembedding_record.json',
                 'matched_configuration_retrieval.tsv','reproduction.tsv']:
        target=evidence_dir/name
        if not target.exists():shutil.copy2(source_dir/name,target)
    audit = evidence_dir/'run_fingerprint.tsv'
    fingerprints = {r['run']:r for r in csv.DictReader(audit.open(), delimiter='\t')}
    artifacts, inputs, absent = [], [], []
    for lane in rows:
        for folder in lane['esm2_artifacts_dirs'].split(';'):
            p = ROOT / folder / 'genome_embeddings.npz'
            inputs.append({'path':p.parent})
            if not p.exists():
                absent.append({'lane_id':lane['lane_id'], 'path':folder, 'status':'no embedding payload; retained registry placeholder'})
                continue
            arr = np.load(p, allow_pickle=True)
            if lane['lane_id'] == 'poc_core':
                evidence = 'Full 662-record retained-FASTA reproduction and final-layer recomputation; verify_and_collate_summary.json'
            else:
                fp = fingerprints[p.parent.parent.name]
                assert fp['inferred_configuration'] == 'final layer (33)'
                assert int(fp['vectors']) == len(arr['embeddings'])
                stats = json.loads((p.parent/'embedding_stats.json').read_text())
                assert stats['model_name'] == 'facebook/esm2_t33_650M_UR50D'
                assert stats['min_aa_len'] == 30 and stats['max_length'] == 1022
                assert stats['max_proteins_per_proteome'] == 6000
                evidence = 'Archived run statistics, code-lineage and same-DNA configuration audit; historical model revision is not recorded in these NPZ files'
            assert arr['embeddings'].shape[1] == 1280 and np.isfinite(arr['embeddings']).all()
            artifacts.append({'lane_id':lane['lane_id'],'path':str(p.relative_to(ROOT)),
                'sha256':file_sha256(p),'records':len(arr['embeddings']),
                'model_name':'facebook/esm2_t33_650M_UR50D','pooling_layers':[33],
                'configuration_evidence':evidence})
    evidence_files=[verification,src,audit,RUN/'pilot_reproduction_check.tsv',
        evidence_dir/'same_dna_46_reembedding_record.json',
        evidence_dir/'matched_configuration_retrieval.tsv',evidence_dir/'reproduction.tsv']
    doc={'schema_version':1,'status':'verified','geometry_date':'2026-09-29',
         'molecular_payload_snapshot':'2026-08-10','model_name':'facebook/esm2_t33_650M_UR50D',
         'pooling_layers':[33], 'token_pooling':'attention-mask mean including CLS/EOS',
         'genome_pooling':'mean protein vector', 'min_aa_len':30,'max_proteins':6000,
         'truncate_residues':1020,'artifacts':artifacts,'absent_registry_paths':absent,
         'verification_files':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p)} for p in evidence_files],
         'remaining_limits':['Historical lane model revisions are not fully recorded; final-layer identity is supported by code lineage and numerical controls.',
                             'Functional pipeline comparability, ecological transfer, exact flux joins and MRV scores are not established by this configuration reconciliation.']}
    CONTRACT.write_text(json.dumps(doc,indent=2)+'\n')
    print(json.dumps(validate_embedding_contract(ROOT,CONTRACT,inputs),indent=2))


if __name__ == '__main__': main()
