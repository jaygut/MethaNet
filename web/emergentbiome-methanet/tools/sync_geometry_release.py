#!/usr/bin/env python3
"""Synchronize/check all landing geometry quantities against the exported audit."""
import argparse
import hashlib
import json
import re
from pathlib import Path

WEB=Path(__file__).resolve().parents[1]


def expected(atlas):
    m=atlas['meta'];g=m['geometry_audit'];n=m['nearest_core_audit']
    z=g['dimension_zscore_reciprocal_pair_counts']
    return {
        'bridgeEdges':len(atlas['bridges']), 'bridgeNodes':m['n_bridge_nodes'],
        'nearestCoreWetland':n['wetland_outside_core_nearest_rumen_units'],
        'wetlandOutsideCore':n['wetland_outside_core_units'],
        'nearestCoreMangrove':n['mangrove_nearest_rumen_units'],
        'nearestCoreMedianCosine':round(n['outside_core_nearest_similarity_median'],4),
        'randomPairMedianCosine':round(g['random_pair_similarity_median'],4),
        'nearestCoreCandidates':n['target_candidate_nearest_rumen_cards'],
        'candidateCards':n['target_candidate_cards'],
        'sampledNeighborLinks':sum(not b['cs'] for b in atlas['bridges']),
        'crossHabitatNeighborEdges':g['raw_cross_domain_directed_edges'],
        'highlightedCandidateLinks':sum(bool(b['cs']) for b in atlas['bridges']),
        'standardizedRumenReciprocalPairs':z.get('rumen↔wetland',0)+z.get('mangrove↔rumen',0),
    }


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--report',type=Path)
    source.add_argument('--published-report',type=Path,help='Verify the report preserved by landing-only CI')
    a=p.parse_args();atlas=json.loads((WEB/'data/atlas.json').read_text())
    config=WEB/'config.js';text=config.read_text();metrics=expected(atlas)
    report_config=atlas['meta'].get('embedding_configuration',{})
    if report_config.get('status')!='verified' or report_config.get('pooling_layers') != [33]:
        raise SystemExit('Site/report embedding configuration mismatch')
    if a.report:
        if json.loads((a.report/'embedding_configuration.json').read_text()) != report_config:
            raise SystemExit('Site/report embedding configuration mismatch')
        for name,key in [('report.html','report_sha256'),('assets/data/niche.json','niche_sha256'),('audit/scientific_audit.json','scientific_audit_sha256')]:
            if hashlib.sha256((a.report/name).read_bytes()).hexdigest()!=report_config.get(key):
                raise SystemExit(f'Geometry release file changed: {name}')
    else:
        if a.write:
            raise SystemExit('--write requires the complete internal --report bundle')
        if hashlib.sha256(a.published_report.read_bytes()).hexdigest()!=report_config.get('report_sha256'):
            raise SystemExit('Published report does not match landing geometry; publish the complete bundle first')
    date_pattern=r'(^    geometryDate:\s*)"([^"\n]+)"'
    date_match=re.search(date_pattern,text,re.M)
    if not date_match:raise SystemExit('Missing geometryDate in landing configuration')
    if a.write:
        text=re.sub(date_pattern,lambda m:m[1]+json.dumps(report_config['geometry_date']),text,flags=re.M)
    elif date_match[2]!=report_config['geometry_date']:
        raise SystemExit('Stale landing geometry date')
    for key,value in metrics.items():
        pattern=r'(^    '+re.escape(key)+r':\s*)([0-9.]+)(,)[^\n]*'
        hit=re.search(pattern,text,re.M)
        if not hit:raise SystemExit(f'Missing configuration quantity {key}')
        if a.write:
            text=re.sub(pattern,lambda m:m[1]+str(value)+m[3]+' // corrected geometry audit, 2026-09-29',text,flags=re.M)
        elif float(hit[2])!=value:raise SystemExit(f'Stale geometry quantity: {key}={hit[2]}, expected {value}')
    if a.write:
        # The recorded example stays source-bound, with its actual recomputed match.
        import csv
        rows=csv.DictReader((a.report/'tables/embedding_context_table.tsv').open(),delimiter='\t')
        row=next(r for r in rows if r['proteome_id']=='mucc_v1__OWC_1885')
        sentence=f"Closest core reference after pooling reconciliation: {row['nearest_poc_id']}, raw cosine {float(row['nearest_poc_similarity']):.4f}. An exploratory match for review."
        text=re.sub(r'"Closest (?:genome in the 625-genome reference core|core reference after pooling reconciliation):[^"\n]+"',lambda m:json.dumps(sentence),text)
        config.write_text(text)
    print(json.dumps({'status':'pass','geometry_date':report_config['geometry_date'],'metrics':metrics},indent=2))


if __name__=='__main__':main()
