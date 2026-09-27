"""Restricted local demo entrypoint over the executed molecular question pack.

This is not a public API. Authenticated loopback credentials and the trusted
internal-review principal are configured outside the request. Display limiting
happens after complete bounded query execution and explicit result filtering.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import re
import time
from neo4j import Query,READ_ACCESS
from .model import C,digest
from .molecular_queries import QUERY_DIR,QUERY_IDS,validate_query,authorize,normalize
from .molecular_domain import PRINCIPAL,fingerprint
from .projection import connect
from .local_runtime import credentials

FILTERS={'lane','habitat','family','site','proteome','process','date_prefix','depth'}
POST={'review_status','minimum_evidence'}

def validate_request(request):
    if not isinstance(request,dict) or set(request)-{'query','purpose','filters','limit'}:raise ValueError('Unknown request fields')
    qid=request.get('query');purpose=request.get('purpose')
    authorize(PRINCIPAL,purpose)
    if qid not in QUERY_IDS:raise ValueError('Unknown fixed query')
    limit=request.get('limit',25)
    if type(limit) is not int or not 1<=limit<=1000:raise ValueError('Display limit must be an integer from 1 to 1000')
    filters=request.get('filters',{})
    if not isinstance(filters,dict) or set(filters)-(FILTERS|POST):raise ValueError('Unknown filters')
    text=(QUERY_DIR/(qid+'.cypher')).read_text();validate_query(qid,text)
    used=set(re.findall(r'\$([A-Za-z_]+)',text))
    for key,value in filters.items():
        if not isinstance(value,str) or not value or len(value)>300:raise ValueError('Filter must be a nonempty bounded string')
        if key in FILTERS and key not in used:raise ValueError('Filter does not apply to this question: '+key)
    if 'review_status' in filters:
        if qid not in ('MQ10','MQ18') or filters['review_status'] not in ('pending','accepted','rejected'):raise ValueError('Review status filter is supported only for explicit claim/packet review rows')
    if 'minimum_evidence' in filters:
        if qid not in ('MQ02','MQ03','MQ06','MQ09') or filters['minimum_evidence'] not in ('source_candidate','sequence_verified'):raise ValueError('Minimum identity evidence filter is not applicable')
    return qid,purpose,filters,limit,text

def run(build,request):
    qid,purpose,filters,limit,text=validate_request(request)
    manifest=json.loads((build/'manifest.json').read_text())
    if digest(build/'graph.nt')!=manifest['graph_sha256']:raise ValueError('Canonical graph drift')
    params={k:filters.get(k,'') for k in FILTERS};params.update(snapshot=manifest['snapshot'],old_snapshot=manifest['summary']['old_snapshot'],purpose=purpose)
    auth=credentials();started=time.monotonic()
    logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)
    with connect(auth['uri'],auth['user'],auth['password']) as driver,driver.session(default_access_mode=READ_ACCESS) as session:
        state=session.run("MATCH (s:MVOSnapshot {id:$sid,status:'validated'}),(d:MVODomainSnapshot {id:$sid,status:'validated',version:'0.1.0'}) RETURN s.rdf_sha256 AS s,d.rdf_sha256 AS d",sid=params['snapshot']).single()
        if not state or state['s']!=manifest['graph_sha256'] or state['d']!=manifest['graph_sha256']:raise ValueError('Both canonical and domain snapshots must be validated')
        result=session.run(Query(text,timeout=60),**params);rows=normalize(result.data());summary=result.consume()
        if summary.query_type!='r' or len(rows)>100000:raise ValueError('Read-only/cardinality guard failed')
    query_count=len(rows)
    if filters.get('minimum_evidence')=='sequence_verified':rows=[r for r in rows if r.get('identity_state')=='source_header_and_coordinate_translation_verified']
    if filters.get('review_status'):rows=[r for r in rows if r.get('review_state')==str(C[filters['review_status']])]
    return dict(status='executed_read_only_not_biological_validation',snapshot=params['snapshot'],query=qid,parameters=params,explicit_post_filters={k:v for k,v in filters.items() if k in POST},
        query_rows_before_post_filters=query_count,full_filtered_result_count=len(rows),full_filtered_result_hash=fingerprint(rows),display_limit=limit,displayed_rows=rows[:limit],
        denominator_policy='Registered/source denominators in each row are unchanged; identity/review filters select result rows, not ecological populations.',
        notifications=[dict(code=s.gql_status,description=s.status_description) for s in summary.gql_status_objects if s.gql_status.startswith('01')],
        query_sha256=digest(QUERY_DIR/(qid+'.cypher')),tool_sha256=digest(Path(__file__)),elapsed_seconds=round(time.monotonic()-started,4),rights='internal_review_only')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--request',required=True);p.add_argument('--out',type=Path)
    a=p.parse_args();result=run(a.build,json.loads(a.request));value=json.dumps(result,indent=2)+'\n'
    if a.out:
        with a.out.open('x') as f:f.write(value)
    else:print(value)
