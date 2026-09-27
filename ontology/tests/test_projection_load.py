"""Synthetic bounded read-back regression; live idempotence has separate receipts."""
import json
import pytest
from rdflib import Graph,URIRef,Literal
from mvo import projection


class Result:
    def __init__(self,rows=()):self.rows=list(rows)
    def data(self):return self.rows
    def single(self):return self.rows[0] if self.rows else None
    def consume(self):return None


class Session:
    def __init__(self,manifest,rows,fault=None):
        self.manifest=manifest;self.rows={r['id']:r for r in rows};self.fault=fault;self.validated=False;self.batch_sizes=[]
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def session(self,**kwargs):return self
    def run(self,q,**p):
        if 'RETURN n.rdf_sha256 AS hash' in q:return Result()
        if 'RETURN count(s) AS n' in q:
            n=self.manifest['resources' if 'MVOResource' in q else 'statements']
            if self.fault=='extra_statement' and 'MVOStatement' in q:n+=1
            return Result([{'n':n}])
        if 'UNWIND $ids AS statement_id' in q:
            self.batch_sizes.append(len(p['ids']));rows=[]
            for sid in p['ids']:
                r=self.rows[sid];rows.append(dict(id=sid,s=r['subject'],p=r['predicate'],o=json.dumps(r['object'])))
            if self.fault=='missing_id':rows=rows[:-1]
            if self.fault=='literal_drift':rows[0]['o']=json.dumps({'kind':'literal','value':'changed','datatype':None,'language':None})
            return Result(rows)
        if "SET n.status='validated'" in q:self.validated=True
        return Result()


@pytest.mark.parametrize('fault',[None,'extra_statement','missing_id','literal_drift'])
def test_exact_bounded_live_reconstruction_contract(tmp_path,monkeypatch,fault):
    g=Graph()
    for i in range(1003):g.add((URIRef('urn:synthetic:s'+str(i)),URIRef('urn:synthetic:p'),Literal(str(i))))
    dest=tmp_path/'projection';manifest=projection.export(g,dest,'synthetic-readback')
    session=Session(manifest,list(projection.read_rows(dest/'statements.jsonl')),fault)
    monkeypatch.setattr(projection,'connect',lambda *a,**kw:session)
    if fault:
        with pytest.raises(ValueError):projection.load_neo4j(dest,'bolt://127.0.0.1','test','synthetic')
        assert not session.validated
    else:
        result=projection.load_neo4j(dest,'bolt://127.0.0.1','test','synthetic')
        assert result['roundtrip']=='exact' and session.validated
        assert session.batch_sizes==[1000,3]
