import pytest
from mvo.molecular_queries import authorize,validate_query,QUERY_DIR,QUERY_IDS,compare_subset,normalize
from mvo.molecular_domain import PRINCIPAL
from mvo.query import AccessDenied


def test_all_fixed_queries_readonly_and_snapshot_guarded():
    for qid in QUERY_IDS:validate_query(qid,(QUERY_DIR/(qid+".cypher")).read_text())


@pytest.mark.parametrize("text",["CREATE (n)","MATCH (n) DELETE n","CALL db.labels()","CALL apoc.cypher.run($text,{})"])
def test_free_form_mutations_and_procedures_rejected(text):
    with pytest.raises(ValueError):validate_query("MQ01",text)


def test_query_tenant_purpose_and_project_guards():
    authorize(PRINCIPAL,"internal_review")
    for principal,purpose in (({**PRINCIPAL,"tenant":"other"},"internal_review"),({**PRINCIPAL,"projects":["other"]},"internal_review"),(PRINCIPAL,"external_export")):
        with pytest.raises(AccessDenied):authorize(principal,purpose)


def test_full_multiset_parity_catches_duplicate_rows_not_just_unique_ids():
    row={"id":"a","value":1}
    assert compare_subset([row],[row],["id","value"])["status"]=="pass"
    with pytest.raises(ValueError):compare_subset([row,row],[row],["id","value"])


def test_component_list_order_is_preserved():
    r=normalize([{"lanes":["b","a"],"component_support":["b","a"]}])[0]
    assert r["lanes"]==["a","b"] and r["component_support"]==["b","a"]
