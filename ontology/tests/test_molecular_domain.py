from mvo.molecular_domain import key, fingerprint
from mvo.model import M


def test_domain_property_keys_do_not_alias_external_namespaces():
    assert key(str(M.laneId))=="mvo_laneId"
    assert key("https://different.example/laneId")!=key(str(M.laneId))
    assert key(str(M)+"unsafe`name").startswith("iri_")


def test_projection_inventory_order_independence_not_value_indifference():
    a=[{"subject":"a","value":["01"]},{"subject":"b","value":["2"]}]
    assert fingerprint(a)==fingerprint(a[::-1])
    assert fingerprint(a)!=fingerprint([{"subject":"a","value":["1"]},a[1]])
