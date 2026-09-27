import pytest
from mvo.molecular_query_tool import validate_request
from mvo.query import AccessDenied

def test_qualified_identity_filter_and_bounded_display():
    result=validate_request({'query':'MQ02','purpose':'internal_review','filters':{'lane':'poc_core','minimum_evidence':'sequence_verified'},'limit':10})
    assert result[0]=='MQ02' and result[3]==10

@pytest.mark.parametrize('payload',[
    {'query':'MQ02','purpose':'internal_review','limit':True},
    {'query':'MQ02','purpose':'internal_review','limit':1001},
    {'query':'MQ02','purpose':'internal_review','tenant':'admin'},
    {'query':'MQ02','purpose':'internal_review','filters':{'review_status':'accepted'}},
    {'query':'MQ18','purpose':'internal_review','filters':{'minimum_evidence':'sequence_verified'}},
    {'query':'MQ01','purpose':'internal_review','filters':{'site':'made-up'}},
    {'query':'MATCH (n) DELETE n','purpose':'internal_review'},
])
def test_invalid_or_inapplicable_filters_rejected(payload):
    with pytest.raises(ValueError):validate_request(payload)

def test_external_request_cannot_override_principal():
    with pytest.raises(AccessDenied):validate_request({'query':'MQ18','purpose':'external_export'})
