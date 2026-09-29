"""Publication must reject a changed vector payload or mixed pooling metadata."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('contract', Path(__file__).resolve().parents[2] / 'scripts/reports/atlas_embedding_contract.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def contract(tmp_path):
    folder = tmp_path/'lane'; folder.mkdir()
    p = folder/'genome_embeddings.npz'; p.write_bytes(b'verified payload')
    doc = {'status':'verified','geometry_date':'2026-09-29',
           'model_name':'facebook/esm2_t33_650M_UR50D','pooling_layers':[33],
           'artifacts':[{'path':'lane/genome_embeddings.npz','sha256':module.file_sha256(p),
             'model_name':'facebook/esm2_t33_650M_UR50D','pooling_layers':[33],
             'configuration_evidence':'retained FASTA recomputation'}]}
    path=tmp_path/'contract.json'; path.write_text(json.dumps(doc))
    return tmp_path,path,[{'path':folder}],doc


def test_verified_contract(contract):
    root,path,inputs,_=contract
    assert module.validate_embedding_contract(root,path,inputs)['status']=='verified'


def test_changed_vectors_rejected(contract):
    root,path,inputs,_=contract
    (inputs[0]['path']/'genome_embeddings.npz').write_bytes(b'old or changed vectors')
    with pytest.raises(ValueError,match='artifact changed'):module.validate_embedding_contract(root,path,inputs)


def test_mixed_pooling_rejected(contract):
    root,path,inputs,doc=contract
    doc['artifacts'][0]['pooling_layers']=list(range(20,34)); path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Mixed pooling'):module.validate_embedding_contract(root,path,inputs)


def test_unlisted_lane_rejected(contract):
    root,path,inputs,_=contract
    other=root/'other'; other.mkdir(); (other/'genome_embeddings.npz').write_bytes(b'extra')
    with pytest.raises(ValueError,match='input set differs'):module.validate_embedding_contract(root,path,inputs+[{'path':other}])
