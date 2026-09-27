MATCH (old:MVOSnapshot {id:$old_snapshot,status:'validated'}),(new:MVOSnapshot {id:$snapshot,status:'validated'})
WHERE $purpose='internal_review'
CALL { WITH old MATCH (n:MVOResource {snapshot:old.id}) RETURN count(n) AS old_resources }
CALL { WITH new MATCH (n:MVOResource {snapshot:new.id}) RETURN count(n) AS new_resources }
CALL { WITH old,new MATCH (c:MVOResource {snapshot:old.id}) WHERE 'https://emergentbiome.earth/ontology/mvo/Claim' IN c.types
MATCH (c)-[:MVO_HAS_STATEMENT]->(t:MVOStatement)
OPTIONAL MATCH (nc:MVOResource {snapshot:new.id,iri:c.iri})-[:MVO_HAS_STATEMENT]->(nt:MVOStatement)
WHERE nt.predicate=t.predicate AND nt.object_json=t.object_json
RETURN count(t) AS original_claim_statements,count(nt) AS preserved_claim_statements }
RETURN old.id AS old_snapshot,new.id AS new_snapshot,old_resources,new_resources,new_resources-old_resources AS resource_growth,
original_claim_statements,preserved_claim_statements,old.rdf_sha256 AS old_rdf_sha256,new.rdf_sha256 AS new_rdf_sha256
