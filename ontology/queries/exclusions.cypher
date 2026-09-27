MATCH (snap:MVOSnapshot {id:$snapshot,status:'validated'})
MATCH (u:MVOResource {snapshot:snap.id})-[:MVO_HAS_STATEMENT]->(excluded:MVOStatement)
WHERE excluded.predicate='https://emergentbiome.earth/ontology/mvo/releaseExcluded' AND excluded.lexical='true'
MATCH (u)-[:MVO_REL {predicate:'https://emergentbiome.earth/ontology/mvo/blockedBy'}]->(gap)
MATCH (gap)-[:MVO_HAS_STATEMENT]->(reason:MVOStatement)
WHERE reason.predicate='https://emergentbiome.earth/ontology/mvo/reason'
RETURN u.iri AS unit,reason.lexical AS reason ORDER BY unit,reason LIMIT 100
