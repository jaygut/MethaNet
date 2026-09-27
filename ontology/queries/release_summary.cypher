// Human-operated query; $snapshot must identify a validated MVOSnapshot.
MATCH (snap:MVOSnapshot {id:$snapshot,status:'validated'})
MATCH (u:MVOResource {snapshot:snap.id})-[:MVO_HAS_STATEMENT]->(lane:MVOStatement)
WHERE 'https://emergentbiome.earth/ontology/mvo/MolecularRecord' IN u.types
AND lane.predicate='https://emergentbiome.earth/ontology/mvo/laneId'
RETURN lane.lexical AS lane, count(u) AS registered ORDER BY lane
