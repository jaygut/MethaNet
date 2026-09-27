MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (e:MVODResource {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_hasValidationAction]->(a:MVD_mvo_ValidationAction)
WHERE ($lane='' OR e.mvo_laneId[0]=$lane) AND ($habitat='' OR e.mvo_habitatLabel[0]=$habitat)
RETURN a.iri AS evidence_id,a.mvo_nextAction[0] AS action,count(DISTINCT e) AS affected_paths,a.mvo_independentUnit[0] AS denominator,
collect(DISTINCT e.mvo_laneId[0]) AS lanes,'Necessary, overlapping blockers: resolving one alone does not complete the chain. Rank only within the same path unit; no carbon-benefit ranking' AS frontier_interpretation
ORDER BY denominator,affected_paths DESC,evidence_id
