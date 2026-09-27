MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (a:MVD_mvo_ValidationAction {snapshot:$snapshot,authorizedInternal:true})
RETURN a.iri AS evidence_id,a.mvo_nextAction[0] AS action,a.mvo_reason[0] AS reason,toInteger(a.mvo_affectedPathCount[0]) AS affected_paths,
a.mvo_independentUnit[0] AS denominator,'Rank only within the same path unit; ties retained; counts are not probabilities or carbon benefit' AS interpretation
ORDER BY denominator,affected_paths DESC,evidence_id
