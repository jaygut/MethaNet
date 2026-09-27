MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (p:MVD_mvo_EvidencePacket {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),
(p)-[:MVD_mvo_hasEvaluation]->(e:MVD_mvo_PanelEvaluation)
WHERE e.mvo_panelStatus[0] IN ['present','ambiguous'] AND ($lane='' OR p.mvo_laneId[0]=$lane) AND ($habitat='' OR p.mvo_habitatLabel[0]=$habitat)
WITH p,m,e ORDER BY e.mvo_familyId[0]
RETURN p.iri AS evidence_id,p.mvo_laneId[0] AS lane,p.mvo_habitatLabel[0] AS habitat,m.mvo_proteomeId[0] AS proteome_id,
collect(e.mvo_familyId[0]) AS co_occurring_candidate_families,collect(e.mvo_componentSupport[0]) AS component_support,
head(collect(e.mvo_assayContract[0])) AS assay,toInteger(head(collect(e.mvo_registeredDenominator[0]))) AS registered_lane_mags,
'Within-source MAG panel co-occurrence only; no community membership/abundance or net process inference' AS boundary
ORDER BY lane,proteome_id
