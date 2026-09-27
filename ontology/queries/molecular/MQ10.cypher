MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (p:MVD_mvo_EvidencePacket {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_hasEvidenceItem]->(claim:MVD_mvo_Claim),
(p)-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),(claim)-[:MVD_mvo_reviewState]->(review:MVODResource),(p)-[:MVD_mvo_blockedBy]->(gap:MVD_mvo_ValidationGap),
(claim)-[:MVD_prov_wasDerivedFrom]->(src:MVODResource),(claim)-[:MVD_mvo_usesMethod]->(method:MVD_mvo_Method)
WHERE ($lane='' OR p.mvo_laneId[0]=$lane) AND ($proteome='' OR m.mvo_proteomeId[0]=$proteome)
RETURN claim.iri AS evidence_id,p.iri AS packet_id,p.mvo_laneId[0] AS lane,m.mvo_proteomeId[0] AS proteome_id,review.iri AS review_state,
method.mvo_version[0] AS mapping_version,src.mvo_sha256[0] AS source_sha256,p.mvo_rightsStatus[0] AS rights,
gap.iri AS gap_id,gap.mvo_reason[0] AS blocking_gap,claim.mvo_allowedWording[0] AS allowed_wording,claim.mvo_nextAction[0] AS next_action
ORDER BY packet_id,gap_id
