MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'})
WHERE $purpose='internal_review'
MATCH (c:MVD_mvo_PanelCoverageSummary {snapshot:$snapshot,authorizedInternal:true})
WHERE ($lane='' OR c.mvo_laneId[0]=$lane) AND ($habitat='' OR c.mvo_habitatLabel[0]=$habitat) AND ($family='' OR c.mvo_familyId[0]=$family)
OPTIONAL MATCH (p:MVD_mvo_EvidencePacket {snapshot:$snapshot,authorizedInternal:true})
WHERE p.mvo_laneId[0]=c.mvo_laneId[0] AND p.mvo_habitatLabel[0]=c.mvo_habitatLabel[0]
RETURN c.iri AS evidence_id,c.mvo_laneId[0] AS lane,c.mvo_habitatLabel[0] AS habitat,c.mvo_familyId[0] AS family,c.mvo_panelStatus[0] AS status,
toInteger(c.mvo_rowCount[0]) AS mag_family_rows,toInteger(c.mvo_registeredDenominator[0]) AS registered_lane_mags,count(p) AS selected_mags_in_habitat,
c.mvo_assayContract[0] AS assay,c.mvo_compatibilityStatus[0] AS comparability
ORDER BY lane,habitat,family,status
