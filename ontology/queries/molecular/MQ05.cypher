MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (c:MVD_mvo_PanelCoverageSummary {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forFamily]->(f:MVD_mvo_PanelFamilyConcept)
WHERE c.mvo_habitatLabel[0]<>'non_wetland_rumen_control' AND ($lane='' OR c.mvo_laneId[0]=$lane) AND ($family='' OR c.mvo_familyId[0]=$family)
RETURN c.iri AS evidence_id,c.mvo_laneId[0] AS lane,c.mvo_habitatLabel[0] AS habitat,c.mvo_regionLabel[0] AS region,c.mvo_familyId[0] AS family,
c.mvo_panelStatus[0] AS status,toInteger(c.mvo_rowCount[0]) AS mag_family_rows,toInteger(c.mvo_registeredDenominator[0]) AS registered_lane_mags,
c.mvo_assayContract[0] AS assay,c.mvo_compatibilityStatus[0] AS comparability,f.mvo_allowedWording[0] AS allowed_wording
ORDER BY lane,habitat,family,status
