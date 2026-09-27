MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (e:MVD_mvo_PanelEvaluation {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord)
WHERE e.mvo_familyId[0]=coalesce(nullIf($family,''),'cummo_complex') AND ($lane='' OR e.mvo_laneId[0]=$lane)
RETURN e.iri AS evidence_id,e.mvo_laneId[0] AS lane,e.mvo_habitatLabel[0] AS habitat,e.mvo_regionLabel[0] AS region,m.mvo_proteomeId[0] AS proteome_id,
e.mvo_familyId[0] AS family,e.mvo_panelStatus[0] AS status,toInteger(e.mvo_registeredDenominator[0]) AS registered_lane_mags,
e.mvo_assayContract[0] AS assay,m.mvo_taxonomicContext[0] AS taxonomy,m.mvo_taxonomyVersion[0] AS taxonomy_version,
m.mvo_magCompleteness[0] AS completeness,m.mvo_magContamination[0] AS contamination,e.mvo_independentlyReviewed[0] AS independently_reviewed,
'Abstain: independent marker review and common assay/version contract pending; selected MAGs are not ecosystem replicates' AS comparison_verdict
ORDER BY lane,habitat,proteome_id
