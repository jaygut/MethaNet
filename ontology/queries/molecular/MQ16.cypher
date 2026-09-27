MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (c:MVD_mvo_SourceContextRecord {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forSample]->(sample:MVD_mvo_SequencingSample)
WHERE ($lane='' OR c.mvo_laneId[0]=$lane) AND ($site='' OR c.mvo_sourceSiteLabel[0]=$site)
AND ($date_prefix='' OR c.mvo_sourceDateLabel[0] STARTS WITH $date_prefix) AND ($depth='' OR c.mvo_sourceDepthLabel[0]=$depth)
OPTIONAL MATCH (m:MVD_mvo_SourceMeasurement {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forSample]->(sample)
WITH c,sample,m ORDER BY m.mvo_status[0]
RETURN c.iri AS evidence_id,c.mvo_laneId[0] AS lane,sample.mvo_sourceIdentifier[0] AS sample_id,c.mvo_sourceSiteLabel[0] AS site,
c.mvo_sourceDateLabel[0] AS date_label,c.mvo_sourceDepthLabel[0] AS depth_label,c.mvo_assayContract[0] AS assay,
collect(m.mvo_status[0]) AS source_matched_environment_variables,collect(m.iri) AS measurement_evidence_ids,
c.mvo_identityState[0] AS context_identity,c.mvo_sampleLinkageStatus[0] AS blocking_join
ORDER BY lane,site,date_label,depth_label,sample_id
