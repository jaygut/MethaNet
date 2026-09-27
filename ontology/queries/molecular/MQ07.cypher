MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (a:MVD_mvo_IdentityAssertion {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forRecord]->(m:MVD_mvo_MolecularRecord),(a)-[:MVD_mvo_forSample]->(sample:MVD_mvo_SequencingSample),
(a)-[:MVD_mvo_hasContextRecord]->(c:MVD_mvo_SourceContextRecord)
WHERE ($lane='' OR a.mvo_laneId[0]=$lane) AND ($site='' OR c.mvo_sourceSiteLabel[0]=$site)
RETURN a.iri AS evidence_id,a.mvo_laneId[0] AS lane,m.mvo_proteomeId[0] AS proteome_id,sample.mvo_sourceIdentifier[0] AS sample_id,
a.mvo_linkResolution[0] AS relation,a.mvo_identityState[0] AS identity_state,c.mvo_sourceSiteLabel[0] AS site_label,c.mvo_sourceDateLabel[0] AS date_label,
c.mvo_sourceDepthLabel[0] AS depth_label,c.mvo_assayContract[0] AS assay,c.mvo_geographicPrecision[0] AS geography_resolution,
c.mvo_sampleLinkageStatus[0] AS physical_pair_status,a.mvo_sourceRow[0] AS source_row
ORDER BY lane,proteome_id,sample_id
