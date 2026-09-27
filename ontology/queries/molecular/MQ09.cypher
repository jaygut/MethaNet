MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (r:MVD_mvo_MolecularSampleObservation {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forLocus]->(l:MVODResource),(r)-[:MVD_mvo_forSample]->(sample:MVD_mvo_SequencingSample),
(sample)-[:MVD_mvo_hasContextRecord]->(c:MVD_mvo_SourceContextRecord),(r)-[:MVD_prov_wasDerivedFrom]->(src:MVODResource),
(r)-[:MVD_mvo_hasEvidenceItem]->(a:MVD_mvo_AnnotationAssertion)
WHERE ($lane='' OR a.mvo_laneId[0]=$lane) AND ($site='' OR c.mvo_sourceSiteLabel[0]=$site)
RETURN r.iri AS evidence_id,a.mvo_laneId[0] AS lane,a.mvo_proteomeId[0] AS proteome_id,l.mvo_geneId[0] AS gene_id,l.mvo_identityState[0] AS identity_state,
sample.mvo_sourceIdentifier[0] AS sample_id,c.mvo_sourceSiteLabel[0] AS source_site_label,c.mvo_regionLabel[0] AS region,
c.mvo_sourceDateLabel[0] AS date_label,c.mvo_sourceDepthLabel[0] AS depth_label,r.mvo_assayModality[0] AS modality,r.mvo_processedValue[0] AS value,
r.mvo_assayContract[0] AS assay,c.mvo_sampleLinkageStatus[0] AS blocking_join,src.mvo_sha256[0] AS source_sha256,r.mvo_sourceRow[0] AS source_row,
0 AS accepted_exact_flux_pairs,'Source RNA plus study context; no physically paired molecular-flux observation' AS allowed_wording
ORDER BY evidence_id
