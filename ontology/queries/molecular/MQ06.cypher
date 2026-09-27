MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (r:MVD_mvo_MolecularSampleObservation {snapshot:$snapshot,authorizedInternal:true})-[:MVD_mvo_forLocus]->(l:MVODResource),(r)-[:MVD_mvo_forSample]->(sample:MVD_mvo_SequencingSample),
(r)-[:MVD_mvo_hasEvidenceItem]->(a:MVD_mvo_AnnotationAssertion),(r)-[:MVD_prov_wasDerivedFrom]->(src:MVODResource)
WHERE ($lane='' OR a.mvo_laneId[0]=$lane)
OPTIONAL MATCH (l)-[:MVD_mvo_forProtein]->(p:MVD_mvo_SequenceProtein)
RETURN r.iri AS evidence_id,a.mvo_laneId[0] AS lane,a.mvo_proteomeId[0] AS proteome_id,a.mvo_familyId[0] AS family,l.mvo_geneId[0] AS gene_id,
l.mvo_identityState[0] AS identity_state,p.mvo_sequenceDigest[0] AS protein_sequence_sha256,sample.mvo_sourceIdentifier[0] AS sample_id,
r.mvo_assayModality[0] AS modality,r.mvo_processedValue[0] AS value,r.mvo_originalUnit[0] AS unit,r.mvo_normalization[0] AS normalization,
r.mvo_assayContract[0] AS assay_reconciliation,r.mvo_sampleLinkageStatus[0] AS field_pair_status,src.mvo_sha256[0] AS source_sha256,r.mvo_sourceRow[0] AS source_row,
'DNA abundance and measured protein abundance are not instantiated' AS missing_modalities
ORDER BY gene_id,sample_id
