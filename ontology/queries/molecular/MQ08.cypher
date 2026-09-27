MATCH (s:MVOSnapshot {id:$snapshot,status:'validated'}), (d:MVODomainSnapshot {id:$snapshot,status:'validated',version:'0.1.0'}) WHERE $purpose='internal_review'
MATCH (m:MVODResource {snapshot:$snapshot,authorizedInternal:true})-[:MVD_prov_wasDerivedFrom]->(src:MVODResource)
WHERE (m:MVD_mvo_SourceMeasurement OR m:MVD_mvo_SourceMeasurementGap) AND ($lane='' OR m.mvo_laneId[0]=$lane) AND ($site='' OR m.mvo_sourceSiteLabel[0]=$site) AND ($process='' OR m.mvo_status[0]=$process)
OPTIONAL MATCH (m)-[:MVD_mvo_forSample]->(sample:MVD_mvo_SequencingSample)
RETURN m.iri AS evidence_id,m.mvo_laneId[0] AS lane,m.mvo_status[0] AS variable,m.mvo_quantityKind[0] AS quantity_kind,m.mvo_processedValue[0] AS value,
CASE WHEN m:MVD_mvo_SourceMeasurementGap THEN 'source_missing_not_zero' ELSE 'reported_numeric' END AS value_status,
m.mvo_originalUnit[0] AS unit,m.mvo_measurementMethod[0] AS method,m.mvo_sourceDateLabel[0] AS date_label,m.mvo_sourceDepthLabel[0] AS depth_label,
m.mvo_sourceSiteLabel[0] AS site,m.mvo_spatialSupport[0] AS spatial_support,m.mvo_sampleLinkageStatus[0] AS sample_linkage,
sample.mvo_sourceIdentifier[0] AS source_metadata_sample,src.mvo_sha256[0] AS source_sha256,m.mvo_sourceRow[0] AS source_row
ORDER BY lane,variable,evidence_id
