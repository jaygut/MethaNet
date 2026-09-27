SELECT a.evidence_id,a.lane,a.habitat,a.proteome_id,a.registered_lane_mags,a.assay,a.family,a.accession,
l.gene_id,l.identity_state,a.source_curation,alt.label AS competing_interpretation,alt.action AS discrimination_action,
s.sha256 AS source_sha256,a.source_row
FROM annotations a JOIN loci l ON a.locus_id=l.id JOIN annotation_alternatives link ON link.annotation_id=a.evidence_id
JOIN alternatives alt ON alt.id=link.alternative_id JOIN sources s ON s.id=a.source_id
WHERE a.family IN ('mcr_complex','cummo_complex') AND ($lane='' OR a.lane=$lane) AND ($habitat='' OR a.habitat=$habitat)
ORDER BY evidence_id,competing_interpretation
