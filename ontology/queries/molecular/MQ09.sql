SELECT r.id AS evidence_id,a.lane,a.proteome_id,l.gene_id,l.identity_state,sample.sample_id,c.source_site_label,c.region,
c.date_label,c.depth_label,r.modality,r.value,r.assay,c.blocking_join,s.sha256 AS source_sha256,r.source_row,
0 AS accepted_exact_flux_pairs,'Source RNA plus study context; no physically paired molecular-flux observation' AS allowed_wording
FROM rna r JOIN loci l ON l.id=r.locus_id JOIN samples sample ON sample.id=r.sample_id
JOIN contexts c ON c.sample_id=sample.id JOIN sources s ON s.id=r.source_id JOIN annotations a ON a.evidence_id=r.annotation_id
WHERE ($lane='' OR a.lane=$lane) AND ($site='' OR c.source_site_label=$site)
ORDER BY evidence_id
