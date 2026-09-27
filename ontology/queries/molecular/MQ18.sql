WITH coverage AS (
 SELECT pe.packet_id,count(*) AS evaluated_families,count(*) FILTER (WHERE e.status='covered_no_accepted_hit') AS covered_no_hit_families,
 count(*) FILTER (WHERE e.status='ambiguous') AS ambiguous_families,min(e.assay) AS assay,min(e.registered_lane_mags) AS registered_lane_mags
 FROM packet_evaluations pe JOIN evaluations e ON e.id=pe.evaluation_id GROUP BY pe.packet_id
), annotations_count AS (SELECT packet_id,count(*) AS annotation_evidence_items FROM packet_annotations GROUP BY packet_id)
SELECT p.id AS evidence_id,p.lane,p.habitat,p.region,p.proteome_id,c.registered_lane_mags,c.assay,c.evaluated_families,c.covered_no_hit_families,c.ambiguous_families,
ac.annotation_evidence_items,claim.review_state,p.rights,g.id AS gap_id,g.reason AS blocking_gap,g.action AS monitoring_or_review_action,p.allowed_wording,
'Proposed verifier diligence workflow; no protocol compliance or credit approval assessed' AS protocol_relevance
FROM packets p JOIN coverage c ON c.packet_id=p.id JOIN annotations_count ac ON ac.packet_id=p.id
JOIN packet_claims pc ON pc.packet_id=p.id JOIN claims claim ON claim.id=pc.claim_id
JOIN packet_gaps pg ON pg.packet_id=p.id JOIN gaps g ON g.id=pg.gap_id
WHERE ($lane='' OR p.lane=$lane) AND ($proteome='' OR p.proteome_id=$proteome) AND ($habitat='' OR p.habitat=$habitat)
ORDER BY evidence_id,gap_id
