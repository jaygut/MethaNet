# MMAG MVP Expert Audit

Date: 2026-06-17

Snapshot audited:

```text
results/attestation/mmag_mvp_20260617/
```

Commands:

```bash
./.venv/bin/python scripts/attestation/build_molecular_attestation_mvp.py \
  --repo-root /home/rsg-jcorre38/Jay_Proyects/MethaNet \
  --snapshot-id mmag_mvp_20260617

./.venv/bin/python scripts/attestation/audit_molecular_attestation_mvp.py \
  --snapshot-dir results/attestation/mmag_mvp_20260617
```

## Verdict

The MVP is audit-solid for its intended scope: a local, embedded,
MAG/proteome-level molecular attestation substrate. It is not a final MRV risk
scorer, and it correctly refuses to behave like one.

It passes the critical expert-facing gates:

- 662 canonical cohort units are present;
- `proteome_id` is unique;
- 625 MAG-bin units are represented;
- 37 assembly-context units are represented and quarantined;
- no completed MAG-bin units are dropped;
- no duplicate primary graph entities are present;
- all graph edge endpoints resolve to exported graph nodes;
- every graph edge has source-artifact or boundary provenance;
- every evidence atom has source-artifact provenance;
- every MAG is blocked from final MRV-risk and sample-risk claims;
- no MAG supports `claim:final_mrv_risk_tier`;
- all 37 assembly-context units have quarantine edges;
- Kuzu builds and executes the query suite.

## Stress Tests

The audit stress suite passed 10/10 additional queries:

| Query | Complexity | Rows | Purpose |
| --- | --- | ---: | --- |
| `simple_node_type_counts` | simple | 8 | Verifies heterogeneous graph schema |
| `simple_claim_status_counts` | simple | 4 | Verifies blocked/provisional/allowed/forbidden claim states |
| `simple_readiness_counts` | simple | 3 | Verifies MAG-bin, QC-caveated, and quarantined units |
| `simple_artifact_fanout` | simple | 7 | Verifies artifact-to-evidence traceability |
| `complex_candidate_to_artifact_to_claim` | complex | 7 | Candidate to evidence to artifact to claim path |
| `complex_bridge_neighbor_with_function_context` | complex | 50 | Cross-domain ESM2 neighbors plus methane evidence |
| `complex_claim_upgrade_paths` | complex | 14 | Claim blockers and upgrade requirements |
| `complex_forbidden_mrv_safety` | complex | 1 | Molecular attestation allowed while final MRV remains blocked |
| `complex_source_taxonomy_caveats` | complex | 30 | Source/taxonomy confounding visibility |
| `complex_quarantine_integrity` | complex | 1 | Assembly-context quarantine integrity |

## Graph Inventory

Node counts:

| Node type | Count |
| --- | ---: |
| EvidenceAtom | 3,968 |
| Feature | 2,644 |
| MAG | 662 |
| Taxon | 397 |
| Artifact | 13 |
| ValidationGap | 8 |
| Claim | 5 |
| SourceDomain | 2 |

Relationship counts:

| Relationship | Count |
| --- | ---: |
| NEAR_IN_ESM2_SPACE | 9,930 |
| HAS_EVIDENCE | 3,968 |
| GENERATED_BY | 3,968 |
| EVIDENCE_SUPPORTS_CLAIM | 3,968 |
| BLOCKED_BY | 3,379 |
| HAS_FEATURE | 2,644 |
| MAG_BLOCKED_FROM_CLAIM | 1,324 |
| FROM_SOURCE | 662 |
| MAG_SUPPORTS_CLAIM | 635 |
| HAS_TAXONOMY | 625 |
| CLAIM_BLOCKED_BY | 14 |

Evidence atom predicates:

| Predicate | Count |
| --- | ---: |
| `has_run_status` | 662 |
| `has_methane_functional_evidence` | 662 |
| `has_sulfur_competition_context` | 662 |
| `has_annotation_coverage` | 662 |
| `blocked_from_final_mrv_risk` | 662 |
| `has_glm2_context` | 648 |
| `near_bridge_in_esm2_space` | 10 |

## Readiness Distribution

| Readiness | MAGs |
| --- | ---: |
| `molecular_attestation_ready_not_mrv` | 437 |
| `molecular_attestation_ready_with_qc_caveat` | 188 |
| `blocked_noncomparable_unit` | 37 |

This distribution is exactly the kind of output an expert reviewer should want:
usable signal is not blurred with QC caution or non-comparable assembly-context
evidence.

## Top Multiview Bridge Evidence

The top queryable multiview bridge candidates include both rumen and wetland
units, with provisional bridge score, methane score, and readiness status
visible in the same graph query. The current top evidence path includes:

| Proteome ID | Source | Bridge score | Methane score | Readiness |
| --- | --- | ---: | ---: | --- |
| `rumen__10674_0004_idba_bin.23` | rumen | 47.992 | 17.000 | ready, not MRV |
| `mucc__GCA_002495465.1_ASM249546v1_genomic` | mucc | 32.136 | 1.000 | ready, not MRV |
| `rumen__10674_0002_idba_bin.8` | rumen | 22.454 | 11.000 | ready with QC caveat |
| `rumen__10674_0004_idba_bin.79` | rumen | 20.995 | 56.000 | ready, not MRV |
| `rumen__10674_0001_idba_bin.23` | rumen | 20.899 | 32.000 | ready with QC caveat |

The important point is not just ranking. The graph can traverse from any one of
these candidates to source domain, taxonomy, methane/sulfur features, gLM2
context, ESM2 neighbors, evidence atoms, source artifacts, and blocked claim
vocabulary.

## Claim-Safety Audit

The graph distinguishes:

- allowed MAG/proteome molecular attestation;
- provisional bridge-candidate attestation;
- blocked sample-level methane-risk claims;
- blocked final A-E MRV tier claims;
- carbon-credit approval forbidden from the current molecular atlas.

Every MAG has edges blocking sample-level methane-risk and final MRV tier
claims. This is not merely documentation; it is queryable graph structure.

## Intelligence Beyond A Monolithic System

A monolithic feature table can answer:

```text
Which rows have high bridge_score and methane_evidence_score?
```

The graph can answer harder questions:

- Which exact evidence atoms support this candidate?
- Which source artifacts generated those atoms?
- Which claim is allowed from those atoms?
- Which claims are blocked, and by which missing evidence?
- Which cross-domain neighbors share methane evidence but differ by source,
  taxonomy, QC, or readiness?
- Which candidates have strong bridge signal but QC caveats?
- Which assembly-context units exist, and are they safely quarantined?
- What measurement would upgrade sample-level or MRV-facing claims?

That is the intelligence moat: the graph stores evidence, provenance, absence,
uncertainty, source caveats, and claim boundaries as first-class relationships.
This creates an auditable molecular-attestation operating layer rather than a
static report or ranking table.

## Remaining Productionization Caveats

The MVP is strong for local attestation, but expert users should not treat it as
finished production infrastructure:

- Gene-level marker/pathway nodes should be added next from MCycDB, SCycDB,
  KOfam, METABOLIC, dbCAN, CAZy, and MEROPS facts.
- Query tests should be expanded once top candidates have reviewed
  candidate-card evidence and source-aware null results.
- Kuzu `0.11.3` should remain pinned and Parquet exports should remain the
  durable fallback.
- Source-independent transfer remains blocked by source/ecosystem confounding.
- Sample-level and registry-facing claims remain blocked until sample mapping,
  abundance/read coverage, environmental covariates, uncertainty propagation,
  and flux/process validation exist.

## Bottom Line

This MVP can stand serious internal expert review for the claim it actually
makes:

> MethaNet now has a local, queryable molecular attestation graph that links
> ESM2 bridge evidence, functional evidence, gLM2 context, QC, taxonomy,
> provenance, missingness, and claim boundaries at MAG/proteome grain.

It also clearly shows what it cannot yet claim, which is precisely what makes it
scientifically defensible.
