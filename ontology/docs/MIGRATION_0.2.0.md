# MVO 0.2.0 molecular-extension migration

Status: local extension validated for source/semantic/query fidelity; not a public release or independent
biological approval. Canonical projection format remains 0.1.0 (lossless RDF
statement projection); ontology, shapes, adapter and mapping versions advance
to 0.2.0. The candidate-panel mapping has its own version 0.1.0.

The additive schema introduces source features, coordinate-verified called
loci, sequence-available proteins, assembly/contig representations, bounded
panel evaluations and coverage ledgers, source-processed gene RNA, contextual
measurements, alternatives and next validation actions. The historical Gene
and Protein classes are retained. New stronger subtypes require stronger
admission, so historical fixtures are not silently reclassified.

No property chain entails function, flux or credit eligibility. A family
accession is not a complete pathway. Component support and pathway completeness
are separate, and this release does not admit a complete-pathway claim. RNA
values cannot be asserted as DNA abundance. Unresolved features may exist
without a sequence digest; `encodes` is used only after a coordinate/translation
check against the specific assembly and source protein.

Data migration is append-only from the full-hash-verified 0.1.0 graph. Existing
record IRIs, old assertions, claim states, exclusions and source evidence are
preserved as exact triples. New IDs include lane, record, original run/feature
namespace and source gene identity. Sequence digests do not merge loci. Named
reference-to-OWC mappings use explicit source `db_id` fields, with sequence and
contig reconciliation tested separately.

The source/event selection was registered before result inspection in
`MOLECULAR_EXPANSION_20260926.md` and the versioned panel mapping. Full coverage
denominators remain in a hash-addressed warehouse ledger; a compact selected
graph does not reduce the registered denominator. Source tables and the August
release pointer are unchanged.

Code inventory now includes the executable query pack. This corrects an omission
in the original snapshot's code inventory; old manifests and hashes are not
rewritten. A new graph's build fingerprint therefore binds its queries as well
as its adapters, mappings and shape definitions.

Rollback is selection of `build/atlas-20260926-validated`, never a database reset
or deletion. A successor Neo4j snapshot must pass full RDF reconstruction before
queries may call it validated. The new snapshot ID, semantic diff, test receipts
and document QA are supplied in the dated handoff and report receipts.

The schema diff adds 71 declared terms and removes none. The two ontology-version
axioms are replaced. Exact source assertions from the old snapshot are unchanged.
The original core file's byte formatting was not recovered; its baseline axioms
were independently reconstructed from the retained reasoner-control artifact
and checked against version-reverted core plus unchanged baseline modules.
This qualification is explicit in the semantic-diff receipt.

Canonical build-time code is archived by manifest checksum. After materialization,
the loader's read-back was changed to bounded, indexed transactions following a
retained timeout; the canonical graph/export meaning is unchanged. Import receipts
bind the operational loader SHA. The separately archived operational pack also
contains the restricted parameterized demo entrypoint. Neither operational change
rewrites the immutable graph manifest or its code inventory.
