# Generated ontology reference

Generated from canonical Turtle; do not edit independently.

Ontology version: 0.2.0. Local development namespace; not yet publicly served.

## Classes

| Term | Meaning | Parent |
| --- | --- | --- |
| `AbundanceObservation` | Read-coverage or abundance measurement under a declared DNA assay and normalization. | mvo:Evidence |
| `AccountingActivity` | Versioned calculation with declared gas conversion factors, baseline, leakage and uncertainty; never inferred from a marker count. | mvo:Activity |
| `Activity` | Execution or event that uses or generates evidence. | http://www.w3.org/ns/prov#Activity |
| `Agent` | Human, organization or software with identified responsibility, not implicit authority. | http://www.w3.org/ns/prov#Agent |
| `AnnotationAssertion` | Tool/database/version/threshold-specific assignment, including accepted-hit state and coverage. | mvo:Assertion |
| `Artifact` | Versioned source or derived digital artifact; provenance does not imply reuse rights. | mvo:Entity |
| `Assay` | Molecular assay execution with specimen and protocol provenance. | mvo:Activity |
| `AssemblyRepresentation` | Byte-addressed assembly file for a lane-scoped molecular record, not an independent specimen or organism. | mvo:Artifact |
| `Assertion` | Reified subject-predicate-object statement with epistemic state, scope, provenance, review, rights and uncertainty. Does not entail the unqualified triple. | mvo:Evidence |
| `AtlasLane` | Registered source cohort with its own denominator and evidence contract. | mvo:Entity |
| `BaselineScenario` | Versioned counterfactual assumptions distinct from observations and project outcomes. | mvo:Entity |
| `BridgeAxiom` | Explicit, bounded proposition connecting molecular evidence to a decision-relevant process; unvalidated bridges remain hypotheses, not OWL inference rules. | mvo:Assertion |
| `CalledLocus` | Source-scoped called feature with tested coordinates, strand and translation against a specific assembly representation. | mvo:Gene |
| `CarbonPool` | Declared accounting stock, such as soil organic carbon or biomass. | mvo:Entity |
| `CarbonProject` | Rights- and boundary-defined intervention under a named methodology; not an automatically eligible project. | mvo:Entity |
| `Claim` | Reviewable communication of a scoped proposition with permitted wording and blocking gaps. | mvo:Assertion |
| `ConcentrationObservation` | Gas concentration measurement; not gas exchange flux. | mvo:Observation |
| `ConsentRecord` | Purpose-bound consent or community governance evidence; lack of a record is not consent. | mvo:Entity |
| `ContigRepresentation` | Named nucleotide sequence within a specific assembly; source names and sequence equality are independently checked. | mvo:Entity |
| `CurationAttempt` | One cohort/proteome/run attempt, including failed, partial and superseded attempts. | mvo:Activity |
| `CustodyTransfer` | Recorded transfer of a specimen or aliquot between responsible agents. | mvo:Activity |
| `DatasetRelease` | Immutable dataset version with explicit denominator and scope. | mvo:Artifact, http://www.w3.org/ns/dcat#Dataset |
| `Decision` | Authorized action or disposition distinct from observations, assertions and hypotheses. | mvo:Entity |
| `Embedding` | Vector payload reference with extraction configuration; neither coordinates nor similarity are biological function labels. | mvo:Artifact |
| `EmbeddingConfiguration` | Model, layer, pooling, tokenization, input sequence and normalization fingerprint. | mvo:Method |
| `Entity` | Identifiable physical, digital or conceptual object with fixed scoped identity. | http://www.w3.org/ns/prov#Entity |
| `EnvironmentalObservation` | Environmental covariate such as salinity, redox, temperature or water level. | mvo:Observation |
| `Evidence` | Source-addressable evidence item; supporting evidence is not automatic truth. | mvo:Entity |
| `EvidenceBundle` | Reconstructable package of supporting and conflicting evidence for review. | mvo:Artifact |
| `EvidencePacket` | Reproducible candidate-level bundle of evidence, counterevidence, rights, allowed wording and next validation actions; not a registry decision. | mvo:Artifact |
| `ExpressionObservation` | Source-processed RNA measurement, not DNA abundance, process rate or measured flux. | mvo:Evidence |
| `FluxObservation` | Measured gas exchange rate under a declared chamber/tower method and spatial/time support. | mvo:Observation |
| `GHGStatement` | Scoped gas-specific or net CO2e accounting proposition, requiring validated measurements and accounting methodology. | mvo:Claim |
| `Gene` | Source-scoped sequence feature on a declared assembly and coordinate system. | mvo:Entity |
| `GraphRelease` | Rebuildable named graph snapshot with independently versioned mappings and shapes. | mvo:DatasetRelease |
| `HabitatConcept` | Controlled habitat category; source labels require reviewed vocabulary mappings. | http://www.w3.org/2004/02/skos/core#Concept |
| `IdentityAssertion` | Reviewable source-to-source identity or candidate crosswalk; does not merge nodes. | mvo:Assertion |
| `InterpretationAlternative` | Scientifically relevant competing interpretation, not an assertion that the alternative is actually active. | mvo:Entity |
| `Intervention` | Restoration, rewetting or management action with spatial and temporal boundaries. | mvo:Activity |
| `Method` | Versioned measurement, computation, curation or decision procedure. | mvo:Entity |
| `MolecularPanel` | Versioned candidate-family selection and assay-mapping specification with explicit coverage and review limits. | mvo:Artifact |
| `MolecularRecord` | Lane-scoped MAG/proteome record, not necessarily a unique genome across lanes. | mvo:Entity |
| `MolecularSampleObservation` | One exact source gene by source RNA column value with its processing contract; not raw counts, DNA abundance, a physical specimen identity or measured activity. | mvo:ExpressionObservation |
| `MonitoringPlan` | Versioned measurement and sampling plan with uncertainty, QA and validation requirements. | mvo:Method |
| `Observation` | Method-specific observation with feature, property, time, result, units, QC and uncertainty. | mvo:Evidence, http://www.w3.org/ns/sosa/Observation |
| `ObservationWindow` | Explicit start/end and support for paired molecular/process validation. | mvo:Entity |
| `PanelCoverageSummary` | Lane/habitat/family/status aggregate whose complete underlying MAG-level denominator is recoverable from a hashed coverage ledger. | mvo:Evidence |
| `PanelEvaluation` | One registered molecular record evaluated for one panel family under one source assay contract, including no-hit, ambiguous, unassayed, missing, failed and excluded states. | mvo:Evidence |
| `PanelFamilyConcept` | Candidate family or bounded component set, distinct from demonstrated enzyme activity and complete biochemical pathways. | http://www.w3.org/2004/02/skos/core#Concept |
| `PathwayConcept` | Controlled concept for a functional mechanism; annotation alone does not establish process direction or magnitude. | http://www.w3.org/2004/02/skos/core#Concept |
| `PhysicalSample` | Collected material linked to a sampling event and custody chain, distinct from library/accession records. | mvo:Entity, http://www.w3.org/ns/sosa/Sample |
| `Policy` | Versioned access and purpose constraint; unknown rights are not public permission. | mvo:Entity |
| `Protein` | Sequence-versioned translated protein; not automatically an experimentally validated function. | mvo:Entity |
| `Protocol` | Versioned external accounting or verification specification. | mvo:Method |
| `ProtocolRequirement` | Version- and section-specific external requirement; local mappings do not imply protocol endorsement. | mvo:Entity |
| `Quantity` | Typed number with an explicit unit and quantity kind; original units retained. | mvo:Entity, http://qudt.org/schema/qudt/QuantityValue |
| `ReportingPeriod` | Bounded period for a project accounting statement. | mvo:ObservationWindow |
| `Review` | Explicit assessment by an identified reviewer under a method and scope. | mvo:Activity |
| `SamplingEvent` | Time, place, depth and method specific collection event. | mvo:Activity |
| `Scope` | Geographic, temporal, population, assay and resolution bounds for evidence use. | mvo:Entity |
| `SequenceProtein` | Protein record with an available source sequence and an explicitly normalized SHA-256 digest; identity remains locus and source scoped. | mvo:Protein |
| `SequencingSample` | Assay/sample accession record distinct from a collected physical specimen. | mvo:Entity |
| `SimilarityAssertion` | Scoped retrieval relation with compatibility assessment; never transitive identity, ecology or causality. | mvo:Assertion |
| `Site` | Spatially identified location with declared precision, not automatically a sampling location. | mvo:Entity, http://www.opengis.net/ont/geosparql#Feature |
| `SourceContextRecord` | A dated/depth/site/assay metadata record at the source-declared resolution, without forcing an unresolved label into an exact sampling event. | mvo:Evidence |
| `SourceFeature` | Source-scoped feature locator whose sequence or locus identity may be unresolved; not a verified called gene merely because a source supplies a name. | mvo:Entity |
| `SourceMeasurement` | Source quantity and unit retained at its recorded temporal and spatial precision; molecular linkage and formal flux-observation admission are separate gates. | mvo:Evidence |
| `SourceMeasurementGap` | Selected source observation or chemistry cell with a missing numeric value, retained with its quantity, unit, source identity and context rather than coerced to zero or omitted. | mvo:ValidationGap |
| `TenureRight` | Documented legal or customary right with holder, area, period and jurisdiction. | mvo:Entity |
| `Uncertainty` | Method-specific uncertainty or explicit lack of quantification, never a universal confidence score. | mvo:Entity |
| `ValidationAction` | Review or measurement action associated with explicit blocked evidence paths; path count is not expected carbon benefit or a calibrated risk score. | mvo:Entity |
| `ValidationGap` | Explicit missing, failed, ambiguous or pending prerequisite with next action. | mvo:Entity |
| `VerificationDecision` | External authorized review disposition; this package does not issue or approve credits. | mvo:Decision |

## Object properties

| Term | Meaning | Parent |
| --- | --- | --- |
| `addressesRequirement` | addresses requirement as supporting evidence |  |
| `annotationCandidate` | candidate-family assignment proposition |  |
| `blockedBy` | blocked by |  |
| `comparisonConfiguration` | other embedding configuration |  |
| `configuration` | embedding configuration |  |
| `contradicts` | contradicts |  |
| `coverageLedger` | complete coverage ledger |  |
| `encodes` | encodes protein record |  |
| `epistemicState` | epistemic state |  |
| `evidenceState` | evidence state |  |
| `exactPhysicalSample` | exact physical sample proposition |  |
| `featureOfInterest` | feature of interest |  |
| `forFamily` | for candidate family |  |
| `forLocus` | for called locus or unresolved source feature |  |
| `forPanel` | for panel |  |
| `forProtein` | for source protein record |  |
| `forRecord` | for molecular record |  |
| `forSample` | for sample |  |
| `fromCustodian` | from custodian |  |
| `functionalPotential` | molecular functional potential proposition |  |
| `gas` | greenhouse gas |  |
| `hasAlternative` | has competing interpretation |  |
| `hasBaseline` | has baseline scenario |  |
| `hasConsent` | has consent record |  |
| `hasContextRecord` | has source context record |  |
| `hasEmbedding` | has embedding |  |
| `hasEvaluation` | has panel evaluation |  |
| `hasEvidenceItem` | has evidence or counterevidence item |  |
| `hasHabitat` | has habitat concept |  |
| `hasMonitoringPlan` | has monitoring plan |  |
| `hasPhysicalSample` | has collected physical sample |  |
| `hasPolicy` | has policy |  |
| `hasProject` | has carbon project |  |
| `hasReportingPeriod` | has reporting period |  |
| `hasRequirement` | has protocol requirement |  |
| `hasResult` | has quantity result |  |
| `hasSamplingEvent` | has sampling event |  |
| `hasScope` | has applicability scope |  |
| `hasSite` | has site context |  |
| `hasTenure` | has tenure evidence |  |
| `hasUncertainty` | has uncertainty |  |
| `hasValidationAction` | has next validation action |  |
| `hasWindow` | has time window |  |
| `inAssembly` | in assembly representation |  |
| `inLane` | in atlas lane |  |
| `inRelease` | in release |  |
| `licenseURI` | license URI |  |
| `molecularSimilarity` | molecular similarity proposition |  |
| `object` | assertion object |  |
| `observedProperty` | observed property |  |
| `onContig` | on contig representation |  |
| `predicate` | assertion predicate |  |
| `processedExpression` | processed expression proposition |  |
| `reviewState` | review state |  |
| `reviewedBy` | reviewed by |  |
| `sameProteinSequence` | same normalized protein sequence proposition, not locus identity |  |
| `sourceContextLink` | source-context relationship proposition |  |
| `sourceMeasurementLink` | source-measurement relationship proposition |  |
| `subject` | assertion subject |  |
| `supersedes` | supersedes without deletion |  |
| `supports` | supports |  |
| `toCustodian` | to custodian |  |
| `usesMethod` | uses method |  |

## Datatype properties

| Term | Meaning | Parent |
| --- | --- | --- |
| `acceptedHit` | method-specific accepted hit |  |
| `accession` | source annotation accession |  |
| `accountingBoundary` | accounting boundary |  |
| `affectedPathCount` | count of blocked source-defined paths, not carbon benefit |  |
| `allowedWording` | allowed wording |  |
| `assayContract` | assay and feature semantics |  |
| `assayModality` | actual molecular assay modality |  |
| `assemblyDigest` | assembly SHA-256 digest |  |
| `callerNamespace` | source caller namespace |  |
| `callerVersion` | caller version or explicit unrecorded state |  |
| `candidateEventCount` | candidate source-event count |  |
| `cohortRunId` | cohort run identifier |  |
| `compatibilityStatus` | compatibility status |  |
| `componentAccession` | assayed panel component accession |  |
| `componentSupport` | bounded component support, not pathway completeness |  |
| `configurationFingerprint` | configuration fingerprint or unresolved state |  |
| `consentStatus` | consent status |  |
| `contigId` | source contig identifier |  |
| `coordinateEnd` | inclusive coordinate end |  |
| `coordinateStart` | inclusive coordinate start |  |
| `coordinateSystem` | coordinate convention and reference |  |
| `coverageProbability` | coverage probability |  |
| `coverageState` | tool coverage state |  |
| `databaseName` | annotation database name |  |
| `databaseVersion` | annotation database version |  |
| `decisionDisposition` | decision disposition |  |
| `depthDescription` | depth description and units |  |
| `displaySelectionRule` | prospectively specified display selection rule |  |
| `expiresAt` | policy expiry |  |
| `externalExportAllowed` | external export permitted |  |
| `familyId` | panel family identifier |  |
| `functionalContract` | functional evidence contract |  |
| `geneId` | source-scoped gene identifier |  |
| `geneticCodeTested` | genetic code used for translation consistency check |  |
| `geographicPrecision` | geographic resolution of source attribution |  |
| `glmProtocol` | gLM2 replicate protocol |  |
| `gwpHorizonYears` | GWP time horizon in years |  |
| `gwpSource` | GWP assessment source and version |  |
| `habitatLabel` | source-qualified habitat label |  |
| `hasESM2` | release ESM-2 availability |  |
| `hasFunctional` | release functional payload availability |  |
| `hasGLM2` | release gLM2 availability |  |
| `identityState` | tested identity state |  |
| `independentUnit` | independent replication unit |  |
| `independentlyReviewed` | independent biological interpretation review completed |  |
| `inputSequenceDigest` | input sequence digest |  |
| `jurisdiction` | jurisdiction |  |
| `laneId` | lane identifier |  |
| `layerSelection` | layer selection |  |
| `linkResolution` | crosswalk resolution |  |
| `literalValue` | assertion literal |  |
| `lowerBound` | lower bound |  |
| `magCompleteness` | source MAG completeness percentage |  |
| `magContamination` | source MAG contamination percentage |  |
| `measurementMethod` | measurement modality |  |
| `mechanismComparable` | release mechanism comparability |  |
| `modelName` | model name |  |
| `nextAction` | next validation action |  |
| `normalization` | normalization method |  |
| `observedAt` | observation time |  |
| `originalUnit` | original source unit |  |
| `panelStatus` | panel evaluation status |  |
| `pathwayComplete` | complete pathway claim admitted |  |
| `pooling` | pooling method |  |
| `processedValue` | source-processed observation value |  |
| `projectKey` | authorization project key |  |
| `proteomeId` | canonical proteome identifier |  |
| `purpose` | permitted purpose |  |
| `qcStatus` | quality-control status |  |
| `quantityKind` | quantity kind |  |
| `reason` | reason |  |
| `recordedAt` | recorded at |  |
| `regionLabel` | source-qualified region label |  |
| `registeredDenominator` | full registered lane denominator, not a display limit |  |
| `releaseExcluded` | release exclusion |  |
| `requirementSection` | external requirement section |  |
| `requiresIndependentReview` | requires independent review |  |
| `rightsStatus` | rights status |  |
| `rowCount` | row count |  |
| `runId` | run identifier |  |
| `sampleLinkageStatus` | sample linkage readiness |  |
| `sampledAt` | collection time |  |
| `scopeDescription` | scope description |  |
| `sequenceAvailable` | source sequence available |  |
| `sequenceDigest` | sequence SHA-256 digest |  |
| `sequenceLength` | sequence length under stated normalization |  |
| `sequenceNormalization` | sequence digest normalization |  |
| `sha256` | SHA-256 digest |  |
| `sourceDateLabel` | original date or temporal-window label without invented precision |  |
| `sourceDepthLabel` | original depth label and ambiguity |  |
| `sourceIdentifier` | original source identifier |  |
| `sourceLocator` | source locator |  |
| `sourceNamespace` | source feature namespace |  |
| `sourcePayload` | bounded source-row JSON retained for provenance |  |
| `sourceRow` | source row or key locator |  |
| `sourceSiteLabel` | original site or land-cover label |  |
| `spatialSupport` | spatial support and precision |  |
| `status` | source status |  |
| `strand` | strand, plus one or minus one |  |
| `taxonomicContext` | source taxonomic context |  |
| `taxonomyVersion` | taxonomy source/version contract |  |
| `tenant` | tenant |  |
| `thresholdDescription` | annotation acceptance threshold |  |
| `timeStatus` | valid-time resolution |  |
| `toolName` | annotation tool name |  |
| `toolVersion` | tool version |  |
| `transactionEnd` | transaction end |  |
| `translationStatus` | translation consistency state |  |
| `triViewReady` | release data-complete tri-view |  |
| `uncertaintyDescription` | uncertainty description |  |
| `uncertaintyKind` | uncertainty kind |  |
| `unitStatus` | unit resolution status |  |
| `upperBound` | upper bound |  |
| `validFrom` | valid from |  |
| `validUntil` | valid until |  |
| `validationState` | bridge validation state |  |
| `version` | version |  |
