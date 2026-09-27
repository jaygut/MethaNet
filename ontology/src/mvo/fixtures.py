"""Small explicitly synthetic scenario spanning CH4, CO2, N2O and review gates."""
from decimal import Decimal
from rdflib import Literal, URIRef
from rdflib.namespace import RDF, XSD
from .model import M, C, PROV, QUDT, iri, graph, node, add, foundation, assertion, gap, bind_policy


def synthetic():
    g = graph()
    ctx = foundation(g, "SYNTHETIC-TEST-ONLY", "2026-09-26T00:00:00Z")
    artifact = node(g, iri("synthetic", "source"), M.Artifact, "SYNTHETIC source; no real project")
    add(g, artifact, M.sourceLocator, "synthetic://fixture")
    release = node(g, iri("synthetic", "release"), M.DatasetRelease)
    add(g, release, M.sourceLocator, "synthetic://release")
    lanes = [node(g, iri("synthetic", "lane", x), M.AtlasLane) for x in ("a", "b")]
    missing = gap(g, iri("synthetic", "gap"), "SYNTHETIC missing ESM2 payload", "Rebuild and validate source payload")
    units = []
    for i in range(3):
        unit = node(g, iri("synthetic", "record", str(i)), M.MolecularRecord)
        units.append(unit)
        for p, v in ((M.laneId, "a" if i < 2 else "b"), (M.proteomeId, "SYNTHETIC-" + str(i)),
                     (M.inLane, lanes[0 if i < 2 else 1]), (M.inRelease, release), (PROV.wasDerivedFrom, artifact),
                     (M.hasESM2, i != 2), (M.hasGLM2, True), (M.hasFunctional, True),
                     (M.triViewReady, i != 2), (M.releaseExcluded, i == 2), (M.mechanismComparable, False)):
            add(g, unit, p, v)
        if i == 2:
            add(g, unit, M.blockedBy, missing)
    site = node(g, iri("synthetic", "site"), M.Site)
    add(g, site, M.hasHabitat, C.mangrove)
    # Habitat is a referenced vocabulary concept, not a source assertion of a real location.
    sample = node(g, iri("synthetic", "physical-sample"), M.PhysicalSample)
    event = node(g, iri("synthetic", "sampling-event"), M.SamplingEvent)
    add(g, event, M.timeStatus, "known")
    add(g, event, M.sampledAt, "2026-09-01T12:00:00Z", XSD.dateTime)
    add(g, event, M.usesMethod, ctx["method"])
    add(g, event, PROV.wasDerivedFrom, artifact)
    add(g, event, M.hasSite, site)
    add(g, sample, M.hasSamplingEvent, event)
    sequencing = node(g, iri("synthetic", "sequencing-sample"), M.SequencingSample)
    ann = assertion(g, iri("synthetic", "annotation"), M.AnnotationAssertion, units[0], M.functionalPotential,
                    C.methanogenesis, artifact, ctx)
    for p, v in ((M.databaseVersion, "SYNTHETIC-database-v1"), (M.databaseName, "SYNTHETIC"), (M.toolVersion, "SYNTHETIC-tool-v1"), (M.thresholdDescription, "SYNTHETIC test threshold"),
                 (M.coverageState, "completed_covered"), (M.evidenceState, C.present)):
        add(g, ann, p, v)
    for gas, process in ((C.CH4, C.methanogenesis), (C.CO2, C.respiration), (C.N2O, C.denitrification)):
        key = str(gas).split("/")[-1]
        obs = node(g, iri("synthetic", "flux", key), M.FluxObservation)
        quantity = node(g, iri("synthetic", "quantity", key), M.Quantity)
        add(g, quantity, QUDT.numericValue, Literal(Decimal("1.25"), datatype=XSD.decimal))
        # QUDT unit IRI verified through unit vocabulary in research ledger.
        add(g, quantity, QUDT.unit, URIRef("http://qudt.org/vocab/unit/MicroMOL-PER-M2-SEC"))
        add(g, quantity, M.quantityKind, "gas_flux")
        for p, v in ((M.hasResult, quantity), (M.gas, gas), (M.featureOfInterest, site), (M.measurementMethod, "chamber"),
                     (M.spatialSupport, "SYNTHETIC chamber footprint"), (M.qcStatus, "synthetic"),
                     (M.usesMethod, ctx["method"]), (M.hasUncertainty, ctx["uncertainty"]), (PROV.wasDerivedFrom, artifact)):
            add(g, obs, p, v)
        add(g, obs, M.observedAt, "2026-09-01T12:00:00Z", XSD.dateTime)
        bridge = assertion(g, iri("synthetic", "bridge", key), M.BridgeAxiom, process, M.supports, obs, artifact, ctx, state=C.hypothetical)
        add(g, bridge, M.validationState, "hypothesis")
    conf_a = node(g, iri("synthetic", "config", "a"), M.EmbeddingConfiguration)
    conf_b = node(g, iri("synthetic", "config", "b"), M.EmbeddingConfiguration)
    add(g, conf_a, M.configurationFingerprint, "a" * 64)
    add(g, conf_b, M.configurationFingerprint, "b" * 64)
    sim = assertion(g, iri("synthetic", "similarity"), M.SimilarityAssertion, units[0], M.molecularSimilarity,
                    units[1], artifact, ctx)
    add(g, sim, M.configuration, conf_a)
    add(g, sim, M.comparisonConfiguration, conf_b)
    add(g, sim, M.compatibilityStatus, "incompatible")
    bind_policy(g, ctx["policy"])
    return g
