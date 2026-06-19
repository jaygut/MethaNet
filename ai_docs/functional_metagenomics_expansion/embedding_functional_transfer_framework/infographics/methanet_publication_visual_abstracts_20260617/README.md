# MethaNet Publication Visual Abstracts

Date: 2026-06-17

Purpose: journal-style visual abstract versions distilled from the generated
MethaNet infographics:

- `methanet_agentic_workflow_moat_v3.png`
- `methanet_attestation_graph_moat_v1.png`
- `methanet_molecular_intelligence_layer_generated.png`
- `methanet_molecular_intelligence_layer_HQ_4k.png`

## Exports

Each visual abstract is exported as PNG, PDF, and SVG.

| Stem | Best use |
| --- | --- |
| `molecular_attestation_layer_visual_abstract_light` | Primary Nature/Science-style visual abstract for the full multi-source data arc. |
| `attestation_graph_visual_abstract_dark` | Cell-style graphical abstract emphasizing graph-versus-table explainability. |
| `agentic_workflow_visual_abstract_light` | Clean companion abstract for the workflow moat and partner-facing development story. |

## Claim Boundary

These graphics intentionally describe MethaNet as a molecular attestation and
MRV feature-readiness layer. They do not claim final sample-level methane-risk
scores, measured flux, final A-E risk tiers, carbon-credit approval, or
source-independent rumen-to-wetland transfer. Stronger claims require sample
mapping, abundance/read coverage, environmental covariates, uncertainty
propagation, source-aware validation, and flux/process validation.

## Reproducibility

Regenerate with:

```bash
.venv/bin/python scripts/figures/functional_metagenomics/plot_publication_visual_abstracts.py
```
