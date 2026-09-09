# Audit registry

A commit or workflow trigger is not evidence. VERIFIED requires a verified evidence file and at least one metrics file; overrides can only downgrade.

| Audit | Status | Evidence | Metrics | Boundary / reason |
|---|---|---|---:|---|
| `committed_subset_audit` | **VERIFIED** | VERIFIED | 3 |  |
| `exact` | **SUPPORTIVE_ONLY** | — | 1 |  |
| `exact_crop` | **SUPPORTIVE_ONLY** | — | 1 | Limited 2-D surrogate; not a complete 512x512 3-D ISAM validation. |
| `exact_fig4_crop` | **PENDING** | — | 1 |  |
| `exact_leaf` | **SUPPORTIVE_ONLY** | — | 1 | Exploratory selected A-lines, not a paired full-volume population analysis. |
| `exact_strip` | **INVALIDATED** | — | 1 | Legacy 2-D ISAM target was evaluated at the wrong axial coordinate; retained only for provenance. |
| `exact_superfactor` | **PENDING** | — | 1 |  |
| `extracts` | **PENDING** | — | 0 |  |
| `order_audit` | **PENDING** | — | 0 |  |
| `order_audit/leaf` | **FAILED_RUN** | — | 0 | Large-download workflow preserved diagnostics but did not produce verified metrics. |
| `order_audit/tio2` | **FAILED_RUN** | — | 0 | Large-download workflow preserved diagnostics but did not produce verified metrics. |
| `published_fig4_point` | **SUPPORTIVE_ONLY** | — | 2 | Registered same-point processed-output comparison; method volumes are separately normalized and not raw physical energy. |
| `published_fig5_aline` | **BOUNDARY_ONLY** | — | 0 | Deposited leaf-output format/reproduction boundary; not an independent raw-data validation. |
| `source_ranges` | **PENDING** | — | 0 |  |
| `superfactor_holdout` | **FAILED_RUN** | — | 0 | Legacy workflow preserved provenance only; use committed_subset_audit support results instead. |
| `synthetic` | **PENDING** | — | 2 |  |
| `uncertainty_audit` | **VERIFIED** | VERIFIED | 1 |  |
| `upstream` | **PENDING** | — | 1 |  |