# FAVR review materials

This snapshot contains the inputs, saved outcomes, and implementation materials used to inspect the reported FAVR experiments and recompute their offline results. It is scoped to the experiments included in the paper. Unused experiments, editing records, API response bodies, credentials, and machine-specific working directories are excluded.

## Contents and coverage

| Paper material | Included scope |
| --- | --- |
| Reports and evaluation references | All 1,567 reports from four collections and the current full evaluation reference. |
| Frozen initializations | The eight HC/LLM starting partitions and the features used by the refinement implementation. |
| FAVR and internal controls | The runtime dependencies, pair-judgment protocol, configuration records, and explicit selector seeds for the reported runs. |
| External pipelines | Selected LLMCluster and TERFUR settings, their required inputs, offline generation code, and selection-protocol notes. |
| A3S | The implementation and the 16 saved endpoints used in the paper. Their original complete per-run configuration records are not included. The different 48-run W2 experiment is excluded. |
| Partitions and usage | FAVR intermediate states through 5%, reported budget checkpoints through 10%, cumulative usage, and matched-query control endpoints. |
| Pair judgments | Saved FAVR judgments through the 5% endpoints; technical failures are retained. |
| Scoring, tables, and figures | Offline checks, necessary derived data, result table/figure generators, and the static explanatory assets identified in `paper_artifacts`. |
| Initialization construction | The HC construction implementation is included. The original LLM initialization menus, menu constructor, dedicated prompt/schema/validator, and preparation manifests are not included. The frozen LLM initializations remain available. |

The two incomplete source-record items above limit reconstruction of the original initialization process and exact A3S invocations. They do not prevent scoring the supplied partitions. This snapshot should not be described as complete fulfillment of every item in the paper's future replication-package commitment.

## Experimental identities

The backends are Qwen3.5 and DeepSeek-v4. Model names in metadata, run identifiers, and filenames are normalized to these names. This metadata correction does not change report texts, reference labels, partitions, judgments, usage, measurements, or stored selector seeds. Explicit stored seeds are authoritative; they must not be regenerated from renamed run identifiers. Some internal method identifiers retain `AQCC`, the development name of FAVR.

The evaluation reference is `FULL_REFERENCE_20260928`, covering all 1,567 reports. Reference agreement is an offline evaluation criterion, not a claim of an independent semantic re-annotation. Historical strict-subset and V5 scoring files are excluded from the active evaluation inputs.

FAVR and its internal controls do not receive evaluation references during query selection or partition refinement. The two external pipelines use the target references for offline parameter selection, as stated in the paper. The selected settings and the relevant selection history are documented in `protocols`; they are not represented as untuned runs.

## Use

The offline entry points and their dependencies are documented in `reproduce/README.md`. The data formats and validation commands are documented with the data. Online code is provided for inspection or explicitly configured use; offline verification does not require an API key or model calls. The included validation environment describes this release's checks, not an assertion that every historical online run used that exact environment.

The result-generation scripts write new outputs; the supplied reports, references, and saved results are treated as frozen inputs. The manuscript itself is not included. Static assets are distinguished from generated results so that the existence of a figure file is not mistaken for an automated figure-generation procedure.

## Provenance and packaging

Opaque record identifiers connect the supplied inputs and outcomes. Original experimental measurements and reference assignments are unchanged. Methodological explanations, selected settings, license notices, and material provenance are retained. In particular, the external-pipeline notes describe the historical FAVR-score-dependent selection condition and the independent check that it did not change the selected results within the supplied candidate grids.

`MANIFEST.json` lists the submitted files and their sizes. `python verify_manifest.py` checks the file inventory and sizes; the separate data, scoring, and reconstruction commands verify the scientific content. `VALIDATION.json` records the checks performed on this snapshot and their scope. Component notices accompanying copied code remain applicable.
