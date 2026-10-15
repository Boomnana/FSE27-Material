# Data and recorded runs

All text files use UTF-8. This package contains the data and recorded results used by the paper, with complete report coverage and the current evaluation reference. Reference labels are offline evaluation inputs; they are not inputs to the FAVR online algorithm.

## Inputs

* `reports/<dataset>.json`: original report array. `id` is a string; `text` preserves the report content exactly; `sourceExcelrow` is the original row locator.
* `report_order.json`: the frozen report-index order for each dataset. Both initializations use this order.
* `reference/<dataset>.json`: `issue_by_source_id` maps every report ID to its evaluation group. Group identifiers are nominal labels.
* `reference/manifest.json`: current reference identity and collection sizes. The reference was supplied and confirmed by the authors; an independent semantic reannotation record was not supplied.
* `initializations/<dataset>/I_hc/` and `I_llm/`: frozen `start_state.json` and its `tfidf_vectors.npz`. Report content, order, vectors and initial partitions are preserved. Release metadata uses portable source locators and opaque record IDs. All eight vector files retain their original bytes.

The LLM frozen states are sufficient to start the recorded refinement task. The original full-menu initializer, menu instances, dedicated grouping prompt/schema/validator and four preparation manifests were not recovered; the frozen states do not stand in for those unavailable materials.

## Partitions

`partitions.jsonl.gz` contains one object per unique partition:

```
{"partition_id":"record_000001","dataset":"jayme","groups":[["report_id", "report_id"], ["report_id"]]}
```

`partition_id` is an opaque lookup key within this release. Original cluster order and member order are preserved. Every partition covers every report in its dataset exactly once. Identical stored states are represented once without changing any run's transitions.

## Main runs

`../runs/main/index.json` enumerates all 48 runs (4 datasets × 2 backends × 2 initializations × 3 repeats). A run ID retains its historical algorithm token `AQCC_LR`; the paper name is FAVR. Model metadata is normalized to the author-confirmed Qwen3.5 and DeepSeek-v4 backends. These metadata corrections do not change observations, seeds or results.

`../runs/main/checkpoints.jsonl` contains 672 rows: 13 budget checkpoints from 0 through 10% and one post-probing state for each run. Fields include `run_id`, `dataset`, `model`, `start`, `repeat`, `stage`, `budget_fraction`, `actual_q`, `partition_id`, `usage`, and `expected_metrics`. The post-probing stage is `Pprobe` with a null nominal budget. `expected_metrics` use the current full reference, not historical scores.

`../runs/main/micro_rounds/<run_id>.jsonl.gz` contains the initial state, every committed micro-round state through the exact 5% endpoint, and the reported budget checkpoints after 5% through 10%. Intermediate states after 5% are omitted because the paper's pair-level process analysis ends at 5%. `actual_q` is cumulative logical pair queries. Every row references its partition and includes cumulative recorded usage. The file also preserves stage, probing status, batch category counts, evidence counts and available decoder/estimator status. An initial zero-token record is defined explicitly at `actual_q=0`.

`usage` is the source's cumulative per-logical-exposure token record, including reuse of a frozen answer. It must not be interpreted as an incremental API bill. Where supplied, judgment records separately retain `physical_usage` and `response_reuse` to expose this distinction.

## Judgments

`../runs/judgments/<run_id>.jsonl.gz` stores all logical judgments through the exact 5% endpoint, in contiguous `query_index` order. Fields are `left`, `right` (report IDs), `status`, `label`, `metadata`, `usage`, `physical_usage`, and available `response_reuse` flags. Selection metadata includes `p_same_before`, stage, category and micro-round identifiers. Successful labels are `SAME` or `DIFFERENT`; failed outcomes remain present. Raw request bodies, model response bodies and free-text explanations are omitted. These omissions do not remove queried pairs, failed observations, labels, costs or reported evidence transitions.

The shared-response fields record the existing response-reuse protocol; they are not a claim that every logical query made a new API request. Pair logs beyond 5% are not included. Partition checkpoints and cumulative usage extend through 10%.

## Comparison runs

`../runs/controls/endpoints.jsonl` contains 240 endpoints: 48 for each of CONST, TEXT, RANDOM, FROZEN_LR and FROZEN_PLAN. Every `matched_main_run` has exactly the same `actual_q` at its 5% endpoint. Historical JAYME control records lack corresponding cumulative usage fields, which remain null.

`../runs/baselines/endpoints.jsonl` contains the paper's 16 single A3S endpoints and 8 selected pipeline endpoints (4 LLMCluster and 4 TERFUR). A3S query counts and stopping status are retained. Exact historical runtime configurations for these 16 A3S runs were not recovered; a different 48-run supplement is excluded. The two pipeline entries retain selected parameters and explicitly record use of target-dataset reference labels for candidate selection. A selected pipeline partition is one observation, not a three-repeat average.

All endpoint records reference `partition_id` and carry `expected_metrics`. `archived_partition_id`, where present, identifies the corresponding source partition record; different serialization conventions can give it a distinct ID from the stored partition lookup key.

## Source and transformation checks

`manifest.json` lists the included files and their sizes, source roles, normalized archive locators, format transformations and unresolved provenance boundaries. Opaque source IDs link records to those source descriptions. The source archives themselves are not redistributed because they contain unrelated runs and raw API payloads.

After installing `reproduce/requirements.txt`, run `python data/verify_data.py` from the package root for file presence and size, report/reference coverage, vector structure, frozen-state consistency, partition membership, source links, query matching, token and judgment-status checks. The independent scoring and paper-output checks are supplied separately in `reproduce/`.
