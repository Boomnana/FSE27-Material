The default review workflow is offline. Run commands from the repository root.

Install the recorded verification environment:
  python -m pip install -r code/requirements.txt

Rebuild the eight selected external-pipeline partitions and compare their memberships and metrics:
  python -B code/baselines/offline_selected.py --output review_selected

The output directory must be new. The check compares complete report membership against the supplied endpoints and independently recomputes their metrics.

Inspect one registered FAVR/control run without model calls:
  python -B code/favr/run_registered.py --job huawei__qwen3.5-flash__R1__I_llm__AQCC_LR

All 288 job identifiers and explicit seeds are in configs/favr_runs.json. Historical HC job identifiers use I_tfidf; distributed initialization directories use I_hc. Frozen starts and vectors are loaded directly. Reference labels are not loaded by this entry point.

A new online FAVR run requires --online, a new --output directory, a shared --cache directory for the matched block, and an environment-provided SILRA_API_KEY. No key is included. Both registered backend names are supported; provider availability for a new live run is not tested by the offline workflow. New calls are not needed to inspect the supplied paper results.

The two FAVR runtime folders retain the archived implementation families. Their folder names do not identify every job's model; the explicit model field does. Backend metadata is standardized to the author-confirmed Qwen3.5 and DeepSeek-v4 identities. Recorded seeds and experimental observations are unchanged. Source and adaptation details are in protocols/code_provenance.json and protocols/PROTOCOL_NOTES.txt. The HC launcher remains in each runtime folder and requires the frozen corpus plus an explicit provenance-only source-workbook file whose cells are not read.

baseline_lib.py contains the selected text/consensus/fuzzy-generation functions and the reference-based candidate evaluator. offline_selected.py regenerates selected settings; it does not repeat the full historical tuning grid. The target-reference selection rule and historical-condition equivalence check are disclosed in protocols/PROTOCOL_NOTES.txt and protocols/baseline_selection_verification.json.

code/llmcluster contains the collected upstream ClusterManager class, prompt and local boundary adapter with the upstream GPL-3.0 license. The recorded protocol is in protocols/llmcluster_native_protocol.json. The offline pipeline uses the 24 recorded upstream partitions and does not call that service again.

code/a3s contains the available local A3S adapter. Exact run-specific configurations for the paper's 16 saved A3S endpoints were not recovered; the distinct 48-run replacement batch is not included. No adapter defaults are presented as those historical settings. The original LLM initialization-menu generator and its prompt/schema are also unavailable; the frozen initialization states are included.
