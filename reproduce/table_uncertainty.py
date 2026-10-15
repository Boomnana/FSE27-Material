"""Sample SDs and recomputed conditional intervals for result tables.

SDs are recomputed from the three run-level values. Saved RQ2 intervals are
accepted only after repeating their specified bootstrap on the bundled data.
"""
from pathlib import Path
import json
import math
import statistics

from verify_ci import verify as verify_rq2_intervals


STATISTICS = Path(__file__).resolve().parents[1] / "paper_artifacts" / "statistics"
RQ2_INTERVALS = STATISTICS / "rq2_conditional_intervals_20261003.json"
REPEATS = (1, 2, 3)
CONTROLS = ("AQCC_CONST", "AQCC_TEXT", "AQCC_RANDOM", "FROZEN_LR", "FROZEN_PLAN")
METRICS = ("delta_ari", "delta_b3_f1")


def require_equal(actual, expected, context):
    if expected is None or not math.isclose(actual, float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Uncertainty/source mismatch for {context}: {actual} != {expected}")


def checked_sample_sd(snapshot, rq, family, dataset, model, start, stage, metric, values):
    if len(values) != 3 or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError(f"A sample SD requires all three runs: {rq}/{dataset}/{model}/{start}/{metric}")
    record = snapshot.config(rq, family, dataset, model, start, stage, metric)
    if int(record["runs"]) != 3 or int(record["valid_before"]) != 3 or int(record["missing_before_runs"]):
        raise ValueError(f"Incomplete SD source coverage: {rq}/{dataset}/{model}/{start}/{metric}")
    require_equal(statistics.mean(values), record["before_mean"], "mean before SD")
    sd = statistics.stdev(values)
    require_equal(sd, record["before_sd"], "sample SD (ddof=1)")
    return sd


def endpoint_sample_sd(snapshot, dataset, model, start, metric):
    values = [snapshot.run_value("RQ1", "score", dataset, model, start, "P_B", metric, repeat)
              for repeat in REPEATS]
    return checked_sample_sd(snapshot, "RQ1", "score", dataset, model, start, "P_B", metric, values)


def within_run_ari_change_sd(snapshot, dataset, model, start, interval):
    first, last = interval.split("->")
    values = []
    for repeat in REPEATS:
        before = snapshot.run_value("RQ3", "score", dataset, model, start, first, "ari", repeat)
        after = snapshot.run_value("RQ3", "score", dataset, model, start, last, "ari", repeat)
        if before is None or after is None:
            raise ValueError(f"Missing within-run endpoint: {dataset}/{model}/{start}/{interval}/R{repeat}")
        change = after - before
        saved = snapshot.run_value("RQ3", "budget_gain", dataset, model, start, interval, "delta_ari", repeat)
        require_equal(change, saved, "within-run ARI difference")
        values.append(change)
    return checked_sample_sd(snapshot, "RQ3", "budget_gain", dataset, model, start, interval, "delta_ari", values)


def load_rq2_intervals(snapshot, path=RQ2_INTERVALS):
    """Reject stale or incomplete CIs rather than silently carrying them forward."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "favr-table-rq2-conditional-ci/v1":
        raise ValueError(f"Unsupported conditional interval schema: {path}")
    if payload["reference_id"] != snapshot.manifest["reference_id"]:
        raise ValueError("The saved conditional intervals use a different reference.")
    expected_files = {"run_metric_diffs.csv", "configuration_diffs.csv", "rq2_overall_diffs.csv", "CURRENT_REFERENCE.json"}
    if set(payload["source_files"]) != expected_files:
        raise ValueError("Conditional interval provenance does not cover the required source files.")
    verify_rq2_intervals(payload, snapshot.data_dir)
    method = payload["method"]
    expected_method = {"interval": "percentile", "confidence_level": 0.95,
                       "coverage": "pointwise marginal, not simultaneous", "strata": 8,
                       "blocks_per_stratum": 3, "draws": 200000, "seed": 20261003,
                       "generator": "PCG64", "quantile_method": "linear",
                       "datasets": ["jayme", "ielts", "hujiang", "huawei"],
                       "backends": ["qwen3.5-flash", "deepseek-v4-flash"],
                       "starts": ["LLM", "HC"], "repetitions": [1, 2, 3]}
    for field, expected in expected_method.items():
        if method.get(field) != expected:
            raise ValueError(f"Saved interval method disagrees with the approved table note: {field}")
    result = {}
    for record in payload["records"]:
        key = (record["control"], record["metric"])
        if key in result:
            raise ValueError(f"Duplicate conditional interval: {key}")
        values = [record[k] for k in ("mean", "ci_low", "ci_high")]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"Nonfinite conditional interval: {key}")
        if record["ci_low"] > record["ci_high"]:
            raise ValueError(f"Reversed conditional interval: {key}")
        require_equal(record["mean"], snapshot.overall_values[key[0], key[1] + "_equal_configuration_mean"], key)
        result[key] = record
    if set(result) != {(control, metric) for control in CONTROLS for metric in METRICS}:
        raise ValueError("All ten planned control/metric intervals must be present.")
    return result
