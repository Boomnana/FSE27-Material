"""Build four Results tables from the bundled current-reference data snapshot.

Run from any working directory with the bundled offline requirements.
The CSV `before` / `before_mean` columns contain the adopted reference scores;
the CSV `delta` column compares an unchanged scenario and is NOT a stage gain.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import math
import statistics

from common import verify_inputs
from table_uncertainty import endpoint_sample_sd, load_rq2_intervals, within_run_ari_change_sd


PACKAGE = Path(__file__).resolve().parents[1]
PAPER = PACKAGE / "generated"
DATA = PACKAGE / "paper_artifacts" / "data"
OUTPUT = PAPER / "results" / "tables"
REFERENCE_ID = "FULL_REFERENCE_20260928"
DATASETS = ("jayme", "ielts", "hujiang", "huawei")
MODELS = ("qwen3.5-flash", "deepseek-v4-flash")
STARTS = ("LLM", "HC")
CONTROLS = ("AQCC_CONST", "AQCC_TEXT", "AQCC_RANDOM", "FROZEN_LR", "FROZEN_PLAN")
MODEL_NAMES = {"qwen3.5-flash": "Qwen", "deepseek-v4-flash": "DeepSeek"}
KEYS = ("rq", "family", "dataset", "model", "start", "stage", "control", "metric")
QUERY_METRICS = ("FN_Q", "FN_E", "FN_C", "FN_C_over_E", "FP_Q", "FP_E", "FP_C", "FP_C_over_E")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    if value in (None, ""):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite metric: {value}")
    return result


def equal(left, right, context):
    if left is None or right is None:
        if left is not right:
            raise ValueError(f"Missing-value disagreement: {context}: {left}, {right}")
    elif not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Numeric disagreement: {context}: {left}, {right}")


def mean(values):
    return statistics.mean(values) if values else None


def key(row):
    return tuple(row.get(field, "") for field in KEYS)


def fmt(value, places=3, signed=False, commas=False, scale=1):
    if value is None:
        return "--"
    value *= scale
    
    if abs(value) < 0.5 * 10 ** -places:
        value = 0.0
    return format(value, ("+" if signed else "") + ("," if commas else "") + f".{places}f")


def tex_row(cells):
    return " & ".join(cells) + r" \\"


def control_name(value):
    return value.removeprefix("AQCC_").replace("_", r"\_")


class Snapshot:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8-sig"))
        self._validate_reference()
        self.runs = read_csv(self.data_dir / "run_metric_diffs.csv")
        self.configurations = read_csv(self.data_dir / "configuration_diffs.csv")
        self.overall = read_csv(self.data_dir / "rq2_overall_diffs.csv")
        self.coverage = json.loads((self.data_dir / "source_coverage.json").read_text(encoding="utf-8-sig"))
        external_path = self.data_dir / "external_baselines.json"
        self.external = json.loads(external_path.read_text(encoding="utf-8-sig")) if external_path.exists() else None
        self.by_config = {}
        self.by_run = {}
        self.grouped_runs = defaultdict(list)
        for row in self.configurations:
            group = key(row)
            if group in self.by_config:
                raise ValueError(f"Duplicate configuration metric: {group}")
            self.by_config[group] = row
        for row in self.runs:
            group = key(row)
            run_key = group + (row["repeat"],)
            if run_key in self.by_run:
                raise ValueError(f"Duplicate run metric: {run_key}")
            self.by_run[run_key] = row
            self.grouped_runs[group].append(row)
            equal(number(row["before"]), number(row["after"]), ("Nonzero scenario", run_key))
            if number(row["delta"]) not in (None, 0):
                raise ValueError("Expected an unchanged current-reference baseline, not a label scenario.")
        self.validated_configuration_metrics = 0
        self._validate_aggregates()
        self._validate_endpoint_gains()
        self.paired_coverage = self._validate_paired_controls()
        self._validate_recomputed_results()

    def _validate_reference(self):
        manifest = verify_inputs(self.data_dir)
        if manifest["reference_id"] != REFERENCE_ID:
            raise ValueError("Snapshot manifest must identify the current reference.")

    def config(self, rq, family, dataset, model, start, stage, metric, control=""):
        return self.by_config[(rq, family, dataset, model, start, stage, control, metric)]

    def value(self, rq, family, dataset, model, start, stage, metric, control=""):
        return number(self.config(rq, family, dataset, model, start, stage, metric, control)["before_mean"])

    def run_value(self, rq, family, dataset, model, start, stage, metric, repeat, control=""):
        return number(self.by_run[(rq, family, dataset, model, start, stage, control, metric, str(repeat))]["before"])

    def _validate_aggregates(self):
        for group, config in self.by_config.items():
            members = self.grouped_runs[group]
            values = [number(row["before"]) for row in members]
            valid = [value for value in values if value is not None]
            missing = sum(row.get("before_status", row["status"]).startswith("missing") for row in members)
            if int(config["runs"]) != len(members) or int(config["valid_before"]) != len(valid):
                raise ValueError(f"Repeat coverage disagreement: {group}")
            if int(config.get("missing_before_runs", config["missing_runs"])) != missing:
                raise ValueError(f"Missing-source disagreement: {group}")
            expected = None if missing else mean(valid)
            equal(number(config["before_mean"]), expected, ("configuration mean", group))
            equal(number(config["before_mean"]), number(config["after_mean"]), ("configuration scenario", group))
            self.validated_configuration_metrics += 1

    def _validate_endpoint_gains(self):
        self.validated_interval_values = 0
        for dataset in DATASETS:
            for model in MODELS:
                for start in STARTS:
                    for interval in ("0->0.05", "0.05->0.1"):
                        first, last = interval.split("->")
                        for output_metric, endpoint_metric in (("delta_ari", "ari"), ("additional_queries", "actual_q")):
                            differences = []
                            for repeat in (1, 2, 3):
                                left = self.run_value("RQ3", "score", dataset, model, start, first, endpoint_metric, repeat)
                                right = self.run_value("RQ3", "score", dataset, model, start, last, endpoint_metric, repeat)
                                if left is None or right is None:
                                    raise ValueError("RQ3 table requires all three current-reference endpoint values.")
                                difference = right - left
                                equal(self.run_value("RQ3", "budget_gain", dataset, model, start, interval, output_metric, repeat), difference, ("endpoint gain", dataset, model, start, interval, repeat))
                                differences.append(difference)
                                self.validated_interval_values += 1
                            equal(self.value("RQ3", "budget_gain", dataset, model, start, interval, output_metric), mean(differences), "mean interval gain")
                    for metric in ("ari", "b3_f1"):
                        for repeat in (1, 2, 3):
                            equal(self.run_value("RQ1", "score", dataset, model, start, "P_B", metric, repeat), self.run_value("RQ3", "score", dataset, model, start, "0.05", metric, repeat), ("RQ1/RQ3 endpoint", dataset, model, start, repeat, metric))

    def _validate_paired_controls(self):
        coverage = {}
        for control in CONTROLS:
            available = complete = 0
            for dataset in DATASETS:
                for model in MODELS:
                    for start in STARTS:
                        valid_repeats = 0
                        for repeat in (1, 2, 3):
                            quality_available = []
                            for metric in ("ari", "b3_f1"):
                                aqcc = self.run_value("RQ2", "score", dataset, model, start, "P_B", metric, repeat)
                                baseline = self.run_value("RQ2", "control_score", dataset, model, start, "P_B", metric, repeat, control)
                                difference = self.run_value("RQ2", "paired_gain", dataset, model, start, "P_B", "delta_" + metric, repeat, control)
                                equal(difference, aqcc - baseline if aqcc is not None and baseline is not None else None, ("paired control", control, dataset, model, start, repeat, metric))
                                quality_available.append(difference is not None)
                            if quality_available[0] != quality_available[1]:
                                raise ValueError("ARI and B3 paired coverage differ.")
                            aqcc_q = self.run_value("RQ2", "score", dataset, model, start, "P_B", "actual_q", repeat)
                            control_q = self.run_value("RQ2", "control_score", dataset, model, start, "P_B", "actual_q", repeat, control)
                            equal(aqcc_q, control_q, ("matched actual query count", control, dataset, model, start, repeat))
                            valid_repeats += int(all(quality_available))
                        available += valid_repeats
                        complete += int(valid_repeats == 3)
            recorded = self.coverage["controls"]["by_group"][control]
            if available != recorded["available"]:
                raise ValueError(f"Control coverage does not match source audit: {control}")
            coverage[control] = {"available_pairs": available, "required_pairs": 48,
                                 "complete_configurations": complete, "required_configurations": 16}
        return coverage

    def _validate_recomputed_results(self):
        self.overall_values = {(r['control'], r['metric']): number(r['before']) for r in self.overall}
        self.opposite_metric_configurations = {}
        for control in CONTROLS:
            groups = {}
            for metric in ('delta_ari', 'delta_b3_f1'):
                values = [self.value('RQ2', 'paired_gain', ds, model, start, 'P_B', metric, control)
                          for ds in DATASETS for model in MODELS for start in STARTS]
                groups[metric] = values
                complete = all(v is not None for v in values)
                equal(self.overall_values[control, metric + '_equal_configuration_mean'],
                      mean(values) if complete else None, ('RQ2 equal-configuration mean', control, metric))
            available = all(v is not None for values in groups.values() for v in values)
            expected = sum(a > 0 and b < 0 for a, b in zip(groups['delta_ari'], groups['delta_b3_f1'])) if available else None
            equal(self.overall_values[control, 'ari_up_b3_down_configurations'], expected, ('RQ2 tradeoff count', control))
            self.opposite_metric_configurations[control] = (
                sum(a * b < 0 for a, b in zip(groups['delta_ari'], groups['delta_b3_f1']))
                if available else None
            )
        for ds in DATASETS:
            for model in MODELS:
                for start in STARTS:
                    for repeat in (1, 2, 3):
                        for kind in ('FN', 'FP'):
                            q, e, c = [self.run_value('RQ4', 'process', ds, model, start, 'Pprobe->P_B', kind + '_' + m, repeat)
                                       for m in ('Q', 'E', 'C')]
                            ratio = self.run_value('RQ4', 'process', ds, model, start, 'Pprobe->P_B', kind + '_C_over_E', repeat)
                            if all(v is not None for v in (q, e, c)):
                                if not 0 <= c <= e <= q:
                                    raise ValueError(f'Invalid RQ4 counts: {ds}/{model}/{start}/{repeat}/{kind}')
                                equal(ratio, c / e if e else None, ('within-run C/E', ds, model, start, repeat, kind))
        if self.external:
            if self.external['reference_id'] != REFERENCE_ID:
                raise ValueError('External baselines use a different reference.')
            if self.external.get('schema') == 'aqcc-external-baselines/v2':
                expected_counts = {'A3S': 16, 'LLMCluster-tuned': 4, 'TERFUR-desc-semantic-seeded-v2': 4}
                actual_counts = dict(Counter(row['method'] for row in self.external['records']))
                if actual_counts != expected_counts or self.coverage['external_baselines'] != expected_counts:
                    raise ValueError('External baseline selection or coverage metadata is stale.')
                for row in self.external['records']:
                    if row['method'] != 'A3S' and (row.get('model') or row.get('start') or not row.get('tuning')):
                        raise ValueError('A selected tuned partition must retain its evaluation unit and tuning provenance.')
            seen = set()
            for row in self.external['records']:
                identity = (row['method'], row['dataset'], row.get('model', ''), row.get('start', ''), row['repeat'])
                if identity in seen:
                    raise ValueError(f'Duplicate external baseline: {identity}')
                seen.add(identity)
                if not row.get('source') or not row.get('archived_partition_id'):
                    raise ValueError(f'External baseline lacks partition provenance: {identity}')
                for metric in ('ari', 'b3_f1'):
                    if number(row[metric]) is None:
                        raise ValueError(f'Unscored external baseline: {identity}/{metric}')
                    if row['method'] == 'A3S':
                        equal(number(row[metric]), self.value('RQ1', 'a3s_score', row['dataset'],
                              row['model'], row['start'], 'P_B', metric), ('A3S external/CSV', identity, metric))

    def external_value(self, method, dataset, model, metric):
        if not self.external:
            return None
        rows = [r for r in self.external['records'] if r['method'] == method
                and r['dataset'] == dataset and r.get('model', '') == model]
        expected = self.external.get('evaluation_protocols', {}).get(method, {}).get('repeats', 3 if method == 'LLMCluster' else 1)
        if not rows:
            raise ValueError(f'Missing external baseline: {method}/{dataset}/{model}')
        if len(rows) != expected:
            raise ValueError(f'Incomplete external baseline repetitions: {method}/{dataset}/{model}')
        if {int(row['repeat']) for row in rows} != set(range(1, expected + 1)):
            raise ValueError(f'Unexpected external repeat identities: {method}/{dataset}/{model}')
        return mean([number(r[metric]) for r in rows])


def head(text):
    return r"\multicolumn{1}{c}{" + text + "}"


def table_start(caption, label, size="8.5", leading="9.6", tabsep="2.2", stretch="1.0", placement="!htbp", width=r"\linewidth"):
    return [r"\begin{table}[" + placement + "]", r"\centering",
            r"\caption{" + caption + "}", r"\label{" + label + "}",
            r"\begin{minipage}{" + width + "}", r"\centering",
            r"\begingroup", rf"\fontsize{{{size}}}{{{leading}}}\selectfont",
            rf"\setlength{{\tabcolsep}}{{{tabsep}pt}}", rf"\renewcommand{{\arraystretch}}{{{stretch}}}",
            r"\setlength{\heavyrulewidth}{0.7pt}", r"\setlength{\lightrulewidth}{0.35pt}",
            r"\setlength{\aboverulesep}{1.4pt}", r"\setlength{\belowrulesep}{1.8pt}"]


def table_end(note):
    return [r"\endgroup", r"\par\vspace{2pt}",
            r"{\fontsize{8}{9.2}\selectfont\raggedright " + note + r"\par}",
            r"\end{minipage}", r"\end{table}"]


def rq1(snapshot):
    lines = table_start(r"Final partition quality.", "tab:rq1-baseline", tabsep="2", leading="9.2", stretch="0.96")
    pair = "cc"
    columns = r"@{\extracolsep{\fill}}lll" + r"@{\hspace{7pt}}".join([pair] * 4) + "@{}"
    lines += [r"\begin{tabular*}{\linewidth}{" + columns + "}", r"\toprule",
              tex_row(["", "", ""] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{{d.upper()}}}}}" for d in DATASETS]),
              r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}",
              tex_row([r"\textbf{Backend}", r"\textbf{Start}", r"\textbf{Method}"] +
                      [head("ARI"), head(r"$B^3$-F1")] * 4), r"\midrule"]
    for mi, model in enumerate(MODELS):
        if mi:
            lines.append(r"\midrule")
        for si, start in enumerate(STARTS):
            if si:
                lines.append(r"\addlinespace[1.5pt]")
            for fi, (family, method) in enumerate((("a3s_score", "A3S"), ("score", r"\textbf{FAVR}"))):
                
                
                stub = [rf"\multirow[t]{{6}}{{*}}{{{MODEL_NAMES[model]}}}" if si == fi == 0 else "",
                        rf"\multirow[t]{{3}}{{*}}{{{start}}}" if fi == 0 else "", method]
                values = []
                other_family = "score" if family == "a3s_score" else "a3s_score"
                for dataset in DATASETS:
                    for metric in ("ari", "b3_f1"):
                        value = snapshot.value("RQ1", family, dataset, model, start, "P_B", metric)
                        other = snapshot.value("RQ1", other_family, dataset, model, start, "P_B", metric)
                        text = fmt(value)
                        if value is not None and other is not None and value > other:
                            text = r"\textbf{" + text + "}"
                        values.append(text)
                if family == "score":
                    
                    lines.append(r"  &    &  \textbf{FAVR}  & " + tex_row(values))
                    deviations = [fmt(endpoint_sample_sd(snapshot, dataset, model, start, metric))
                                  for dataset in DATASETS for metric in ("ari", "b3_f1")]
                    lines.append(r" & & \quad\textit{SD} & " + tex_row(deviations))
                else:
                    lines.append(tex_row(stub + values))
    lines += [r"\midrule", r"\multicolumn{11}{@{}l}{\textit{Reference-tuned pipelines (in-sample)}} \\", r"\addlinespace[1pt]"]
    for method, name in (("LLMCluster-tuned", "LLMCluster"), ("TERFUR-desc-semantic-seeded-v2", "TERFUR")):
        values = [fmt(snapshot.external_value(method, dataset, "", metric))
                  for dataset in DATASETS for metric in ("ari", "b3_f1")]
        lines.append(tex_row([rf"\multicolumn{{3}}{{@{{}}l}}{{{name}}}"] + values))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += table_end(
        r"FAVR: three-run means at $5\%$, with sample SD below. "
        r"A3S: one run at the same budget ceiling; actual queries and resource use are unmatched. "
        r"Reference-tuned pipelines: one partition per dataset. "
        r"SDs below 0.0005 round to 0.000. Bold: higher A3S--FAVR score.")
    return "\n".join(lines) + "\n"


def rq2(snapshot):
    intervals = load_rq2_intervals(snapshot)
    lines = table_start(r"Paired quality gains over selection controls.", "tab:rq2-control",
                        tabsep="2")
    lines += [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrc@{\hspace{9pt}}rcc@{}}", r"\toprule",
              r" & \multicolumn{2}{c}{$\Delta$ARI} & \multicolumn{2}{c}{$\Delta B^3$-F1} & \\",
              r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
              tex_row([r"\textbf{Control}", "Mean", r"95\% CI", "Mean", r"95\% CI", r"\textbf{Opp.}"]), r"\midrule"]
    for control in CONTROLS:
        cover = snapshot.paired_coverage[control]
        if cover['available_pairs'] != 48 or cover['complete_configurations'] != 16:
            raise ValueError("The shared coverage note requires 48/48 pairs and 16/16 configurations for every control.")
        if control == "FROZEN_LR":
            lines.append(r"\addlinespace[2pt]")
        values = []
        for metric in ('delta_ari', 'delta_b3_f1'):
            value = snapshot.overall_values[control, metric + '_equal_configuration_mean']
            text = fmt(value, 4, signed=True)
            if value is not None and value > 0:
                text = r"\textbf{" + text + "}"
            values.append(text)
            interval = intervals[control, metric]
            values.append("[" + fmt(interval["ci_low"], 4, signed=True) + ", " +
                          fmt(interval["ci_high"], 4, signed=True) + "]")
        values.append(fmt(snapshot.opposite_metric_configurations[control], 0))
        lines.append(tex_row([control_name(control)] + values))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += table_end(
        r"FAVR minus control at matched actual $q$ ($5\%$): 48 matched run pairs per control, 16 equally weighted configurations. "
        r"Pointwise 95\% CIs are conditional on the fixed datasets, backends, and starts; "
        r"bootstrap details are in Section~\ref{sec:eval-metrics}. "
        r"Opp.: configurations with opposite-sign metric gains. Bold: positive means.")
    return "\n".join(lines) + "\n"


def rq3(snapshot):
    lines = table_start(r"Quality changes and logical-query counts across budget intervals.", "tab:rq3-budget",
                        tabsep="1.6", leading="9.2", stretch="0.96")
    columns = r"@{\extracolsep{\fill}}lll c S[table-format=5.1]@{\hspace{9pt}}c S[table-format=5.1]@{}"
    lines += [r"\begin{tabular*}{\linewidth}{" + columns + "}", r"\toprule",
              r" & & & \multicolumn{2}{c}{\textbf{Qwen}} & \multicolumn{2}{c}{\textbf{DeepSeek}} \\",
              r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
              tex_row([r"\textbf{Dataset}", r"\textbf{Start}", r"\textbf{Budget}"] +
                      [r"$\Delta$ARI (mean $\pm$ SD)", r"$\Delta q$"] * 2), r"\midrule"]
    for index, dataset in enumerate(DATASETS):
        if index:
            lines.append(r"\addlinespace[2.5pt]")
        for si, start in enumerate(STARTS):
            for ii, interval in enumerate(("0->0.05", "0.05->0.1")):
                values = [rf"\multirow[t]{{4}}{{*}}{{{dataset.upper()}}}" if si == ii == 0 else "",
                          rf"\multirow[t]{{2}}{{*}}{{{start}}}" if ii == 0 else "",
                          r"0--5\%" if ii == 0 else r"5--10\%"]
                for model in MODELS:
                    value = snapshot.value("RQ3", "budget_gain", dataset, model, start, interval, "delta_ari")
                    text = fmt(value, 4, signed=True)
                    if value is not None:
                        if interval == "0->0.05" and value > 0:
                            text = r"\textbf{" + text + "}"
                        elif interval == "0.05->0.1" and value < 0:
                            text = r"\underline{" + text + "}"
                    sd = within_run_ari_change_sd(snapshot, dataset, model, start, interval)
                    values.append(text + r"\,$\pm$\," + fmt(sd, 4))
                    queries = snapshot.value("RQ3", "budget_gain", dataset, model, start, interval, "additional_queries")
                    values.append(fmt(queries, 1, commas=True))
                lines.append(tex_row(values))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += table_end(
        r"$\Delta$ARI: mean $\pm$ sample SD of three within-run changes; $\Delta q$: mean additional queries. "
        r"The $0$--$5\%$ interval includes probing. "
        r"Bold: positive $0$--$5\%$ means; underlining: negative $5$--$10\%$ means.")
    return "\n".join(lines) + "\n"


def rq4(snapshot):
    lines = table_start(r"Query-time errors and endpoint agreement from Probe to $5\%$.", "tab:rq4-evidence", tabsep="2")
    block = r"*{3}{S[table-format=3.1]}@{\hspace{5pt}}S[table-format=3.1]"
    columns = r"@{\extracolsep{\fill}}lll" + block + r"@{\hspace{9pt}}" + block + "@{}"
    lines += [r"\begin{tabular*}{\linewidth}{" + columns + "}", r"\toprule",
              r" & & & \multicolumn{4}{c}{\textbf{FN at query time}} & \multicolumn{4}{c}{\textbf{FP at query time}} \\",
              r"\cmidrule(lr){4-7}\cmidrule(l){8-11}",
              tex_row([r"\textbf{Dataset}", r"\textbf{Backend}", r"\textbf{Start}"] +
                      [head(r"$Q$"), head(r"$E$"), head(r"$C$"), head(r"$C/E$ (\%)")] * 2), r"\midrule"]
    for index, dataset in enumerate(DATASETS):
        if index:
            lines.append(r"\midrule")
        for mi, model in enumerate(MODELS):
            if mi:
                lines.append(r"\addlinespace[1.5pt]")
            for si, start in enumerate(STARTS):
                values = [dataset.upper() if mi == si == 0 else "", MODEL_NAMES[model] if si == 0 else "", start]
                for metric in QUERY_METRICS:
                    record = snapshot.config("RQ4", "process", dataset, model, start, "Pprobe->P_B", metric)
                    value = number(record['before_mean'])
                    text = fmt(value, 1, scale=100 if metric.endswith("_over_E") else 1)
                    if metric.endswith('_over_E') and value is not None:
                        text = r"\bfseries " + text
                    if metric.endswith('_over_E') and not int(record['missing_before_runs']):
                        n = int(record['valid_before'])
                        if not n:
                            text = head('NA')
                        elif n < 3:
                            text = head(rf'{{{text}}}$^{{(n={n})}}$')
                    if value is None:
                        text = head('--')
                    values.append(text)
                lines.append(tex_row(values))
    lines += [r"\bottomrule", r"\end{tabular*}"]
    lines += table_end(
        r"$Q$: query-time erroneous pairs; $E$: their valid reference-consistent judgments; "
        r"$C$: pairs in $E$ correct at $5\%$. Counts are three-run means; "
        r"$C/E$ averages within-run ratios with $E>0$. Bold highlights endpoint agreement ($C/E$).")
    return "\n".join(lines) + "\n"
def build(data_dir=DATA, output_dir=OUTPUT, check=False):
    snapshot = Snapshot(data_dir)
    output_dir = Path(output_dir)
    header = ""
    rendered = {"rq1.tex": rq1(snapshot), "rq2.tex": rq2(snapshot), "rq3.tex": rq3(snapshot), "rq4.tex": rq4(snapshot)}
    if not check:
        output_dir.mkdir(parents=True, exist_ok=True)
    for filename, body in rendered.items():
        path = output_dir / filename
        content = header + body
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Generated table is stale: {path}")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    report = {
        "status": "PASS", "reference_id": REFERENCE_ID,
        "configuration_metrics_checked_against_runs": snapshot.validated_configuration_metrics,
        "rq3_individual_interval_values_checked_against_endpoints": snapshot.validated_interval_values,
        "rq1_rq3_five_percent_endpoints": "consistent",
        "rq2_actual_query_counts": "matched for all five controls and all 48 paired runs",
        "rq2_coverage": snapshot.paired_coverage,
        "rq2_opposite_metric_configurations": snapshot.opposite_metric_configurations,
        "external_baseline_records": len(snapshot.external['records']) if snapshot.external else 0,
        "rq4_query_metric_means_available": sum(number(r['before_mean']) is not None for r in snapshot.configurations
                                                if r['rq'] == 'RQ4' and r['metric'] in QUERY_METRICS),
        "missing_source_run_metrics": sum(r.get('before_status', '').startswith('missing') for r in snapshot.runs),
        "value_columns": "before / before_mean; named delta_ari is a stage or paired gain; CSV delta is never used as a quality gain",
        "generated_tables": list(rendered),
    }
    if not check:
        (output_dir / "table_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="Validate data and require generated tables to be up to date.")
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.output_dir, args.check), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
