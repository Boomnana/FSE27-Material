"""Plot current-reference stage scores, retaining unavailable stage gaps.

Run from any directory with ``python path/to/plot_rq1.py``.  Plot inputs are
validated copies of the current-reference scores; CSV ``delta`` is never
used as a stage score.  Every displayed mean requires all three runs.
"""
import math
from collections import defaultdict
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from common import (
    COLORS, DATASETS, MODEL_NAMES, MODELS, STARTS,
    read_csv, save, style, verify_inputs, write_rows,
)

STAGES = ("P0", "Pprobe", "P_B")
STAGE_NAMES = ("Init.", "Probe", "5%")
METRICS = ("ari", "pair_fp", "pair_fn")
METRIC_NAMES = {"ari": "ARI", "pair_fp": "FP pairs", "pair_fn": "FN pairs"}


def load_points():
    verify_inputs()
    runs = defaultdict(list)
    for row in read_csv("run_metric_diffs.csv"):
        if row["rq"] == "RQ1" and row["family"] == "score" and row["metric"] in METRICS:
            key = tuple(row[k] for k in ("dataset", "model", "start", "stage", "metric"))
            runs[key].append(row)
    points = {}
    for row in read_csv("configuration_diffs.csv"):
        if row["rq"] != "RQ1" or row["family"] != "score" or row["metric"] not in METRICS:
            continue
        key = tuple(row[k] for k in ("dataset", "model", "start", "stage", "metric"))
        assert key not in points, ("Duplicate configuration", key)
        source = runs[key]
        assert len(source) == 3 and {r["repeat"] for r in source} == {"1", "2", "3"}, key
        available = [float(r["before"]) for r in source if r["before"] != ""]
        assert int(row["runs"]) == 3 and int(row["valid_before"]) == len(available), key
        if len(available) == 3:
            value = float(row["before_mean"])
            assert math.isclose(value, mean(available), rel_tol=1e-12, abs_tol=1e-9), key
        else:
            assert row["before_mean"] == "", ("Partial mean must remain unavailable", key)
            value = math.nan
        points[key] = {
            "dataset": key[0], "model": key[1], "start": key[2],
            "stage": key[3], "metric": key[4],
            "current_reference_mean": "" if math.isnan(value) else value,
            "current_reference_sd": row["before_sd"],
            "available_runs": len(available), "required_runs": 3,
            "plotted": not math.isnan(value), "status": row["status"],
        }
    expected = {(d, m, s, t, metric) for d in DATASETS for m in MODELS
                for s in STARTS for t in STAGES for metric in METRICS}
    assert set(points) == expected, "RQ1 configurations do not cover the expected study design"
    return points


def main():
    points = load_points()
    style()
    
    fig, axes = plt.subplots(3, 4, figsize=(5.5, 3.05), sharex=True)
    fig.subplots_adjust(left=0.088, right=0.994, top=0.845, bottom=0.10,
                        hspace=0.26, wspace=0.42)
    missing_stages = []
    for col, dataset in enumerate(DATASETS):
        axes[0, col].set_title(dataset.upper(), pad=5, fontweight="bold", fontsize=8.5)
        for row_index, metric in enumerate(METRICS):
            ax = axes[row_index, col]
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#DCDCDC", linewidth=0.5)
            ax.axvline(2, color="#A8A8A8", linestyle=(0, (3, 3)), linewidth=0.75, zorder=0)
            if col == 0:
                ax.set_ylabel(METRIC_NAMES[metric], labelpad=3, fontsize=8)
            values = []
            for model in MODELS:
                for start in STARTS:
                    selected = [points[(dataset, model, start, stage, metric)] for stage in STAGES]
                    y = [float(p["current_reference_mean"]) if p["plotted"] else math.nan for p in selected]
                    values.extend(v for v in y if not math.isnan(v))
                    is_llm = start == "LLM"
                    ax.plot(range(3), y, color=COLORS[model],
                            marker="o" if is_llm else "s", markersize=3.3,
                            markerfacecolor=COLORS[model] if is_llm else "white",
                            markeredgewidth=0.75,
                            linestyle="-" if is_llm else (0, (3, 2)), zorder=3)
            for stage_index, stage in enumerate(STAGES):
                stage_points = [points[(dataset, m, s, stage, metric)] for m in MODELS for s in STARTS]
                if not any(p["plotted"] for p in stage_points):
                    ax.axvspan(stage_index - 0.22, stage_index + 0.22, color="#EEEEEE", zorder=0)
                    ax.text(stage_index, 0.49, "unavailable", rotation=90,
                            transform=ax.get_xaxis_transform(), ha="center", va="center",
                            color="#666666", fontsize=6.8)
                    if row_index == 0:
                        available = sum(p["available_runs"] for p in stage_points)
                        expected = sum(p["required_runs"] for p in stage_points)
                        missing_stages.append(f"{dataset.upper()} {STAGE_NAMES[stage_index]}: {available}/{expected} runs")
            ax.set_xlim(-0.18, 2.18)
            ax.set_xticks(range(3), STAGE_NAMES)
            ax.tick_params(axis="both", length=2.5, pad=2, labelsize=7.4)
            if metric == "ari":
                ax.set_ylim(0, 1)
                ax.set_yticks((0, 0.5, 1.0))
            else:
                ax.set_ylim(0, max(values) * 1.1 if values and max(values) > 0 else 1)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=3))
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    
    handles = [Line2D([], [], color=COLORS[m], label=MODEL_NAMES[m]) for m in MODELS]
    handles.extend(Line2D([], [], color="#444444", marker="o" if s == "LLM" else "s",
                          markerfacecolor="#444444" if s == "LLM" else "white",
                          linestyle="-" if s == "LLM" else (0, (3, 2)),
                          markersize=3.5, label=f"{s} start") for s in STARTS)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.535, 1.0),
               ncol=4, frameon=False, handlelength=2.0, columnspacing=1.5,
               handletextpad=0.5, fontsize=7.5)
    if missing_stages:
        fig.subplots_adjust(bottom=0.16)
        fig.text(0.535, 0.023, "; ".join(missing_stages) + ". No values substituted.",
                 ha="center", fontsize=7.5, color="#555555")
    write_rows("F1_stage_quality_and_errors", list(points.values()))
    save(fig, "F1_stage_quality_and_errors")
    print(f"RQ1: {sum(p['plotted'] for p in points.values())}/{len(points)} metric means plotted; "
          + "; ".join(missing_stages))


if __name__ == "__main__":
    main()
