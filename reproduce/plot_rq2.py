"""Plot available current-reference FAVR-minus-control paired ARI gains.

The ``before`` value of a ``paired_gain / delta_ari`` row is the paired
control contrast.  CSV ``delta`` instead measures a label-scenario change
and must not be plotted here.  Incomplete study-wide means remain absent.
"""
import math
from collections import Counter, defaultdict
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from common import MODEL_NAMES, MODELS, PAPER, read_csv, save, style, verify_inputs, write_rows

CONTROLS = ("AQCC_CONST", "AQCC_TEXT", "AQCC_RANDOM", "FROZEN_LR", "FROZEN_PLAN")
CONTROL_NAMES = dict(zip(CONTROLS, ("CONST", "TEXT", "RANDOM", "FROZEN LR", "FROZEN PLAN")))
MODEL_COLORS = dict(zip(MODELS, ("#425A89", "#D95743")))


def load_points():
    verify_inputs()
    selected = [r for r in read_csv("run_metric_diffs.csv")
                if r["rq"] == "RQ2" and r["family"] == "paired_gain"
                and r["metric"] == "delta_ari" and r["stage"] == "P_B"]
    assert len(selected) == len(CONTROLS) * 48
    keys = [(r["control"], r["dataset"], r["model"], r["start"], r["repeat"]) for r in selected]
    assert len(keys) == len(set(keys)), "Duplicate paired run"
    assert Counter(r["control"] for r in selected) == Counter({c: 48 for c in CONTROLS})
    for row in selected:
        if row["before"]:
            assert math.isfinite(float(row["before"]))
            assert not row["before_status"].startswith("missing"), row
    overall = [r for r in read_csv("rq2_overall_diffs.csv")
               if r["metric"] == "delta_ari_equal_configuration_mean"]
    assert {r["control"] for r in overall} == set(CONTROLS)
    for summary in overall:
        members = [r for r in selected if r["control"] == summary["control"]]
        groups = defaultdict(list)
        for row in members:
            if row["before"]:
                groups[row["dataset"], row["model"], row["start"]].append(float(row["before"]))
        complete = len(groups) == 16 and all(len(v) == 3 for v in groups.values())
        assert bool(summary["before"]) == complete, summary
        if complete:
            assert math.isclose(float(summary["before"]), mean(mean(v) for v in groups.values()),
                                abs_tol=1e-12), summary
    return selected, {r["control"]: r for r in overall}


def swarm_offsets(ax, values, max_offset=0.11):
    """Deterministic point packing using displayed distances, without data jitter."""
    
    placed = []
    result = [0.0] * len(values)
    dy = 0.055
    candidates = [0.0] + [sign * level * dy for level in range(1, 4) for sign in (-1, 1)]
    candidates = [y for y in candidates if abs(y) <= max_offset + 1e-9]
    for i in sorted(range(len(values)), key=lambda k: (values[k], k)):
        x = values[i]
        ranked = []
        for y in candidates:
            xy = ax.transData.transform((x, y))
            separation = min((math.dist(xy, ax.transData.transform(p)) for p in placed), default=math.inf)
            ranked.append((separation >= 7.0, -abs(y), separation, y))
        feasible = [r for r in ranked if r[0]]
        y = max(feasible or ranked, key=lambda r: (r[0], r[1] if feasible else r[2], r[2]))[3]
        placed.append((x, y))
        result[i] = y
    return result


def density_curve(values):
    """Gaussian KDE with Scott bandwidth, displayed within the observed range."""
    values = np.asarray(values, dtype=float)
    bandwidth = max(float(values.std(ddof=1)) * len(values) ** (-0.2), 1e-6)
    grid = np.linspace(float(values.min()), float(values.max()), 200)
    kernels = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * kernels ** 2).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return grid, density, bandwidth


def main():
    rows, overall = load_points()
    available = [r for r in rows if r["before"]]
    coverage = Counter(r["control"] for r in available)
    datasets = Counter(r["dataset"] for r in available)
    style()
    fig, ax = plt.subplots(figsize=(5.5, 3.02))
    fig.subplots_adjust(left=0.155, right=0.89, top=0.84, bottom=0.15)
    values = [float(r["before"]) for r in available]
    ax.set_xlim(min(-0.1, math.floor((min(values, default=0) - 0.02) * 10) / 10),
                max(0.1, math.ceil((max(values, default=0) + 0.02) * 10) / 10))
    ax.set_ylim(len(CONTROLS) - 0.5, -0.5)
    ax.set_yticks(range(len(CONTROLS)), [CONTROL_NAMES[c] for c in CONTROLS])
    ax.tick_params(axis="y", length=0, pad=5, labelsize=8)
    ax.tick_params(axis="x", length=0, labelsize=8, pad=4)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="x", color="#E1E6EA", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.axvline(0, color="#3D474D", linewidth=0.9, zorder=2)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel(r"$\Delta$ARI (FAVR $-$ control)", labelpad=4, fontsize=8.5)
    
    fig.canvas.draw()
    plotted_rows = []
    distribution_rows = []
    density_rows = []
    for control_index, control in enumerate(CONTROLS):
        group = sorted([r for r in available if r["control"] == control],
                       key=lambda r: (r["dataset"], r["model"], r["start"], int(r["repeat"])))
        run_values = [float(r["before"]) for r in group]
        grid, density, bandwidth = density_curve(run_values)
        baseline = control_index - 0.09
        ridge = baseline - 0.25 * density / density.max()
        ax.fill_between(grid, ridge, baseline, facecolor="#B9DCE4", edgecolor="#6985A3",
                        linewidth=0.65, zorder=2)
        q1, median, q3 = np.percentile(run_values, [25, 50, 75])
        stats = {"q1": q1, "med": median, "q3": q3,
                 "whislo": min(run_values), "whishi": max(run_values), "fliers": []}
        ax.bxp([stats], positions=[control_index], widths=0.13, vert=False,
               showfliers=False, showcaps=False, patch_artist=True, manage_ticks=False,
               boxprops={"facecolor": "white", "edgecolor": "#59656F", "linewidth": 0.65},
               medianprops={"color": "#3D474D", "linewidth": 0.8},
               whiskerprops={"color": "#7C8993", "linewidth": 0.7}, zorder=3)
        offsets = swarm_offsets(ax, run_values)
        for row, offset in zip(group, offsets):
            value = float(row["before"])
            color = MODEL_COLORS[row["model"]]
            ax.scatter(value, control_index + 0.28 + offset, s=8.5, zorder=4,
                       marker="o" if row["model"] == MODELS[0] else "s",
                       facecolor=color if row["start"] == "LLM" else "white",
                       edgecolor=color, linewidth=0.6)
        summary = overall[control]
        if summary["before"]:
            ax.scatter(float(summary["before"]), control_index, s=24, zorder=5,
                       marker="D", facecolor="#7756A4", edgecolor="white", linewidth=0.65)
            ax.text(1.025, control_index, f"{float(summary['before']):+.3f}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=8, color="#303B43")
        distribution_rows.append({"control": control, "n": len(run_values),
                                  "min": min(run_values), "q1": q1, "median": median,
                                  "q3": q3, "max": max(run_values),
                                  "equal_configuration_mean": summary["before"],
                                  "kde_bandwidth": bandwidth})
        density_rows.extend({"control": control, "delta_ari": x, "density": y}
                            for x, y in zip(grid, density))
        if coverage[control] != 48 or not summary["before"]:
            note = f"{coverage[control]}/48; mean unavailable"
            ax.text(0.99, control_index - 0.36, note,
                    transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color="#666666")
        for row in [r for r in rows if r["control"] == control]:
            plotted_rows.append({
                "control": control, "dataset": row["dataset"], "model": row["model"],
                "start": row["start"], "repeat": row["repeat"], "run": row["run"],
                "metric": "delta_ari", "current_reference_aqcc_minus_control": row["before"],
                "plotted": bool(row["before"]), "status": row["before_status"],
            })
    handles = [Line2D([], [], marker="o" if m == MODELS[0] else "s", color=MODEL_COLORS[m],
                      linestyle="None", markerfacecolor=MODEL_COLORS[m],
                      markersize=3.5, label=MODEL_NAMES[m]) for m in MODELS]
    handles.extend(Line2D([], [], marker="o", color="#555555", linestyle="None",
                          markerfacecolor="#555555" if s == "LLM" else "white",
                          markersize=3.5, label=f"{s} start") for s in ("LLM", "HC"))
    handles.append(Line2D([], [], marker="D", color="#7756A4", linestyle="None",
                          markersize=4, label="Mean"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.51, 0.99),
               ncol=5, frameon=False, columnspacing=1.2, handletextpad=0.35,
               handlelength=1.0, fontsize=8)
    write_rows("F2_paired_control_effects", plotted_rows)
    write_rows("F2_configuration_means", [{"control": c,
               "equal_configuration_mean": overall[c]["before"],
               "plotted": bool(overall[c]["before"]), "status": overall[c]["before_status"]}
               for c in CONTROLS])
    write_rows("F2_distribution_summary", distribution_rows)
    write_rows("F2_density_curves", density_rows)
    export_dir = PAPER / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(export_dir / "Figure4_FAVR.svg", bbox_inches="tight")
    save(fig, "F2_paired_control_effects")
    print(f"RQ2: {len(available)}/{len(rows)} paired runs plotted; "
          f"coverage={dict(coverage)}; datasets={dict(datasets)}; "
          f"{sum(bool(r['before']) for r in overall.values())} overall means")


if __name__ == "__main__":
    main()
