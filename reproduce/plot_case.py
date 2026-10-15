"""Plot recorded judgment conflicts and A-B co-membership in the running case.

Reference-based evaluations remain in the unchanged audit CSVs.  The drawing
distinguishes observed LLM responses from partition states, without inferring
defect identity from the abbreviated report texts.
"""
import json
import textwrap

import matplotlib.pyplot as plt
from common import DATA, PAPER, style, save, verify_inputs, write_rows

SUMMARIES = {
    "A": "A complete phone number is entered; Next still requests a complete number.",
    "B": "A 178-prefix number cannot register; the app reports an incomplete number.",
    "C": "A 178-prefix number is rejected as incomplete; newer numbers are mentioned.",
    "D": "Some correctly entered numbers still prompt: 'Enter a complete phone number.'",
    "E": "A correct phone number cannot proceed; a prompt asks for a number for verification.",
}

def main():
    manifest = verify_inputs()
    case = json.loads((DATA / "running_case.json").read_text(encoding="utf-8"))
    assert case["reference_id"] == manifest["reference_id"]
    style()
    reports = case["reports"]
    aid, bid = reports["A"]["id"], reports["B"]["id"]
    same_reference = reports["A"]["reference_label"] == reports["B"]["reference_label"]
    rows = []
    for row in case["milestones"]:
        partition = row["partition"]
        ids = [rid for cluster in partition for rid in cluster]
        assert len(ids) == len(set(ids)) == len(case["reference_labels"])
        assert set(ids) == set(case["reference_labels"])
        index = {rid: c for c, cluster in enumerate(partition) for rid in cluster}
        together = index[aid] == index[bid]
        assert together == row["together"]
        status = ("TP" if same_reference else "FP") if together else ("FN" if same_reference else "TN")
        rows.append({"q": row["q"], "micro_round": row["micro_round"],
                     "together": together, "reference_same": same_reference, "status": status,
                     "milestone": "Probe" if row["is_probe"] else ("5%" if row["is_endpoint"] else f"M{row['micro_round']}")})
    write_rows("F5_running_case", rows)
    events = sorted(case.get('query_events', []), key=lambda event: event['query_index'])
    assert len(events) <= 10, 'Review layout before adding more than ten case-query events.'
    event_rows = []
    for event in events:
        first, second = event['pair']
        same = reports[first]['reference_label'] == reports[second]['reference_label']
        assert event['reference_relation'] == ('SAME' if same else 'DIFFERENT')
        outcome = event['outcome']
        consistent = outcome == event['reference_relation'] if outcome in ('SAME', 'DIFFERENT') else None
        assert event['reference_consistent'] == consistent
        event_rows.append({'q': event['query_index'], 'pair': '-'.join(event['pair']),
                           'outcome': outcome, 'reference_relation': event['reference_relation'],
                           'reference_consistent': consistent})
    write_rows('F5_running_case_queries', event_rows)
    
    
    by_pair = {frozenset(e["pair"]): e for e in events}
    conflict_edges = (("B", "C"), ("C", "A"), ("A", "D"))
    assert all(by_pair[frozenset(p)]["outcome"] == "SAME"
               and by_pair[frozenset(p)]["query_index"] <= 255
               for p in conflict_edges)
    assert by_pair[frozenset(("B", "D"))]["outcome"] == "DIFFERENT"
    assert by_pair[frozenset(("B", "D"))]["query_index"] == 255
    assert by_pair[frozenset(("A", "B"))]["outcome"] == "FAILED"

    fig = plt.figure(figsize=(5.6, 3.15))
    ax = fig.add_axes([0.015, 0.425, 0.57, 0.56])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, .99, "Report summaries", fontsize=8.2, fontweight="bold", va="top")
    ax.plot([0, .97], [.905, .905], color="#A7B0B7", linewidth=.6)
    for i, key in enumerate(("A", "B", "C", "D", "E")):
        y = .865 - i * .171
        ax.text(0, y, f"{key}\n...{reports[key]['id'][-5:]}", fontsize=7.5,
                fontweight="bold", color="#303B43", va="top", linespacing=1.1)
        ax.text(.195, y, textwrap.fill(SUMMARIES[key], width=42),
                fontsize=7.5, va="top", linespacing=1.08)
    relation = "same" if same_reference else "different"
    query_ax = fig.add_axes([0.62, 0.425, 0.365, 0.56])
    query_ax.set_axis_off()
    query_ax.set_xlim(0, 1); query_ax.set_ylim(0, 1)
    columns = (0, .19, .41)
    for x, label in zip(columns, ("q", "Pair", "LLM response")):
        query_ax.text(x, .99, label, fontsize=8.2, fontweight="bold", va="top")
    query_ax.plot([0, .99], [.905, .905], color="#A7B0B7", linewidth=.6)
    for i, event in enumerate(event_rows):
        returned = event["outcome"] if event["outcome"] in ("SAME", "DIFFERENT") else "Failed (no label)"
        color = {"SAME": "#0072B2", "DIFFERENT": "#D55E00"}.get(event["outcome"], "#666666")
        for j, (x, value) in enumerate(zip(columns, (str(event["q"]), event["pair"], returned))):
            query_ax.text(x, .862 - i * .0845, value, fontsize=7.6, va="top",
                          color=color if j == 2 else "#202020")

    conflict = fig.add_axes([0.02, .327, .96, .08]); conflict.set_axis_off()
    conflict.text(.5, .52, "At q=255: SAME along B-C-A-D conflicts with DIFFERENT on B-D.",
                  fontsize=8.0, ha="center", va="center", color="#303B43")
    timeline = fig.add_axes([0.10, 0.09, 0.86, 0.18])
    x = list(range(len(rows)))
    y = [int(row["together"]) for row in rows]
    timeline.plot(x, y, color="#77838A", linewidth=1.0, zorder=1)
    for i, row in enumerate(rows):
        timeline.scatter(i, y[i], s=30, color="#425A89", edgecolor="white", linewidth=0.5, zorder=2)
    timeline.set_yticks([0, 1], ["Apart", "Together"])
    timeline.set_ylim(-0.25, 1.30); timeline.set_xlim(-0.25, len(rows) - 0.75)
    tick_labels = [f"q={row['q']}" + (f"\n{row['milestone']}" if row["milestone"] in ("Probe", "5%") else "")
                   for row in rows]
    timeline.set_xticks(x, tick_labels)
    timeline.tick_params(axis="both", length=0, labelsize=8)
    timeline.spines[["left", "bottom"]].set_visible(False)
    timeline.set_title("A–B partition states", fontsize=8.5, pad=4)
    export = PAPER / "exports" / "Figure7_Judgment_Conflicts.svg"
    export.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"svg.fonttype": "none"}):
        fig.savefig(export, bbox_inches="tight")
    save(fig, "F5_running_case")
    print(f"Running case: {len(rows)} milestones, {len(event_rows)} verified queries; "
          f"current A-B relation is {relation}; endpoint={rows[-1]['status']}")

if __name__ == "__main__":
    main()
