"""Shared, local-only inputs and plot conventions for current-reference figures."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parents[1]
PAPER = PACKAGE / "generated"
DATA = PACKAGE / "paper_artifacts" / "data"
FIGURES = PAPER / "figures"
PREVIEWS = PAPER / "results" / "previews"
DATASETS = ("jayme", "ielts", "hujiang", "huawei")
MODELS = ("qwen3.5-flash", "deepseek-v4-flash")
STARTS = ("LLM", "HC")
MODEL_NAMES = {"qwen3.5-flash": "Qwen", "deepseek-v4-flash": "DeepSeek"}
COLORS = {"qwen3.5-flash": "#0072B2", "deepseek-v4-flash": "#D55E00"}
DATASET_COLORS = dict(zip(DATASETS, ("#0072B2", "#009E73", "#D55E00", "#CC79A7")))

def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def verify_inputs(data_dir=DATA):
    data_dir = Path(data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    reference = json.loads((data_dir / "CURRENT_REFERENCE.json").read_text(encoding="utf-8"))
    assert manifest["reference_id"] == reference["reference_id"] == "FULL_REFERENCE_20260928"
    entries = manifest["files"]
    assert entries and len({entry["file"] for entry in entries}) == len(entries)
    for entry in entries:
        path = (data_dir / entry["file"]).resolve()
        assert path.is_relative_to(data_dir) and path.is_file(), path
        if "bytes" in entry:
            assert path.stat().st_size == entry["bytes"], (path, "file size")
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = csv.reader(stream)
                header = next(rows)
                assert header and len(header) == len(set(header)), path
                count = 0
                for row in rows:
                    assert len(row) == len(header), (path, count)
                    count += 1
                assert count, (path, "empty data")
    return manifest

def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": 0.6,
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })

def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight",
                metadata={"Creator": "FAVR result plotting scripts"})
    plt.close(fig)

def write_rows(stem, rows):
    output = PAPER / "results" / "plot_data"
    output.mkdir(parents=True, exist_ok=True)
    if rows:
        with (output / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
