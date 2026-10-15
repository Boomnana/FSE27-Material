Offline checks and paper artifacts

Run the following from the FAVR directory with Python 3.12:

```text
python -m pip install -r reproduce/requirements.txt
python -B reproduce/verify_scores.py
python -B reproduce/verify_ci.py
python -B reproduce/build_paper.py
```

The verification and build commands use local files only. They write results to
generated/. They do not contact an LLM, launch an experiment, change reference
labels, or modify the supplied paper_artifacts.

verify_scores.py independently calculates ARI, report-level B3 scores, and
pairwise error counts from the supplied report partitions and references. It
checks endpoint/checkpoint metrics and the corresponding derived score rows.
verify_ci.py recomputes all ten Table 4 conditional intervals with the recorded
PCG64 seed, 200,000 draws, and joint repetition blocks within eight strata.
These are pointwise intervals conditional on the fixed datasets and backends;
they do not estimate uncertainty for new datasets.

build_paper.py rebuilds Tables 3-6 and Figures 3-7. The table bodies must match
their supplied versions. Table generation also recomputes the saved bootstrap
intervals before using them. Tables 1-2 are supplied as static LaTeX. Figure 1 is
supplied as PDF and editable SVG; Figure 2 is supplied as PDF. This package
does not claim automatic regeneration of the entire manuscript. Numerical
figure inputs are fixed; PDF metadata, font rendering and tight cropping can
vary across runtime versions.
The table files are manuscript fragments that use its acmart, booktabs,
tabularx and siunitx setup; they are not standalone LaTeX documents.

The data snapshots retain historical AQCC identifiers for exact run matching;
the method is named FAVR in the paper. In the metric CSV files, before and
before_mean hold the current-reference values. The duplicate after fields
come from the archived scoring schema. A column named delta describes that
unchanged reference scenario, not a refinement gain. Named delta_ari metrics
represent the stated within-run or paired comparison.

The two external aggregation pipelines were selected using the target
reference labels, as stated in the paper. Their setting and evaluation-unit
metadata remain in external_baselines.json. These offline pipeline comparisons
are not query-matched controls. A3S records are single endpoints per setting.

The requirements describe the validated offline distribution environment,
not a claim about every historical online execution environment. File origins
and transformations are described in paper_artifacts/provenance.json.
Published metadata uses the paper's backend names and opaque record identifiers;
model-name corrections do not change partitions, measurements, or seeds.
The checks validate file structure, reference identity, partition membership,
scores, matched queries, and bootstrap intervals. Generated files are
supplementary checks, not raw observations.
