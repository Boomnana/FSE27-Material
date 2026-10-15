"""Supported offline launcher for frozen-corpus character TF-IDF and HC P1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqcc_final.initialization import prepare_tfidf_hc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="Annotation-free JSON array: id, text, sourceExcelrow only")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-workbook", type=Path, required=True,
                        help="Provenance only: this file is byte-hashed; its cells are never read")
    parser.add_argument("--source-sheet", required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New preparation directory; existing directories are rejected")
    args = parser.parse_args(argv)
    metadata = prepare_tfidf_hc(corpus_path=args.corpus, dataset_id=args.dataset_id,
                                source_workbook=args.source_workbook, source_sheet=args.source_sheet,
                                output_dir=args.output_dir)
    print(json.dumps({"status": metadata["status"], "dataset_id": metadata["dataset_id"],
                      "reports": metadata["reports"], "p1_clusters": metadata["p1_clusters"],
                      "selected_silhouette": metadata["selected_silhouette"],
                      "candidate_k_count": metadata["candidate_k_count"],
                      "representation": metadata["representation"],
                      "preparation_metadata": str((args.output_dir / "preparation_metadata.json").resolve())},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
