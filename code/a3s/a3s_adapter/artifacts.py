from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".write-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    
    
    
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt + 1), 0.5))


def dump_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n")


def dump_jsonl(path: Path, rows) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    atomic_text(path, text)


def persist_engine(output_dir: Path, engine) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "pair_queries.jsonl", engine.oracle.records)
    attempt_rows = []
    for query in engine.oracle.records:
        attempts = query.get("physical_attempts") or []
        if not attempts and (query.get("shared_oracle") or {}).get("physical_cost_new"):
            attempts = query.get("attempts") or []
        for attempt in attempts:
            attempt_rows.append({
                "query_index": query["query_index"],
                "report_ids": query["report_ids"],
                "pair_indices": query["pair_indices"],
                "phase": query["phase"],
                "attempt": attempt,
                "request_sha256": query.get("request_sha256"),
                "response_sha256": query.get("response_sha256"),
                "raw_response_sha256": query.get("raw_response_sha256"),
                "usage": query.get("physical_usage") or {},
            })
    dump_jsonl(output_dir / "api_attempts.jsonl", attempt_rows)
    dump_jsonl(output_dir / "events.jsonl", engine.events)
    dump_jsonl(output_dir / "decision_and_inference.jsonl", engine.events)
    dump_jsonl(output_dir / "partition_submissions.jsonl", [
        {key: row[key] for key in (
            "logical_queries", "action", "clusters", "partition",
            "partition_sha256") if key in row}
        for row in engine.events if "partition" in row
    ])
    dump_json(output_dir / "checkpoints.json", engine.checkpoints)
    dump_json(output_dir / "runtime_state.json", {
        "schema": "a3s-p0-runtime-state-v1",
        "logical_queries": engine.oracle.logical_queries,
        "budget": engine.oracle.budget,
        "partition": [list(block) for block in engine.partition()],
        "constraint_sha256": engine.oracle.constraints.digest(),
        "stop_reason": engine.stop_reason,
    })
