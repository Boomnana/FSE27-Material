from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from . import silra_transport_v2 as transport
from .evidence import EvidenceLedger, Observation

PAIR_ORACLE_CONTRACT = "ATOMIC_UNORDERED_BINARY_PAIR_ONE_SEMANTIC_SAMPLE"


def _unique_object(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out


def strict_json(text: str):
    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))


def canonical_sha256(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value) -> bool:
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _envelope_usage(envelope) -> dict:
    if type(envelope) is not dict or type(envelope.get("usage")) is not dict:
        return {}
    
    return dict(envelope["usage"])


def request_payload(left_id: str, left_text: str, right_id: str, right_text: str) -> dict:
    return {
        "schema": "aqcc-final-atomic-pair-v2",
        "report_a": {"id": left_id, "text": left_text},
        "report_b": {"id": right_id, "text": right_text},
    }


def request_hash(system_prompt: str, payload: dict) -> str:
    raw = json.dumps({"system": system_prompt, "payload": payload}, ensure_ascii=False,
                     sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def parse_pair_content(content: str) -> tuple[str, str]:
    obj = strict_json(content)
    if type(obj) is not dict or set(obj) != {"label", "reason"}:
        raise ValueError("pair response must contain exactly label and reason")
    label, reason = obj["label"], obj["reason"]
    if label not in {"SAME", "DIFFERENT"}:
        raise ValueError("label must be SAME or DIFFERENT")
    if type(reason) is not str or not reason.strip() or len(reason) > 800:
        raise ValueError("invalid reason")
    return label, reason.strip()


@dataclass
class SilraPairJudge:
    key: str
    system_prompt: str
    max_attempts: int = 3
    connect_timeout: float = 20
    read_timeout: float = 60
    total_timeout: float = 90
    record_raw_responses: bool = False
    
    model: str = transport.MODEL

    def __post_init__(self):
        if type(self.key) is not str or not self.key or any(c.isspace() for c in self.key):
            raise ValueError("invalid SILRA_API_KEY")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if type(self.model) is not str or self.model not in transport.ALLOWED_MODELS:
            raise ValueError("unsupported atomic judge model")

    @property
    def identity(self):
        return {"kind":"SILRA_ATOMIC_LLM","model":self.model,"endpoint":transport.ENDPOINT}

    def _with_response_trace(self, result: dict, body: str) -> dict:
        if self.record_raw_responses:
            
            
            safe_body = body.replace(self.key, "[REDACTED_CREDENTIAL]")
            if isinstance(result.get("reason"), str):
                result["reason"] = result["reason"].replace(self.key, "[REDACTED_CREDENTIAL]")
            result["raw_response_body"] = safe_body
            result["raw_response_sha256"] = hashlib.sha256(safe_body.encode("utf-8")).hexdigest()
            result["raw_response_redacted"] = safe_body != body
        return result

    def __call__(self, left_id: str, left_text: str, right_id: str, right_text: str) -> dict:
        payload = request_payload(left_id, left_text, right_id, right_text)
        req_sha = request_hash(self.system_prompt, payload)
        wire = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                          sort_keys=True, separators=(",", ":"))},
            ],
            "temperature": 0,
            "max_tokens": 512,
        }
        attempts = []
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            try:
                status, body = transport.send(
                    wire, self.key, transport.ENDPOINT,
                    connect_timeout=self.connect_timeout,
                    read_timeout=self.read_timeout,
                    total_timeout=self.total_timeout,
                )
            except transport.TransportError as exc:
                attempts.append({"attempt": attempt, "kind": "TRANSPORT", "phase": exc.phase,
                                 "error_class": exc.error_class, "http_status": exc.http_status,
                                 "elapsed_seconds": time.monotonic() - started})
                
                
                
                before_send = exc.phase == "connect" and exc.http_status is None
                if before_send and attempt < self.max_attempts:
                    time.sleep(2 ** (attempt - 1)); continue
                return {"status": "FAILED_TRANSPORT" if before_send else "FAILED_TRANSPORT_UNCERTAIN",
                        "label": None, "reason": "",
                        "request_sha256": req_sha, "response_sha256": None,
                        "attempts": attempts, "usage": {}}
            response_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
            usage = {}
            try:
                envelope = strict_json(body)
                usage = _envelope_usage(envelope)
            except (ValueError, TypeError):
                envelope = None
            if status != 200:
                attempts.append({"attempt": attempt, "kind": "HTTP", "http_status": status,
                                 "response_sha256": response_sha,
                                 "elapsed_seconds": time.monotonic() - started})
                
                
                return self._with_response_trace({"status": "FAILED_HTTP", "label": None, "reason": "",
                        "request_sha256": req_sha, "response_sha256": response_sha,
                        "attempts": attempts, "usage": usage}, body)
            try:
                if type(envelope) is not dict or envelope.get("error"):
                    raise ValueError("provider envelope error")
                response_model = envelope.get("model")
                versioned_deepseek_alias = (
                    self.model == "deepseek-v4-flash"
                    and type(response_model) is str
                    and response_model.startswith("deepseek-v4-flash-")
                )
                if response_model not in (None, self.model) and not versioned_deepseek_alias:
                    raise ValueError("provider model mismatch")
                choices = envelope.get("choices")
                if type(choices) is not list or len(choices) != 1:
                    raise ValueError("expected exactly one choice")
                choice = choices[0]
                if choice.get("finish_reason") not in ("stop", None):
                    raise ValueError("incomplete finish_reason")
                content = choice.get("message", {}).get("content")
                if type(content) is not str:
                    raise ValueError("missing assistant content")
                label, reason = parse_pair_content(content)
            except Exception as exc:
                
                
                attempts.append({"attempt": attempt, "kind": "PROTOCOL",
                                 "error_class": type(exc).__name__,
                                 "response_sha256": response_sha,
                                 "elapsed_seconds": time.monotonic() - started})
                return self._with_response_trace({"status": "FAILED_PROTOCOL", "label": None, "reason": "",
                        "request_sha256": req_sha, "response_sha256": response_sha,
                        "attempts": attempts, "usage": usage}, body)
            attempts.append({"attempt": attempt, "kind": "SUCCESS", "http_status": 200,
                             "response_sha256": response_sha,
                             "elapsed_seconds": time.monotonic() - started})
            return self._with_response_trace({"status": "SUCCESS", "label": label, "reason": reason,
                    "request_sha256": req_sha, "response_sha256": response_sha,
                    "attempts": attempts, "usage": usage}, body)
        raise AssertionError("unreachable")


@dataclass
class GoldJudge:
    issue_by_id: dict[str, str]
    system_prompt: str = "OFFLINE_GOLD_ORACLE_NOT_LLM"

    @property
    def identity(self):
        return {"kind":"OFFLINE_GOLD_ORACLE","model":None}

    def __call__(self, left_id: str, left_text: str, right_id: str, right_text: str) -> dict:
        if left_id not in self.issue_by_id or right_id not in self.issue_by_id:
            raise ValueError("GoldJudge called outside strict labeled population")
        label = "SAME" if self.issue_by_id[left_id] == self.issue_by_id[right_id] else "DIFFERENT"
        payload = request_payload(left_id, left_text, right_id, right_text)
        req_sha = request_hash(self.system_prompt, payload)
        raw = json.dumps({"label": label, "reason": "OFFLINE_GOLD_ORACLE"}, sort_keys=True)
        return {"status": "SUCCESS", "label": label, "reason": "OFFLINE_GOLD_ORACLE",
                "request_sha256": req_sha,
                "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "attempts": [{"attempt": 1, "kind": "OFFLINE_GOLD"}], "usage": {}}


class ReplayJudge:
    """Returns the exact frozen atomic response from an exhaustive reference run."""
    system_prompt = "REPLAY_OF_FROZEN_EXHAUSTIVE_REFERENCE"

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        terminal = strict_json((run_dir / "terminal.json").read_text(encoding="utf-8"))
        manifest = strict_json((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if type(terminal) is not dict or type(manifest) is not dict:
            raise ValueError("reference terminal and manifest must be objects")
        config = manifest.get("config")
        if type(config) is not dict:
            raise ValueError("reference manifest lacks configuration")
        self.reference_config = config
        self.reference_terminal = terminal
        if (terminal.get("status") != "VALID_AQCC_FINAL_TERMINAL"
                or terminal.get("reference_eligible") is not True):
            raise ValueError("reference run is not a valid AQCC-Final terminal")
        ids = terminal.get("report_ids")
        if (type(ids) is not list or len(ids) < 2
                or any(type(rid) is not str or not rid for rid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("reference has an invalid report population")
        self.report_ids = tuple(ids)
        n = len(ids)
        total = n * (n - 1) // 2
        for container, field, expected in (
            (terminal, "logical_pairs_queried", total),
            (terminal, "total_possible_pairs", total),
            (config, "reports", n),
            (config, "total_pairs", total),
        ):
            if type(container.get(field)) is not int or container[field] != expected:
                raise ValueError(f"reference {field} differs from its population")
        if (not config.get("dataset_id") or config["dataset_id"] != terminal.get("dataset_id")
                or type(config.get("source_sha256")) is not str or not config["source_sha256"]
                or not _is_sha256(config.get("prompt_sha256"))
                or config.get("pair_oracle_contract") != PAIR_ORACLE_CONTRACT
                or type(config.get("judge_identity")) is not dict):
            raise ValueError("reference source/prompt/oracle configuration is invalid")
        for field in ("execution_mode", "strategy"):
            if config.get(field) != terminal.get(field):
                raise ValueError(f"reference terminal {field} differs from manifest")
        ledger = EvidenceLedger(n)
        self.rows = {}
        qpath = run_dir / "pair_queries.jsonl"
        for line_number, line in enumerate(qpath.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = strict_json(line)
            if type(row) is not dict or type(row.get("observation")) is not dict:
                raise ValueError(f"invalid reference observation at line {line_number}")
            try:
                obs = Observation(**row["observation"])
            except TypeError as exc:
                raise ValueError(f"invalid reference observation at line {line_number}") from exc
            if (type(obs.query_index) is not int or obs.query_index != len(self.rows) + 1
                    or type(obs.left) is not int or type(obs.right) is not int
                    or not 0 <= obs.left < obs.right < n
                    or obs.status != "SUCCESS" or type(obs.label) is not str
                    or type(obs.reason) is not str or len(obs.reason) > 800
                    or not _is_sha256(obs.request_sha256) or not _is_sha256(obs.response_sha256)
                    or type(obs.metadata) is not dict):
                raise ValueError(f"invalid successful reference pair at line {line_number}")
            ledger.add(obs)
            pair_index = obs.left * (2 * n - obs.left - 1) // 2 + obs.right - obs.left - 1
            pair_ids = [self.report_ids[obs.left], self.report_ids[obs.right]]
            if (obs.metadata.get("report_ids") != pair_ids
                    or type(obs.metadata.get("pair_index")) is not int
                    or obs.metadata["pair_index"] != pair_index):
                raise ValueError(f"reference pair metadata mismatch at line {line_number}")
            result = row.get("judge_result")
            if type(result) is not dict or any(
                result.get(field) != getattr(obs, field)
                for field in ("status", "label", "reason", "request_sha256", "response_sha256")
            ):
                raise ValueError(f"reference judge result differs from observation at line {line_number}")
            self.rows[tuple(sorted(pair_ids))] = row
        if len(self.rows) != total:
            raise ValueError("reference query log does not contain every pair")
        evidence_sha = ledger.snapshot_sha256()
        if (terminal.get("evidence_sha256") != evidence_sha
                or terminal.get("evidence_counts") != ledger.counts()):
            raise ValueError("reference terminal evidence does not match query log")
        evidence = strict_json((run_dir / "evidence.json").read_text(encoding="utf-8"))
        if (type(evidence) is not dict or evidence.get("sha256") != evidence_sha
                or evidence.get("history") != ledger.history()):
            raise ValueError("reference evidence snapshot does not match query log")
        self.reference_evidence_sha256 = evidence_sha
        self.reference_config_sha256 = canonical_sha256(config)

    @property
    def identity(self):
        return {"kind":"REPLAY_EXHAUSTIVE_REFERENCE",
                "reference_judge_identity":self.reference_config.get("judge_identity"),
                "reference_evidence_sha256":self.reference_evidence_sha256,
                "reference_config_sha256":self.reference_config_sha256}

    def __call__(self, left_id: str, left_text: str, right_id: str, right_text: str) -> dict:
        key = tuple(sorted((left_id, right_id)))
        row = self.rows.get(key)
        if row is None:
            raise ValueError("pair absent from exhaustive reference")
        obs = row["observation"]
        return {"status": "SUCCESS", "label": obs["label"], "reason": obs["reason"],
                "request_sha256": obs["request_sha256"],
                "response_sha256": obs["response_sha256"],
                "attempts": [{"kind": "REPLAY", "original_query_index": obs["query_index"]}],
                "usage": {}}
