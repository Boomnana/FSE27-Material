"""A block-local, first-response atomic LLM cache with conservative recovery.

This is an on-demand shared live oracle, not an exhaustive reference. Each arm
may only request its selected pair. A durable claim without a durable answer
consumes the key as uncertain; neither recovery nor a later arm resamples it.
The global response records, deduplicated by cache key, are the cost ledger.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid

from .provider import (
    PAIR_ORACLE_CONTRACT, SilraPairJudge, canonical_sha256, request_hash,
    request_payload, strict_json,
)


SHARED_ORACLE_CONTRACT = "BLOCK_FIRST_RESPONSE_ATOMIC_LLM_V1"
TRANSPORT_CONTRACT = "SILRA_ONE_SEMANTIC_SAMPLE_CONNECT_ONLY_RETRY_V1"
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class SharedOracleIntegrityError(ValueError):
    """Fatal cache corruption or incompatible identity, never a model label."""


def _thread_lock(path: Path):
    name = os.path.normcase(str(path.resolve()))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(name, threading.Lock())


@contextmanager
def _locked(path: Path):
    """Wait in short intervals; OS locks are released on process death."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _thread_lock(path):
        
        
        with path.open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                while True:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                            raise
                        time.sleep(0.025)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                while True:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.025)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _dump_new(path: Path, content: dict):
    """Caller holds the corresponding lock; do not overwrite old evidence."""
    if path.exists():
        raise SharedOracleIntegrityError("shared oracle record already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"content": content, "sha256": canonical_sha256(content)}
    payload = json.dumps(envelope, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n"
    
    
    temporary = path.with_name(".write-" + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _load(path: Path):
    try:
        envelope = strict_json(path.read_text(encoding="utf-8"))
        if (type(envelope) is not dict or set(envelope) != {"content", "sha256"}
                or type(envelope["content"]) is not dict
                or envelope["sha256"] != canonical_sha256(envelope["content"])):
            raise SharedOracleIntegrityError("shared oracle record hash mismatch")
        return envelope["content"]
    except (ValueError, TypeError, UnicodeError, KeyError) as exc:
        if isinstance(exc, SharedOracleIntegrityError):
            raise
        raise SharedOracleIntegrityError("shared oracle record is malformed") from None


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED_CREDENTIAL]")
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact(key, secret): _redact(item, secret) for key, item in value.items()}
    return value


class SharedFirstResponseJudge:
    """Wrap a SilraPairJudge in one immutable experiment-block oracle.

    ``block_identity`` is a public nonempty string or JSON object, for example
    an experiment id and block number. Different blocks require different cache
    directories and identities; reusing a directory with another identity fails.
    Top-level ``usage`` and ``attempts`` retain the frozen answer's provenance.
    Only a first capture has nonempty ``physical_usage``/``physical_attempts``;
    aggregate cost from global responses, not sums of arm-local answer usage.
    """

    fatal_exception_types = (SharedOracleIntegrityError, OSError)

    def __init__(self, delegate: SilraPairJudge, cache_dir: Path, block_identity=None, *, block_id=None):
        if block_id is not None:
            if block_identity is not None:
                raise ValueError("provide block_identity or block_id, not both")
            block_identity = block_id
        if not isinstance(delegate, SilraPairJudge):
            raise TypeError("shared live oracle requires SilraPairJudge")
        if not isinstance(block_identity, (str, dict)) or not block_identity:
            raise ValueError("block identity must be a nonempty public string or object")
        self.delegate = delegate
        self.system_prompt = delegate.system_prompt
        self.cache_dir = Path(cache_dir).resolve()
        
        public_identity = json.loads(json.dumps(block_identity, allow_nan=False))
        if delegate.key in json.dumps(public_identity, ensure_ascii=False):
            raise SharedOracleIntegrityError("block identity contains a credential")
        self.block_identity_sha256 = canonical_sha256(public_identity)
        self._delegate_identity = deepcopy(delegate.identity)
        self._transport_configuration = dict(
            contract=TRANSPORT_CONTRACT, atomic_oracle_contract=PAIR_ORACLE_CONTRACT,
            max_attempts=delegate.max_attempts,
            connect_timeout=delegate.connect_timeout, read_timeout=delegate.read_timeout,
            total_timeout=delegate.total_timeout,
            provider_implementation_sha256=hashlib.sha256(
                Path(__file__).with_name("provider.py").read_bytes()).hexdigest(),
            transport_implementation_sha256=hashlib.sha256(
                Path(__file__).with_name("silra_transport_v2.py").read_bytes()).hexdigest(),
        )
        self._protocol_identity = dict(
            delegate=self._delegate_identity, system_prompt_sha256=hashlib.sha256(
                self.system_prompt.encode("utf-8")).hexdigest(),
            temperature=0, max_tokens=512,
            transport=self._transport_configuration,
        )
        if delegate.key in json.dumps(self._protocol_identity, ensure_ascii=False):
            raise SharedOracleIntegrityError("public oracle identity contains a credential")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema="shared-oracle-block-v1", contract=SHARED_ORACLE_CONTRACT,
                        block_identity=public_identity,
                        block_identity_sha256=self.block_identity_sha256,
                        
                        
                        oracle_protocol_sha256=canonical_sha256(self._protocol_identity))
        self._block_manifest = deepcopy(manifest)
        manifest_path = self.cache_dir / "block.json"
        with _locked(self.cache_dir / ".block.lock"):
            if manifest_path.exists():
                if _load(manifest_path) != manifest:
                    raise SharedOracleIntegrityError("cache belongs to a different experiment block")
            else:
                if any(self.cache_dir.glob("claims/*.json")) or any(self.cache_dir.glob("responses/*.json")):
                    raise SharedOracleIntegrityError("cache records exist without their block manifest")
                _dump_new(manifest_path, manifest)

    @property
    def identity(self):
        return dict(kind="LIVE_BLOCK_SHARED_FIRST_RESPONSE_ATOMIC_LLM",
                    contract=SHARED_ORACLE_CONTRACT,
                    block_identity_sha256=self.block_identity_sha256,
                    protocol_sha256=canonical_sha256(self._protocol_identity),
                    delegate=deepcopy(self._delegate_identity),
                    exhaustive_reference=False)

    def _context(self, left_id, left_text, right_id, right_text):
        if (self.delegate.identity != self._delegate_identity
                or self.delegate.system_prompt != self.system_prompt
                or any(getattr(self.delegate, key) != self._transport_configuration[key]
                       for key in ("max_attempts", "connect_timeout", "read_timeout", "total_timeout"))):
            raise SharedOracleIntegrityError("delegate changed after shared oracle identity was frozen")
        values = (left_id, left_text, right_id, right_text)
        if not all(type(value) is str for value in values) or left_id == right_id:
            raise ValueError("atomic shared oracle requires ordered distinct string report IDs and text")
        payload = request_payload(*values)
        wire = dict(model=self._delegate_identity["model"], messages=[
            dict(role="system", content=self.system_prompt),
            dict(role="user", content=json.dumps(payload, ensure_ascii=False,
                                                 sort_keys=True, separators=(",", ":"))),
        ], temperature=0, max_tokens=512)
        wire_identity = dict(wire=wire, endpoint=self._delegate_identity["endpoint"],
                             transport=self._transport_configuration)
        wire_hash = canonical_sha256(wire_identity)
        cache_key = canonical_sha256(dict(contract=SHARED_ORACLE_CONTRACT,
                                         block=self.block_identity_sha256, wire=wire_hash))
        return dict(cache_key=cache_key, wire_identity_sha256=wire_hash,
                    block_identity_sha256=self.block_identity_sha256,
                    request_sha256=request_hash(self.system_prompt, payload),
                    protocol_sha256=canonical_sha256(self._protocol_identity))

    def _claim(self, context, origin):
        record = dict(schema="shared-oracle-claim-v1", state="CLAIMED", **context,
                      origin=origin, created_utc=_now())
        _dump_new(self.cache_dir / "claims" / (context["cache_key"] + ".json"), record)
        return record

    def _read_claim(self, context):
        claim = _load(self.cache_dir / "claims" / (context["cache_key"] + ".json"))
        if (claim.get("schema") != "shared-oracle-claim-v1" or claim.get("state") != "CLAIMED"
                or any(claim.get(key) != value for key, value in context.items())):
            raise SharedOracleIntegrityError("shared oracle claim identity mismatch")
        return claim

    def _validate_result(self, result, context):
        if (type(result) is not dict or result.get("request_sha256") != context["request_sha256"]
                or type(result.get("status")) is not str
                or type(result.get("attempts")) is not list or type(result.get("usage")) is not dict):
            raise SharedOracleIntegrityError("shared oracle result identity or structure mismatch")
        if result["status"] == "SUCCESS":
            if (result.get("label") not in ("SAME", "DIFFERENT")
                    or type(result.get("reason")) is not str or not result["reason"].strip()
                    or type(result.get("response_sha256")) is not str
                    or len(result["response_sha256"]) != 64):
                raise SharedOracleIntegrityError("invalid successful shared oracle result")
        elif not result["status"].startswith("FAILED_") or result.get("label") is not None:
            raise SharedOracleIntegrityError("invalid failed shared oracle result")

    def _complete(self, context, claim, result, origin, unknown_attempts=False):
        self._validate_result(result, context)
        safe = _redact(deepcopy(result), self.delegate.key)
        
        
        if isinstance(safe.get("raw_response_body"), str):
            safe["raw_response_sha256"] = hashlib.sha256(safe["raw_response_body"].encode("utf-8")).hexdigest()
        record = dict(schema="shared-oracle-response-v1", state="COMPLETED", **context,
                      claim_sha256=canonical_sha256(claim), result=safe,
                      result_sha256=canonical_sha256(safe), origin=origin,
                      completed_utc=_now(), physical_attempts_unknown=unknown_attempts,
                      physical_attempt_count=None if unknown_attempts else len(safe["attempts"]),
                      recorded_usage=deepcopy(safe["usage"]))
        _dump_new(self.cache_dir / "responses" / (context["cache_key"] + ".json"), record)
        return record

    def _read_response(self, context, claim):
        record = _load(self.cache_dir / "responses" / (context["cache_key"] + ".json"))
        if (record.get("schema") != "shared-oracle-response-v1" or record.get("state") != "COMPLETED"
                or any(record.get(key) != value for key, value in context.items())
                or record.get("claim_sha256") != canonical_sha256(claim)
                or record.get("result_sha256") != canonical_sha256(record.get("result"))):
            raise SharedOracleIntegrityError("shared oracle response binding mismatch")
        self._validate_result(record["result"], context)
        return record

    def _uncertain(self, context):
        return dict(status="FAILED_INTERRUPTED_UNCERTAIN", label=None, reason="",
                    request_sha256=context["request_sha256"], response_sha256=None,
                    attempts=[dict(kind="RECOVERED_UNCERTAIN", semantic_resample=False)], usage={})

    def _returned(self, record, origin, first_capture=False):
        result = deepcopy(record["result"])
        result["physical_usage"] = deepcopy(result["usage"]) if first_capture else {}
        result["physical_attempts"] = deepcopy(result["attempts"]) if first_capture else []
        result["shared_oracle"] = dict(
            contract=SHARED_ORACLE_CONTRACT, cache_key=record["cache_key"],
            block_identity_sha256=self.block_identity_sha256, origin=origin,
            original_capture_origin=record["origin"], original_result_sha256=record["result_sha256"],
            global_response_sha256=canonical_sha256(record),
            is_cache_hit=not first_capture, physical_cost_new=first_capture,
            physical_attempts_unknown=record["physical_attempts_unknown"],
            usage_semantics="FROZEN_ANSWER_USAGE_NOT_INCREMENTAL_ARM_CHARGE",
            cost_ledger="GLOBAL_RESPONSES_DEDUPLICATED_BY_CACHE_KEY",
        )
        return result

    def _resolve(self, values, recover):
        if _load(self.cache_dir / "block.json") != self._block_manifest:
            raise SharedOracleIntegrityError("shared oracle block identity changed")
        context = self._context(*values)
        key = context["cache_key"]
        with _locked(self.cache_dir / "locks" / (key + ".lock")):
            claim_path = self.cache_dir / "claims" / (key + ".json")
            response_path = self.cache_dir / "responses" / (key + ".json")
            if response_path.exists() and not claim_path.exists():
                raise SharedOracleIntegrityError("shared response is missing its durable claim")
            if claim_path.exists():
                claim = self._read_claim(context)
                if response_path.exists():
                    return self._returned(self._read_response(context, claim),
                                          "RECOVERED_COMPLETED" if recover else "CACHE_HIT")
                
                
                record = self._complete(context, claim, self._uncertain(context),
                                        "CLAIM_RECOVERY", unknown_attempts=True)
                return self._returned(record, "FROZEN_UNCERTAIN")
            claim = self._claim(context, "LOCAL_CLAIM_RECOVERY" if recover else "FIRST_CAPTURE")
            if recover:
                
                
                record = self._complete(context, claim, self._uncertain(context),
                                        "LOCAL_CLAIM_RECOVERY_NO_GLOBAL_CLAIM", unknown_attempts=True)
                return self._returned(record, "FROZEN_UNCERTAIN")
            try:
                result = self.delegate(*values)
            except Exception:
                
                
                record = self._complete(context, claim, self._uncertain(context),
                                        "DELEGATE_EXCEPTION", unknown_attempts=True)
                return self._returned(record, "FROZEN_UNCERTAIN")
            record = self._complete(context, claim, result, "FIRST_CAPTURE")
            return self._returned(record, "FIRST_CAPTURE", first_capture=True)

    def __call__(self, left_id, left_text, right_id, right_text):
        return self._resolve((left_id, left_text, right_id, right_text), recover=False)

    def recover_cached_result(self, left_id, left_text, right_id, right_text):
        """No provider calls, including when no global record exists yet."""
        return self._resolve((left_id, left_text, right_id, right_text), recover=True)




SharedPairJudge = SharedFirstResponseJudge
