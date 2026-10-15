from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

from aqcc_final.provider import canonical_sha256
from aqcc_final.shared_oracle import (
    SharedFirstResponseJudge, SharedOracleIntegrityError, _load, _locked,
)


class LayeredSharedFirstResponseJudge(SharedFirstResponseJudge):
    """Write to a new cache while lazily reading immutable paired-control records.

    A completed fallback response and its durable claim are hard-linked into the
    new block on first access. Incomplete claims, locks and temporary files are
    never imported. A miss is delegated to the live provider and saved only in
    the new cache through the parent implementation.
    """

    def __init__(self, delegate, cache_dir: Path, fallback_cache_dir: Path | None,
                 block_identity=None, *, block_id=None):
        self.fallback_cache_dir = (
            Path(fallback_cache_dir).resolve() if fallback_cache_dir is not None else None
        )
        super().__init__(delegate, cache_dir, block_identity, block_id=block_id)
        if self.fallback_cache_dir is not None:
            source_manifest = self.fallback_cache_dir / "block.json"
            if source_manifest.exists() and _load(source_manifest) != self._block_manifest:
                raise SharedOracleIntegrityError(
                    "fallback cache belongs to a different experiment block")

    @property
    def identity(self):
        value = deepcopy(super().identity)
        value["fallback_mode"] = "LAZY_IMMUTABLE_COMPLETED_RECORDS_ONLY"
        value["fallback_block_present"] = bool(
            self.fallback_cache_dir is not None
            and (self.fallback_cache_dir / "block.json").exists()
        )
        return value

    def _validate_fallback_pair(self, context, claim, response) -> None:
        if (claim.get("schema") != "shared-oracle-claim-v1"
                or claim.get("state") != "CLAIMED"
                or any(claim.get(key) != value for key, value in context.items())):
            raise SharedOracleIntegrityError("fallback claim identity mismatch")
        if (response.get("schema") != "shared-oracle-response-v1"
                or response.get("state") != "COMPLETED"
                or any(response.get(key) != value for key, value in context.items())
                or response.get("claim_sha256") != canonical_sha256(claim)
                or response.get("result_sha256") != canonical_sha256(response.get("result"))):
            raise SharedOracleIntegrityError("fallback response binding mismatch")
        self._validate_result(response["result"], context)

    def _seed_completed_fallback(self, values) -> None:
        if self.fallback_cache_dir is None:
            return
        context = self._context(*values)
        name = context["cache_key"] + ".json"
        source_claim = self.fallback_cache_dir / "claims" / name
        source_response = self.fallback_cache_dir / "responses" / name
        
        
        if not source_response.exists():
            return
        if not source_claim.exists():
            raise SharedOracleIntegrityError("fallback response is missing its durable claim")
        claim, response = _load(source_claim), _load(source_response)
        self._validate_fallback_pair(context, claim, response)
        target_claim = self.cache_dir / "claims" / name
        target_response = self.cache_dir / "responses" / name
        with _locked(self.cache_dir / "seed_locks" / (context["cache_key"] + ".lock")):
            if target_claim.exists() or target_response.exists():
                return
            target_claim.parent.mkdir(parents=True, exist_ok=True)
            target_response.parent.mkdir(parents=True, exist_ok=True)
            
            
            os.link(source_claim, target_claim)
            os.link(source_response, target_response)

    def __call__(self, left_id, left_text, right_id, right_text):
        values = (left_id, left_text, right_id, right_text)
        self._seed_completed_fallback(values)
        return super().__call__(*values)

    def recover_cached_result(self, left_id, left_text, right_id, right_text):
        values = (left_id, left_text, right_id, right_text)
        self._seed_completed_fallback(values)
        return super().recover_cached_result(*values)
