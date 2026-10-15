"""Non-learning scores for acquisition-only surrogate value controls."""
from __future__ import annotations

from dataclasses import asdict
import numpy as np

from .surrogate import SurrogateSummary


CONTROL_MODES = {"constant", "text_cosine"}


def score_semantics(mode):
    return {
        "constant": "UNCALIBRATED_CONSTANT_SCORE_NOT_LLM_PROBABILITY",
        "text_cosine": "UNCALIBRATED_TEXT_COSINE_NOT_LLM_PROBABILITY",
    }[mode]


def control_fields(mode):
    return {"acquisition_mode": mode, "surrogate_enabled": False,
            "acquisition_score_semantics": score_semantics(mode)}


def control_config(mode):
    return dict(control_fields(mode), acquisition_control={
        "schema": "aqcc-nonlearning-acquisition-v1",
        "unknown_score": "CONSTANT_0_5" if mode == "constant" else "CLIP_LEXICAL_CHAR_TFIDF_COSINE_0_1",
        "score_dtype": "float32", "successful_direct_scores": "SAME_1_DIFFERENT_0",
        "failed_pair_policy": "CONSUMED_WITHOUT_DIRECT_LABEL",
        "probing_pending_unknown_score": 0.5,
        "calibrated_llm_probability": False, "lr_instantiated": False,
        "lr_training": False, "lr_prediction": False, "lr_dynamic_features": False,
        "selector": "UNCHANGED_ADAPTIVE_DISCOVERY_RANKING_AND_DIVERSITY",
    })


class DisabledSurrogate:
    """An inert record, not an LLMSurrogate instance or model wrapper.

    Successful observations remain in the evidence ledger. Zero training counts
    describe the absent training operation, not the amount of collected evidence.
    """

    def __init__(self, mode, *, feature_policy):
        if mode not in CONTROL_MODES:
            raise ValueError("invalid nonlearning acquisition mode")
        self.mode = mode
        self.feature_policy = feature_policy
        self.model = None
        self.summary = SurrogateSummary(False, 0, 0, 0, tuple(), tuple(), None)

    def unknown_scores(self, store):
        if self.mode == "constant":
            return np.full(store.total_pairs, .5, dtype=np.float32)
        return np.clip(store.lexical, 0., 1.).astype(np.float32, copy=True)

    def state(self):
        return {"enabled": False, "status": "DISABLED_BY_ACQUISITION_CONTROL",
                "acquisition_mode": self.mode, "summary": asdict(self.summary), "model": None,
                "requested_feature_policy": self.feature_policy, "feature_names_used": [],
                "frozen_scaler_state": None,
                "training_performed": False, "prediction_performed": False,
                "dynamic_features_computed": False}
