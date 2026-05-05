"""Layout scoring package.

Top-level package today (sibling of ``src/offscroll``); will be
moved into ``src/offscroll/scoring/`` when the layout optimizer
is wired into the main pipeline (planned in the follow-on brief
to #379). Code that consumes this module today should add the
repo root to ``sys.path`` and ``import scoring.technical_scorer``,
or import ``technical_scorer`` directly by file path.

Public surface:
    score_spread(features) -> float
    hard_constraints_violated(features) -> list[str]
    evaluate_candidate(features) -> CandidateScore
    select_best_candidate(candidates, ...) -> int | None
    ScoringMonitor
"""

from .technical_scorer import (
    CandidateScore,
    FEATURE_ORDER,
    RIDGE_COEFS,
    RIDGE_INTERCEPT,
    ScoringMonitor,
    evaluate_candidate,
    hard_constraints_violated,
    rank_candidates,
    rule_score,
    score_spread,
    select_best_candidate,
)

__all__ = [
    "CandidateScore",
    "FEATURE_ORDER",
    "RIDGE_COEFS",
    "RIDGE_INTERCEPT",
    "ScoringMonitor",
    "evaluate_candidate",
    "hard_constraints_violated",
    "rank_candidates",
    "rule_score",
    "score_spread",
    "select_best_candidate",
]
