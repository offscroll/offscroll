"""Technical proficiency scorer for layout candidates.

Backed by Ridge(alpha=0.1) trained on 200 graded spreads with 18
features (EXP-004 / brief #344). The model's coefficients ship as
constants — no joblib pickle, no model file on disk.

Performance envelope (from #344):
    5-fold CV R^2 ............ 0.498 +/- 0.114
    Bootstrap median R^2 ..... 0.499  (95% CI [0.310, 0.646])
    Val MAE .................. ~1.14 grade points (1-7 scale)

Use the score for *relative* ranking inside a candidate set. Do
not use it as an absolute grade prediction (MAE is too wide for
that). Pair with rule-based hard constraints to reject obviously
broken layouts that the smooth model would otherwise let pass.

Feature contract:
    score_spread(features) takes a dict[str, float] keyed on the
    18 names in FEATURE_ORDER. Missing keys default to 0.0 (the
    same NA handling the training pipeline uses). The same feature
    computation path that produced training features must be used
    at runtime — see compute_features_004.py in the training
    pipeline.

Three of the eighteen features have zero coefficients in the
trained model:
    - d2_widows        : feature was always-zero in training data
                         (computation bug, fixed in brief #377).
                         Will activate after a re-fit on corrected
                         features.
    - est_image_count  : zero across the training corpus (no
                         image-bearing editions).
    - d3_image_fraction: zero across the training corpus.

These keep the constant block at 18 entries so the schema lines
up with FEATURE_ORDER. They contribute nothing to the score
today.

Stability:
    Coefficients reproduce sklearn's Ridge(alpha=0.1) prediction
    on the training feature matrix to within ~1e-15. Verified by
    fit-and-compare against features-004.csv at module import time
    in the test suite.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trained-model constants (Ridge alpha=0.1, n=200, features-004 schema)
# ---------------------------------------------------------------------------

# Feature order matches the training pipeline's reduced feature set
# (compute_features_004.py + fit_batch004.py redundancy-removal step).
# Order is load-bearing for any downstream code that materializes a
# feature vector — name lookup is preferred over positional access.
FEATURE_ORDER: tuple[str, ...] = (
    "is_solo",
    "edition_page_count",
    "page_position_frac",
    "edition_brief_frac",
    "edition_template_entropy",
    "est_items_per_page",
    "est_words_per_page",
    "est_brief_count",
    "est_image_count",
    "est_source_count",
    "d3_image_fraction",
    "d5_fill_fraction",
    "d7_template_entropy",
    "anchor_strength",
    "d2_orphans",
    "d2_widows",
    "d4_col_balance",
    "d6_dead_space",
)

RIDGE_INTERCEPT: float = 2.32198434404878

RIDGE_COEFS: Mapping[str, float] = {
    "is_solo":                   +0.828162018587711,
    "edition_page_count":        -0.0207078793645214,
    "page_position_frac":        -0.441311789435324,
    "edition_brief_frac":        -0.878674152769393,
    "edition_template_entropy":  -0.120424510029847,
    "est_items_per_page":        +0.0718802482523443,
    "est_words_per_page":        +1.30321301228503e-05,
    "est_brief_count":           -0.124309716613182,
    "est_image_count":            0.0,
    "est_source_count":          -0.0804153711395041,
    "d3_image_fraction":          0.0,
    "d5_fill_fraction":          +6.29276217479481,
    "d7_template_entropy":       +0.226757888114596,
    "anchor_strength":           +0.0815454087277855,
    "d2_orphans":                -0.427305264989429,
    "d2_widows":                  0.0,
    "d4_col_balance":            +0.00202433141840321,
    "d6_dead_space":             -0.170072318224925,
}

assert set(RIDGE_COEFS) == set(FEATURE_ORDER), (
    "RIDGE_COEFS keys must match FEATURE_ORDER exactly"
)
assert len(FEATURE_ORDER) == 18

# Grade is on a 1-7 integer scale during grading; the model is a
# regression and can predict outside that range. Clamp at the
# boundaries when comparing model output to grader scale.
GRADE_MIN: float = 1.0
GRADE_MAX: float = 7.0


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def score_spread(features: Mapping[str, float]) -> float:
    """Predict technical proficiency grade (~1-7) for a spread.

    Args:
        features: dict-like keyed by feature name. Missing keys are
            treated as 0.0 (matches training pipeline's NA fill).
            Extra keys are ignored. Boolean-ish ints (e.g., is_solo)
            should be 0 or 1.

    Returns:
        Continuous score; expected mostly within [1.0, 7.0] but not
        clamped. Use the raw value for ranking. For human-facing
        display call ``clamp_grade(score_spread(features))``.
    """
    s = RIDGE_INTERCEPT
    for name, coef in RIDGE_COEFS.items():
        if coef == 0.0:
            # Skip feature lookup entirely for zero-coef features.
            continue
        v = features.get(name, 0.0)
        if v is None:
            v = 0.0
        s += coef * float(v)
    return s


def clamp_grade(score: float) -> float:
    """Clamp a raw model score to the [GRADE_MIN, GRADE_MAX] band."""
    if score != score:  # NaN guard
        return GRADE_MIN
    return max(GRADE_MIN, min(GRADE_MAX, score))


# ---------------------------------------------------------------------------
# Hard constraints (Tier-1 rule violations the model agrees with)
# ---------------------------------------------------------------------------

# Model-agreement rationale for each rule (from EXP-004 §10):
#   d2_orphans         coef = -0.427  -> orphans hurt; reject any > 0
#   d2_widows          coef =  0.000  -> feature broken in training set
#                                        (fixed in #377); rule stands
#                                        independently of the model
# d6_dead_space and d4_col_balance are *not* hard rules — they are
# soft signals the model already weights. Rejecting on them would
# double-count.

# Allow a tiny tolerance so that minor rendering jitter at the
# grading-pixel level doesn't fail the gate.
_ORPHAN_TOLERANCE = 0
_WIDOW_TOLERANCE = 0


def hard_constraints_violated(features: Mapping[str, float]) -> list[str]:
    """Return the list of hard-rule violations (empty list = pass).

    These are absolute rejections — a candidate that violates any
    of these is removed from contention before model scoring.
    """
    violations: list[str] = []

    orphans = features.get("d2_orphans", 0.0) or 0.0
    if orphans > _ORPHAN_TOLERANCE:
        violations.append(f"d2_orphans={orphans:g} > {_ORPHAN_TOLERANCE}")

    widows = features.get("d2_widows", 0.0) or 0.0
    if widows > _WIDOW_TOLERANCE:
        violations.append(f"d2_widows={widows:g} > {_WIDOW_TOLERANCE}")

    fill = features.get("d5_fill_fraction", 0.0) or 0.0
    if not (0.0 <= fill <= 1.001):
        violations.append(
            f"d5_fill_fraction={fill:g} out of [0, 1] (computation bug)"
        )

    return violations


def rule_score(features: Mapping[str, float]) -> float:
    """Pure rule-based score, used as a tiebreaker and as a
    monitoring signal compared against the model.

    Higher is better. Composition mirrors the dimensions Neville's
    LAYOUT-OPTIMIZATION.md treats as Tier-2 deterministic checks:
    page fill rewards, dead-space penalty, column-balance penalty,
    orphan/widow penalty.
    """
    fill = features.get("d5_fill_fraction", 0.0) or 0.0
    dead = features.get("d6_dead_space", 0.0) or 0.0
    col_balance_pt = features.get("d4_col_balance", 0.0) or 0.0
    orphans = features.get("d2_orphans", 0.0) or 0.0
    widows = features.get("d2_widows", 0.0) or 0.0

    score = 0.0
    score += 6.0 * fill
    score -= 1.5 * dead
    # d4_col_balance is in points; 720pt is the printable column
    # height. Penalize larger imbalance.
    score -= (col_balance_pt / 720.0) * 1.0
    score -= 0.5 * orphans
    score -= 0.5 * widows
    return score


# ---------------------------------------------------------------------------
# Hybrid candidate evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateScore:
    """Evaluation result for a single candidate layout.

    `model_score` is the raw Ridge output; `rule_score_value` is
    the deterministic rule-based score. `violations` is non-empty
    iff the candidate fails a hard constraint and should not be
    deployed.
    """

    model_score: float
    rule_score_value: float
    violations: tuple[str, ...]

    @property
    def passes(self) -> bool:
        return not self.violations


def evaluate_candidate(features: Mapping[str, float]) -> CandidateScore:
    """Compute the full evaluation tuple for one candidate."""
    violations = hard_constraints_violated(features)
    return CandidateScore(
        model_score=score_spread(features),
        rule_score_value=rule_score(features),
        violations=tuple(violations),
    )


def select_best_candidate(
    candidates: Sequence[Mapping[str, float]],
    *,
    top_k: int = 3,
    monitor: "ScoringMonitor | None" = None,
) -> int | None:
    """Hybrid scoring per Ada's design (#344 §9):

    1. Filter out candidates with hard-constraint violations.
    2. Rank survivors by model score; take top-K.
    3. Within top-K, tie-break by deterministic rule score, then
       by candidate index for stability across runs.
    4. Return the index (into the input sequence) of the winner,
       or None if every candidate violates a hard constraint.

    `monitor`, if supplied, receives per-candidate evaluations and
    a final disagreement record (set when the model winner is not
    the rule winner).
    """
    if not candidates:
        return None

    survivors: list[tuple[int, CandidateScore]] = []
    for i, feats in enumerate(candidates):
        ev = evaluate_candidate(feats)
        if monitor is not None:
            monitor.record(i, ev)
        if ev.passes:
            survivors.append((i, ev))

    if not survivors:
        if monitor is not None:
            monitor.note_all_rejected(len(candidates))
        return None

    # Top-K by model score; tiebreak by rule score, then index.
    survivors.sort(
        key=lambda iv: (-iv[1].model_score, -iv[1].rule_score_value, iv[0])
    )
    top = survivors[:max(1, top_k)]

    # Final tiebreak inside top-K: rule score then index.
    top.sort(key=lambda iv: (-iv[1].rule_score_value, iv[0]))
    winner_idx, winner_score = top[0]

    if monitor is not None:
        # Compare against pure-rule winner among survivors for the
        # disagreement metric.
        rule_ranked = sorted(
            survivors, key=lambda iv: (-iv[1].rule_score_value, iv[0])
        )
        rule_winner_idx = rule_ranked[0][0]
        monitor.record_disagreement(
            model_winner=winner_idx,
            rule_winner=rule_winner_idx,
            top_k_indices=tuple(i for i, _ in top),
        )

    return winner_idx


# ---------------------------------------------------------------------------
# Monitoring (#344 §"What to monitor in production")
# ---------------------------------------------------------------------------

@dataclass
class ScoringMonitor:
    """Collects per-candidate evaluations during a layout decision.

    What's tracked, in order of importance:

    1. Score distribution: min / median / max of model scores
       across the candidate set. Drift in median over a 20-edition
       window beyond +/- 0.5 means the feature distribution has
       shifted; trigger a re-train review.
    2. Model-vs-rule disagreement: every time the model winner
       differs from the rule winner, log both indices and both
       scores. A high disagreement rate is fine (the model exists
       to add signal beyond rules); but a *zero* disagreement rate
       means the model isn't adding anything and should be audited.
    3. Hard-constraint rejections: how many candidates were thrown
       out, and on what rule. If the rejection rate spikes above
       baseline, the candidate generator has regressed.

    The monitor is stateful across one optimization decision. For
    cross-edition aggregation, persist the JSON dict from
    `summary()` and roll it up in the calling pipeline.
    """

    edition_id: str | None = None
    spread_id: str | None = None
    candidates_seen: int = 0
    candidates_passed: int = 0
    model_scores: list[float] = field(default_factory=list)
    rule_scores: list[float] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    disagreement: dict[str, object] | None = None

    def record(self, candidate_idx: int, ev: CandidateScore) -> None:
        self.candidates_seen += 1
        self.model_scores.append(ev.model_score)
        self.rule_scores.append(ev.rule_score_value)
        if ev.passes:
            self.candidates_passed += 1
        else:
            for v in ev.violations:
                # Bucket rejections by rule name (text before "=")
                key = v.split("=", 1)[0].strip()
                self.rejections[key] = self.rejections.get(key, 0) + 1

    def note_all_rejected(self, total: int) -> None:
        logger.warning(
            "All %d candidates failed hard constraints (edition=%s spread=%s); "
            "rejections=%s",
            total, self.edition_id, self.spread_id, self.rejections,
        )

    def record_disagreement(
        self,
        *,
        model_winner: int,
        rule_winner: int,
        top_k_indices: tuple[int, ...],
    ) -> None:
        self.disagreement = {
            "model_winner": model_winner,
            "rule_winner": rule_winner,
            "agree": model_winner == rule_winner,
            "top_k": list(top_k_indices),
        }
        if model_winner != rule_winner:
            logger.info(
                "Model and rule rankings disagree (edition=%s spread=%s): "
                "model picked #%d, rules picked #%d",
                self.edition_id, self.spread_id, model_winner, rule_winner,
            )

    def summary(self) -> dict[str, object]:
        if not self.model_scores:
            return {
                "edition_id": self.edition_id,
                "spread_id": self.spread_id,
                "candidates_seen": 0,
                "candidates_passed": 0,
            }
        ms = sorted(self.model_scores)
        rs = sorted(self.rule_scores)
        n = len(ms)
        median = ms[n // 2] if n % 2 == 1 else 0.5 * (ms[n // 2 - 1] + ms[n // 2])
        rule_median = rs[n // 2] if n % 2 == 1 else 0.5 * (rs[n // 2 - 1] + rs[n // 2])
        return {
            "edition_id": self.edition_id,
            "spread_id": self.spread_id,
            "candidates_seen": self.candidates_seen,
            "candidates_passed": self.candidates_passed,
            "model_score_min": ms[0],
            "model_score_median": median,
            "model_score_max": ms[-1],
            "model_score_range": ms[-1] - ms[0],
            "rule_score_median": rule_median,
            "rejections": dict(self.rejections),
            "disagreement": self.disagreement,
        }


# ---------------------------------------------------------------------------
# Convenience: ranking helper for offline analysis / tests
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: Iterable[Mapping[str, float]],
) -> list[tuple[int, CandidateScore]]:
    """Return all candidates sorted by (passes, model_score) desc.

    Failing candidates sink to the bottom in their original order.
    Useful for diagnostics and tests.
    """
    items = [(i, evaluate_candidate(f)) for i, f in enumerate(candidates)]
    items.sort(
        key=lambda iv: (
            0 if iv[1].passes else 1,
            -iv[1].model_score,
            -iv[1].rule_score_value,
            iv[0],
        )
    )
    return items


# ---------------------------------------------------------------------------
# Math helper kept local so the module has zero hard dependencies
# beyond the stdlib. Required by ScoringMonitor.summary() in some
# downstream callers that aggregate variances.
# ---------------------------------------------------------------------------

def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)
