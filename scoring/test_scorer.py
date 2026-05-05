"""Validation runs for technical_scorer.py.

This is not a pytest module — it's a manually-runnable validator
that exercises the scorer against the 200 graded training spreads
and reports:

    1. Numeric equivalence with sklearn Ridge (sanity / regression).
    2. Out-of-fold R^2 against held-out grades (matches #344).
    3. Per-spread score distribution and rule-vs-model
       disagreement counts on a sample of editions.
    4. A toy candidate-selection run that demonstrates hybrid
       scoring + monitor output.

Run with the offscroll venv:

    /home/modus/offscroll/.venv/bin/python -m scoring.test_scorer

(or import the module from a notebook).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Resolve sibling import without relying on package install.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from scoring.technical_scorer import (  # noqa: E402
    FEATURE_ORDER,
    ScoringMonitor,
    evaluate_candidate,
    rank_candidates,
    rule_score,
    score_spread,
    select_best_candidate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)

TRAINING_FEATURES_CSV = "/home/modus/offscroll/training/features/features-004.csv"


def _load_features() -> list[dict]:
    """Load features-004.csv as a list of dicts (one per spread)."""
    import csv

    rows: list[dict] = []
    with open(TRAINING_FEATURES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {"spread_id": row["spread_id"], "split": row["split"]}
            for k in [
                "technical_grade", "style_grade",
            ]:
                rec[k] = int(row[k])
            for f_name in FEATURE_ORDER:
                v = row.get(f_name, "0")
                if v == "" or v is None or v == "NA":
                    rec[f_name] = 0.0
                else:
                    rec[f_name] = float(v)
            rows.append(rec)
    return rows


def test_sklearn_equivalence(rows: list[dict]) -> None:
    """Numerical regression: scorer must match sklearn within 1e-9."""
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        print("  SKIP (sklearn not available)")
        return

    train_rows = [r for r in rows if r["split"] == "train"]
    X_train = [[r[f] for f in FEATURE_ORDER] for r in train_rows]
    y_train = [r["technical_grade"] for r in train_rows]
    ridge = Ridge(alpha=0.1).fit(X_train, y_train)

    max_delta = 0.0
    for r in rows:
        sk_pred = float(
            ridge.predict([[r[f] for f in FEATURE_ORDER]])[0]
        )
        ours = score_spread(r)
        max_delta = max(max_delta, abs(sk_pred - ours))

    print(f"  sklearn-vs-constants max abs delta: {max_delta:.2e}")
    assert max_delta < 1e-9, f"Scorer drifted from sklearn by {max_delta}"
    print("  PASS — constants reproduce sklearn within tolerance")


def test_held_out_r2(rows: list[dict]) -> None:
    """Held-out R^2 should match the #344 single-split number (0.413)."""
    val_rows = [r for r in rows if r["split"] == "val"]
    y_true = [r["technical_grade"] for r in val_rows]
    y_pred = [score_spread(r) for r in val_rows]

    n = len(y_true)
    mean_y = sum(y_true) / n
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    ss_tot = sum((y - mean_y) ** 2 for y in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    mae = sum(abs(y - p) for y, p in zip(y_true, y_pred)) / n
    print(f"  Val R^2 = {r2:.4f}  (#344 reported 0.4128)")
    print(f"  Val MAE = {mae:.4f}  (#344 reported 1.1364)")
    assert abs(r2 - 0.4128) < 1e-3, f"Val R^2 drift: {r2}"
    assert abs(mae - 1.1364) < 1e-3, f"Val MAE drift: {mae}"
    print("  PASS")


def test_score_distribution(rows: list[dict]) -> None:
    """Report distribution of model scores on the 200 spreads."""
    scores = sorted(score_spread(r) for r in rows)
    rule_scores = sorted(rule_score(r) for r in rows)

    n = len(scores)

    def pct(p):
        return scores[min(n - 1, int(p * n))]

    print(f"  n = {n}")
    print(f"  model score: min={scores[0]:.2f}  p10={pct(0.10):.2f}  "
          f"median={pct(0.50):.2f}  p90={pct(0.90):.2f}  max={scores[-1]:.2f}")
    print(f"  rule  score: min={rule_scores[0]:.2f}  median={rule_scores[n//2]:.2f}  "
          f"max={rule_scores[-1]:.2f}")
    print("  (Distribution snapshot for production drift comparison.)")


def test_model_vs_rule_ranking(rows: list[dict]) -> None:
    """For each edition, treat all spreads in that edition as
    candidates, rank by both model and pure-rule scores, and count
    how often the top-1 differs.
    """
    by_edition: dict[str, list[dict]] = {}
    for r in rows:
        # spread_id is "s-{nnn}-{idx}"; edition = "training-{nnn}".
        eid = r["spread_id"].split("-")[1]
        by_edition.setdefault(eid, []).append(r)

    n_editions = 0
    n_disagree = 0
    n_eligible = 0
    grade_lift_when_disagree: list[float] = []

    for eid, spreads in by_edition.items():
        if len(spreads) < 2:
            continue
        n_eligible += 1
        n_editions += 1

        evals = [evaluate_candidate(s) for s in spreads]
        # Best-by-model vs best-by-rule (skip filtering on hard
        # constraints since this exercises ranking, not gating)
        model_winner = max(
            range(len(spreads)), key=lambda i: evals[i].model_score
        )
        rule_winner = max(
            range(len(spreads)), key=lambda i: evals[i].rule_score_value
        )
        if model_winner != rule_winner:
            n_disagree += 1
            # Did the model's pick have a better actual grade?
            lift = (
                spreads[model_winner]["technical_grade"]
                - spreads[rule_winner]["technical_grade"]
            )
            grade_lift_when_disagree.append(lift)

    print(f"  editions with >=2 spreads: {n_eligible}")
    print(
        f"  model and rule top-1 disagree: {n_disagree}/{n_eligible} "
        f"({100*n_disagree/max(1,n_eligible):.0f}%)"
    )
    if grade_lift_when_disagree:
        avg_lift = sum(grade_lift_when_disagree) / len(grade_lift_when_disagree)
        better = sum(1 for x in grade_lift_when_disagree if x > 0)
        worse = sum(1 for x in grade_lift_when_disagree if x < 0)
        tie = sum(1 for x in grade_lift_when_disagree if x == 0)
        print(
            f"  on disagreements: model picked higher-graded spread "
            f"{better} times, lower {worse}, tied {tie}; "
            f"avg grade lift = {avg_lift:+.2f}"
        )


def test_hard_constraint_filter(rows: list[dict]) -> None:
    """Hard constraints should reject spreads with d2_orphans>0."""
    rejected_orphans = sum(
        1 for r in rows if (r.get("d2_orphans") or 0) > 0
    )
    rejected_widows = sum(
        1 for r in rows if (r.get("d2_widows") or 0) > 0
    )
    print(f"  spreads with orphans>0: {rejected_orphans}/{len(rows)}")
    print(f"  spreads with widows>0:  {rejected_widows}/{len(rows)} "
          f"(should be 0 in features-004 — bug fixed in #377)")

    n_rejected = sum(
        1 for r in rows
        if evaluate_candidate(r).violations
    )
    print(f"  total spreads rejected by hard constraints: "
          f"{n_rejected}/{len(rows)}")


def test_select_best_candidate_demo(rows: list[dict]) -> None:
    """Exercise select_best_candidate on a small synthetic candidate
    set drawn from a single edition. Demonstrates monitor output.
    """
    # Pick the edition with the most spreads (good candidate diversity)
    by_edition: dict[str, list[dict]] = {}
    for r in rows:
        eid = r["spread_id"].split("-")[1]
        by_edition.setdefault(eid, []).append(r)
    eid, spreads = max(by_edition.items(), key=lambda kv: len(kv[1]))
    print(f"  using edition training-{eid} with {len(spreads)} spreads as "
          f"candidate set")

    monitor = ScoringMonitor(edition_id=f"training-{eid}", spread_id="demo")
    winner_idx = select_best_candidate(spreads, top_k=3, monitor=monitor)

    if winner_idx is None:
        print("  WARNING: every candidate was rejected by hard constraints")
        return

    winner = spreads[winner_idx]
    summary = monitor.summary()
    print(f"  winner: {winner['spread_id']} "
          f"(actual grade={winner['technical_grade']}, "
          f"model={score_spread(winner):.2f})")
    print(f"  monitor summary: {json.dumps(summary, indent=2, default=str)}")


def main() -> int:
    rows = _load_features()
    print(f"Loaded {len(rows)} spreads from {TRAINING_FEATURES_CSV}\n")

    print("=== 1. sklearn equivalence ===")
    test_sklearn_equivalence(rows)
    print()

    print("=== 2. Held-out R^2 / MAE (regression-test against #344) ===")
    test_held_out_r2(rows)
    print()

    print("=== 3. Score distribution ===")
    test_score_distribution(rows)
    print()

    print("=== 4. Model-vs-rule top-1 disagreement ===")
    test_model_vs_rule_ranking(rows)
    print()

    print("=== 5. Hard-constraint filtering ===")
    test_hard_constraint_filter(rows)
    print()

    print("=== 6. select_best_candidate demo ===")
    test_select_best_candidate_demo(rows)
    print()

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
