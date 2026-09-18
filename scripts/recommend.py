"""
recommend.py
------------
The point of the whole system: given a build that has not run yet, print the
test cases worth running and why.

Usage:
  python scripts/recommend.py
  python scripts/recommend.py --data data/sample_new_build.json
  python scripts/recommend.py --top 10 --min-score 0.2
  python scripts/recommend.py --release              # forces full regression
  python scripts/recommend.py --json                 # machine-readable, for CI

The --json form is the one a pipeline consumes: it emits the ranked TC ids so a
runner can feed them straight to the test framework. The human form is for an
SDET deciding whether to trust the selection, which is why it leads with
evidence rather than with scores.
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import (
    load_build_query_from_json,
    retrieve_similar_builds,
    retrieve_relevant_test_cases,
)
from src.ranker import rank
from src.policy import decide_strategy
from src.chunking import build_query_to_text

DATA_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_new_build.json"
)

CONFIDENCE_MARK = {"high": "HIGH  ", "medium": "MEDIUM", "low": "LOW   "}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recommend which test cases to run for an incoming build."
    )
    parser.add_argument("--data", default=DATA_DEFAULT, help="Path to the incoming build JSON")
    parser.add_argument("--builds-k", type=int, default=5, help="How many similar builds to retrieve")
    parser.add_argument("--tc-k", type=int, default=15, help="How many test cases to retrieve semantically")
    parser.add_argument("--top", type=int, default=12, help="How many recommendations to show")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop recommendations below this score")
    parser.add_argument("--release", action="store_true", help="Mark as a release candidate (forces full regression)")
    parser.add_argument("--full", action="store_true", help="Force full regression for this run")
    parser.add_argument("--same-service-only", action="store_true",
                        help="Restrict semantic matches to services this build touched")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON instead of a report")
    return parser.parse_args()


def emit_json(build, strategy, ranked):
    """Machine-readable output for CI consumption."""
    payload = {
        "build_id": build.build_id,
        "build_number": build.build_number,
        "strategy": {
            "mode": strategy.mode,
            "reason": strategy.reason,
            "run_suggested_first": strategy.run_suggested_first,
        },
        "retrieval": {
            "neighbour_builds": ranked.neighbour_build_ids,
            "best_neighbour_similarity": ranked.best_neighbour_similarity,
            "quality": ranked.retrieval_quality,
        },
        "warnings": ranked.warnings,
        "suggested_tc_ids": [r.tc_id for r in ranked.recommendations],
        "recommendations": [
            {
                "tc_id": r.tc_id,
                "title": r.title,
                "score": r.score,
                "confidence": r.confidence,
                "signals": r.signals,
                "history_score": r.history_score,
                "semantic_score": r.semantic_score,
                "failed_in_builds": r.failed_in_builds,
                "evidence": r.evidence,
            }
            for r in ranked.recommendations
        ],
    }
    print(json.dumps(payload, indent=2))


def emit_report(build, strategy, ranked):
    """Human-readable report."""
    line = "=" * 74

    print(line)
    print(f"  TEST-SCALPEL RECOMMENDATION  |  {build.build_id}  (build #{build.build_number})")
    print(line)
    print()

    print("INCOMING BUILD")
    for raw_line in build_query_to_text(build).splitlines()[1:]:
        print(f"  {raw_line}")
    print()
    for mr in build.mrs:
        print(f"  {mr.mr_id}  {mr.title}")
    print()

    print("STRATEGY")
    print(f"  Mode:   {strategy.mode.upper().replace('_', ' ')}")
    print(f"  Reason: {strategy.reason}")
    print()

    print("RETRIEVAL")
    print(f"  Neighbour builds:  {', '.join(ranked.neighbour_build_ids) or 'none'}")
    print(f"  Best similarity:   {ranked.best_neighbour_similarity:.3f} ({ranked.retrieval_quality})")
    print()

    if ranked.warnings:
        print("WARNINGS")
        for warning in ranked.warnings:
            print(f"  ! {warning}")
        print()

    if not ranked.recommendations:
        print("  No recommendations produced. Run the full suite.")
        print(line)
        return

    if strategy.is_full_regression:
        heading = f"EXECUTION ORDER  (full suite runs; these {len(ranked.recommendations)} go first)"
    else:
        heading = f"SELECTED TEST CASES  ({len(ranked.recommendations)} of the suite)"

    print(heading)
    print("  " + "-" * 70)
    print(f"  {'#':>2}  {'TC':<8} {'SCORE':>6}  {'CONF':<7} {'SIGNAL':<17} TITLE")
    print("  " + "-" * 70)

    for i, r in enumerate(ranked.recommendations, 1):
        mark = CONFIDENCE_MARK.get(r.confidence, r.confidence)
        title = r.title[:30] if r.title else "(title not indexed)"
        print(f"  {i:>2}. {r.tc_id:<8} {r.score:>6.3f}  {mark:<7} {r.signals:<17} {title}")

    print()
    print("EVIDENCE")
    for r in ranked.recommendations:
        print(f"  {r.tc_id}  (history {r.history_score:.2f} / semantic {r.semantic_score:.2f})")
        for item in r.evidence:
            print(f"      - {item}")
    print()

    high = sum(1 for r in ranked.recommendations if r.confidence == "high")
    cold = sum(1 for r in ranked.recommendations if r.is_cold_start)
    print("SUMMARY")
    print(f"  {len(ranked.recommendations)} recommended  |  {high} high confidence  |  {cold} cold-start (no failure history)")
    if not strategy.is_full_regression:
        print(f"  Run: {' '.join(r.tc_id for r in ranked.recommendations)}")
    print(line)


def main():
    args = parse_args()

    build = load_build_query_from_json(args.data)

    strategy = decide_strategy(
        build_number=build.build_number,
        is_release=args.release,
        force_full=args.full,
    )

    similar_builds = retrieve_similar_builds(build, k=args.builds_k)
    relevant_tcs = retrieve_relevant_test_cases(
        build,
        k=args.tc_k,
        restrict_to_touched_services=args.same_service_only,
    )

    ranked = rank(
        similar_builds,
        relevant_tcs,
        top_n=args.top,
        min_score=args.min_score,
    )

    if args.as_json:
        emit_json(build, strategy, ranked)
    else:
        emit_report(build, strategy, ranked)


if __name__ == "__main__":
    main()
