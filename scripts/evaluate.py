"""
evaluate.py
-----------
Leave-one-out evaluation of the recommender against the indexed build history.

Without this, every claim the system makes is unfalsifiable. A test selector
that misses real failures is worse than no selector at all -- it converts a
slow-but-honest pipeline into a fast one that lies -- so the number that
matters most here is recall, not precision. Precision saves minutes. A recall
miss ships a defect.

Method
------
For each historical build:
  1. Strip its results, leaving only the files it touched.
  2. Retrieve neighbours with that build EXCLUDED from its own index.
     Without the exclusion a build retrieves itself at ~0.92 similarity,
     hands back its own answer sheet, and every metric reads perfect.
  3. Rank, take the top k.
  4. Compare against the product failures that actually occurred.

Baselines
---------
Recall@k in isolation means nothing -- selecting the entire suite scores 1.0.
Every run is therefore reported against two reference points:

  RANDOM        k test cases drawn at random. The floor. Anything not clearly
                above this is noise dressed up as intelligence.

  SAME-SERVICE  every test case owned by a service this build touched. This is
                the naive folder/tag-based selection the README dismisses, and
                it is the honest competitor: it needs no embeddings, no vector
                store and no API key. If Test-Scalpel cannot beat it on recall
                at a comparable suite size, the RAG machinery is not paying for
                itself.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --k 5 10 15
  python scripts/evaluate.py --verbose      # per-build breakdown with misses
"""

import sys
import os
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import load_builds_from_json, load_test_cases_from_json
from src.retriever import (
    build_document_to_query,
    retrieve_similar_builds,
    retrieve_relevant_test_cases,
)
from src.ranker import rank

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDS_DEFAULT = os.path.join(BASE, "data", "sample_builds.json")
TCS_DEFAULT = os.path.join(BASE, "data", "sample_test_cases.json")

RANDOM_SEED = 42          # fixed so the baseline is reproducible across runs
RANDOM_TRIALS = 200       # averaged, so one lucky draw cannot flatter the floor


def recall(selected: list[str], actual: set[str]) -> float:
    """Fraction of real product failures that the selection caught."""
    if not actual:
        return float("nan")
    return len(set(selected) & actual) / len(actual)


def precision(selected: list[str], actual: set[str]) -> float:
    """Fraction of the selection that turned out to be a real failure."""
    if not selected:
        return 0.0
    return len(set(selected) & actual) / len(selected)


def random_baseline_recall(all_tc_ids: list[str], actual: set[str], k: int, rng) -> float:
    """Mean recall of k randomly chosen test cases, over RANDOM_TRIALS draws."""
    if not actual:
        return float("nan")
    total = 0.0
    for _ in range(RANDOM_TRIALS):
        picked = rng.sample(all_tc_ids, min(k, len(all_tc_ids)))
        total += recall(picked, actual)
    return total / RANDOM_TRIALS


def same_service_baseline(build, tc_by_service: dict[str, list[str]]) -> list[str]:
    """Every test case owned by a service this build touched."""
    selected = []
    for repo in build.all_repos:
        selected.extend(tc_by_service.get(repo, []))
    return sorted(set(selected))


def parse_args():
    parser = argparse.ArgumentParser(description="Leave-one-out evaluation of the recommender.")
    parser.add_argument("--builds", default=BUILDS_DEFAULT)
    parser.add_argument("--test-cases", default=TCS_DEFAULT)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15],
                        help="Cut-offs to evaluate recall/precision at")
    parser.add_argument("--builds-k", type=int, default=8, help="Neighbours to retrieve per query")
    parser.add_argument("--tc-k", type=int, default=30, help="Test cases to retrieve semantically")
    parser.add_argument("--verbose", action="store_true", help="Per-build breakdown including misses")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(RANDOM_SEED)

    builds = load_builds_from_json(args.builds)
    test_cases = load_test_cases_from_json(args.test_cases)

    all_tc_ids = [tc.tc_id for tc in test_cases]
    tc_by_service: dict[str, list[str]] = {}
    for tc in test_cases:
        tc_by_service.setdefault(tc.service, []).append(tc.tc_id)

    service_of: dict[str, str] = {tc.tc_id: tc.service for tc in test_cases}

    # How often each TC failed across the whole history. Used only to report
    # how thin the history signal is -- a suite where everything failed once
    # cannot support frequency-based ranking, and the caveat should say so
    # with the real number rather than a remembered one.
    tc_failure_counts: dict[str, int] = {}
    for b in builds:
        for tc in b.product_failures:
            tc_failure_counts[tc.tc_id] = tc_failure_counts.get(tc.tc_id, 0) + 1

    evaluable = [b for b in builds if b.product_failures]
    skipped = len(builds) - len(evaluable)

    if not args.as_json:
        print("=" * 74)
        print("  LEAVE-ONE-OUT EVALUATION")
        print("=" * 74)
        print(f"  Builds indexed:        {len(builds)}")
        print(f"  Builds with failures:  {len(evaluable)}  ({skipped} clean build(s) skipped)")
        print(f"  Suite size:            {len(all_tc_ids)} test cases")
        print(f"  Cut-offs:              {', '.join('k=' + str(k) for k in args.k)}")
        print()

    per_build = []

    for build in evaluable:
        actual = {tc.tc_id for tc in build.product_failures}
        query = build_document_to_query(build)

        neighbours = retrieve_similar_builds(
            query, k=args.builds_k, exclude_build_ids=[build.build_id]
        )
        relevant = retrieve_relevant_test_cases(query, k=args.tc_k)
        ranked = rank(neighbours, relevant, top_n=None)
        ordered = [r.tc_id for r in ranked.recommendations]

        service_pick = same_service_baseline(build, tc_by_service)

        # A failure is 'cross-service' when the test belongs to a service this
        # build never touched. Those are the ones no folder-based rule can
        # reach, so they are scored separately below.
        touched = set(build.all_repos)
        cross = {tc for tc in actual if service_of.get(tc) not in touched}

        record = {
            "build_id": build.build_id,
            "actual_failures": sorted(actual),
            "cross_service_failures": sorted(cross),
            "neighbours": ranked.neighbour_build_ids,
            "scalpel": {},
            "random": {},
            "same_service": {
                "k": len(service_pick),
                "recall": recall(service_pick, actual),
                "precision": precision(service_pick, actual),
            },
        }

        for k in args.k:
            top_k = ordered[:k]
            record["scalpel"][k] = {
                "recall": recall(top_k, actual),
                "precision": precision(top_k, actual),
                "missed": sorted(actual - set(top_k)),
            }
            record["random"][k] = {
                "recall": random_baseline_recall(all_tc_ids, actual, k, rng)
            }

        # Per-failure hit/miss, so the cross vs same split can be aggregated
        # over individual failures rather than averaged over builds. Averaging
        # per build would let a build with one cross-service failure outweigh a
        # build with four.
        record["hits"] = [
            {
                "tc_id": tc,
                "cross_service": tc in cross,
                "scalpel_hit": {k: tc in ordered[:k] for k in args.k},
                "baseline_hit": tc in set(service_pick),
            }
            for tc in sorted(actual)
        ]

        per_build.append(record)

        if args.verbose and not args.as_json:
            print(f"  {build.build_id}  actual failures: {', '.join(sorted(actual))}")
            print(f"    neighbours: {', '.join(record['neighbours'])}")
            for k in args.k:
                s = record["scalpel"][k]
                miss = f"  missed: {', '.join(s['missed'])}" if s["missed"] else ""
                print(f"    k={k:<3} recall {s['recall']:.2f}  precision {s['precision']:.2f}{miss}")
            print()

    # --- Aggregate ---
    summary = {"k": {}, "same_service": {}}

    for k in args.k:
        summary["k"][k] = {
            "scalpel_recall": sum(r["scalpel"][k]["recall"] for r in per_build) / len(per_build),
            "scalpel_precision": sum(r["scalpel"][k]["precision"] for r in per_build) / len(per_build),
            "random_recall": sum(r["random"][k]["recall"] for r in per_build) / len(per_build),
            "perfect_builds": sum(1 for r in per_build if r["scalpel"][k]["recall"] == 1.0),
            "suite_fraction": min(k, len(all_tc_ids)) / len(all_tc_ids),
        }

    # Cross vs same split, counted per failure across every build.
    best_k = max(args.k)
    all_hits = [h for r in per_build for h in r["hits"]]
    cross_hits = [h for h in all_hits if h["cross_service"]]
    same_hits = [h for h in all_hits if not h["cross_service"]]

    def _scalpel_rate(hits, k):
        if not hits:
            return float("nan")
        return sum(1 for h in hits if h["scalpel_hit"][k]) / len(hits)

    def _baseline_rate(hits):
        if not hits:
            return float("nan")
        return sum(1 for h in hits if h["baseline_hit"]) / len(hits)

    summary["split"] = {
        "cross_total": len(cross_hits),
        "same_total": len(same_hits),
        "cross_baseline": _baseline_rate(cross_hits),
        "same_baseline": _baseline_rate(same_hits),
        "cross_scalpel": {k: _scalpel_rate(cross_hits, k) for k in args.k},
        "same_scalpel": {k: _scalpel_rate(same_hits, k) for k in args.k},
    }

    summary["same_service"] = {
        "recall": sum(r["same_service"]["recall"] for r in per_build) / len(per_build),
        "precision": sum(r["same_service"]["precision"] for r in per_build) / len(per_build),
        "avg_k": sum(r["same_service"]["k"] for r in per_build) / len(per_build),
        "suite_fraction": (
            sum(r["same_service"]["k"] for r in per_build) / len(per_build) / len(all_tc_ids)
        ),
    }

    if args.as_json:
        print(json.dumps({"per_build": per_build, "summary": summary}, indent=2))
        return

    print("RESULTS  (averaged over builds with at least one product failure)")
    print("  " + "-" * 70)
    print(f"  {'SELECTION':<22} {'SUITE RUN':>10} {'RECALL':>8} {'PRECISION':>10} {'PERFECT':>9}")
    print("  " + "-" * 70)

    for k in args.k:
        s = summary["k"][k]
        print(
            f"  {'Test-Scalpel k=' + str(k):<22} "
            f"{s['suite_fraction'] * 100:>9.0f}% "
            f"{s['scalpel_recall']:>8.2f} "
            f"{s['scalpel_precision']:>10.2f} "
            f"{str(s['perfect_builds']) + '/' + str(len(per_build)):>9}"
        )

    print("  " + "-" * 70)

    for k in args.k:
        s = summary["k"][k]
        print(
            f"  {'Random k=' + str(k):<22} "
            f"{s['suite_fraction'] * 100:>9.0f}% "
            f"{s['random_recall']:>8.2f} "
            f"{'-':>10} {'-':>9}"
        )

    ss = summary["same_service"]
    print("  " + "-" * 70)
    print(
        f"  {'Same-service (naive)':<22} "
        f"{ss['suite_fraction'] * 100:>9.0f}% "
        f"{ss['recall']:>8.2f} "
        f"{ss['precision']:>10.2f} "
        f"{'-':>9}"
    )
    print("  " + "-" * 70)
    print()

    # --- Interpretation ---
    print("READING THIS")
    best_k = max(args.k)
    s = summary["k"][best_k]
    lift = s["scalpel_recall"] - s["random_recall"]
    print(f"  At k={best_k} the selector runs {s['suite_fraction'] * 100:.0f}% of the suite "
          f"and catches {s['scalpel_recall'] * 100:.0f}% of real failures.")
    print(f"  That is {lift * 100:+.0f} points of recall over random selection at the same cost.")

    if s["scalpel_recall"] >= ss["recall"] and s["suite_fraction"] <= ss["suite_fraction"]:
        print("  It beats the naive same-service baseline on recall at equal or lower cost.")
    elif s["scalpel_recall"] < ss["recall"]:
        print(f"  WARNING: the naive same-service baseline reaches {ss['recall']:.2f} recall "
              f"vs Test-Scalpel's {s['scalpel_recall']:.2f}.")
        print("  On this dataset the embedding machinery is not yet earning its keep.")

    # --- The split that decides whether retrieval is worth its cost ---------
    #
    # Aggregate recall hides the only comparison that matters. A same-service
    # failure is catchable by grepping a folder name; a cross-service one is
    # not, and it is the entire reason this project embeds anything. Reporting
    # them separately is what stops a good aggregate number from concealing
    # total failure on the hard half.
    split = summary["split"]
    if split["cross_total"]:
        k_cols = "".join(f"{'k=' + str(k):>8}" for k in args.k)
        print()
        print("CROSS-SERVICE vs SAME-SERVICE FAILURES  (recall, counted per failure)")
        print("  " + "-" * 70)
        print(f"  {'FAILURE TYPE':<31} {'COUNT':>6} {'SAME-SVC':>10}{k_cols}")
        print("  " + "-" * 70)
        print(
            f"  {'Same-service (folder-findable)':<31} {split['same_total']:>6} "
            f"{split['same_baseline']:>10.2f}"
            + "".join(f"{split['same_scalpel'][k]:>8.2f}" for k in args.k)
        )
        print(
            f"  {'Cross-service (needs retrieval)':<31} {split['cross_total']:>6} "
            f"{split['cross_baseline']:>10.2f}"
            + "".join(f"{split['cross_scalpel'][k]:>8.2f}" for k in args.k)
        )
        print("  " + "-" * 70)
        print()
        print(f"  The naive baseline catches {split['cross_baseline'] * 100:.0f}% of cross-service failures. It cannot do")
        print("  better: a test owned by a service the build never touched is outside the set")
        print("  it considers at all. Test-Scalpel reaches "
              f"{split['cross_scalpel'][best_k] * 100:.0f}% at k={best_k}.")
        print()
        print("  That column is the whole argument for the vector store. On same-service")
        print("  failures both approaches saturate, and the embeddings are an expensive way")
        print("  to reach a result folder-based tagging already gives you for free.")

    print()
    n_failing_tcs = len({tc for r in per_build for tc in r["actual_failures"]})
    once = sum(1 for tc, c in tc_failure_counts.items() if c == 1)
    print(f"  Caveat: {len(builds)} builds and {n_failing_tcs} distinct failing test cases is a")
    print(f"  small sample, and {once} of those failed exactly once, so the history signal is")
    print("  thin by construction. Treat these numbers as directional, not as a measurement.")
    print("  The data is synthetic -- see data/README.md for what was put in on purpose.")
    print("=" * 74)


if __name__ == "__main__":
    main()
