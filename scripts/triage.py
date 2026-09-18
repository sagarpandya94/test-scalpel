"""
triage.py
---------
Runs LLM triage over test failures.

Two modes:

  default     Classify failures that no human has triaged yet, and report what
              would be written back. This is the pipeline use: clear the triage
              backlog so builds become eligible for embedding.

  --evaluate  Re-classify failures that DO have human labels and score against
              them. This is the only way to know whether the classifier can be
              trusted with the index.

Usage:
  python scripts/triage.py --evaluate
  python scripts/triage.py --evaluate --verbose     # show every misclassification
  python scripts/triage.py --build BUILD-011        # triage one build's backlog
  python scripts/triage.py --apply                  # write verdicts back to the JSON

Reading the evaluation
----------------------
Accuracy alone is not the number to optimise. The two error directions carry
very different costs (see src/triage.py), so they are reported separately:

  INDEX POISONING   a script or environment failure labelled 'product'. Writes
                    a false coupling into the build history index, which is
                    then retrieved and acted on forever. Unrecoverable.

  SIGNAL LOSS       a product failure labelled script or environment. Loses one
                    real coupling. Recoverable -- it gets learned the next time
                    that code breaks that test.

A classifier with 90% accuracy and a 10% poisoning rate is worse than one with
80% accuracy that abstains instead. Watch the poisoning row first.
"""

import sys
import os
import json
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import load_builds_from_json
from src.triage import (
    classify_failure, get_triage_model, VALID_TYPES, MIN_CONFIDENCE,
)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDS_DEFAULT = os.path.join(BASE, "data", "sample_builds.json")

MAX_WORKERS = 8


def parse_args():
    p = argparse.ArgumentParser(description="LLM triage of test failures.")
    p.add_argument("--data", default=BUILDS_DEFAULT)
    p.add_argument("--evaluate", action="store_true",
                   help="Score against existing human labels instead of triaging the backlog")
    p.add_argument("--build", help="Restrict to one build id")
    p.add_argument("--apply", action="store_true",
                   help="Write verdicts back into the builds JSON (default is dry run)")
    p.add_argument("--verbose", action="store_true", help="Show every disagreement")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    return p.parse_args()


def collect(builds, build_filter, need_labels):
    """Gathers (build, tc) pairs to classify."""
    out = []
    for b in builds:
        if build_filter and b.build_id != build_filter:
            continue
        for tc in b.tc_results:
            if tc.status != "failed":
                continue
            if need_labels and not (tc.triaged and tc.failure_type in VALID_TYPES):
                continue
            if not need_labels and tc.triaged:
                continue
            out.append((b, tc))
    return out


def run_batch(pairs, workers):
    """Classifies pairs concurrently, preserving input order."""
    model = get_triage_model()

    def one(pair):
        build, tc = pair
        return classify_failure(tc, build, model)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, pairs))


# ---------------------------------------------------------------------------
# Evaluation mode
# ---------------------------------------------------------------------------

def evaluate(pairs, verdicts, verbose):
    truth = [tc.failure_type for _, tc in pairs]
    pred = [v.failure_type for v in verdicts]
    n = len(pairs)

    committed = [i for i, p in enumerate(pred) if p != "unknown"]
    abstained = n - len(committed)
    correct = sum(1 for i in committed if pred[i] == truth[i])

    majority = collections.Counter(truth).most_common(1)[0]

    # The two asymmetric error types.
    poisoning = [i for i in committed if truth[i] != "product" and pred[i] == "product"]
    signal_loss = [i for i in committed if truth[i] == "product" and pred[i] != "product"]

    print("=" * 74)
    print("  TRIAGE EVALUATION")
    print("=" * 74)
    print(f"  Failures with human labels:  {n}")
    print(f"  Confidence floor:            {MIN_CONFIDENCE:.2f}")
    print()

    print("COVERAGE")
    print(f"  Committed to a label:  {len(committed):>3} / {n}  ({len(committed)/n:.0%})")
    print(f"  Abstained (unknown):   {abstained:>3} / {n}  ({abstained/n:.0%})  -> routed to a human")
    downgraded = sum(1 for v in verdicts if v.downgraded)
    if downgraded:
        print(f"    of which downgraded by the confidence floor: {downgraded}")
    print()

    print("ACCURACY")
    if committed:
        print(f"  On committed labels:   {correct}/{len(committed)}  ({correct/len(committed):.0%})")
    print(f"  Over all failures:     {correct}/{n}  ({correct/n:.0%})")
    print(f"  Majority-class baseline (always '{majority[0]}'): {majority[1]/n:.0%}")
    print()

    print("ERROR COST  (the rows that decide whether this is safe to automate)")
    print("  " + "-" * 70)
    non_product = sum(1 for t in truth if t != "product")
    products = sum(1 for t in truth if t == "product")
    print(f"  INDEX POISONING   {len(poisoning):>3} / {non_product:<3} non-product failures "
          f"labelled 'product'   ({len(poisoning)/non_product:.0%})" if non_product else "")
    print(f"  SIGNAL LOSS       {len(signal_loss):>3} / {products:<3} product failures "
          f"labelled otherwise    ({len(signal_loss)/products:.0%})" if products else "")
    print("  " + "-" * 70)
    print()

    print("CONFUSION MATRIX  (rows = truth, columns = predicted)")
    labels = list(VALID_TYPES) + ["unknown"]
    matrix = collections.Counter(zip(truth, pred))
    header = "".join(f"{l[:7]:>12}" for l in labels)
    print(f"  {'':<14}{header}")
    for t in VALID_TYPES:
        row = "".join(f"{matrix.get((t, p), 0):>12}" for p in labels)
        print(f"  {t:<14}{row}")
    print()

    print("PER-CLASS")
    print(f"  {'CLASS':<14}{'RECALL':>9}{'PRECISION':>12}{'SUPPORT':>10}")
    print("  " + "-" * 45)
    for t in VALID_TYPES:
        support = sum(1 for x in truth if x == t)
        tp = sum(1 for i in range(n) if truth[i] == t and pred[i] == t)
        predicted_t = sum(1 for p in pred if p == t)
        rec = tp / support if support else float("nan")
        prec = tp / predicted_t if predicted_t else float("nan")
        print(f"  {t:<14}{rec:>9.2f}{prec:>12.2f}{support:>10}")
    print()

    if verbose:
        disagreements = [i for i in range(n) if pred[i] != truth[i]]
        if disagreements:
            print("DISAGREEMENTS")
            for i in disagreements:
                b, tc = pairs[i]
                v = verdicts[i]
                kind = ("POISONING" if i in poisoning else
                        "signal loss" if i in signal_loss else
                        "abstained" if pred[i] == "unknown" else "misread")
                print(f"  [{kind}] {b.build_id} {tc.tc_id}: truth={truth[i]} "
                      f"pred={pred[i]} (model said {v.raw_type} @ {v.raw_confidence:.2f})")
                print(f"      error:  {(tc.error_message or '')[:100]}")
                print(f"      reason: {v.reasoning}")
            print()

    verdict = (
        "SAFE TO AUTOMATE" if not poisoning
        else "NOT SAFE TO AUTOMATE UNATTENDED"
    )
    print(f"  Verdict: {verdict}")
    if poisoning:
        print("  Poisoning errors write false couplings that are never detected or")
        print("  corrected. Raise MIN_CONFIDENCE or keep a human in the loop for")
        print("  anything the model labels 'product'.")
    else:
        print("  No non-product failure was labelled 'product'. The expensive error")
        print("  direction did not occur on this sample.")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Backlog mode
# ---------------------------------------------------------------------------

def report_backlog(pairs, verdicts, args):
    if not pairs:
        print("No untriaged failures. Nothing to do.")
        return

    print("=" * 74)
    print(f"  TRIAGE BACKLOG  ({len(pairs)} untriaged failure(s))")
    print("=" * 74)
    print()

    counts = collections.Counter(v.failure_type for v in verdicts)
    for (b, tc), v in zip(pairs, verdicts):
        flag = "-> human" if v.abstained else "auto"
        print(f"  {b.build_id} {tc.tc_id:<8} {v.failure_type:<12} "
              f"conf {v.raw_confidence:.2f}  [{flag}]")
        print(f"      {v.reasoning}")
    print()
    print(f"  {dict(counts)}")
    print()

    if args.apply:
        apply_verdicts(args.data, pairs, verdicts)
        print(f"  Written back to {args.data}.")
        print("  Verdicts marked triaged=True only where the model did not abstain.")
    else:
        print("  Dry run. Re-run with --apply to write these back.")
    print("=" * 74)


def apply_verdicts(path, pairs, verdicts):
    """
    Writes verdicts back into the builds JSON.

    Abstentions are written as failure_type='unknown' with triaged=False, so
    they stay out of the index and remain visible as outstanding human work.
    Only confident verdicts set triaged=True.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    index = {(b.build_id, tc.tc_id): v for (b, tc), v in zip(pairs, verdicts)}

    for build in raw:
        for result in build["tc_results"]:
            key = (build["build_id"], result["tc_id"])
            if key not in index:
                continue
            v = index[key]
            result["failure_type"] = v.failure_type
            result["triaged"] = not v.abstained
            result["triage_source"] = "llm"
            result["triage_confidence"] = round(v.raw_confidence, 2)
            result["triage_reasoning"] = v.reasoning

    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
        f.write("\n")


def main():
    args = parse_args()
    # skip_untriaged=False: the untriaged builds are the entire point of the
    # backlog mode, and the default loader would filter them out.
    builds = load_builds_from_json(args.data, skip_untriaged=False)

    pairs = collect(builds, args.build, need_labels=args.evaluate)
    if not pairs:
        print("Nothing to classify with the given filters.")
        return

    if args.evaluate:
        print(f"Classifying {len(pairs)} labelled failures with {args.workers} workers...\n")

    verdicts = run_batch(pairs, args.workers)

    if args.evaluate:
        evaluate(pairs, verdicts, args.verbose)
    else:
        report_backlog(pairs, verdicts, args)


if __name__ == "__main__":
    main()
