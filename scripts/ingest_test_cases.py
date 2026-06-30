"""
ingest_test_cases.py
--------------------
Ingestion script for TestRail test cases.

Run this:
  - Once initially (or with --reset) to build the test case index
  - Whenever test cases are added, updated, or retired in TestRail

The test case index is more stable than the build index — you don't need
to run this after every build, only when the test suite itself changes.

Usage:
  python scripts/ingest_test_cases.py
  python scripts/ingest_test_cases.py --reset
  python scripts/ingest_test_cases.py --data path/to/test_cases.json
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import load_test_cases_from_json, index_test_cases, TC_STORE_PATH

DATA_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_test_cases.json"
)


def main():
    parser = argparse.ArgumentParser(description="Ingest TestRail test cases into the test_cases vector index.")
    parser.add_argument("--data", default=DATA_DEFAULT, help="Path to test cases JSON file")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the index from scratch")
    args = parser.parse_args()

    print("=" * 60)
    print("TEST CASE INGESTION")
    print("=" * 60)
    print(f"  Source:  {args.data}")
    print(f"  Store:   {TC_STORE_PATH}")
    print(f"  Mode:    {'RESET (full rebuild)' if args.reset else 'INCREMENTAL (append/upsert)'}")
    print()

    # --- Load ---
    print("Step 1 / 3  Loading test cases from JSON...")
    test_cases = load_test_cases_from_json(args.data)
    print(f"  Loaded: {len(test_cases)} test case(s)")

    # --- Preview ---
    print()
    print("Step 2 / 3  Preview (first 3 test cases):")
    from src.chunking import test_case_to_text
    for tc in test_cases[:3]:
        print()
        print(f"  {'─' * 50}")
        print(f"  {test_case_to_text(tc)}")
    print(f"  {'─' * 50}")

    # --- Index ---
    print()
    print("Step 3 / 3  Embedding and indexing...")
    vector_store = index_test_cases(test_cases, reset=args.reset)

    # --- Verify ---
    count = vector_store._collection.count()
    print()
    print("=" * 60)
    print(f"  Done. {count} document(s) in test_cases index.")
    print("=" * 60)


if __name__ == "__main__":
    main()
