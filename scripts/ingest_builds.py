"""
ingest_builds.py
----------------
Ingestion script for build history.

Run this:
  - Once initially with --reset to backfill all historical builds
  - After every build cycle completes and SDETs have triaged the failures

Usage:
  python scripts/ingest_builds.py
  python scripts/ingest_builds.py --reset         # wipe and rebuild from scratch
  python scripts/ingest_builds.py --data path/to/builds.json
"""

import sys
import os
import argparse

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexer import load_builds_from_json, index_builds, BUILD_STORE_PATH

DATA_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample_builds.json"
)


def main():
    parser = argparse.ArgumentParser(description="Ingest build history into the build_history vector index.")
    parser.add_argument("--data", default=DATA_DEFAULT, help="Path to builds JSON file")
    parser.add_argument("--reset", action="store_true", help="Wipe and rebuild the index from scratch")
    args = parser.parse_args()

    print("=" * 60)
    print("BUILD HISTORY INGESTION")
    print("=" * 60)
    print(f"  Source:  {args.data}")
    print(f"  Store:   {BUILD_STORE_PATH}")
    print(f"  Mode:    {'RESET (full rebuild)' if args.reset else 'INCREMENTAL (append/upsert)'}")
    print()

    # --- Load ---
    print("Step 1 / 3  Loading builds from JSON...")
    builds = load_builds_from_json(args.data)
    print(f"  Ready to embed: {len(builds)} build(s)")
    if not builds:
        print("  Nothing to index. Exiting.")
        return

    # --- Preview ---
    print()
    print("Step 2 / 3  Preview (first 3 builds):")
    from src.chunking import build_to_text
    for build in builds[:3]:
        print()
        print(f"  {'─' * 50}")
        print(f"  {build_to_text(build)}")
    print(f"  {'─' * 50}")

    # --- Index ---
    print()
    print("Step 3 / 3  Embedding and indexing...")
    vector_store = index_builds(builds, reset=args.reset)

    # --- Verify ---
    count = vector_store._collection.count()
    print()
    print("=" * 60)
    print(f"  Done. {count} document(s) in build_history index.")
    print("=" * 60)


if __name__ == "__main__":
    main()
