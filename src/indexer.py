"""
indexer.py
----------
Handles embedding and persisting documents into two separate Chroma collections:

  Collection 1: 'build_history'
    One document per build. Captures which repos/files were touched and which
    test cases failed (product failures only, after triage).
    Used at query time to find historically similar builds to the new one.

  Collection 2: 'test_cases'
    One document per TestRail test case. Captures what each test covers
    semantically. Used to surface relevant test cases for areas with no
    historical failure data yet (cold-start coverage).

Key design decisions:
  - Builds with untriaged failures are skipped (is_ready_to_embed guard).
  - Metadata fields are stored separately from the embedded text, so you can
    filter at retrieval time (e.g. filter by repo, by failure_type, by date).
  - Passed TCs are embedded as part of the build text — 'passed' is also
    signal that can weaken spurious couplings over time.
  - Upsert semantics: re-indexing the same build_id overwrites the existing
    entry rather than creating duplicates.
"""

import os
import json
import shutil
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.schemas import BuildDocument, MRRecord, TCResult, TestCaseDocument
from src.chunking import build_to_text, test_case_to_text
from src.embedder import get_embedder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
BUILD_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "build_history")
TC_STORE_PATH = os.path.join(VECTOR_STORE_DIR, "test_cases")


# ---------------------------------------------------------------------------
# Loaders: JSON → domain objects
# ---------------------------------------------------------------------------

def load_builds_from_json(path: str, skip_untriaged: bool = True) -> list[BuildDocument]:
    """
    Loads build records from JSON and deserialises into BuildDocument objects.

    Args:
        skip_untriaged:
            When True (the default) builds with failures still pending triage
            are dropped, because embedding an untriaged failure is exactly the
            index poisoning the triage gate exists to prevent.

            Triage tooling must pass False. Those builds are precisely the ones
            it needs to see, and with the default it would be handed an empty
            list and report 'nothing to do' about the backlog it was invoked to
            clear -- the gate would hide its own queue.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    builds = []
    skipped = []

    for item in raw:
        mrs = [
            MRRecord(
                mr_id=mr["mr_id"],
                title=mr["title"],
                description=mr["description"],
                repo=mr["repo"],
                files_changed=mr["files_changed"],
            )
            for mr in item["mrs"]
        ]

        tc_results = [
            TCResult(
                tc_id=tc["tc_id"],
                tc_name=tc["tc_name"],
                status=tc["status"],
                failure_type=tc.get("failure_type"),
                triaged=tc["triaged"],
                error_message=tc.get("error_message"),
            )
            for tc in item["tc_results"]
        ]

        build = BuildDocument(
            build_id=item["build_id"],
            build_number=item["build_number"],
            created_at=item["created_at"],
            mrs=mrs,
            tc_results=tc_results,
        )

        if build.is_ready_to_embed or not skip_untriaged:
            builds.append(build)
        else:
            skipped.append(build.build_id)

    if skipped:
        print(f"  Skipped (untriaged failures): {', '.join(skipped)}")

    return builds


def load_test_cases_from_json(path: str) -> list[TestCaseDocument]:
    """Loads test cases from JSON and deserialises into TestCaseDocument objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        TestCaseDocument(
            tc_id=item["tc_id"],
            title=item["title"],
            category=item["category"],
            service=item["service"],
            description=item["description"],
            steps=item["steps"],
            expected_result=item["expected_result"],
        )
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Build index
# ---------------------------------------------------------------------------

def index_builds(builds: list[BuildDocument], reset: bool = False) -> Chroma:
    """
    Embeds and indexes BuildDocuments into the 'build_history' Chroma collection.

    Args:
        builds:  List of BuildDocument objects (already filtered for triage readiness).
        reset:   If True, wipes and rebuilds the entire collection from scratch.
                 Use this when backfilling historical data. For incremental runs
                 (one new build per CI cycle), leave reset=False.

    Returns:
        The Chroma vector store instance (useful for quick post-index verification).

    Metadata stored per document (all filterable at retrieval time):
        build_id      - unique identifier
        build_number  - sequential build number
        created_at    - ISO timestamp of the build
        repos         - comma-separated list of repos touched
        failed_tc_ids - comma-separated list of product-failure TC IDs
        passed_tc_ids - comma-separated list of passed TC IDs
    """
    embedder = get_embedder()

    if reset and os.path.exists(BUILD_STORE_PATH):
        shutil.rmtree(BUILD_STORE_PATH)
        print(f"  Cleared existing build index at {BUILD_STORE_PATH}")

    os.makedirs(BUILD_STORE_PATH, exist_ok=True)

    documents = []
    for build in builds:
        text = build_to_text(build)

        failed_ids = [tc.tc_id for tc in build.product_failures]
        failed_names = {tc.tc_id: tc.tc_name for tc in build.product_failures}

        metadata = {
            "build_id": build.build_id,
            "build_number": build.build_number,
            "created_at": build.created_at,
            "repos": ",".join(sorted(build.all_repos)),
            # Comma-separated strings — Chroma metadata values must be scalar
            "failed_tc_ids": ",".join(failed_ids) if failed_ids else "",
            "failed_tc_names": ",".join(
                f"{tc_id}:{failed_names[tc_id]}" for tc_id in failed_ids
            ) if failed_ids else "",
            "passed_tc_ids": ",".join(build.passed_tc_ids) if build.passed_tc_ids else "",
            "mr_ids": ",".join(mr.mr_id for mr in build.mrs),
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
                id=build.build_id,   # Use build_id as doc ID → upsert semantics
            )
        )

    vector_store = Chroma(
        collection_name="build_history",
        embedding_function=embedder,
        persist_directory=BUILD_STORE_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    vector_store.add_documents(documents)

    return vector_store


# ---------------------------------------------------------------------------
# Test case index
# ---------------------------------------------------------------------------

def index_test_cases(test_cases: list[TestCaseDocument], reset: bool = False) -> Chroma:
    """
    Embeds and indexes TestCaseDocuments into the 'test_cases' Chroma collection.

    Args:
        test_cases:  List of TestCaseDocument objects.
        reset:       If True, wipes and rebuilds the collection.
                     The test case index is relatively stable — rebuild only
                     when TestRail cases are added, renamed, or restructured.

    Returns:
        The Chroma vector store instance.

    Metadata stored per document (all filterable):
        tc_id     - TestRail test case ID
        title     - Human-readable name
        category  - Functional area (e.g. 'Checkout', 'Payments')
        service   - Owning microservice (e.g. 'pricing-service')
    """
    embedder = get_embedder()

    if reset and os.path.exists(TC_STORE_PATH):
        shutil.rmtree(TC_STORE_PATH)
        print(f"  Cleared existing test case index at {TC_STORE_PATH}")

    os.makedirs(TC_STORE_PATH, exist_ok=True)

    documents = []
    for tc in test_cases:
        text = test_case_to_text(tc)

        metadata = {
            "tc_id": tc.tc_id,
            "title": tc.title,
            "category": tc.category,
            "service": tc.service,
        }

        documents.append(
            Document(
                page_content=text,
                metadata=metadata,
                id=tc.tc_id,  # Use TC ID as doc ID → upsert semantics
            )
        )

    vector_store = Chroma(
        collection_name="test_cases",
        embedding_function=embedder,
        persist_directory=TC_STORE_PATH,
        collection_metadata={"hnsw:space": "cosine"},
    )

    vector_store.add_documents(documents)

    return vector_store
