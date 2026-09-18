"""
retriever.py
------------
The read half of the RAG loop. Phase 1 wrote two indexes; this module queries
them.

Given an incoming build (repos + files touched, no results yet), we run two
independent retrievals:

  1. retrieve_similar_builds()      -> build_history index
     'Which past builds touched code like this, and what broke on them?'
     This is the historical-failure signal. Strong when the area has history.

  2. retrieve_relevant_test_cases() -> test_cases index
     'Which tests semantically cover what this build is doing?'
     This is the cold-start signal. Works even for code with no failure history.

Neither is ranked here. Both return scored raw hits; ranker.py fuses them.
Keeping retrieval and ranking separate means the ranking policy can be tuned
or replaced without touching vector store code, and each half can be evaluated
on its own.

A note on scores
----------------
Chroma is configured with cosine space, and langchain_chroma's
similarity_search_with_score returns the raw *distance*, not a similarity:

    distance = 1 - cosine_similarity      range [0, 2]

so we convert with  similarity = 1 - distance  and clamp to [0, 1]. Negative
similarity (vectors pointing opposite ways) is meaningless for ranking here --
it just means 'unrelated' -- so clamping loses nothing and keeps every
downstream weight in a predictable range.
"""

import os
import json
from dataclasses import dataclass

from langchain_chroma import Chroma

from src.schemas import BuildDocument, MRRecord
from src.chunking import build_query_to_text, build_intent_to_text
from src.embedder import get_embedder
from src.indexer import BUILD_STORE_PATH, TC_STORE_PATH


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SimilarBuild:
    """
    One hit from the build_history index.

    failed_tc_ids are the product failures recorded on that historical build --
    these are the actual predictions this hit contributes. failed_tc_names is
    carried alongside purely so downstream output can be human-readable without
    a second lookup into the test case index.
    """
    build_id: str
    build_number: int
    created_at: str
    similarity: float
    repos: list[str]
    failed_tc_ids: list[str]
    failed_tc_names: dict[str, str]
    passed_tc_ids: list[str]

    @property
    def date(self) -> str:
        return self.created_at[:10]


@dataclass
class RelevantTestCase:
    """One hit from the test_cases index -- a semantic match, no history implied."""
    tc_id: str
    title: str
    category: str
    service: str
    similarity: float


# ---------------------------------------------------------------------------
# Incoming build loading
# ---------------------------------------------------------------------------

def load_build_query_from_json(path: str) -> BuildDocument:
    """
    Loads an incoming build to generate recommendations for.

    Accepts either a single build object or a list (in which case the last
    entry is taken -- CI systems tend to append). tc_results is optional and
    ignored if present: an incoming build has no outcome yet, and letting a
    caller pass results here would silently leak the answer into the query.

    The result is a BuildDocument with tc_results=[], which makes every derived
    property (files_by_repo, all_repos) work unchanged, while product_failures
    and passed_tc_ids correctly come back empty.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"{path} contains an empty list -- nothing to query with.")
        raw = raw[-1]

    mrs = [
        MRRecord(
            mr_id=mr["mr_id"],
            title=mr["title"],
            description=mr["description"],
            repo=mr["repo"],
            files_changed=mr["files_changed"],
        )
        for mr in raw["mrs"]
    ]

    return BuildDocument(
        build_id=raw["build_id"],
        build_number=raw["build_number"],
        created_at=raw["created_at"],
        mrs=mrs,
        tc_results=[],   # unknown by definition -- this is what we are predicting
    )


def build_document_to_query(build: BuildDocument) -> BuildDocument:
    """
    Strips results off an already-loaded BuildDocument.

    Used by evaluate.py, which loads historical builds *with* their known
    outcomes and must query using only the cause side. Doing the strip here
    rather than at each call site means there is one place where the
    answer-leak guard lives.
    """
    return BuildDocument(
        build_id=build.build_id,
        build_number=build.build_number,
        created_at=build.created_at,
        mrs=build.mrs,
        tc_results=[],
    )


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------

def _open_store(collection_name: str, path: str) -> Chroma:
    """
    Opens an existing persisted Chroma collection for reading.

    Fails loudly if the directory is missing. A silently empty collection is
    the worst outcome here -- it would return zero recommendations and read as
    'nothing to run' rather than 'the index was never built'.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No index at {path}. Run the ingestion scripts first:\n"
            f"  python scripts/ingest_test_cases.py --reset\n"
            f"  python scripts/ingest_builds.py --reset"
        )

    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedder(),
        persist_directory=path,
        collection_metadata={"hnsw:space": "cosine"},
    )


def _to_similarity(distance: float) -> float:
    """Cosine distance -> similarity in [0, 1]. See module docstring."""
    return max(0.0, min(1.0, 1.0 - distance))


def _split_csv(value: str) -> list[str]:
    """Chroma metadata values must be scalar, so lists were stored comma-joined."""
    if not value:
        return []
    return [part for part in value.split(",") if part]


def _parse_failed_names(value: str) -> dict[str, str]:
    """
    Unpacks the 'TC-009:Apply valid promo code,TC-011:Bulk order discount'
    metadata format written by the indexer.

    Test case names can themselves contain a colon, so we split on the first
    one only. Entries without a colon are skipped rather than guessed at.
    """
    result: dict[str, str] = {}
    for part in _split_csv(value):
        if ":" not in part:
            continue
        tc_id, name = part.split(":", 1)
        result[tc_id] = name
    return result


# ---------------------------------------------------------------------------
# Retrieval 1 -- similar builds (historical failure signal)
# ---------------------------------------------------------------------------

def retrieve_similar_builds(
    build: BuildDocument,
    k: int = 8,
    exclude_build_ids: list[str] | None = None,
) -> list[SimilarBuild]:
    """
    Finds the k historical builds most similar to this one by files touched.

    On the default of k=8
    ---------------------
    Neighbour depth is the single biggest lever on cross-service recall, because
    a cross-service coupling can only be predicted if some retrieved neighbour
    actually recorded it. Measured by sweeping k over the 24-build dataset:

        k=3   reachable ceiling 0.67   cross-service recall@15  0.80
        k=5   reachable ceiling 0.73   cross-service recall@15  0.80
        k=8   reachable ceiling 0.93   cross-service recall@15  0.93
        k=12  reachable ceiling 1.00   cross-service recall@15  0.87

    'Reachable ceiling' is the fraction of cross-service failures that appear in
    ANY retrieved neighbour -- the best score achievable no matter how good the
    ranking is. It rises monotonically with k, but measured recall does not:
    at k=12 every coupling is technically reachable and the score still falls,
    because the extra neighbours contribute more unrelated failures than real
    ones and the ranking dilutes. More retrieval is not more signal.

    k=8 sits at the turn. Worth re-running the sweep as the history grows --
    the right depth depends on how many builds exist and how varied they are,
    so this number should not be treated as permanent.

    Args:
        build:              The incoming build to find neighbours for.
        k:                  How many neighbours to return.
        exclude_build_ids:  Build IDs to filter out. Used by evaluate.py to hold
                            a build out of its own retrieval -- without this, a
                            build already in the index retrieves itself at
                            similarity 1.0 and every metric looks perfect.

    Returns hits sorted by similarity, highest first.
    """
    store = _open_store("build_history", BUILD_STORE_PATH)
    query_text = build_query_to_text(build)

    where = None
    excluded = list(exclude_build_ids or [])
    if excluded:
        # Chroma needs $nin for a list, $ne for a single value.
        where = (
            {"build_id": {"$nin": excluded}} if len(excluded) > 1
            else {"build_id": {"$ne": excluded[0]}}
        )

    hits = store.similarity_search_with_score(query_text, k=k, filter=where)

    results = []
    for doc, distance in hits:
        meta = doc.metadata
        results.append(
            SimilarBuild(
                build_id=meta.get("build_id", "unknown"),
                build_number=meta.get("build_number", 0),
                created_at=meta.get("created_at", ""),
                similarity=_to_similarity(distance),
                repos=_split_csv(meta.get("repos", "")),
                failed_tc_ids=_split_csv(meta.get("failed_tc_ids", "")),
                failed_tc_names=_parse_failed_names(meta.get("failed_tc_names", "")),
                passed_tc_ids=_split_csv(meta.get("passed_tc_ids", "")),
            )
        )

    return sorted(results, key=lambda r: r.similarity, reverse=True)


# ---------------------------------------------------------------------------
# Retrieval 2 -- relevant test cases (cold-start semantic signal)
# ---------------------------------------------------------------------------

def retrieve_relevant_test_cases(
    build: BuildDocument,
    k: int = 10,
    restrict_to_touched_services: bool = False,
) -> list[RelevantTestCase]:
    """
    Finds the k test cases whose described coverage best matches this build's
    intent (MR titles and descriptions -- see build_intent_to_text).

    Args:
        restrict_to_touched_services:
            If True, only return TCs owned by a service this build touched.
            Off by default, and deliberately so: cross-service regressions are
            exactly the failures a human reviewer misses, and a pricing change
            breaking an orders test is the case this tool exists to catch.
            Turn it on only when you want a fast, narrow smoke selection.
    """
    store = _open_store("test_cases", TC_STORE_PATH)
    query_text = build_intent_to_text(build)

    where = None
    if restrict_to_touched_services:
        repos = build.all_repos
        where = (
            {"service": {"$in": repos}} if len(repos) > 1
            else {"service": repos[0]}
        )

    hits = store.similarity_search_with_score(query_text, k=k, filter=where)

    results = [
        RelevantTestCase(
            tc_id=doc.metadata.get("tc_id", "unknown"),
            title=doc.metadata.get("title", ""),
            category=doc.metadata.get("category", ""),
            service=doc.metadata.get("service", ""),
            similarity=_to_similarity(distance),
        )
        for doc, distance in hits
    ]

    return sorted(results, key=lambda r: r.similarity, reverse=True)
