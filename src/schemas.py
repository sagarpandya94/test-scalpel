"""
schemas.py
----------
Dataclass definitions for the two core domain objects in this system:

  1. BuildDocument  - one record per deployed build, capturing which repos/files
                      were touched and which test cases failed (after triage).

  2. TestCaseDocument - one record per TestRail test case, capturing what the
                        test covers semantically.

These are the two indexes we maintain:
  - builds_index       → used at query time to find builds similar to the new one
  - test_cases_index   → used to surface test cases that semantically cover
                         the areas touched by the new build
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Build document types
# ---------------------------------------------------------------------------

@dataclass
class MRRecord:
    """
    Represents a single Merge Request included in a build.

    repo is kept separate from files_changed so that at embedding time
    we can namespace each file as  <repo>/<path>  — critical in a
    microservice architecture where the same filename can exist in
    multiple repos.
    """
    mr_id: str
    title: str
    description: str
    repo: str
    files_changed: list[str]


@dataclass
class TCResult:
    """
    Represents the outcome of one test case execution in a build run.

    failure_type is the key field that guards index quality:
      - 'product'     → real defect; included in the failure signal
      - 'script'      → test automation bug; excluded from signal
      - 'environment' → infra/data issue; excluded from signal
      - 'unknown'     → not yet triaged; excluded until updated

    Only results where triaged=True are safe to embed.
    Passed results (status='passed') don't need failure_type — a pass
    is always clean signal regardless of environment.
    """
    tc_id: str
    tc_name: str
    status: str                         # 'passed' | 'failed'
    failure_type: Optional[str]         # 'product' | 'script' | 'environment' | 'unknown' | None
    triaged: bool                       # False = pending triage, skip embedding
    error_message: Optional[str]        # Raw error from CI — useful for LLM classification later


@dataclass
class BuildDocument:
    """
    Top-level record for one deployed build.

    Computed fields (files_by_repo, product_failures, passed_tc_ids)
    are derived from the raw mrs and tc_results at ingestion time,
    not stored in source data. This keeps the raw data clean and the
    derived representation explicit.
    """
    build_id: str
    build_number: int
    created_at: str
    mrs: list[MRRecord]
    tc_results: list[TCResult]

    # --- derived at ingestion time ---

    @property
    def files_by_repo(self) -> dict[str, list[str]]:
        """
        Returns files grouped by repo, with duplicates removed.

        Example:
          {
            'pricing-service': ['src/discount/engine.py', 'src/models/price_rule.py'],
            'auth-service':    ['src/session/manager.py']
          }
        """
        result: dict[str, list[str]] = {}
        for mr in self.mrs:
            if mr.repo not in result:
                result[mr.repo] = []
            for f in mr.files_changed:
                if f not in result[mr.repo]:
                    result[mr.repo].append(f)
        return result

    @property
    def all_repos(self) -> list[str]:
        """Unique list of repos touched by this build."""
        return list({mr.repo for mr in self.mrs})

    @property
    def product_failures(self) -> list[TCResult]:
        """
        Only the test results that are:
          - status = 'failed'
          - failure_type = 'product'
          - triaged = True

        These are the only failures we embed as real signal.
        Script and environment failures are excluded.
        """
        return [
            tc for tc in self.tc_results
            if tc.status == "failed"
            and tc.failure_type == "product"
            and tc.triaged
        ]

    @property
    def passed_tc_ids(self) -> list[str]:
        """
        Test cases that passed — also valuable signal.
        'TC-09 passed when pricing files changed' weakens that coupling over time.
        """
        return [tc.tc_id for tc in self.tc_results if tc.status == "passed"]

    @property
    def is_ready_to_embed(self) -> bool:
        """
        A build is ready to embed only when all its failures have been triaged.
        Untriaged failures mean an SDET hasn't reviewed the run yet.
        Passed results don't require triage.
        """
        untriaged_failures = [
            tc for tc in self.tc_results
            if tc.status == "failed" and not tc.triaged
        ]
        return len(untriaged_failures) == 0


# ---------------------------------------------------------------------------
# Test case document type
# ---------------------------------------------------------------------------

@dataclass
class TestCaseDocument:
    """
    Represents one TestRail test case.

    The text we embed is constructed from title + category + service +
    description + expected_result. This gives the embedder enough semantic
    context to know *what* the test covers, so it can surface relevant
    cases even for builds with no historical failure data for those cases.

    steps is stored in metadata but not embedded — step-level detail adds
    noise to semantic similarity without improving retrieval quality.
    """
    tc_id: str
    title: str
    category: str
    service: str
    description: str
    steps: str
    expected_result: str
