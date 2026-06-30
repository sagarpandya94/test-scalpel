"""
chunking.py
-----------
Converts domain objects into text strings suitable for embedding.

This is not chunking in the traditional sense (splitting large documents
into smaller pieces). Our documents are naturally small — one build record
and one test case each fit comfortably in a single embedding call.

What we are doing here is *text construction*: deciding what information
to include, in what format, and what to leave out. This directly controls
the quality of semantic similarity at retrieval time.

Two functions:
  build_to_text()      → text representation of a BuildDocument
  test_case_to_text()  → text representation of a TestCaseDocument

Design principles applied:
  1. Repo-namespaced file paths     - 'pricing-service/src/discount/engine.py'
     prevents false matches across services with similar file structures.

  2. Only product failures in build text - script/environment failures are
     excluded so the embedder never learns spurious failure associations.

  3. Passed TCs listed separately   - 'these passed' is a negative signal
     that weakens a coupling over time. We include it but label it clearly.

  4. Test case text focuses on WHAT is tested, not HOW - steps are excluded
     from the embedded text because they add procedural noise without
     improving semantic matching to code areas.
"""

from src.schemas import BuildDocument, TestCaseDocument


def build_to_text(build: BuildDocument) -> str:
    """
    Constructs the embedding text for a single BuildDocument.

    Structure (each section on its own line for LLM readability):
      Build ID and date
      Repos changed
      Files changed (grouped by repo, fully namespaced)
      Product failures (tc_id + tc_name only — error messages excluded to
                        avoid embedding noise from implementation-specific errors)
      Passed test cases (tc_ids only — lighter weight, just the signal)

    Example output:
      Build BUILD-002 | 2024-01-12
      Repos changed: pricing-service, orders-service
      Files changed:
        [pricing-service] src/discount/engine.py, src/discount/tier_calculator.py, src/models/price_rule.py
        [orders-service] src/checkout/promo_validator.py, src/api/checkout_api.py, src/models/promo_code.py
      Product failures: TC-009 (Apply valid promo code at checkout), TC-011 (Bulk order discount tier validation)
      Passed: TC-010, TC-024, TC-025
    """
    lines = []

    # --- Header ---
    date = build.created_at[:10]  # YYYY-MM-DD only
    lines.append(f"Build {build.build_id} | {date}")

    # --- Repos ---
    repos = ", ".join(sorted(build.all_repos))
    lines.append(f"Repos changed: {repos}")

    # --- Files by repo (namespaced) ---
    lines.append("Files changed:")
    for repo, files in sorted(build.files_by_repo.items()):
        namespaced = ", ".join(f"{repo}/{f}" for f in files)
        lines.append(f"  [{repo}] {namespaced}")

    # --- Product failures only ---
    failures = build.product_failures
    if failures:
        failure_parts = [f"{tc.tc_id} ({tc.tc_name})" for tc in failures]
        lines.append(f"Product failures: {', '.join(failure_parts)}")
    else:
        lines.append("Product failures: none")

    # --- Passed TCs (lightweight) ---
    passed = build.passed_tc_ids
    if passed:
        lines.append(f"Passed: {', '.join(passed)}")

    return "\n".join(lines)


def test_case_to_text(tc: TestCaseDocument) -> str:
    """
    Constructs the embedding text for a single TestCaseDocument.

    Structure:
      TC ID | Title | Category | Service
      Description (what the test validates)
      Expected result (the success condition)

    Steps are deliberately excluded — they describe HOW the test runs,
    not WHAT it covers. Including them would make two tests with similar
    steps but different domains look more similar than they are.

    Example output:
      TC-009 | Apply valid promo code at checkout | Category: Checkout | Service: orders-service
      Verifies that entering a valid promotional code at checkout correctly reduces
      the order subtotal by the specified discount percentage before tax calculation.
      Expected: 10% discount applied to subtotal. Promo code label shown. Savings amount displayed.
    """
    lines = []

    # --- Header ---
    lines.append(
        f"{tc.tc_id} | {tc.title} | Category: {tc.category} | Service: {tc.service}"
    )

    # --- What this test validates ---
    lines.append(tc.description)

    # --- Success condition ---
    lines.append(f"Expected: {tc.expected_result}")

    return "\n".join(lines)
