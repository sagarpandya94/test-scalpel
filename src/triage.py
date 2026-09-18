"""
triage.py
---------
LLM classification of test failures into product / script / environment.

This is the generation half of the system. Everything else in Test-Scalpel is
retrieval: embeddings, cosine similarity, ranking arithmetic. Nothing generates.
Triage is where a model actually reads something and produces a judgement.

It is also the feedback loop, not a side feature. The triage gate decides what
enters the build history index:

    product      -> embedded as failure signal, drives future selection
    script       -> stored, never embedded
    environment  -> stored, never embedded
    unknown      -> excluded until a human reviews it

So this module's output determines what the retriever will learn. A classifier
that works turns a manual SDET bottleneck into a pipeline step. A classifier
that quietly mislabels poisons the index that every future recommendation is
built on.


Why abstention is the central design concern
--------------------------------------------
The two error directions are not symmetric.

Labelling a script or environment failure as *product* is the expensive
mistake. It writes a false coupling into the index -- "this test broke when
these files changed" -- and that lie is then retrieved, ranked and acted on for
every future build touching similar code. One bad label degrades selection
indefinitely, and nothing in the pipeline will ever flag it.

Labelling a product failure as script/environment is cheaper. It loses one
piece of real signal. The coupling stays invisible until the next time that
code breaks that test, at which point it can be learned properly.

Missing signal is recoverable. A poisoned index is not. So the classifier is
built to abstain rather than guess: it may return 'unknown', and any verdict
below MIN_CONFIDENCE is downgraded to 'unknown' regardless of what the model
claimed. An unknown failure simply waits for a human, which is exactly the
status quo this replaces -- abstaining costs nothing that was not already
being paid.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from src.schemas import BuildDocument, TCResult

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VALID_TYPES = ("product", "script", "environment")

# Any verdict the model returns below this confidence becomes 'unknown'.
# Set deliberately high. The cost asymmetry documented above means a confident
# wrong answer is far worse than a shrug, and the fallback for a shrug is the
# human triage that happens today anyway.
MIN_CONFIDENCE = 0.70

DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class TriageVerdict:
    """
    One classification result.

    failure_type is what should be written back to the TCResult. It is
    'unknown' whenever the model abstained or fell below MIN_CONFIDENCE, which
    keeps the record out of the index until a human looks at it.

    raw_type and raw_confidence preserve what the model actually said, so a
    reviewer can see the difference between 'the model had no idea' and 'the
    model said script at 0.65 and we declined to act on it'. Without that
    distinction, tuning MIN_CONFIDENCE later would be guesswork.
    """
    tc_id: str
    failure_type: str
    confidence: float
    reasoning: str
    raw_type: str
    raw_confidence: float

    @property
    def abstained(self) -> bool:
        return self.failure_type == "unknown"

    @property
    def downgraded(self) -> bool:
        """True when the model committed to a label but confidence was too low."""
        return self.abstained and self.raw_type in VALID_TYPES


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You triage automated test failures for a microservices e-commerce platform.
Given one failure, decide what actually caused it.

Categories:

- product
  A real defect in application code. The test worked correctly and caught
  broken behaviour. Typical shape: an assertion on a business value that came
  back wrong, an unexpected null or type from application logic, a contract
  the application stopped honouring.

- script
  A defect in the test automation, not the application. The application is
  fine. Typical shape: stale selectors or page objects, missing waits, stale
  element references, hardcoded ids or dates or locales, fixture and seed-data
  mismatches, errors raised inside test helpers, imports broken by test-side
  refactors, comparisons that are wrong in the test itself.

- environment
  An infrastructure or test-data problem. Neither the application nor the test
  is at fault. Typical shape: unreachable or unhealthy services and stubs,
  exhausted connection pools, expired certificates, DNS failures, evicted or
  OOMKilled pods, disk exhaustion, third-party quota or maintenance windows,
  staging data being rebuilt or restored mid-run.

- unknown
  The evidence does not clearly support one category.

Use the list of files the build changed as evidence. A failure whose error
points at application code the build touched leans product. A failure whose
error points at test infrastructure or a page object leans script, even when
the build touched related application code.

Be careful with failures that look like one category and are another. A
timeout on a loaded CI agent is 'script' if the test had no wait or retry,
and 'environment' if the infrastructure genuinely was unavailable. Read for
the root cause, not the symptom.

IMPORTANT -- when to abstain:
A wrong 'product' label is the worst outcome available to you. It writes a
false association between this code change and this test into a retrieval
index, and that association is then used to select tests on every future
build. It is never detected or corrected. Missing a real product defect is
cheaper: the signal is simply learned later.

So do not guess. If the error message is vague, generic, or consistent with
more than one category, return 'unknown' with low confidence. Abstaining
routes the failure to a human, which is what happens today regardless.

Respond with JSON only:
{"failure_type": "product|script|environment|unknown",
 "confidence": <float 0.0-1.0>,
 "reasoning": "<one sentence, max 25 words>"}"""


def _build_user_prompt(tc: TCResult, build: Optional[BuildDocument]) -> str:
    """Assembles the per-failure prompt, including build context when available."""
    parts = [
        f"Test case: {tc.tc_id} - {tc.tc_name}",
        f"Error message: {tc.error_message or '(none recorded)'}",
    ]

    if build is not None:
        parts.append(f"Build: {build.build_id}")
        changed = []
        for repo, files in sorted(build.files_by_repo.items()):
            changed.append(f"  [{repo}] " + ", ".join(files))
        if changed:
            parts.append("Files changed in this build:")
            parts.extend(changed)
        if build.mrs:
            parts.append("Change descriptions:")
            for mr in build.mrs:
                parts.append(f"  {mr.title}: {mr.description}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------

def get_triage_model() -> ChatOpenAI:
    """
    Returns the chat model used for triage.

    Kept separate from embedder.get_embedder() because the two have genuinely
    different requirements -- classification wants a low temperature and a
    capable-enough reasoning model, embeddings want throughput and cost. Tying
    them to one config would force a compromise on both.

    temperature=0 because this is classification, not composition. Two runs
    over the same failure should not disagree, or the index becomes a function
    of when ingestion happened to run.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    model = os.getenv("OPENAI_TRIAGE_MODEL", DEFAULT_MODEL)

    return ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _parse_verdict(tc_id: str, content: str) -> TriageVerdict:
    """
    Turns the model's JSON response into a verdict, applying the confidence
    floor.

    Every failure mode here resolves to 'unknown' rather than raising. A
    malformed response during a nightly ingestion run should cost one
    unclassified failure, not the whole batch -- and 'unknown' is the safe
    default, because it is the value that keeps a record out of the index.
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return TriageVerdict(
            tc_id=tc_id, failure_type="unknown", confidence=0.0,
            reasoning="Model returned unparseable output.",
            raw_type="unparseable", raw_confidence=0.0,
        )

    raw_type = str(data.get("failure_type", "unknown")).strip().lower()
    reasoning = str(data.get("reasoning", "")).strip()

    try:
        raw_confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        raw_confidence = 0.0
    raw_confidence = max(0.0, min(1.0, raw_confidence))

    # A label outside the known set is treated as an abstention, not coerced
    # into the nearest match -- guessing at the model's intent here would
    # reintroduce exactly the confident-wrong-answer risk the design avoids.
    if raw_type not in VALID_TYPES:
        return TriageVerdict(
            tc_id=tc_id, failure_type="unknown", confidence=raw_confidence,
            reasoning=reasoning or "Model did not commit to a category.",
            raw_type=raw_type, raw_confidence=raw_confidence,
        )

    if raw_confidence < MIN_CONFIDENCE:
        return TriageVerdict(
            tc_id=tc_id, failure_type="unknown", confidence=raw_confidence,
            reasoning=reasoning, raw_type=raw_type, raw_confidence=raw_confidence,
        )

    return TriageVerdict(
        tc_id=tc_id, failure_type=raw_type, confidence=raw_confidence,
        reasoning=reasoning, raw_type=raw_type, raw_confidence=raw_confidence,
    )


def classify_failure(
    tc: TCResult,
    build: Optional[BuildDocument] = None,
    model: Optional[ChatOpenAI] = None,
) -> TriageVerdict:
    """
    Classifies a single failed test result.

    Args:
        tc:     The failed TCResult. Its error_message carries most of the signal.
        build:  Optional build context. Passing it materially improves accuracy,
                because 'does this error point at code the build actually
                touched' is one of the strongest available cues.
        model:  Reuse an existing client across a batch rather than constructing
                one per call.
    """
    if tc.status != "failed":
        raise ValueError(f"{tc.tc_id} did not fail - nothing to triage.")

    client = model or get_triage_model()

    response = client.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", _build_user_prompt(tc, build)),
    ])

    return _parse_verdict(tc.tc_id, response.content)


def classify_build(
    build: BuildDocument,
    model: Optional[ChatOpenAI] = None,
    only_untriaged: bool = True,
) -> list[TriageVerdict]:
    """
    Classifies the failures on one build.

    Args:
        only_untriaged: When True (the default) this skips failures a human has
                        already triaged. Human verdicts outrank model verdicts
                        and are never overwritten -- the point is to clear the
                        backlog, not to relitigate decisions an SDET has made.
                        Set False to re-classify everything, which is what the
                        evaluation harness does to score against known labels.
    """
    client = model or get_triage_model()

    targets = [
        tc for tc in build.tc_results
        if tc.status == "failed" and (not only_untriaged or not tc.triaged)
    ]

    return [classify_failure(tc, build, client) for tc in targets]
