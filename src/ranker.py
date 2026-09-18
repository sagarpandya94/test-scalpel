"""
ranker.py
---------
Fuses the two retrieval signals into one ranked list of test cases to run.

Input is the raw output of retriever.py:
  - similar builds  (each carrying the TCs that actually failed on it)
  - relevant test cases (each carrying a semantic similarity)

Output is a ranked list of Recommendations, each with a score, a confidence
band, and human-readable evidence explaining why it was selected. The evidence
is not decoration: an SDET who cannot see why a test was chosen will not trust
the selection, and a tool nobody trusts gets bypassed with a full regression
run -- which defeats the entire purpose.


Why the two signals are normalised separately
---------------------------------------------
The two indexes return similarity in visibly different ranges. Measured on the
sample data:

    build_history index   non-self neighbours land in ~0.775 - 0.790
    test_cases index      hits land in ~0.43 - 0.64

That difference is structural, not a quirk of this dataset. Build documents are
long, and any two of them share a lot of surface -- the same header format, the
same repo vocabulary, the same file path shapes -- so cosine similarity between
any two builds is high and tightly clustered. Test case documents are short
prose with genuinely different subject matter, so they spread out.

Adding those numbers together would let the build signal dominate purely
because of its offset, and the tight 0.015-wide band on the build side means
raw similarity barely distinguishes a good neighbour from a poor one.

So each signal is normalised *within its own candidate set* before fusion,
which throws away absolute similarity and keeps relative ordering -- the only
part that carries information here.

The two signals are normalised differently, and the difference matters:

  - Neighbour build similarities are scaled *proportionally* (_proportional),
    because their band is too narrow for the differences between them to mean
    anything. See that function for the measurement.

  - Per-test-case scores are scaled with *min-max* (_min_max), because there
    the spread is real: one test case genuinely is far more relevant than
    another, and that separation should survive into the final ranking.

The cost of relative normalisation is that a weak retrieval still produces
confident-looking numbers: if every retrieved neighbour is poor, the best of a
bad set still scores 1.0 on the history axis. absolute_similarity is preserved
on every Recommendation so a caller can spot exactly that, and the
`retrieval_quality` field on RankedResult flags it explicitly.
"""

from dataclasses import dataclass, field

from src.retriever import SimilarBuild, RelevantTestCase


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Relative trust in each signal. History is weighted higher because 'this test
# actually broke when this code last changed' is direct evidence, while
# semantic match is an inference about coverage. Both are kept well away from
# 0 and 1 -- a pure-history ranker cannot cold-start, and a pure-semantic one
# never learns from what actually broke.
W_HISTORY = 0.65
W_SEMANTIC = 0.35

# A test that PASSED on a similar build is weak evidence *against* selecting it:
# it is the negative signal the README calls out as weakening a coupling over
# time. Kept deliberately small. A pass means 'did not break that time', not
# 'cannot break' -- the same test may guard a boundary this build crosses and
# the last one did not.
PASS_PENALTY = 0.25

# Confidence band thresholds, applied to the fused score.
HIGH_CONFIDENCE = 0.70
MEDIUM_CONFIDENCE = 0.40

# Below this normalised build similarity, treat the whole retrieval as weak and
# say so rather than presenting confident-looking output built on poor matches.
WEAK_RETRIEVAL_SIMILARITY = 0.60


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """One test case, with the reasoning that put it on the list."""
    tc_id: str
    title: str
    score: float                  # fused, 0-1
    confidence: str               # 'high' | 'medium' | 'low'
    signals: str                  # 'history+semantic' | 'history' | 'semantic'
    history_score: float          # normalised, 0-1
    semantic_score: float         # normalised, 0-1
    absolute_similarity: float    # raw cosine similarity from the TC index, pre-normalisation
    failed_in_builds: list[str] = field(default_factory=list)
    passed_in_builds: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def is_cold_start(self) -> bool:
        """True when nothing in retrieved history has ever failed on this TC."""
        return not self.failed_in_builds


@dataclass
class RankedResult:
    """The full ranked output plus the context needed to judge it."""
    recommendations: list[Recommendation]
    neighbour_build_ids: list[str]
    best_neighbour_similarity: float
    retrieval_quality: str        # 'good' | 'weak'
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _min_max(values: dict[str, float]) -> dict[str, float]:
    """
    Min-max normalises a {key: score} map into [0, 1].

    Two degenerate cases are handled explicitly rather than left to divide-by-
    zero:
      - empty input      -> empty output
      - all values equal -> every entry becomes 1.0, not 0.0. If three builds
        are equally similar, they are equally *good* evidence; collapsing them
        all to zero would silently delete the entire history signal.
    """
    if not values:
        return {}

    lo = min(values.values())
    hi = max(values.values())

    if hi - lo < 1e-9:
        return {key: 1.0 for key in values}

    return {key: (value - lo) / (hi - lo) for key, value in values.items()}


def _proportional(values: dict[str, float]) -> dict[str, float]:
    """
    Rescales a {key: score} map so the largest value becomes 1.0 and the rest
    keep their true proportion to it.

    This is the right tool when the input values sit in a tight band and their
    *differences* are not meaningful, only their ordering and rough magnitude.
    Min-max would be wrong there: it stretches whatever spread exists to fill
    [0, 1] regardless of how narrow it was, inventing confidence the data does
    not support.

    Measured on the sample data, five retrieved neighbour builds spanned
    0.7397 - 0.8381, a band 0.098 wide. Min-max turned that into weights of
    1.00 / 0.50 / 0.26 / 0.14 / 0.00 -- so the single nearest build cast a vote
    seven times heavier than the fifth, on a similarity difference of under a
    tenth. In practice that meant whatever happened to fail on the top
    neighbour won the ranking outright, even against test cases far more
    obviously related to the change.

    Proportional scaling gives 1.00 / 0.94 / 0.91 / 0.90 / 0.88 instead: the
    neighbours stay near-equal, and what separates test cases is how many
    neighbours they failed on rather than which one happened to sort first.
    Repetition across builds is the more trustworthy signal at this resolution.
    """
    if not values:
        return {}

    hi = max(values.values())
    if hi <= 0:
        return {key: 0.0 for key in values}

    return {key: value / hi for key, value in values.items()}


# ---------------------------------------------------------------------------
# Signal 1 -- history
# ---------------------------------------------------------------------------

def _history_scores(
    similar_builds: list[SimilarBuild],
) -> tuple[dict[str, float], dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    """
    Accumulates a per-TC history score from the retrieved neighbour builds.

    Each neighbour votes with its own normalised similarity as the weight, so a
    failure on a very similar build counts for more than one on a marginal
    match. A TC that failed across several neighbours accumulates several votes
    -- that repetition is the strongest signal this system has.

    Returns (raw scores, failed-in map, passed-in map, tc_id -> name).
    """
    # Proportional, not min-max -- see _proportional() for the measurement that
    # drove this. Neighbour similarities arrive in too tight a band for their
    # differences to carry real information.
    build_weights = _proportional({b.build_id: b.similarity for b in similar_builds})

    raw: dict[str, float] = {}
    failed_in: dict[str, list[str]] = {}
    passed_in: dict[str, list[str]] = {}
    names: dict[str, str] = {}

    for build in similar_builds:
        weight = build_weights.get(build.build_id, 0.0)

        for tc_id in build.failed_tc_ids:
            raw[tc_id] = raw.get(tc_id, 0.0) + weight
            failed_in.setdefault(tc_id, []).append(build.build_id)
            if tc_id in build.failed_tc_names:
                names.setdefault(tc_id, build.failed_tc_names[tc_id])

        for tc_id in build.passed_tc_ids:
            passed_in.setdefault(tc_id, []).append(build.build_id)

    # Apply the pass penalty only to TCs that already have a history score.
    # A TC that merely passed on a neighbour and never failed should be absent
    # from this signal entirely, not carry a negative score -- if it belongs on
    # the list at all, that is the semantic signal's call to make.
    for tc_id in list(raw.keys()):
        for build in similar_builds:
            if tc_id in build.passed_tc_ids:
                weight = build_weights.get(build.build_id, 0.0)
                raw[tc_id] = max(0.0, raw[tc_id] - PASS_PENALTY * weight)

    return raw, failed_in, passed_in, names


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

def _confidence_band(score: float, failed_count: int) -> str:
    """
    Maps a fused score to a confidence band.

    Repeated historical failure overrides the score thresholds: a TC that broke
    on two or more retrieved neighbours is high confidence regardless of what
    the arithmetic says. That pattern is the clearest evidence available, and a
    scoring weight should not be able to bury it.
    """
    if failed_count >= 2:
        return "high"
    if score >= HIGH_CONFIDENCE:
        return "high"
    if score >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


def _build_evidence(
    failed_in: list[str],
    passed_in: list[str],
    semantic: float,
    absolute_similarity: float,
) -> list[str]:
    """Composes the human-readable 'why was this selected' lines."""
    evidence = []

    if failed_in:
        builds = ", ".join(failed_in)
        times = "1 similar build" if len(failed_in) == 1 else f"{len(failed_in)} similar builds"
        evidence.append(f"Product failure on {times} ({builds})")

    if passed_in:
        evidence.append(f"Passed on {len(passed_in)} similar build(s) ({', '.join(passed_in)})")

    if semantic > 0:
        evidence.append(
            f"Semantic coverage match to this build's change intent "
            f"(cosine {absolute_similarity:.2f})"
        )

    if not failed_in:
        evidence.append("No failure history in retrieved builds -- cold-start selection")

    return evidence


def rank(
    similar_builds: list[SimilarBuild],
    relevant_test_cases: list[RelevantTestCase],
    top_n: int | None = None,
    min_score: float = 0.0,
) -> RankedResult:
    """
    Fuses both signals into a ranked recommendation list.

    Args:
        similar_builds:       hits from retrieve_similar_builds()
        relevant_test_cases:  hits from retrieve_relevant_test_cases()
        top_n:                truncate to this many (None = all candidates)
        min_score:            drop anything scoring below this

    A TC present in only one signal scores 0 on the other rather than being
    excluded. That is what lets a never-before-failed test surface on semantic
    grounds alone, and what keeps a historically flaky area on the list even
    when this build's MR descriptions say nothing about it.
    """
    hist_raw, failed_in, passed_in, hist_names = _history_scores(similar_builds)
    hist_norm = _min_max(hist_raw)

    sem_raw = {tc.tc_id: tc.similarity for tc in relevant_test_cases}
    sem_norm = _min_max(sem_raw)

    titles = {tc.tc_id: tc.title for tc in relevant_test_cases}
    titles.update({k: v for k, v in hist_names.items() if k not in titles})

    candidates = set(hist_norm) | set(sem_norm)

    recommendations = []
    for tc_id in candidates:
        h = hist_norm.get(tc_id, 0.0)
        s = sem_norm.get(tc_id, 0.0)
        score = W_HISTORY * h + W_SEMANTIC * s

        if score < min_score:
            continue

        failed = failed_in.get(tc_id, [])
        passed = passed_in.get(tc_id, [])

        if failed and s > 0:
            signals = "history+semantic"
        elif failed:
            signals = "history"
        else:
            signals = "semantic"

        recommendations.append(
            Recommendation(
                tc_id=tc_id,
                title=titles.get(tc_id, ""),
                score=round(score, 4),
                confidence=_confidence_band(score, len(failed)),
                signals=signals,
                history_score=round(h, 4),
                semantic_score=round(s, 4),
                absolute_similarity=round(sem_raw.get(tc_id, 0.0), 4),
                failed_in_builds=failed,
                passed_in_builds=passed,
                evidence=_build_evidence(failed, passed, s, sem_raw.get(tc_id, 0.0)),
            )
        )

    # Sort by score, then by historical failure count, then by TC id. The
    # tiebreakers keep output stable across runs -- an unstable recommendation
    # list is impossible to diff between builds or to trust in CI.
    recommendations.sort(
        key=lambda r: (-r.score, -len(r.failed_in_builds), r.tc_id)
    )

    if top_n is not None:
        recommendations = recommendations[:top_n]

    best_similarity = max((b.similarity for b in similar_builds), default=0.0)
    quality = "good" if best_similarity >= WEAK_RETRIEVAL_SIMILARITY else "weak"

    warnings = []
    if not similar_builds:
        warnings.append(
            "No similar builds retrieved -- every recommendation is semantic-only."
        )
    elif quality == "weak":
        warnings.append(
            f"Closest historical build scored only {best_similarity:.2f} similarity. "
            "This build touches unfamiliar code; treat history-based picks with caution."
        )

    if not relevant_test_cases:
        warnings.append(
            "Test case index returned nothing -- cold-start coverage is unavailable."
        )

    cold = sum(1 for r in recommendations if r.is_cold_start)
    if recommendations and cold == len(recommendations):
        warnings.append(
            "No retrieved neighbour has any recorded product failure. "
            "Selection rests entirely on semantic coverage."
        )

    return RankedResult(
        recommendations=recommendations,
        neighbour_build_ids=[b.build_id for b in similar_builds],
        best_neighbour_similarity=round(best_similarity, 4),
        retrieval_quality=quality,
        warnings=warnings,
    )
