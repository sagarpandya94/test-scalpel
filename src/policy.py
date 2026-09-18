"""
policy.py
---------
Decides *whether* to trust the RAG selection for a given build, before we get
to the question of which tests it picked.

This implements the safety rule the README describes but Phase 1 never coded:

    Build 1 -> RAG-selected cases only
    Build 2 -> RAG-selected cases only
    Build 3 -> Full regression, RAG-suggested cases run first
    ...
    Hard rule: full regression always required before any production release

The rule exists to break a feedback loop that would otherwise quietly destroy
the system. If selection were purely RAG-driven forever, tests that were never
selected would never run, never produce a pass or fail, never enter the build
history index, and so never become selectable. The unselected set would calcify
into a permanent blind spot -- and, worse, the index would keep looking healthy,
because the only tests reporting results would be the ones already trusted.

Periodic full regression is what keeps data flowing across the whole suite.
It is a data-coverage mechanism at least as much as a safety net.

Separated from ranker.py on purpose: ranking is a scoring question that will
keep being tuned, while this is a release-safety question that should change
rarely and visibly.
"""

from dataclasses import dataclass


# Every Nth build runs the full suite. 3 is the README's number. Raising it
# buys more time per cycle and widens the blind-spot window; lowering it does
# the reverse. It is the single most important safety dial in the system.
FULL_REGRESSION_EVERY = 3


@dataclass
class Strategy:
    """What to run for this build, and why."""
    mode: str              # 'rag_only' | 'full_regression'
    reason: str
    run_suggested_first: bool

    @property
    def is_full_regression(self) -> bool:
        return self.mode == "full_regression"


def decide_strategy(
    build_number: int,
    is_release: bool = False,
    force_full: bool = False,
    cadence: int = FULL_REGRESSION_EVERY,
) -> Strategy:
    """
    Decides the execution strategy for a build.

    Args:
        build_number:  Sequential build number. The cadence keys off this
                       rather than off a run counter so the schedule is a
                       property of the build history itself -- it cannot drift
                       if the tool is skipped, re-run, or run from a different
                       machine.
        is_release:    True if this build is a production release candidate.
        force_full:    Manual override for an operator who wants everything.
        cadence:       Run full regression every N builds.

    The checks are ordered by how non-negotiable they are: a release build runs
    everything no matter what the cadence says, and no argument to this function
    can produce a 'rag_only' answer for a release.

    Note that full_regression never means 'ignore the recommendations'. The
    ranked list still determines execution *order*, so the tests most likely to
    fail run first and a real defect surfaces minutes into the run rather than
    hours. Fail-fast ordering is valuable even when nothing is being skipped.
    """
    if is_release:
        return Strategy(
            mode="full_regression",
            reason="Release candidate -- full regression is mandatory before production.",
            run_suggested_first=True,
        )

    if force_full:
        return Strategy(
            mode="full_regression",
            reason="Full regression requested manually (--full).",
            run_suggested_first=True,
        )

    if cadence > 0 and build_number % cadence == 0:
        return Strategy(
            mode="full_regression",
            reason=(
                f"Build {build_number} falls on the every-{cadence}-builds full "
                "regression cadence -- keeps the whole suite generating data."
            ),
            run_suggested_first=True,
        )

    position = build_number % cadence if cadence > 0 else 0
    remaining = (cadence - position) if cadence > 0 else 0
    return Strategy(
        mode="rag_only",
        reason=(
            f"Build {build_number} runs the RAG-selected subset. "
            f"Next full regression in {remaining} build(s)."
        ),
        run_suggested_first=True,
    )
