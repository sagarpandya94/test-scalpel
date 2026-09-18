# Synthetic Data Design

These files are hand-built, not sampled from a real pipeline. That makes them
useful for development and dangerous for evaluation: a synthetic dataset will
happily confirm whatever its author wanted to believe. This document records
what was put in on purpose, so anyone reading an evaluation result can judge
whether the benchmark is fair.

---

## The problem with the original 10 builds

`sample_builds.json` originally held builds 1-10. Every one of its 25 product
failures belonged to a service the build had directly touched. **Zero
cross-service failures.**

Under that condition the naive baseline — "run every test owned by a service
this build touched" — achieves 1.00 recall *by construction*. It is not a good
heuristic on that data; it is a tautology. No retrieval system can beat it, and
Test-Scalpel measured 0.92 against its 1.00 and looked worse than folder-based
tagging.

That made the benchmark unable to measure the thing the project exists to do.
Builds 11-24 were added to fix it.

---

## Service dependency graph

The added builds derive their cross-service failures from an explicit
architecture, not from what the retriever happens to find. Consumer → provider:

```
orders-service        → pricing-service      (checkout applies discounts)
orders-service        → inventory-service    (stock reserved at order creation)
payments-service      → orders-service       (payment charges the order total)
notifications-service → orders-service       (confirmation email renders order content)
everything            → auth-service         (session validation)
```

Each cross-service failure in builds 11-24 traces to one of those edges. For
example, a change to `pricing-service/src/api/pricing_api.py` breaks
`TC-009 Apply valid promo code at checkout`, which is owned by
**orders-service** — because orders calls pricing to compute the discount, and
the build never touches a single orders file.

That is the case the whole project is premised on, and the case folder-based
tagging structurally cannot catch.

---

## What was deliberately varied

**Coupling repeats, but not uniformly.** Real coupling recurs — that is what
distinguishes it from coincidence — so some edges appear several times and are
learnable from history:

| Coupling | Appears in | Learnable? |
|---|---|---|
| pricing → orders checkout | BUILD-011, 016, 020 | yes, 3 occurrences |
| orders → payments | BUILD-013, 022 | yes, 2 occurrences |
| inventory → orders | BUILD-012, 024 | yes, 2 occurrences |
| orders → notifications | BUILD-018 | no, one-off |
| auth → checkout | BUILD-014 | no, one-off |

The two one-off edges are there on purpose. A dataset where every pattern
repeats would overstate what retrieval can do; a selector should be caught
missing things it has never seen.

**Failure mix.** 15 of the 32 product failures in the new builds are
cross-service (~47%). Across all 24 builds that is 15 of 57 (~26%), which is
meant to be a plausible rather than a flattering proportion. Same-service
failures still dominate, as they should.

**Triage noise retained.** Builds 17 and 23 carry a `script` and an
`environment` failure respectively, so the triage gate keeps being exercised
rather than quietly bypassed.

---

## What was NOT done

The data was written before the evaluation was re-run, and was not adjusted
afterwards. No coupling was added, removed or reweighted in response to a
score. If a later change to the ranker makes these numbers look bad, the
correct response is to fix the ranker or report the number — not to edit this
directory.

---

## Files

| File | Contents |
|---|---|
| `sample_builds.json` | 24 builds. 1-10 original, 11-24 add cross-service coupling. |
| `sample_test_cases.json` | 30 TestRail-style cases across 6 services. |
| `sample_new_build.json` | One un-run build, used as the input to `recommend.py`. |
