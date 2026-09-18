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

## Triage labels (added for Phase 3)

The LLM triage classifier is scored against the `failure_type` labels in this
file, and the original distribution made that impossible: 57 `product` against
2 `script` and 2 `environment`. A classifier that always answered "product"
would have scored 93%.

24 further failures were added across the existing builds - 12 `script`, 12
`environment` - bringing the split to **57 / 14 / 14** and the majority-class
baseline down to 67%.

These additions do **not** affect retrieval. `build_to_text()` embeds only
product failures and passed test-case ids, and a script or environment failure
appears in neither, so the Phase 2 numbers are unchanged by construction. That
was verified by re-running `scripts/evaluate.py` before and after.

Two of them are deliberately ambiguous, to give the abstention path something
real to work on:

| Build | Test | Label | Why it is hard |
|---|---|---|---|
| BUILD-022 | TC-019 | `script` | Reads as an environment problem (loaded CI agent), but the root cause is a hardcoded sleep with no polling. |
| BUILD-014 | TC-018 | `environment` | Reads as a product search defect, but the staging index was mid-rebuild. |

### A known bad label, left in place

`BUILD-004 / TC-023` is labelled `script` in the original Phase 1 fixture, but
its own error message reads *"SMTP sandbox was unavailable in CI environment"* -
which is this schema's definition of `environment`. The label contradicts its
own evidence.

The classifier disagrees with it and is arguably right, which would make true
accuracy 95% rather than 94%.

**It has been left unchanged on purpose.** Correcting a label *after* seeing the
model disagree with it is the exact failure this document exists to prevent, and
a reader who saw that edit would have no way to know it was the only one. One
point of accuracy is cheaper than the credibility of every other number here.

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
