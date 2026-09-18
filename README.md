# Test-Scalpel ✂️

> Surgical test selection powered by RAG. Stop running 4000 tests on every build — run the ones that actually matter.

---

## The Problem

At scale, full regression suites become a bottleneck. 4000 test cases. Hours of execution. Developers waiting. Releases delayed.

The uncomfortable truth: the vast majority of those 4000 cases have zero relevance to what actually changed in any given build. Running them all is not thoroughness — it's noise.

But naive solutions (folder-based tagging, keyword matching, gut instinct) miss real failures and erode trust. You need something smarter.

---

## What Test-Scalpel Does

Test-Scalpel is a RAG (Retrieval-Augmented Generation) powered test impact advisor. Given a new build, it analyses what the build actually touched across your microservices and surfaces the test cases most likely to be affected — based on historical failure patterns, not guesswork.

**The goal is not to skip tests permanently. It's to run the right ones first, fast, on every build.**

---

## How It Works

Test-Scalpel maintains two vector indexes:

### 1. Build History Index
Every deployed build is indexed as a document capturing:
- Which repos and files were touched (union across all MRs in the build)
- Which test cases failed — **product failures only**, after SDET triage

When a new build arrives, Test-Scalpel finds historically similar builds and surfaces the test cases that failed on those builds.

### 2. Test Case Index
Every TestRail test case is indexed semantically — what it covers, which service it belongs to, what the expected behaviour is.

This handles the cold-start problem: even for code areas with no failure history yet, Test-Scalpel can surface semantically relevant test cases.

---

## Key Design Decisions

### Build-level attribution, not MR-level
In most organisations, 10–15 MRs merge before a single build deploys. You cannot cleanly attribute a test failure to one specific MR. Test-Scalpel operates at the **build level** — the union of all files changed across every MR in the build.

### Microservice-aware file paths
File paths are namespaced by repo to prevent false matches across services with similar structures:
```
pricing-service/src/discount/engine.py
orders-service/src/checkout/promo_validator.py
```

### Triage gate before embedding
Not every test failure is a product defect. Test-Scalpel classifies failures before indexing:

| `failure_type` | Description | Embedded as signal? |
|---|---|---|
| `product` | Real product defect | ✅ Yes |
| `script` | Test automation bug | ❌ No |
| `environment` | Infra/data issue | ❌ No |
| `unknown` | Pending triage | ❌ No (until reviewed) |

Script and environment failures are stored for audit purposes but never drive future suggestions.

### Every 3rd build is a full regression
Pure RAG-only selection creates a feedback loop — unselected tests never run, never generate data, and become permanently invisible. The strategy:

```
Build 1 → RAG-selected cases only
Build 2 → RAG-selected cases only
Build 3 → Full regression, RAG-suggested cases run first
...
Hard rule: full regression always required before any production release
```

This maintains data coverage across the full test suite while still dramatically reducing execution time on most builds.

---

## Project Structure

```
test-scalpel/
├── data/
│   ├── sample_builds.json          # Synthetic build history (10 builds, 6 microservices)
│   ├── sample_test_cases.json      # Synthetic TestRail test cases (30 cases)
│   └── sample_new_build.json       # An un-run build to generate recommendations for
├── src/
│   ├── schemas.py                  # BuildDocument, MRRecord, TCResult, TestCaseDocument
│   ├── chunking.py                 # Text construction — index side and query side
│   ├── embedder.py                 # OpenAI embeddings wrapper
│   ├── indexer.py                  # Chroma vector store indexing
│   ├── retriever.py                # Queries both indexes for an incoming build
│   ├── ranker.py                   # Fuses both signals into a ranked list
│   └── policy.py                   # Full-regression cadence and release gate
├── scripts/
│   ├── ingest_builds.py            # CLI: ingest build history
│   ├── ingest_test_cases.py        # CLI: ingest TestRail test cases
│   ├── recommend.py                # CLI: which tests to run for a new build
│   └── evaluate.py                 # CLI: leave-one-out recall vs baselines
├── vector_store/                   # Chroma persisted indexes (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.12
- OpenAI API key

### 2. Set up environment
```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 4. Ingest test cases (do this once, or when TestRail changes)
```bash
python scripts/ingest_test_cases.py --reset
```

### 5. Ingest build history (backfill, then run after each build cycle)
```bash
python scripts/ingest_builds.py --reset    # first time (full backfill)
python scripts/ingest_builds.py            # subsequent runs (incremental)
```

### 6. Get recommendations for a new build
```bash
python scripts/recommend.py                            # human-readable report
python scripts/recommend.py --json                     # for CI consumption
python scripts/recommend.py --release                  # forces full regression
python scripts/recommend.py --data path/to/build.json
```

### 7. Evaluate the selector
```bash
python scripts/evaluate.py             # recall vs random and same-service baselines
python scripts/evaluate.py --verbose   # per-build breakdown, including misses
```

---

## Current Results

Leave-one-out over 24 indexed builds, each excluded from its own retrieval.
`builds_k=8` neighbours, ranked, cut at k.

| Selection | Suite run | Recall | Precision | Builds fully covered |
|---|---|---|---|---|
| Test-Scalpel k=5 | 17% | 0.71 | 0.33 | 9/23 |
| Test-Scalpel k=7 | 23% | 0.77 | 0.26 | 12/23 |
| Test-Scalpel k=10 | 33% | 0.92 | 0.22 | 19/23 |
| Test-Scalpel k=15 | 50% | 0.99 | 0.16 | 22/23 |
| Random k=10 | 33% | 0.33 | - | - |
| Same-service (naive) | 22% | 0.76 | 0.33 | - |

### The split that actually matters

Aggregate recall hides the only comparison worth making. Counted per failure:

| Failure type | Count | Naive baseline | k=5 | k=10 | k=15 |
|---|---|---|---|---|---|
| Same-service (folder-findable) | 42 | 1.00 | 0.76 | 0.95 | 1.00 |
| **Cross-service (needs retrieval)** | **15** | **0.00** | **0.40** | **0.73** | **0.93** |

On same-service failures both approaches saturate - embeddings are an expensive
way to reach a result folder-based tagging gives you for free. The cross-service
row is the entire argument for the vector store. The naive baseline scores 0.00
there and structurally cannot do better: a test owned by a service the build
never touched is outside the set it considers at all.

### Retrieval depth is the biggest lever

Sweeping neighbour count `builds_k` moves cross-service recall far more than any
ranking change tested:

| `builds_k` | Reachable ceiling | Cross-service recall@15 |
|---|---|---|
| 3 | 0.67 | 0.80 |
| 5 | 0.73 | 0.80 |
| **8** | **0.93** | **0.93** |
| 12 | 1.00 | 0.87 |

"Reachable ceiling" is the fraction of cross-service failures appearing in *any*
retrieved neighbour - the best score achievable regardless of ranking quality. It
rises monotonically with depth, but measured recall does not. At `builds_k=12`
every coupling is technically reachable and the score still falls, because the
extra neighbours contribute more unrelated failures than real ones. More
retrieval is not more signal. The default is 8; re-run the sweep as history grows.

### How close to the ceiling is 0.93?

Of the 15 cross-service failures, 4 come from couplings occurring exactly once in
the dataset, with no prior occurrence to learn from. Those are reachable only if
the semantic index happens to surface them. 0.93 means the system is recovering
nearly everything the data makes recoverable.

### What was measured and rejected

Five fusion strategies were compared on identical cached retrieval results. The
hypothesis going in was that a weighted sum penalises cross-service tests, since
they carry history evidence but no semantic evidence, and a sum treats absent
evidence as negative. **That hypothesis was wrong.** Evidence-normalisation, the
direct fix for it, was the worst variant tested - it inflates weak history-only
candidates and crowds out genuinely relevant ones:

| Fusion | k=10 overall | k=10 cross-service |
|---|---|---|
| weighted sum (current) | 0.928 | 0.67 |
| noisy-or | 0.942 | 0.73 |
| max | 0.913 | 0.60 |
| weighted + agreement bonus | 0.928 | 0.67 |
| evidence-normalised | 0.796 | 0.47 |

noisy-or edges the weighted sum, but by a single caught failure out of 15 -
inside the noise of a 24-build sample. The fusion was left unchanged rather than
tuned to a difference this dataset cannot resolve. Depth was changed instead,
because that effect was large enough to be real.

### The confidence column was wrong, and the fix was not the obvious one

Thresholding the fused score into high/medium/low produced a band that was
**inverted** - `medium` picks hit more often than `high` ones. Hit rate by score:

| Score | Hit rate | | Score | Hit rate |
|---|---|---|---|---|
| 0.0-0.2 | 0.01 | | 0.5-0.6 | 0.37 |
| 0.2-0.3 | 0.09 | | 0.6-0.7 | 0.41 |
| 0.3-0.4 | 0.24 | | 0.7-0.85 | 0.33 |
| 0.4-0.5 | 0.15 | | **0.85-1.0** | **0.11** |

The highest scores are the least reliable. That is the min-max artefact: the top
candidate of every build normalises to ~1.0 whether or not it is any good, so the
0.85+ bucket fills with the top picks of builds where retrieval went badly.
Dropping the normalisation does not help - ordering is identical and calibration
stays non-monotonic.

Banding on evidence shape instead surfaced a genuinely surprising result. Failure
history is only useful when **focused**:

| Failed on | Hit rate |
|---|---|
| 1 neighbour build | 0.16 |
| 2 neighbour builds | 0.30 |
| **3+ neighbour builds** | **0.09** |

Failing on many neighbours is *worse* evidence than failing on two - a test that
breaks on everything is flaky or ubiquitously wired, and its history says little
about any particular build. This contradicts the "repetition is the strongest
signal" assumption the ranker was built on.

That finding did **not** translate into a ranking improvement: damping repeat
occurrences by `0.7^(n-1)` moved recall@10 by +1.5 points and harder damping hurt,
so the ranker was left alone. And banding on evidence shape alone, though
monotonic in aggregate (0.23 / 0.08 / 0.04), produced incoherent output - a rank-1
pick labelled MEDIUM beside a rank-8 pick labelled HIGH. Two columns that
contradict each other destroy the trust the evidence column exists to build.

The shipped banding uses rank position with a demotion for cold-start picks:
**0.33 / 0.11 / 0.04**, monotonic, 8x separation, and it contradicted the rank
order once across 23 builds.

> All numbers come from synthetic data. See `data/README.md` for what was put in
> on purpose and what was deliberately left unlearnable.

---

## Phases

### ✅ Phase 1 — Embedding, Chunking & Indexing *(complete)*
- Domain schemas for builds and test cases
- Text construction optimised for semantic similarity
- Two Chroma vector indexes (build history + test cases)
- Triage-gated ingestion pipeline

### ✅ Phase 2 — Retrieval & Ranking *(complete)*
- Asymmetric query construction — an incoming build has no results to embed
- Dual retrieval: file paths → build history, MR prose → test case index
- Signal fusion with per-signal normalisation and a pass penalty
- Confidence bands and per-recommendation evidence
- Full-regression cadence and release gate implemented
- Leave-one-out evaluation against random and same-service baselines
- Dataset extended to 24 builds with cross-service coupling, so the benchmark
  can distinguish retrieval from folder-based tagging (see `data/README.md`)
- Cross-service vs same-service recall reported separately

### 🔲 Phase 3 — Feedback Loop *(next)*
- SDET feedback on suggestions (thumbs up/down per TC), fed back as a ranking prior
- LLM-based auto-classification of `failure_type` from `error_message`
  (the field is already captured at ingestion for exactly this)
- Recency weighting — an 8-month-old coupling should not count as much as
  last week's, and nothing currently decays

---

## What Test-Scalpel Is Not

- It does not replace your test suite — it selects from it
- It does not guarantee zero missed defects on RAG-only builds — it reduces risk intelligently
- It is not a one-shot solution — it gets smarter with every build it indexes

---

## Tech Stack

- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector store:** ChromaDB (cosine similarity)
- **Orchestration:** LangChain
- **Language:** Python 3.12
