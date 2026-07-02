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
│   └── sample_test_cases.json      # Synthetic TestRail test cases (30 cases)
├── src/
│   ├── schemas.py                  # BuildDocument, MRRecord, TCResult, TestCaseDocument
│   ├── chunking.py                 # Text construction for embedding
│   ├── embedder.py                 # OpenAI embeddings wrapper
│   └── indexer.py                  # Chroma vector store indexing
├── scripts/
│   ├── ingest_builds.py            # CLI: ingest build history
│   └── ingest_test_cases.py        # CLI: ingest TestRail test cases
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

---

## Phases

### ✅ Phase 1 — Embedding, Chunking & Indexing *(complete)*
- Domain schemas for builds and test cases
- Text construction optimised for semantic similarity
- Two Chroma vector indexes (build history + test cases)
- Triage-gated ingestion pipeline

### 🔲 Phase 2 — Retrieval & Ranking *(next)*
- Given a new build's file list, query both indexes
- Aggregate and rank test case recommendations by historical failure frequency
- Confidence scoring alongside suggestions

### 🔲 Phase 3 — Feedback Loop
- SDET feedback on suggestions (thumbs up/down per TC)
- LLM-based auto-classification of failure types
- Recency weighting for older builds

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
