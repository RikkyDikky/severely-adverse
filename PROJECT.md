# Topic 2 Project: CCAR Stress Testing Assistant

_Scope locked: 2026-09-12_

## What this proves

Per `AI_Strategy_Learning_Path.pdf`, Topic 2's project must be a RAG system: a document corpus + a chatbot that answers questions with citations, deployed publicly, on public data only. This project does that using five real Federal Reserve documents from the 2026 CCAR/DFAST stress-testing cycle, with hybrid search (dense + sparse) and citation-bearing generation, evaluated on both the RAG triad and a hand-verified numeric-fact test.

## Concept

A chatbot grounded in the Fed's 2026 CCAR/DFAST publications that answers questions about stress testing scenarios, methodology, results, and governance guidance, citing the exact source document and page for every claim, and explicitly declining to answer questions outside its five-document scope rather than guessing.

## Why this project (vs. CECL, vs. an F1 steward-decisions bot)

- CCAR sits closer to actual day-to-day work (home lending stress-loss forecasting) than CECL would have — CECL is the accounting standard for loss provisioning, CCAR/DFAST is the stress-testing exercise itself.
- An F1 steward-decisions RAG bot (grounded in FIA regulations + penalty decisions) was seriously considered as a more LinkedIn-shareable alternative, but ultimately treated as a separate concern: this project's job is to be strong, credible portfolio/interview material, not necessarily a LinkedIn hook. Topic 2's LinkedIn content instead comes from the technical findings below, independent of which project got built. The F1 idea is shelved as a possible unrelated personal project, not part of this learning path.
- Still finance-flavored, keeping the "domain expert who builds AI" positioning intact.

## Scope

Five public Federal Reserve documents, ~188 pages total:

| Document | Pages | Role |
|---|---|---|
| 2026 Supervisory Stress Test Scenarios (Feb 2026) | 42 | The macroeconomic scenarios (baseline/adverse/severely adverse) banks are tested against |
| 2026 Supervisory Stress Test Methodology (Feb 2026) | 3 | Year-over-year model changes only — confirmed the Fed no longer publishes a full methodology document post-2020 |
| 2026 Federal Reserve Stress Test Results (Jun 2026) | 68 | Per-bank, per-portfolio loss rate and capital ratio tables — the numeric "case-level" data |
| SR 15-18 (Capital Planning Governance Guidance, rev. Jan 2021) | 41 | Board/senior management governance expectations for capital planning |
| CCAR Q&As (current, Jul 2026 update) | 32 | Interpretive FAQ clarifications on CCAR/DFAST reporting and rules |

**Deliberately excluded:** the full Basel III framework and any pre-2020 comprehensive methodology documents — scoping to a bounded, current document set rather than "all technical rules," mirroring the same discipline the F1 idea's scoping discussion arrived at.

## Architecture

1. **Ingest** (`01_ingest.py`) — pdfplumber extracts text and tables per page from each PDF, tags every chunk with its source document and page number, embeds with Voyage (`voyage-4-lite`), stores in a dedicated Chroma collection (`ccar_2026`, 788 chunks).
2. **Retrieve + generate** (`02_search_and_generate.py`) — hybrid search (dense via Chroma + sparse via BM25, combined with reciprocal rank fusion), then Claude generates an answer constrained to the retrieved context, citing document + page for every claim, explicitly declining rather than guessing when the context doesn't cover the question.
3. **Evaluate** (`03_evaluate.py`) — the RAG triad (context relevance / faithfulness / answer relevance) via LLM-as-judge on a broad question set, plus a hand-verified numeric-fact test against the Results report's tables.
4. **Deploy** (`app.py`) — a Streamlit chat interface wrapping the same pipeline, with retrieved sources shown per answer.

## Design decisions

- **pdfplumber over pypdf.** Chosen specifically for `extract_tables()`, needed for the Results report's (and Scenarios document's) numeric tables — pypdf has no comparable table-reconstruction capability, and there was no reason to add a second library once pdfplumber covered both plain text and tables.
- **Per-document `use_text_flow` override.** SR 15-18 is an OCR-reflowed PDF (Producer: ABBYY FineReader, vs. the other four Fed-native PDFs) whose footnote markers extracted scrambled under pdfplumber's default position-based text ordering. `use_text_flow=True` (reads the PDF's stored content-stream order instead) fixed it completely; applied only to this one document since the other four already extracted cleanly. Full write-up in `LEARNINGS.md`.
- **Numeric-fact accuracy was deliberately not pre-engineered.** A common concern with RAG is that it handles precise numeric/tabular facts poorly. Rather than pre-building mitigations (e.g., restructuring table rows into full natural-language sentences before chunking), the project shipped the simpler pipeline first and tested it — see Evaluation below.
- **Ingestion pipeline reused across steps, not duplicated.** `02_search_and_generate.py` and `03_evaluate.py` both load `01_ingest.py`'s finished Chroma collection rather than re-extracting or re-chunking; `app.py` and `03_evaluate.py` both dynamically import `02_search_and_generate.py`'s functions rather than reimplementing retrieval or generation.

## Evaluation results

**RAG triad**, five questions spanning all five documents: four scored 4-5/5 across all three dimensions. The fifth, a deliberately out-of-scope question ("What is Basel III's minimum leverage ratio requirement?", not covered by any of the five documents), scored context_relevance=3, faithfulness=5, answer_relevance=2 — the system gave a careful, non-hallucinated answer that distinguished U.S. domestic requirements from Basel III, but the judge itself penalized it for not volunteering Basel III's actual figure from its own training knowledge. See `LEARNINGS.md` for the full write-up — this is a finding about the judge's own blind spot, not the system's.

**Numeric-fact test**, six questions against the Results report's tables (loss rates for first-lien mortgages, credit cards, C&I, CRE; starting and stressed-minimum aggregate CET1 ratios), each answer hand-verified against the source PDF beforehand: **6/6 correct**. A genuine, tested data point against the assumption that RAG can't handle precise numeric extraction — with the honest caveat that six questions against one table is a spot-check, not a large sample.

## Build steps

1. [x] Source and verify the five documents (page counts, producer metadata)
2. [x] Build the ingestion pipeline (extraction + tables + chunking + embedding + storage)
3. [x] Build hybrid search + citation-bearing generation
4. [x] Evaluate (RAG triad + numeric-fact test)
5. [x] Deploy as a Streamlit app
6. [ ] Push to GitHub with a README
7. [ ] At least one LinkedIn post about this topic (no fixed category/timing)

## Data rule reminder

Public data only — all five documents are publicly published Federal Reserve materials. No JPMorgan internal data, models, or processes used anywhere in this project.
