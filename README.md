# Severely Adverse

A CCAR/DFAST stress-testing Q&A assistant, grounded in five public Federal Reserve documents from the 2026 CCAR/DFAST stress-testing cycle. Every answer cites its source document and page, and the system explicitly declines to answer questions outside its document set rather than guessing.

## What it does

Ask a question about the Fed's 2026 stress test scenarios, methodology, results, or capital-planning governance guidance, and the system:

1. Searches the document set using hybrid search (dense semantic search + BM25 keyword search, combined via reciprocal rank fusion)
2. Generates an answer from Claude, constrained to only the retrieved passages
3. Cites the source document and page for every claim
4. States explicitly when a question falls outside the five documents, instead of fabricating an answer

### Example

**Q: What was the aggregate loss rate for first-lien mortgages, domestic, in the 2026 severely adverse scenario?**

> According to the context, the aggregate loss rate for first-lien mortgages, domestic, in the severely adverse scenario was **1.5 percent** (2026 Federal Reserve Stress Test Results, p. 26).

**Q: What is Basel III's minimum leverage ratio requirement?** *(not covered by any of the five source documents)*

> The context does not contain information about Basel III's minimum leverage ratio requirement. The documents show minimum regulatory capital ratios, including a Tier 1 leverage ratio minimum requirement of 4.0% and a Supplementary leverage ratio minimum requirement of 3.0%, but these are identified as U.S. regulatory requirements for stress testing purposes, not explicitly as Basel III requirements.

## Documents

| Document | Pages | Source |
|---|---|---|
| 2026 Supervisory Stress Test Scenarios | 42 | [federalreserve.gov](https://www.federalreserve.gov/publications/files/2026-final-supervisory-stress-test-scenarios-20260204.pdf) |
| 2026 Supervisory Stress Test Methodology | 3 | [federalreserve.gov](https://www.federalreserve.gov/publications/files/2026-february-supervisory-stress-test-methodology.pdf) |
| 2026 Federal Reserve Stress Test Results | 68 | [federalreserve.gov](https://www.federalreserve.gov/publications/files/2026-dfast-results-20260624.pdf) |
| SR 15-18 (Capital Planning Governance Guidance) | 41 | [federalreserve.gov](https://www.federalreserve.gov/supervisionreg/srletters/sr1518a1.pdf) |
| CCAR Q&As | 32 | [federalreserve.gov](https://www.federalreserve.gov/publications/ccar-qas/comprehensive-capital-analysis-and-review-questions-and-answers.htm) |

## Architecture

```
PDFs --(pdfplumber: text + tables)--> chunks with document/page metadata
    --(Voyage embeddings)--> Chroma vector DB
    --(hybrid search: dense + BM25 + RRF)--> top-k relevant chunks
    --(Claude, constrained to context)--> cited answer
```

- **Ingestion** (`01_ingest.py`): extracts text and tables per page, chunks with overlapping word windows, embeds, and stores in Chroma with per-chunk document/page metadata.
- **Retrieval + generation** (`02_search_and_generate.py`): hybrid dense/sparse search, then citation-constrained generation.
- **Evaluation** (`03_evaluate.py`): LLM-as-judge RAG-triad scoring plus a hand-verified numeric-fact test.
- **App** (`app.py`): Streamlit chat interface.

## Evaluation

- **RAG triad** (context relevance / faithfulness / answer relevance), five general questions across all five documents: four scored 4-5/5 on every dimension. The one deliberately out-of-scope question surfaced a finding about LLM-as-judge evaluation itself — see below.
- **Numeric-fact accuracy**: 6/6 correct on a hand-verified test set against the Results report's loss-rate and capital-ratio tables (a common concern with RAG systems is unreliable extraction of precise numeric facts from tables; this is a real, tested data point against that, not a large-sample proof).

Full write-ups of two notable findings from building this project (a PDF-extraction bug and a subtlety in how LLM-as-judge evaluation can leak the judge's own outside knowledge into a supposedly grounded score) are in [`FINDINGS.md`](./FINDINGS.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with:

```
VOYAGE_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

Then run, in order:

```bash
python 01_ingest.py              # builds the vector database (run once)
python 02_search_and_generate.py # sanity-check retrieval + generation
python 03_evaluate.py            # RAG-triad + numeric-fact evaluation
streamlit run app.py             # launch the chat interface
```

## Stack

Python, [Voyage AI](https://www.voyageai.com/) (embeddings), [Chroma](https://www.trychroma.com/) (vector database), [rank_bm25](https://github.com/dorianbrown/rank_bm25) (sparse search), [Anthropic Claude](https://www.anthropic.com/) (generation + evaluation), [pdfplumber](https://github.com/jsvine/pdfplumber) (PDF extraction), [Streamlit](https://streamlit.io/) (interface).

## Disclaimer

Built for educational/portfolio purposes using only publicly available Federal Reserve documents. Not affiliated with, endorsed by, or reviewed by the Federal Reserve. Not financial, legal, or regulatory advice.
