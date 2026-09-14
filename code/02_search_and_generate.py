"""
CCAR project, step 2: hybrid search over the ingested corpus, then Claude
generates an answer from the retrieved chunks with citations back to the
source document and page number.

Reuses the dense (Chroma) + sparse (BM25) + reciprocal rank fusion
approach from Week 3, but pulls the corpus back out of Chroma rather than
re-chunking the source files, since ingestion already did that once and
stored the result with document/page metadata attached.
"""

import re
import pathlib
import chromadb
import voyageai
from anthropic import Anthropic
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()
voyage_client = voyageai.Client()
anthropic_client = Anthropic()

RRF_K = 60

# Anchored to this file's own folder rather than the working directory --
# Streamlit Community Cloud always runs apps with the working directory set
# to the repo root, not the app's own folder, so a bare relative path like
# "./chroma_db" would look in the wrong place once deployed.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def tokenize(text):
    return re.findall(r"\w+", text.lower())


# --- Load the whole corpus back out of Chroma (BM25 needs every chunk to
# compute its term-frequency statistics, not just the top matches) ---
chroma_client = chromadb.PersistentClient(path=str(SCRIPT_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(name="ccar_2026")

all_data = collection.get(include=["documents", "metadatas"])
all_ids = all_data["ids"]
all_docs = all_data["documents"]
all_metas = all_data["metadatas"]
id_to_index = {doc_id: i for i, doc_id in enumerate(all_ids)}

bm25 = BM25Okapi([tokenize(d) for d in all_docs])


def hybrid_search(query, top_k=5, candidate_pool=15):
    # Dense: embed the query and ask Chroma for its nearest chunks.
    query_embedding = voyage_client.embed(
        texts=[query], model="voyage-4-lite", input_type="query"
    ).embeddings[0]
    dense_results = collection.query(query_embeddings=[query_embedding], n_results=candidate_pool)
    dense_ranked_ids = dense_results["ids"][0]

    # Sparse: score every chunk against the query's keywords, take the top ones.
    scores = bm25.get_scores(tokenize(query))
    sparse_ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_pool]
    sparse_ranked_ids = [all_ids[i] for i in sparse_ranked_indices]

    # Combine both rankings with reciprocal rank fusion (same formula as Week 3).
    rrf_scores = {}
    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)
    for rank, doc_id in enumerate(sparse_ranked_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (RRF_K + rank)

    top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
    results = []
    for doc_id in top_ids:
        idx = id_to_index[doc_id]
        results.append({
            "text": all_docs[idx],
            "document": all_metas[idx]["document"],
            "page": all_metas[idx]["page"],
        })
    return results


def generate_answer(question, chunks):
    context = "\n\n".join(
        f"[Source: {c['document']}, page {c['page']}]\n{c['text']}" for c in chunks
    )
    prompt = f"""Answer the question using ONLY the context below. Every claim in your
answer must cite its source in the format (Source Name, p. X). If the
context does not contain the answer, say so explicitly -- do not guess.

Context:
{context}

Question: {question}"""

    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    TEST_QUESTIONS = [
        # Should be answerable from SR 15-18 / the scenarios or methodology docs.
        "What are the board of directors' responsibilities in capital planning under SR 15-18?",
        "What is the stress capital buffer and how is the current supervisory stress test used to set it?",
        # Numeric-fact test against the Results report's tables -- the thing we
        # flagged as a real open question, not something we've fixed yet.
        "What was the aggregate loss rate for first-lien mortgages, domestic, across all banks in the 2026 stress test results?",
        # Deliberately out of scope -- not covered by any of the 5 documents.
        "What is Basel III's minimum leverage ratio requirement?",
    ]

    for q in TEST_QUESTIONS:
        print(f"\n{'=' * 80}\nQuestion: {q}\n{'=' * 80}")
        retrieved = hybrid_search(q, top_k=5)
        print("Retrieved from:")
        for c in retrieved:
            print(f"  - {c['document']}, page {c['page']}")
        answer = generate_answer(q, retrieved)
        print(f"\nAnswer:\n{answer}")
