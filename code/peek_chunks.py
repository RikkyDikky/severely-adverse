"""
Quick sanity check on the ingested chunks -- prints a few sample chunks
per document so we can eyeball whether extraction actually produced
clean, real content (same kind of check that caught the Motley Fool ad
back in Week 2), before trusting the corpus enough to build retrieval
and generation on top of it.

Chunk counts per page varied a lot across the five documents (CCAR Q&As
came out to ~10-11 chunks/page vs. ~3/page for the others), which is
worth eyeballing specifically -- either the Q&A PDF is genuinely denser
text, or something is being extracted messily.
"""

import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ccar_2026")

DOCS = [
    "2026 Supervisory Stress Test Scenarios",
    "2026 Supervisory Stress Test Methodology",
    "2026 Federal Reserve Stress Test Results",
    "SR 15-18 (Capital Planning Governance Guidance)",
    "CCAR Q&As",
]

for doc_name in DOCS:
    result = collection.get(where={"document": doc_name}, limit=3, include=["documents", "metadatas"])
    print(f"\n=== {doc_name} ({len(result['ids'])} of many shown) ===")
    for text, meta in zip(result["documents"], result["metadatas"]):
        print(f"--- page {meta['page']} ---")
        print(text[:500])
        print()
