"""
CCAR project, step 1: ingest the five Federal Reserve source documents
into a Chroma vector database.

Extracts text (and tables) from each PDF with pdfplumber, chunks each
document into overlapping word windows (same approach as Week 2), tracks
which page each chunk started on, embeds everything with Voyage, and
stores it all in Chroma along with document/page metadata -- so answers
can cite a real source later instead of just returning bare text.
"""

import os
import pdfplumber
import voyageai
import chromadb
from dotenv import load_dotenv

load_dotenv()
client = voyageai.Client()

DATA_DIR = "../data"

# Maps each source PDF to a short, readable name used in citations.
DOCUMENTS = {
    "scenarios.pdf": "2026 Supervisory Stress Test Scenarios",
    "methodology.pdf": "2026 Supervisory Stress Test Methodology",
    "results.pdf": "2026 Federal Reserve Stress Test Results",
    "sr_15-18.pdf": "SR 15-18 (Capital Planning Governance Guidance)",
    "ccar_qas.pdf": "CCAR Q&As",
}

# SR 15-18 is an OCR/reflow PDF (Producer: ABBYY FineReader, unlike the
# other four Fed-native PDFs) whose footnote markers extract scrambled
# with the default text ordering -- use_text_flow=True reads text in the
# PDF's stored content-stream order instead of reconstructing it by
# on-page position, which fixes this specific document's footnotes
# without changing behavior for the other four, which already extract
# cleanly.
TEXT_FLOW_FILES = {"sr_15-18.pdf"}


def extract_pages(path, use_text_flow=False):
    """Returns a list of page texts. Any tables found on a page are
    appended as pipe-separated rows under a [TABLE DATA] marker, so
    column values stay attached to each other on one line instead of
    being scattered the way plain text extraction would leave them."""
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text(use_text_flow=use_text_flow) or ""
            tables = page.extract_tables()
            table_lines = []
            for table in tables:
                for row in table:
                    cleaned = [(cell or "").strip() for cell in row]
                    table_lines.append(" | ".join(cleaned))
            if table_lines:
                text += "\n[TABLE DATA]\n" + "\n".join(table_lines)
            pages.append(text)
    return pages


def chunk_with_pages(pages, chunk_size=200, overlap=50):
    """Same sliding-window chunking as Week 2, but tracks which page each
    chunk's first word came from, so every chunk can cite a page number."""
    words = []
    word_pages = []
    for page_num, page_text in enumerate(pages, start=1):
        for w in page_text.split():
            words.append(w)
            word_pages.append(page_num)

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append({"text": " ".join(chunk_words), "page": word_pages[start]})
        start += step
    return chunks


# --- Step 1: extract + chunk every document ---
all_chunks = []
all_metadatas = []
all_ids = []

for filename, doc_name in DOCUMENTS.items():
    path = os.path.join(DATA_DIR, filename)
    pages = extract_pages(path, use_text_flow=(filename in TEXT_FLOW_FILES))
    chunks = chunk_with_pages(pages)
    print(f"{doc_name}: {len(pages)} pages -> {len(chunks)} chunks")

    for i, c in enumerate(chunks):
        all_chunks.append(c["text"])
        all_metadatas.append({"document": doc_name, "page": c["page"]})
        all_ids.append(f"{filename}_{i}")

print(f"\nTotal chunks across all 5 documents: {len(all_chunks)}\n")

# --- Step 2: embed everything ---
# Voyage limits how much text one call can embed, and this corpus is much
# bigger than Week 2's single transcript, so we batch instead of sending
# everything at once (this is the rate-limit lesson from Week 2, applied
# proactively instead of waiting to hit the same error again).
BATCH_SIZE = 128
all_embeddings = []
for i in range(0, len(all_chunks), BATCH_SIZE):
    batch = all_chunks[i:i + BATCH_SIZE]
    result = client.embed(texts=batch, model="voyage-4-lite", input_type="document")
    all_embeddings.extend(result.embeddings)
    print(f"Embedded {min(i + BATCH_SIZE, len(all_chunks))}/{len(all_chunks)} chunks")

# --- Step 3: store in Chroma, with document + page metadata attached ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ccar_2026")

collection.upsert(
    ids=all_ids,
    embeddings=all_embeddings,
    documents=all_chunks,
    metadatas=all_metadatas,
)
print(f"\nStored {collection.count()} chunks in Chroma collection 'ccar_2026'.")
