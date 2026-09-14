"""
CCAR project, step 3: evaluation.

Two separate things, run back to back:

1. The RAG-triad judge from Week 4 (context relevance / faithfulness /
   answer relevance), run against a broader set of general CCAR
   questions spanning all five documents.
2. A hand-verified numeric-fact test against the Results report --
   the thing we deliberately did NOT pre-solve earlier. Each expected
   value below was independently confirmed against the actual PDF
   (Table 9, p.22 for loss rates; Table 1, p.2 for CET1 ratios -- note
   those are the document's own printed page numbers, which differ from
   the raw PDF page index our citations use, since front matter like the
   Preface/Contents is numbered separately).

Reuses hybrid_search() and generate_answer() from 02_search_and_generate.py
rather than duplicating that logic here.
"""

import importlib.util
import pathlib
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
anthropic_client = Anthropic()

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def _load_search_and_generate():
    spec = importlib.util.spec_from_file_location(
        "search_and_generate", str(SCRIPT_DIR / "02_search_and_generate.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sg = _load_search_and_generate()
hybrid_search = sg.hybrid_search
generate_answer = sg.generate_answer


# --- Part 1: RAG-triad judge ---

JUDGE_TOOL = {
    "name": "record_scores",
    "description": "Record RAG triad evaluation scores for one question/answer pair.",
    "input_schema": {
        "type": "object",
        "properties": {
            dim: {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "reason": {"type": "string"},
                },
                "required": ["score", "reason"],
            }
            for dim in ["context_relevance", "faithfulness", "answer_relevance"]
        },
        "required": ["context_relevance", "faithfulness", "answer_relevance"],
    },
}


def judge(question, context_chunks, answer):
    context = "\n\n".join(c["text"] for c in context_chunks)
    prompt = f"""Score this question/context/answer triple on three dimensions, each 1-5:

- context_relevance: how relevant is the retrieved context to the question?
- faithfulness: is the answer fully supported by the context (no hallucination)?
- answer_relevance: does the answer actually address the question asked?

Question: {question}

Context:
{context}

Answer:
{answer}"""
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_scores"},
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].input


GENERAL_QUESTIONS = [
    "What are the board of directors' responsibilities in capital planning under SR 15-18?",
    "What is the stress capital buffer and how is the current supervisory stress test used to set it?",
    "What macroeconomic variables define the severely adverse scenario in the 2026 stress test scenarios?",
    "What changed in the supervisory stress test methodology for the 2026 cycle compared to prior years?",
    "What is Basel III's minimum leverage ratio requirement?",  # deliberately out of scope
]

print("=" * 80)
print("PART 1: RAG-triad evaluation")
print("=" * 80)

for q in GENERAL_QUESTIONS:
    retrieved = hybrid_search(q, top_k=5)
    answer = generate_answer(q, retrieved)
    scores = judge(q, retrieved, answer)
    print(f"\nQ: {q}")
    print(f"   context_relevance={scores['context_relevance']['score']}  "
          f"faithfulness={scores['faithfulness']['score']}  "
          f"answer_relevance={scores['answer_relevance']['score']}")
    print(f"   answer_relevance reason: {scores['answer_relevance']['reason']}")


# --- Part 2: hand-verified numeric-fact test ---

NUMERIC_QUESTIONS = [
    ("What was the aggregate loss rate for first-lien mortgages, domestic, in the 2026 severely adverse scenario?", "1.5"),
    ("What was the aggregate loss rate for credit cards in the 2026 severely adverse scenario?", "17.1"),
    ("What was the aggregate loss rate for commercial and industrial loans in the 2026 severely adverse scenario?", "9.0"),
    ("What was the aggregate loss rate for commercial real estate, domestic, in the 2026 severely adverse scenario?", "8.8"),
    ("What was the aggregate starting Common Equity Tier 1 (CET1) capital ratio for the 32 banks tested, as of 2025:Q4?", "12.8"),
    ("What was the projected minimum aggregate CET1 capital ratio under the severely adverse scenario?", "11.2"),
]

print("\n\n" + "=" * 80)
print("PART 2: Numeric-fact accuracy test (hand-verified against the source PDF)")
print("=" * 80)

correct_count = 0
for q, expected in NUMERIC_QUESTIONS:
    retrieved = hybrid_search(q, top_k=5)
    answer = generate_answer(q, retrieved)
    got_it = expected in answer
    correct_count += got_it
    print(f"\nQ: {q}")
    print(f"   Expected: {expected}  |  {'CORRECT' if got_it else 'MISS'}")
    print(f"   Answer: {answer}")

print(f"\nNumeric-fact accuracy: {correct_count}/{len(NUMERIC_QUESTIONS)}")
