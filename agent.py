from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from typing_extensions import Annotated
import operator

class State(TypedDict):
    question: str
    sources: list
    answer: str
    status: str
    verdict: str
    indexed: list
    attempts: int
    top_k: int

from google import genai
from google.genai import errors
import math, time

client = genai.Client()

# --- global pacing + retry ---
_last_call_time = [0.0]
MIN_GAP_SECONDS = 4.5

def call_with_retry(fn, max_attempts=5):
    for attempt in range(max_attempts):
        elapsed = time.time() - _last_call_time[0]
        if elapsed < MIN_GAP_SECONDS:
            time.sleep(MIN_GAP_SECONDS - elapsed)
        _last_call_time[0] = time.time()
        try:
            return fn()
        except (errors.ServerError, errors.ClientError) as e:
            code = getattr(e, "code", None)
            if code in (429, 503) and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b)
def build_index_from_text(text):
    """Chunk document text on section headings and embed each chunk."""
    import re
    parts = re.split(r'\n(?=\d+\.\s)', text)
    chunks = [p.strip() for p in parts if p.strip()]
    indexed = []
    for chunk in chunks:
        emb = call_with_retry(lambda: client.models.embed_content(
            model="gemini-embedding-001", contents=chunk
        )).embeddings[0].values
        indexed.append({"text": chunk, "embedding": emb})
    return indexed

# ---------- NODE 1: retrieve ----------
def retrieve_node(state: State):
    question = state["question"]
    indexed = state["indexed"]
    top_k = state.get("top_k", 3)
    attempts = state.get("attempts", 0)

    q_emb = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001", contents=question
    )).embeddings[0].values

    scored = [{"score": cosine_similarity(q_emb, it["embedding"]), "text": it["text"]}
              for it in indexed]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"sources": scored[:top_k], "attempts": attempts + 1}

# ---------- NODE 2: generate ----------
def generate_node(state: State):
    question = state["question"]
    sources = state["sources"]

    context = "\n\n".join(f"[Source {i}]\n{it['text']}"
                          for i, it in enumerate(sources, 1))
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present in the context, reply with exactly: NO_ANSWER
Do not explain, do not guess — just output NO_ANSWER if the information is not in the context.

Context:
{context}

Question: {question}

Answer:"""
    answer = call_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )).text
    return {"answer": answer}

# ---------- NODE 3: verify ----------
def verify_node(state: State):
    answer = state["answer"]
    sources = state["sources"]
    question = state["question"]

    if answer.strip().upper().startswith("NO_ANSWER"):
        return {"status": "no_answer",
                "verdict": "The document does not contain this information."}

    context = "\n\n".join(f"[Source {i}]\n{it['text']}"
                          for i, it in enumerate(sources, 1))
    prompt = f"""You are a strict fact-checker. Decide whether the ANSWER
is fully supported by the SOURCES below. Do not use any outside knowledge.

Reply in this exact format:
- First line: either GROUNDED or NOT GROUNDED
- Second line: one short sentence explaining why.

SOURCES:
{context}

QUESTION: {question}

ANSWER: {answer}

Verdict:"""
    verdict_text = call_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )).text.strip()

    status = "verified" if verdict_text.upper().startswith("GROUNDED") else "unverified"
    return {"status": status, "verdict": verdict_text}

# ---------- ROUTER ----------
def should_retry(state: State):
    if state["status"] == "unverified" and state["attempts"] < 3:
        return "retry"
    return "done"

# ---------- NODE: widen ----------
def widen_node(state: State):
    return {"top_k": state.get("top_k", 3) + 2}

# ---------- build the graph ----------
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("verify", verify_node)
builder.add_node("widen", widen_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "verify")
builder.add_conditional_edges(
    "verify",
    should_retry,
    {"retry": "widen", "done": END},
)
builder.add_edge("widen", "retrieve")

graph = builder.compile()

def run_agent(question, indexed):
    return graph.invoke({"question": question, "indexed": indexed})

# ============ PLANNER LAYER ============

def plan(question):
    prompt = f"""You are a research planner. Break the user's question into the
minimal set of focused sub-questions needed to answer it.

Rules:
- If the question is already simple and focused, return it UNCHANGED as a single line.
- Never invent more than 4 sub-questions.
- Each sub-question must be self-contained and answerable on its own.
- Output ONE sub-question per line. No numbering, no extra text.

Question: {question}

Sub-questions:"""
    text = call_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )).text
    subqs = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    return subqs if subqs else [question]


def synthesize(original_question, sub_results):
    parts = [f"Sub-question: {r['question']}\nAnswer: {r['answer']}" for r in sub_results]
    combined = "\n\n".join(parts)
    prompt = f"""You are answering a user's research question by combining the
findings from several sub-questions. Write one clear, coherent answer to the
ORIGINAL question using only the sub-answers provided. If some sub-answers were
not found, acknowledge what could not be determined.

ORIGINAL QUESTION: {original_question}

FINDINGS:
{combined}

Final answer:"""
    return call_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )).text


def run_research(question, indexed):
    subqs = plan(question)
    sub_results = []
    for sq in subqs:
        result = run_agent(sq, indexed)
        sub_results.append({
            "question": sq,
            "answer": result["answer"],
            "status": result["status"],
            "sources": result["sources"],
        })
    if len(sub_results) == 1:
        final = sub_results[0]["answer"]
    else:
        final = synthesize(question, sub_results)
    return {"final_answer": final, "sub_results": sub_results, "sub_questions": subqs}

# --- temporary test ---
if __name__ == "__main__":
    from pypdf import PdfReader
    import re

    reader = PdfReader("Lumora_Robotics_Annual_Report_2024.pdf")
    text = "".join(page.extract_text() for page in reader.pages)
    parts = re.split(r'\n(?=\d+\.\s)', text)
    chunks = [p.strip() for p in parts if p.strip()]

    indexed = []
    for chunk in chunks:
        emb = call_with_retry(lambda: client.models.embed_content(
            model="gemini-embedding-001", contents=chunk
        )).embeddings[0].values
        indexed.append({"text": chunk, "embedding": emb})

    result = run_research(
        "Compare Lumora's 2024 revenue growth to its employee headcount, "
        "and say whether the company is growing efficiently.",
        indexed
    )
    print("SUB-QUESTIONS:")
    for sq in result["sub_questions"]:
        print("  -", sq)
    print("\nFINAL ANSWER:\n", result["final_answer"])