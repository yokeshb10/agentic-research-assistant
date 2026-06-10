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
from groq import Groq
import math, time

client = genai.Client()        # Gemini — used ONLY for embeddings
groq_client = Groq()           # Groq — used for all generation (reads GROQ_API_KEY)

GROQ_MODEL = "llama-3.3-70b-versatile"

# --- global pacing + retry (handles both providers' rate-limit errors) ---
_last_call_time = [0.0]
MIN_GAP_SECONDS = 2.0          # Groq's free tier is more generous, so shorter gap

def call_with_retry(fn, max_attempts=5):
    for attempt in range(max_attempts):
        elapsed = time.time() - _last_call_time[0]
        if elapsed < MIN_GAP_SECONDS:
            time.sleep(MIN_GAP_SECONDS - elapsed)
        _last_call_time[0] = time.time()
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = "429" in msg or "rate" in msg or "503" in msg or "quota" in msg
            if transient and attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            else:
                raise

def groq_generate(prompt):
    """Generate text via Groq (Llama). Hides the message-format difference."""
    resp = call_with_retry(lambda: groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ))
    return resp.choices[0].message.content

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    return dot / (mag_a * mag_b)

def build_index_from_text(text):
    """Chunk document text on section headings and embed each chunk (Gemini embeddings)."""
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

# ---------- NODE 1: retrieve (Gemini embeddings) ----------
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

# ---------- NODE 2: generate (Groq) ----------
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
    answer = groq_generate(prompt)
    return {"answer": answer}

# ---------- NODE 3: verify (Groq) ----------
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
    verdict_text = groq_generate(prompt).strip()

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

# ============ QUICK PATH (no planner) ============

def run_quick(question, indexed):
    """Single-question path: no planner. Fewer calls, no planner to mangle simple questions."""
    result = run_agent(question, indexed)
    return {
        "final_answer": result["answer"],
        "sub_results": [{
            "question": question,
            "answer": result["answer"],
            "status": result["status"],
            "sources": result["sources"],
        }],
        "sub_questions": [question],
    }

# ============ PLANNER LAYER (deep research) ============

def plan(question):
    prompt = f"""You are a research planner. Decide whether the user's question needs to be broken down.

CRITICAL RULES:
- If the question asks for ONE fact or is already simple, return it EXACTLY AS WRITTEN, unchanged, as a single line. Do NOT rephrase it.
- Only split if the question genuinely contains MULTIPLE distinct asks (e.g. "compare X and Y", "what is A and also B").
- Never change the meaning of the question. Never invent a different question.
- Output ONE question per line. No numbering, no extra text.

Question: {question}

Output:"""
    text = groq_generate(prompt)
    subqs = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. )").strip()
        if cleaned:
            subqs.append(cleaned)
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
    return groq_generate(prompt)


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

    reader = PdfReader("Lumora_Robotics_Annual_Report_2024.pdf")
    text = "".join(page.extract_text() for page in reader.pages)
    indexed = build_index_from_text(text)

    result = run_quick("Who is the CEO of Lumora Robotics?", indexed)
    print("ANSWER:", result["final_answer"])
    print("STATUS:", result["sub_results"][0]["status"])