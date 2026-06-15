from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
import os, math, time, hashlib, io, re

from google import genai
from groq import Groq
import vecs

client = genai.Client()        # Gemini — embeddings only
groq_client = Groq()           # Groq — generation
GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------- vector database ----------
DB_CONNECTION = os.getenv("DB_CONNECTION")
vx = vecs.create_client(DB_CONNECTION)
docs_collection = vx.get_or_create_collection(name="documents", dimension=3072)

class State(TypedDict):
    question: str
    sources: list
    answer: str
    status: str
    verdict: str
    doc_id: str
    attempts: int
    top_k: int

# --- global pacing + retry (handles both providers) ---
_last_call_time = [0.0]
MIN_GAP_SECONDS = 2.0

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
    resp = call_with_retry(lambda: groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    ))
    return resp.choices[0].message.content

# ---------- indexing: embed chunks and store in the vector DB ----------
def build_index(file_bytes):
    """Embed a PDF's chunks and store them in the vector DB, tagged by document id.
    Returns (doc_id, num_chunks). Same file bytes -> same doc_id -> no duplicates."""
    from pypdf import PdfReader

    doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

    reader = PdfReader(io.BytesIO(file_bytes))
    text = "".join(page.extract_text() for page in reader.pages)
    parts = re.split(r'\n(?=\d+\.\s)', text)
    chunks = [p.strip() for p in parts if p.strip()]

    records = []
    for i, chunk in enumerate(chunks):
        emb = call_with_retry(lambda c=chunk: client.models.embed_content(
            model="gemini-embedding-001", contents=c
        )).embeddings[0].values
        records.append((
            f"{doc_id}_{i}",                        # unique chunk id
            emb,                                     # 3072-number vector
            {"doc_id": doc_id, "text": chunk},       # metadata
        ))

    docs_collection.upsert(records=records)
    # fast similarity search
    return doc_id, len(chunks)

# ---------- NODE 1: retrieve (query the DB, filtered to this document) ----------
def retrieve_node(state: State):
    question = state["question"]
    doc_id = state["doc_id"]
    top_k = state.get("top_k", 3)
    attempts = state.get("attempts", 0)

    q_emb = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001", contents=question
    )).embeddings[0].values

    results = docs_collection.query(
        data=q_emb,
        limit=top_k,
        filters={"doc_id": {"$eq": doc_id}},
        include_metadata=True,
        include_value=True,
    )

    sources = []
    for record_id, distance, metadata in results:
        sources.append({
            "text": metadata["text"],
            "score": 1 - distance,    # vecs returns distance; convert to similarity
        })

    return {"sources": sources, "attempts": attempts + 1}

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

# ---------- ROUTER + widen ----------
def should_retry(state: State):
    if state["status"] == "unverified" and state["attempts"] < 3:
        return "retry"
    return "done"

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

def run_agent(question, doc_id):
    return graph.invoke({"question": question, "doc_id": doc_id})

# ============ QUICK PATH ============

def run_quick(question, doc_id):
    result = run_agent(question, doc_id)
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

# ============ PLANNER LAYER ============

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


def run_research(question, doc_id):
    subqs = plan(question)
    sub_results = []
    for sq in subqs:
        result = run_agent(sq, doc_id)
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