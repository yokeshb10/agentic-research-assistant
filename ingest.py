from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from google import genai
from google.genai import errors
import math
import re
import time

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

# --- load PDF ---
reader = PdfReader("Lumora_Robotics_Annual_Report_2024.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()
print("Total characters extracted:", len(text))

# --- chunk on section headings ---
def chunk_text(text):
    parts = re.split(r'\n(?=\d+\.\s)', text)
    return [p.strip() for p in parts if p.strip()]

chunks = chunk_text(text)
print("Number of chunks:", len(chunks))

# --- embed all chunks ---
indexed_chunks = []
for i, chunk in enumerate(chunks):
    result = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001", contents=chunk
    ))
    indexed_chunks.append({"text": chunk, "embedding": result.embeddings[0].values})
    print(f"Embedded chunk {i+1}/{len(chunks)}")
print(f"\nDone. Indexed {len(indexed_chunks)} chunks.")

# --- interactive question loop ---
while True:
    question = input("\nAsk a question (or type 'quit' to exit): ")
    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    q_emb = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001", contents=question
    )).embeddings[0].values

    scored = [{"score": cosine_similarity(q_emb, it["embedding"]), "text": it["text"]}
              for it in indexed_chunks]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored[:3]

    context = "\n\n".join(f"[Source {i}]\n{it['text']}"
                          for i, it in enumerate(top_chunks, 1))
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I cannot find that in the document."

Context:
{context}

Question: {question}

Answer:"""
    answer = call_with_retry(lambda: client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )).text

    print("\n=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    for i, it in enumerate(top_chunks, 1):
        snippet = it["text"][:150].replace("\n", " ")
        print(f"[Source {i}] (score {it['score']:.3f}) {snippet}...")