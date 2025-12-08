print("Script is starting…")
import sys
print("Python version:", sys.version)
import os
print("OPENROUTER_API_KEY:", os.getenv("OPENROUTER_API_KEY"))
print("Script started successfully")
# query_with_llm_openrouter.py
import os
import traceback
import textwrap
import requests
from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer

# ---------- Config (via env or defaults) ----------
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "pdf_vectors")
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")  # must match collection
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # set your API key in env
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "256"))

OPENROUTER_MODEL = os.getenv("LLM_MODEL")  # OpenRouter LLaMA-2 model

def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        print("(print error)")

def load_resources():
    safe_print("Connecting to Milvus...", MILVUS_HOST, MILVUS_PORT)
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found in Milvus.")

    coll = Collection(COLLECTION_NAME)

    safe_print("Loading embedding model:", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    safe_print("Using OpenRouter LLaMA-2-7B-Instruct")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not found in environment variables!")

    return coll, embed_model

def encode_query(embed_model, query):
    vec = embed_model.encode([query], normalize_embeddings=True)
    return vec[0].tolist()

def search_milvus(collection, qvec, top_k=TOP_K):
    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    results = collection.search(
        data=[qvec],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "source", "page"],
    )
    hits = results[0]
    retrieved = []
    for hit in hits:
        ent = hit.entity
        retrieved.append({
            "text": ent.get("text") if ent else "",
            "source": ent.get("source") if ent else "unknown",
            "page": ent.get("page") if ent else "N/A",
            "score": hit.score,
        })
    return retrieved

def build_prompt(query, retrieved, max_chars=4000):
    parts = []
    for i, r in enumerate(retrieved, start=1):
        txt = r["text"][:1000]
        parts.append(f"[{i}] Source: {r['source']} (Page {r['page']})\n{txt}")
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        context = context[:max_chars]
    
    prompt = f"""You are a helpful, intelligent AI assistant. Answer the user's question using the information found in the provided context from multiple documents.

CONTEXT:
{context}

INSTRUCTIONS:
- Use the context as your primary reference while applying deep reasoning
- Connect information logically and provide analytical insights
- Explain naturally, clearly, and in a conversational tone
- You may summarize, reorganize, and reason through the information
- Cite sources using brackets [1], [2], etc. for factual statements
- If context doesn't contain sufficient information, state: "I cannot fully answer this question based on the provided documents."
- Apply your reasoning to provide meaningful, well-structured responses

USER QUESTION:
{query}

ANSWER:"""
    return prompt

def generate_answer_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    # The generated text is in choices[0].message.content
    return data["choices"][0]["message"]["content"].strip()

def main():
    try:
        collection, embed_model = load_resources()
    except Exception:
        safe_print("Failed to load resources:")
        safe_print(traceback.format_exc())
        return

    safe_print("Ready. Type a question (or 'exit'):")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("Exiting.")
            break

        if not q or q.lower() in ("exit", "quit"):
            break

        try:
            qvec = encode_query(embed_model, q)
            hits = search_milvus(collection, qvec, TOP_K)
            if not hits:
                safe_print("No documents retrieved from Milvus.")
                continue

            prompt = build_prompt(q, hits)
            safe_print("Calling OpenRouter LLaMA-2-7B-Instruct to generate answer...")
            answer = generate_answer_openrouter(prompt)
            safe_print("\n=== Answer ===\n")
            safe_print(answer)
            safe_print("\n=== Sources ===")
            for i, h in enumerate(hits, start=1):
                safe_print(f"[{i}] {h['source']} (Page {h['page']}) score={h['score']:.4f}")

        except Exception:
            safe_print("Error while processing query:")
            safe_print(traceback.format_exc())

if __name__ == "__main__":
    main()
