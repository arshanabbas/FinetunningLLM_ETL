"""
04_verify.py — Pipeline Verification & Test Queries
=====================================================
Verifies the full RAG pipeline is working end-to-end:
  1. Checks Qdrant collection stats
  2. Runs test embedding queries
  3. Performs similarity search with sample IT problems
  4. Generates a full RAG response using qwen2.5:32b

Run after 03_embed_load.py is complete.
Run: python 04_verify.py
"""

import os
import json
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL     = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "qwen2.5:32b")
QDRANT_HOST     = os.getenv("QDRANT_HOST", "http://172.19.4.206")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "otrs_tickets")
TOP_K           = 5   # number of similar tickets to retrieve

# ── Test queries — replace with real examples from your ticket history ────────
TEST_QUERIES = [
    "Drucker druckt nicht",
    "VPN connection fails from home office",
    "Outlook cannot connect to Exchange server",
    "Windows update breaks printer driver",
    "User cannot login after password reset",
]

RAG_SYSTEM_PROMPT = """Du bist ein erfahrener IT-Support-Assistent.
Du erhältst eine IT-Support-Anfrage und relevante historische Tickets aus der Wissensdatenbank.
Antworte strukturiert auf Deutsch (oder Englisch, falls die Anfrage auf Englisch ist) mit:
1. Letztes bekanntes Auftreten (Datum und Ticket-Nummer falls verfügbar)
2. Ursache (aus historischen Tickets)
3. Häufige Probleme dieser Art in der Organisation
4. Empfohlene Lösungsschritte

Zitiere immer die Ticket-Nummern als Referenz.
Sei präzise und technisch korrekt."""


def embed_text(text):
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("embedding", [])


def search_similar(client, query_vector, top_k=TOP_K):
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        score_threshold=0.5,  # ignore very dissimilar results
    )
    return results


def build_rag_context(results):
    """Format search results as context for the LLM."""
    if not results:
        return "No relevant historical tickets found."

    context_parts = []
    for i, r in enumerate(results, 1):
        p = r.payload
        ticket_info = [
            f"Ticket #{p.get('ticket_number', 'N/A')} (Score: {r.score:.2f})",
            f"Title: {p.get('title', 'N/A')}",
            f"Queue: {p.get('queue', 'N/A')}",
            f"Date: {p.get('create_time', 'N/A')[:10] if p.get('create_time') else 'N/A'}",
        ]
        if p.get("question"):
            ticket_info.append(f"Problem: {p['question']}")
        if p.get("answer"):
            ticket_info.append(f"Resolution: {p['answer'][:300]}")

        context_parts.append(f"[{i}] " + " | ".join(ticket_info[:2]) + "\n" +
                              "\n".join(ticket_info[2:]))

    return "\n\n".join(context_parts)


def generate_rag_response(query, context):
    """Generate a RAG response using the LLM."""
    prompt = f"""Historische Tickets aus der Wissensdatenbank:

{context}

---
Aktuelle IT-Support-Anfrage: {query}

Bitte analysiere die historischen Tickets und gib eine strukturierte Antwort."""

    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model":  LLM_MODEL,
            "prompt": prompt,
            "system": RAG_SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "num_predict": 600,
            }
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def check_services():
    """Verify all services are running."""
    print("CHECKING SERVICES")
    print("-" * 40)

    # Ollama
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        models = [m["name"] for m in r.json().get("models", [])]
        embed_ok = any(EMBED_MODEL.split(":")[0] in m for m in models)
        llm_ok   = any(LLM_MODEL.split(":")[0] in m for m in models)
        print(f"  Ollama:            ✅ reachable")
        print(f"  {EMBED_MODEL}:  {'✅' if embed_ok else '❌'} {'loaded' if embed_ok else 'NOT FOUND'}")
        print(f"  {LLM_MODEL}:       {'✅' if llm_ok else '❌'} {'loaded' if llm_ok else 'NOT FOUND'}")
    except Exception as e:
        print(f"  Ollama:            ❌ {e}")
        return False

    # Qdrant
    try:
        r = requests.get(f"{QDRANT_HOST}:{QDRANT_PORT}/healthz", timeout=10)
        print(f"  Qdrant:            ✅ reachable at {QDRANT_HOST}:{QDRANT_PORT}")
    except Exception as e:
        print(f"  Qdrant:            ❌ {e}")
        return False

    return True


def check_collection(client):
    """Check Qdrant collection statistics."""
    print("\nCOLLECTION STATISTICS")
    print("-" * 40)

    try:
        info = client.get_collection(COLLECTION_NAME)
        count = info.points_count
        print(f"  Collection:        '{COLLECTION_NAME}'")
        print(f"  Total vectors:     {count:,}")
        print(f"  Vector size:       {info.config.params.vectors.size}")
        print(f"  Distance metric:   {info.config.params.vectors.distance}")

        if count == 0:
            print("  ⚠️  Collection is empty — run 03_embed_load.py first")
            return False

        # Sample a few points
        sample = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        print(f"\n  Sample tickets in Qdrant:")
        for point in sample[0]:
            p = point.payload
            print(f"    #{p.get('ticket_number')} — {p.get('title', '')[:50]} [{p.get('queue', '')}]")

        return True
    except Exception as e:
        print(f"  ❌ Collection error: {e}")
        return False


def run_test_queries(client):
    """Run similarity search for test queries."""
    print("\nSIMILARITY SEARCH TESTS")
    print("-" * 40)

    for query in TEST_QUERIES:
        print(f"\n  Query: '{query}'")

        # Embed the query
        t0 = time.time()
        vector = embed_text(query)
        embed_ms = (time.time() - t0) * 1000

        # Search Qdrant
        t1 = time.time()
        results = search_similar(client, vector)
        search_ms = (time.time() - t1) * 1000

        print(f"  Embed: {embed_ms:.0f}ms | Search: {search_ms:.0f}ms | Results: {len(results)}")

        for r in results[:3]:
            p = r.payload
            print(f"    [{r.score:.3f}] #{p.get('ticket_number')} — {p.get('title', '')[:50]}")

        if not results:
            print("    No results above threshold 0.5")


def run_full_rag_test(client):
    """Run a complete RAG pipeline test."""
    print("\nFULL RAG PIPELINE TEST")
    print("-" * 40)

    test_query = TEST_QUERIES[0]
    print(f"Query: '{test_query}'")
    print("Generating response...\n")

    # 1. Embed
    t0 = time.time()
    vector = embed_text(test_query)
    print(f"  Step 1 — Embedding:   {(time.time()-t0)*1000:.0f}ms")

    # 2. Search
    t1 = time.time()
    results = search_similar(client, vector)
    print(f"  Step 2 — Qdrant search: {(time.time()-t1)*1000:.0f}ms ({len(results)} results)")

    # 3. Build context
    context = build_rag_context(results)

    # 4. Generate response
    t2 = time.time()
    response = generate_rag_response(test_query, context)
    print(f"  Step 3 — LLM response:  {(time.time()-t2):.1f}s")
    print(f"  Total pipeline:         {(time.time()-t0):.1f}s")

    print("\n" + "=" * 60)
    print("RAG RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)

    return response


def main():
    print("=" * 60)
    print("OTRS RAG PIPELINE VERIFIER — 04_verify.py")
    print("=" * 60)
    print()

    # 1. Check all services
    services_ok = check_services()
    if not services_ok:
        print("\n❌ Service check failed. Fix services before running.")
        return

    # 2. Check Qdrant collection
    client = QdrantClient(
        host=QDRANT_HOST.replace("http://", ""),
        port=QDRANT_PORT,
        timeout=30,
    )
    collection_ok = check_collection(client)
    if not collection_ok:
        return

    # 3. Run similarity search tests
    run_test_queries(client)

    # 4. Full RAG pipeline test
    print()
    run_rag = input("Run full RAG pipeline test with LLM? (y/n): ").strip().lower()
    if run_rag == "y":
        run_full_rag_test(client)

    print("\n✅ Verification complete")
    print("\nNEXT STEP: Configure n8n on R1 to route ticket queries to R2")
    print(f"  Ollama endpoint: {OLLAMA_HOST}/api/embeddings  (embed)")
    print(f"  Ollama endpoint: {OLLAMA_HOST}/api/generate    (generate)")
    print(f"  Qdrant endpoint: {QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/points/search")


if __name__ == "__main__":
    main()
