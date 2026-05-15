"""
03_embed_load.py — Embedder & Qdrant Loader
=============================================
Reads tickets_enriched.jsonl, generates 768-dim embeddings using
nomic-embed-text via Ollama, and upserts them into Qdrant.

Each Qdrant point contains:
  - vector:   768-dim embedding of combined_text
  - payload:  ticket metadata for filtering and display

Run: python 03_embed_load.py
"""

import os
import json
import time
import uuid
import requests
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    OptimizersConfigDiff, HnswConfigDiff
)
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST       = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "nomic-embed-text")
QDRANT_HOST       = os.getenv("QDRANT_HOST", "http://172.19.4.206")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "otrs_tickets")
OUTPUT_DIR        = os.getenv("OUTPUT_DIR", "./data")
BATCH_SIZE        = int(os.getenv("BATCH_SIZE", 50))  # smaller for embedding

INPUT_FILE        = os.path.join(OUTPUT_DIR, "tickets_enriched.jsonl")
PROGRESS_FILE     = os.path.join(OUTPUT_DIR, "embed_progress.json")

VECTOR_SIZE       = 768   # nomic-embed-text output dimensions
EMBED_TIMEOUT     = 60    # seconds per embedding request


def get_qdrant_client():
    return QdrantClient(
        host=QDRANT_HOST.replace("http://", "").replace("https://", ""),
        port=QDRANT_PORT,
        timeout=30,
    )


def ensure_collection(client):
    """Create Qdrant collection if it doesn't exist."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        info = client.get_collection(COLLECTION_NAME)
        count = info.points_count
        print(f"✅ Collection '{COLLECTION_NAME}' exists — {count:,} points already loaded")
        return count
    else:
        print(f"🆕 Creating collection '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
            # Tuned for our dataset size (~43K tickets)
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000,
            ),
        )
        print(f"✅ Collection created")
        return 0


def embed_text(text, retries=3):
    """Generate embedding vector using nomic-embed-text via Ollama."""
    # Truncate to avoid token limit issues (nomic-embed-text: 8192 tokens)
    if len(text) > 6000:
        text = text[:6000]

    payload = {
        "model":  EMBED_MODEL,
        "prompt": text,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json=payload,
                timeout=EMBED_TIMEOUT,
            )
            resp.raise_for_status()
            vector = resp.json().get("embedding", [])
            if len(vector) == VECTOR_SIZE:
                return vector
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)
            else:
                raise

    return None


def build_qdrant_payload(doc):
    """Build the metadata payload stored alongside each vector in Qdrant."""
    return {
        # Core identifiers
        "ticket_id":      doc.get("ticket_id"),
        "ticket_number":  doc.get("ticket_number"),

        # Display fields (shown in RAG response)
        "title":          doc.get("title", "")[:200],
        "queue":          doc.get("queue", ""),
        "state":          doc.get("state", ""),
        "priority":       doc.get("priority", ""),
        "service":        doc.get("service", ""),
        "owner":          doc.get("owner", ""),

        # Generated Q&A
        "question":       (doc.get("question") or "")[:500],
        "answer":         (doc.get("answer") or "")[:1000],

        # Timestamps
        "create_time":    doc.get("create_time", ""),
        "close_time":     doc.get("close_time", ""),

        # Quality flags
        "enriched":       doc.get("enriched", False),
        "word_count":     doc.get("word_count", 0),

        # Article count — useful for filtering
        "article_count":  len(doc.get("articles", [])),

        # Dynamic fields flattened for filtering
        "dynamic_fields": json.dumps(doc.get("dynamic_fields", {}), ensure_ascii=False),
    }


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": 0, "loaded": 0, "failed": 0}


def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def count_lines(filepath):
    count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count


def main():
    print("=" * 60)
    print("OTRS EMBEDDER & QDRANT LOADER — 03_embed_load.py")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Run 02_enrich.py first.")
        return

    total_lines = count_lines(INPUT_FILE)
    progress    = load_progress()
    already_done = progress["processed"]

    print(f"📂 Input:      {INPUT_FILE} ({total_lines:,} tickets)")
    print(f"🔢 Embed model: {EMBED_MODEL} ({VECTOR_SIZE} dims)")
    print(f"🗄️  Qdrant:     {QDRANT_HOST}:{QDRANT_PORT} → '{COLLECTION_NAME}'")

    # Verify Ollama embedding works
    print("\n🔍 Testing embedding model...")
    try:
        test_vec = embed_text("test connection")
        if test_vec and len(test_vec) == VECTOR_SIZE:
            print(f"✅ nomic-embed-text working — {len(test_vec)} dimensions")
        else:
            print("❌ Embedding test failed — wrong dimensions")
            return
    except Exception as e:
        print(f"❌ Cannot reach Ollama: {e}")
        return

    # Connect to Qdrant
    print("\n🔍 Connecting to Qdrant...")
    try:
        client = get_qdrant_client()
        existing_count = ensure_collection(client)
    except Exception as e:
        print(f"❌ Cannot reach Qdrant: {e}")
        return

    if already_done > 0:
        print(f"\n▶️  Resuming from ticket {already_done:,}")
    print()

    stats = {k: v for k, v in progress.items()}
    batch_points = []
    batch_count  = 0

    pbar = tqdm(
        total=total_lines,
        initial=already_done,
        desc="Embedding + loading",
        unit="ticket",
        dynamic_ncols=True,
        postfix={"loaded": stats["loaded"], "failed": stats["failed"]},
    )

    def flush_batch(points):
        """Upsert a batch of points into Qdrant."""
        if not points:
            return
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )
        except Exception as e:
            tqdm.write(f"  ❌ Batch upsert failed: {e}")
            raise

    try:
        with open(INPUT_FILE, encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                if i < already_done:
                    continue

                doc = json.loads(line.strip())

                # Use combined_text for embedding
                text_to_embed = doc.get("combined_text", "")
                if not text_to_embed.strip():
                    stats["failed"] += 1
                    pbar.update(1)
                    continue

                # Generate embedding
                vector = embed_text(text_to_embed)

                if vector is None:
                    stats["failed"] += 1
                    tqdm.write(f"  ⚠️  Embedding failed for ticket {doc['ticket_id']}")
                    pbar.update(1)
                    continue

                # Build Qdrant point
                # Use ticket_id as the point ID (deterministic — allows re-runs)
                point = PointStruct(
                    id=doc["ticket_id"],
                    vector=vector,
                    payload=build_qdrant_payload(doc),
                )
                batch_points.append(point)
                stats["loaded"] += 1

                # Flush batch when full
                if len(batch_points) >= BATCH_SIZE:
                    flush_batch(batch_points)
                    batch_points = []
                    batch_count += 1

                stats["processed"] += 1
                pbar.update(1)
                pbar.set_postfix({
                    "loaded": stats["loaded"],
                    "failed": stats["failed"],
                })

                # Save progress every 200 tickets
                if stats["processed"] % 200 == 0:
                    save_progress(stats)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — flushing remaining batch...")
    finally:
        # Flush any remaining points
        if batch_points:
            flush_batch(batch_points)
        pbar.close()
        save_progress(stats)

    # Final collection count
    try:
        final_count = client.get_collection(COLLECTION_NAME).points_count
    except:
        final_count = stats["loaded"]

    print()
    print("=" * 60)
    print("EMBEDDING & LOADING COMPLETE")
    print("=" * 60)
    print(f"  ✅ Processed:        {stats['processed']:,}")
    print(f"  📥 Loaded to Qdrant: {stats['loaded']:,}")
    print(f"  ❌ Failed:           {stats['failed']:,}")
    print(f"  🗄️  Qdrant total:     {final_count:,} points in '{COLLECTION_NAME}'")

    # Estimate VRAM usage note
    print()
    print("NOTE: During embedding, nvidia-smi may show increased VRAM usage.")
    print("This is normal — nomic-embed-text loads alongside qwen2.5:32b.")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed/60:.1f} min")
