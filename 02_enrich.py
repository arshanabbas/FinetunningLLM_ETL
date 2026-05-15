"""
02_enrich.py — LLM Ticket Enricher
=====================================
Reads tickets_raw.jsonl and enriches sparse tickets (< MIN_WORD_COUNT words)
by asking qwen2.5:32b to generate a structured Question and Answer
based on whatever context is available (title, queue, dynamic fields, history).

Rich tickets pass through unchanged.
Results saved to ./data/tickets_enriched.jsonl

Run: python 02_enrich.py
"""

import os
import json
import time
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL      = os.getenv("LLM_MODEL", "qwen2.5:32b")
MIN_WORD_COUNT = int(os.getenv("MIN_WORD_COUNT", 50))
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "./data")

INPUT_FILE     = os.path.join(OUTPUT_DIR, "tickets_raw.jsonl")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "tickets_enriched.jsonl")
PROGRESS_FILE  = os.path.join(OUTPUT_DIR, "enrich_progress.json")

# How long to wait for LLM response (seconds)
LLM_TIMEOUT    = 120

# System prompt for enrichment
SYSTEM_PROMPT = """You are an IT support knowledge base assistant.
You receive incomplete IT support ticket information and must generate:
1. A clear, specific QUESTION that summarizes what the problem was
2. A helpful ANSWER that describes the likely solution or resolution

Be concise and technical. Focus on IT support terminology.
Always respond in the SAME LANGUAGE as the ticket content (German or English).
Respond ONLY with valid JSON in this exact format:
{"question": "...", "answer": "..."}
Do not add any explanation, markdown, or extra text."""


def build_enrichment_prompt(doc):
    """Build a prompt for the LLM from sparse ticket data."""
    parts = ["Ticket information available:"]

    if doc.get("title"):
        parts.append(f"Title: {doc['title']}")
    if doc.get("queue"):
        parts.append(f"Queue/Category: {doc['queue']}")
    if doc.get("service"):
        parts.append(f"Service: {doc['service']}")
    if doc.get("priority"):
        parts.append(f"Priority: {doc['priority']}")
    if doc.get("create_time"):
        parts.append(f"Created: {doc['create_time']}")

    # Dynamic fields often have the most useful info for sparse tickets
    if doc.get("dynamic_fields"):
        parts.append("Custom fields:")
        for k, v in doc["dynamic_fields"].items():
            parts.append(f"  {k}: {v}")

    # History events give clues about resolution
    if doc.get("history"):
        parts.append("History events:")
        for h in doc["history"][:5]:  # max 5 events
            parts.append(f"  [{h['event']}] {h['detail']} by {h['agent']} at {h['time']}")

    # Any partial article content
    if doc.get("articles"):
        parts.append("Available message content:")
        for article in doc["articles"][:3]:
            body = article.get("body", "")[:300]
            if body:
                parts.append(f"  [{article['sender_type']}]: {body}")

    parts.append("\nGenerate a question and answer for this IT ticket.")
    return "\n".join(parts)


def call_ollama_enrich(prompt, retries=3):
    """Call Ollama LLM to generate Q&A for a sparse ticket."""
    payload = {
        "model":  LLM_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temp = more consistent structured output
            "num_predict": 300,   # Q&A doesn't need to be long
        }
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip()

            # Try to parse JSON response
            # Sometimes LLM wraps in ```json ... ``` — strip that
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            parsed = json.loads(raw)
            question = parsed.get("question", "").strip()
            answer   = parsed.get("answer", "").strip()

            if question and answer:
                return question, answer, True

        except json.JSONDecodeError:
            pass  # retry
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(5)
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(3)

    return None, None, False


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": 0, "enriched": 0, "failed": 0, "passed_through": 0}


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
    print("OTRS TICKET ENRICHER — 02_enrich.py")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        print("   Run 01_extract.py first.")
        return

    total_lines = count_lines(INPUT_FILE)
    progress    = load_progress()
    already_done = progress["processed"]

    print(f"📂 Input:    {INPUT_FILE} ({total_lines:,} tickets)")
    print(f"📁 Output:   {OUTPUT_FILE}")
    print(f"🤖 Model:    {LLM_MODEL}")
    print(f"⚙️  Min words for enrichment: {MIN_WORD_COUNT}")

    # Count how many need enrichment
    sparse_count = 0
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("needs_enrichment"):
                sparse_count += 1
    print(f"⚠️  Sparse tickets needing enrichment: {sparse_count:,}")
    print(f"✅ Rich tickets (pass-through):        {total_lines - sparse_count:,}")

    if already_done > 0:
        print(f"▶️  Resuming from ticket {already_done:,}")
    print()

    # Check Ollama is reachable
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
        r.raise_for_status()
        print(f"✅ Ollama reachable at {OLLAMA_HOST}")
    except Exception as e:
        print(f"❌ Cannot reach Ollama: {e}")
        return

    stats = {k: v for k, v in progress.items()}
    mode = "a" if already_done > 0 else "w"
    outfile = open(OUTPUT_FILE, mode, encoding="utf-8")

    pbar = tqdm(
        total=total_lines,
        initial=already_done,
        desc="Enriching tickets",
        unit="ticket",
        dynamic_ncols=True,
    )

    try:
        with open(INPUT_FILE, encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                # Skip already processed lines on resume
                if i < already_done:
                    continue

                doc = json.loads(line.strip())

                if doc.get("needs_enrichment"):
                    prompt = build_enrichment_prompt(doc)
                    question, answer, success = call_ollama_enrich(prompt)

                    if success:
                        doc["question"] = question
                        doc["answer"]   = answer
                        doc["enriched"] = True
                        # Update combined_text with the generated Q&A
                        doc["combined_text"] = (
                            f"Question: {question}\n\n"
                            f"Answer: {answer}\n\n"
                            f"Original context:\n{doc['combined_text']}"
                        )
                        doc["word_count"] = len(doc["combined_text"].split())
                        stats["enriched"] += 1
                    else:
                        # LLM failed — keep as-is, flag it
                        doc["enriched"] = False
                        doc["question"] = None
                        doc["answer"]   = None
                        stats["failed"] += 1
                        tqdm.write(f"  ⚠️  Enrichment failed for ticket {doc['ticket_id']}")
                else:
                    # Rich ticket — generate Q&A from existing content too
                    # for better retrieval quality
                    doc["question"] = doc.get("title", "")
                    doc["answer"]   = ""
                    # Use first agent reply as the answer if available
                    for article in doc.get("articles", []):
                        if article.get("sender_type") in ("agent", "system"):
                            doc["answer"] = article["body"][:500]
                            break
                    doc["enriched"] = True
                    stats["passed_through"] += 1

                outfile.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
                stats["processed"] += 1
                pbar.update(1)

                # Save progress every 50 tickets
                if stats["processed"] % 50 == 0:
                    save_progress(stats)
                    outfile.flush()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — progress saved, run again to resume")
    finally:
        outfile.close()
        pbar.close()
        save_progress(stats)

    print()
    print("=" * 60)
    print("ENRICHMENT COMPLETE")
    print("=" * 60)
    print(f"  ✅ Total processed:     {stats['processed']:,}")
    print(f"  🤖 LLM enriched:        {stats['enriched']:,}")
    print(f"  📝 Rich pass-through:   {stats['passed_through']:,}")
    print(f"  ❌ Enrichment failed:   {stats['failed']:,}")

    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"  💾 Output size:         {size_mb:.1f} MB")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed/60:.1f} min")
