"""
01_extract.py — OTRS Ticket Extractor
======================================
Extracts closed tickets from MySQL otrs_restore and saves them
as structured JSON files. One JSON file per ticket containing:
  - ticket metadata (queue, priority, state, dates)
  - all article bodies (conversation history)
  - ticket history events (state changes, notes)
  - dynamic field values (custom fields, asset IDs)

Run: python 01_extract.py
Output: ./data/tickets_raw.jsonl  (one JSON object per line)
"""

import os
import json
import time
import pymysql
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT     = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER     = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
OUTPUT_DIR     = os.getenv("OUTPUT_DIR", "./data")
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", 100))
MIN_WORD_COUNT = int(os.getenv("MIN_WORD_COUNT", 50))
MIN_CREATE_TIME = os.getenv("MIN_CREATE_TIME", "2020-01-01 00:00:00")
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "tickets_raw.jsonl")
PROGRESS_FILE  = os.path.join(OUTPUT_DIR, "extract_progress.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def decode_body(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value

def get_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=30,
    )


def get_total_closed_tickets(conn):
    """Count all closed tickets for progress bar."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) as cnt
            FROM ticket t
            JOIN ticket_state ts ON t.ticket_state_id = ts.id
            JOIN ticket_state_type tst ON ts.type_id = tst.id
            WHERE tst.name IN ('closed', 'removed')
                    AND t.create_time >= %s
        """, (MIN_CREATE_TIME,))
        return cur.fetchone()["cnt"]


def get_ticket_ids_batch(conn, offset, limit):
    """Fetch a batch of closed ticket IDs."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT t.id
            FROM ticket t
            JOIN ticket_state ts ON t.ticket_state_id = ts.id
            JOIN ticket_state_type tst ON ts.type_id = tst.id
            WHERE tst.name IN ('closed', 'removed')
                AND t.create_time >= %s       
            ORDER BY t.id ASC
            LIMIT %s OFFSET %s
        """, (MIN_CREATE_TIME, limit, offset))
        return [row["id"] for row in cur.fetchall()]


def get_ticket_metadata(conn, ticket_id):
    """Get core ticket fields."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                t.id,
                t.tn                        AS ticket_number,
                t.title,
                t.customer_id,
                t.customer_user_id,
                t.create_time,
                t.change_time,
                q.name                      AS queue,
                ts.name                     AS state,
                tst.name                    AS state_type,
                tp.name                     AS priority,
                tt.name                     AS ticket_type,
                CONCAT(u.first_name, ' ', u.last_name) AS owner,
                s.name                      AS service
            FROM ticket t
            LEFT JOIN queue         q  ON t.queue_id           = q.id
            LEFT JOIN ticket_state  ts ON t.ticket_state_id    = ts.id
            LEFT JOIN ticket_state_type tst ON ts.type_id      = tst.id
            LEFT JOIN ticket_priority   tp ON t.ticket_priority_id = tp.id
            LEFT JOIN ticket_type       tt ON t.type_id         = tt.id
            LEFT JOIN users             u  ON t.user_id         = u.id
            LEFT JOIN service           s  ON t.service_id      = s.id
            WHERE t.id = %s
        """, (ticket_id,))
        return cur.fetchone()


def get_ticket_articles(conn, ticket_id):
    """Get all article bodies for a ticket in chronological order."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                a.id                        AS article_id,
                a.create_time,
                ast.name                    AS sender_type,
                admp.body                   AS body_plain,
                adm.a_from                  AS from_addr,
                adm.a_to                    AS to_addr,
                adm.a_subject               AS subject,
                adm.a_content_type          AS content_type
            FROM article a
            LEFT JOIN article_sender_type ast ON a.article_sender_type_id = ast.id
            LEFT JOIN article_data_mime   adm ON adm.article_id = a.id
            LEFT JOIN article_data_mime_plain admp ON admp.article_id = a.id
            WHERE a.ticket_id = %s
            ORDER BY a.create_time ASC
        """, (ticket_id,))
        rows = cur.fetchall()
        articles = []
        for row in rows:
            body = decode_body(row.get("body_plain")).strip()
            if body:
                articles.append({
                    "article_id":   row["article_id"],
                    "sender_type":  row["sender_type"],
                    "from":         row.get("from_addr", ""),
                    "subject":      row.get("subject", ""),
                    "body":         body,
                    "create_time":  str(row["create_time"]),
                })
        return articles


def get_ticket_history_summary(conn, ticket_id):
    """Get key history events — state changes and close events."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                th.name,
                th.create_time,
                tht.name AS history_type,
                CONCAT(u.first_name, ' ', u.last_name) AS agent
            FROM ticket_history th
            LEFT JOIN ticket_history_type tht ON th.history_type_id = tht.id
            LEFT JOIN users u ON th.create_by = u.id
            WHERE th.ticket_id = %s
              AND tht.name IN (
                'StateUpdate','Close','Move','AddNote',
                'PhoneCallAgent','PhoneCallCustomer','EmailAgent',
                'FollowUp','Merged','SetPendingTime'
              )
            ORDER BY th.create_time ASC
        """, (ticket_id,))
        return [
            {
                "event":       row["history_type"],
                "detail":      row["name"],
                "agent":       row["agent"],
                "time":        str(row["create_time"]),
            }
            for row in cur.fetchall()
        ]


def get_dynamic_fields(conn, ticket_id):
    """Get custom dynamic field values for this ticket."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                df.name         AS field_name,
                df.label        AS field_label,
                dfv.value_text,
                dfv.value_int,
                dfv.value_date
            FROM dynamic_field_value dfv
            JOIN dynamic_field df ON dfv.field_id = df.id
            JOIN dynamic_field_obj_id_name dfo
                ON dfv.object_id = dfo.object_id
                AND dfo.object_type = 'Ticket'
            WHERE dfo.object_id = %s
              AND df.object_type = 'Ticket'
              AND df.valid_id = 1
        """, (ticket_id,))
        fields = {}
        for row in cur.fetchall():
            value = row["value_text"] or (
                str(row["value_int"]) if row["value_int"] is not None
                else str(row["value_date"]) if row["value_date"] else None
            )
            if value:
                fields[row["field_label"] or row["field_name"]] = value
        return fields


def build_combined_text(ticket, articles, dynamic_fields):
    """
    Combine all text into one searchable string.
    This is what gets embedded into the vector.
    """
    parts = []

    # Title
    if ticket.get("title"):
        parts.append(f"Ticket: {ticket['title']}")

    # Queue and metadata
    meta = []
    if ticket.get("queue"):    meta.append(f"Queue: {ticket['queue']}")
    if ticket.get("service"):  meta.append(f"Service: {ticket['service']}")
    if ticket.get("priority"): meta.append(f"Priority: {ticket['priority']}")
    if meta:
        parts.append(" | ".join(meta))

    # Dynamic fields (asset IDs, device names, locations, etc.)
    if dynamic_fields:
        df_parts = [f"{k}: {v}" for k, v in dynamic_fields.items()]
        parts.append("Custom fields: " + ", ".join(df_parts))

    # Conversation — all article bodies in order
    for i, article in enumerate(articles):
        role = article["sender_type"] or "unknown"
        body = article["body"]
        if len(body) > 2000:
            body = body[:2000] + "..."
        parts.append(f"[{role.upper()}]: {body}")

    return "\n\n".join(parts)


def count_words(text):
    return len(text.split()) if text else 0


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"last_offset": 0, "processed": 0, "skipped": 0, "errors": 0}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    print("=" * 60)
    print("OTRS TICKET EXTRACTOR — 01_extract.py")
    print("=" * 60)

    conn = get_connection()
    print(f"✅ Connected to MySQL: {MYSQL_DATABASE}@{MYSQL_HOST}")

    total = get_total_closed_tickets(conn)
    print(f"📊 Total closed tickets: {total:,}")
    print(f"📁 Output: {OUTPUT_FILE}")
    print(f"⚙️  Batch size: {BATCH_SIZE} | Min words: {MIN_WORD_COUNT}")
    print()

    # Resume support — don't restart from scratch if interrupted
    progress = load_progress()
    start_offset = progress["last_offset"]
    if start_offset > 0:
        print(f"▶️  Resuming from offset {start_offset:,} ({progress['processed']:,} already done)")

    # Open output file in append mode for resume support
    mode = "a" if start_offset > 0 else "w"
    outfile = open(OUTPUT_FILE, mode, encoding="utf-8")

    stats = {
        "processed": progress["processed"],
        "skipped":   progress["skipped"],
        "errors":    progress["errors"],
        "rich":      0,   # tickets with enough content
        "sparse":    0,   # tickets needing enrichment
    }

    offset = start_offset
    pbar = tqdm(
        total=total,
        initial=offset,
        desc="Extracting tickets",
        unit="ticket",
        dynamic_ncols=True,
    )

    try:
        while True:
            ticket_ids = get_ticket_ids_batch(conn, offset, BATCH_SIZE)
            if not ticket_ids:
                break

            for tid in ticket_ids:
                try:
                    meta     = get_ticket_metadata(conn, tid)
                    articles = get_ticket_articles(conn, tid)
                    history  = get_ticket_history_summary(conn, tid)
                    dyn      = get_dynamic_fields(conn, tid)

                    if not meta:
                        stats["skipped"] += 1
                        continue

                    combined_text = build_combined_text(meta, articles, dyn)
                    word_count    = count_words(combined_text)

                    doc = {
                        "ticket_id":      tid,
                        "ticket_number":  meta.get("ticket_number"),
                        "title":          meta.get("title", ""),
                        "queue":          meta.get("queue", ""),
                        "state":          meta.get("state", ""),
                        "state_type":     meta.get("state_type", ""),
                        "priority":       meta.get("priority", ""),
                        "ticket_type":    meta.get("ticket_type", ""),
                        "owner":          meta.get("owner", ""),
                        "service":        meta.get("service", ""),
                        "customer_id":    meta.get("customer_id", ""),
                        "create_time":    str(meta.get("create_time", "")),
                        "close_time":     str(meta.get("change_time", "")),
                        "articles":       articles,
                        "history":        history,
                        "dynamic_fields": dyn,
                        "combined_text":  combined_text,
                        "word_count":     word_count,
                        # Flag: does this ticket have enough content?
                        "needs_enrichment": word_count < MIN_WORD_COUNT,
                        "enriched":         False,
                        "question":         None,
                        "answer":           None,
                    }

                    outfile.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
                    stats["processed"] += 1

                    if doc["needs_enrichment"]:
                        stats["sparse"] += 1
                    else:
                        stats["rich"] += 1

                except Exception as e:
                    stats["errors"] += 1
                    tqdm.write(f"  ⚠️  Error on ticket {tid}: {e}")

                pbar.update(1)

            offset += BATCH_SIZE
            progress["last_offset"] = offset
            progress["processed"]   = stats["processed"]
            progress["skipped"]     = stats["skipped"]
            progress["errors"]      = stats["errors"]
            save_progress(progress)

            # Flush to disk every batch
            outfile.flush()

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — progress saved, run again to resume")
    finally:
        outfile.close()
        pbar.close()
        conn.close()

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  ✅ Processed:          {stats['processed']:,}")
    print(f"  📝 Rich tickets:       {stats['rich']:,}  (≥{MIN_WORD_COUNT} words)")
    print(f"  ⚠️  Sparse tickets:     {stats['sparse']:,}  (needs LLM enrichment)")
    print(f"  ⏭️  Skipped:            {stats['skipped']:,}")
    print(f"  ❌ Errors:             {stats['errors']:,}")
    print(f"  📁 Output:             {OUTPUT_FILE}")

    # Show file size
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
        print(f"  💾 File size:          {size_mb:.1f} MB")

    # Quick sample of first extracted ticket
    print()
    print("SAMPLE — first extracted ticket:")
    print("-" * 40)
    with open(OUTPUT_FILE) as f:
        sample = json.loads(f.readline())
    print(f"  Ticket #:   {sample['ticket_number']}")
    print(f"  Title:      {sample['title'][:60]}")
    print(f"  Queue:      {sample['queue']}")
    print(f"  Articles:   {len(sample['articles'])}")
    print(f"  Words:      {sample['word_count']}")
    print(f"  Needs enrichment: {sample['needs_enrichment']}")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")