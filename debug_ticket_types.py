import os
import pymysql
from dotenv import load_dotenv

load_dotenv(".env")

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

ticket_ids = [47246, 47248, 47249, 47250, 47251]

conn = pymysql.connect(
    host=MYSQL_HOST,
    port=MYSQL_PORT,
    user=MYSQL_USER,
    password=MYSQL_PASSWORD,
    database=MYSQL_DATABASE,
    charset="utf8mb4",
    cursorclass=pymysql.cursors.DictCursor,
)

queries = {
    "metadata": """
        SELECT t.id, t.tn, t.title, t.customer_id, t.customer_user_id,
               t.create_time, t.change_time
        FROM ticket t
        WHERE t.id = %s
    """,

    "articles": """
        SELECT
            a.id AS article_id,
            a.create_time,
            ast.name AS sender_type,
            admp.body AS body_plain,
            adm.a_from AS from_addr,
            adm.a_to AS to_addr,
            adm.a_subject AS subject,
            adm.a_content_type AS content_type
        FROM article a
        LEFT JOIN article_sender_type ast ON a.article_sender_type_id = ast.id
        LEFT JOIN article_data_mime adm ON adm.article_id = a.id
        LEFT JOIN article_data_mime_plain admp ON admp.article_id = a.id
        WHERE a.ticket_id = %s
        ORDER BY a.create_time ASC
    """,

    "dynamic_fields": """
        SELECT
            df.name AS field_name,
            df.label AS field_label,
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
    """,

    "history": """
        SELECT
            th.name,
            th.create_time,
            tht.name AS history_type
        FROM ticket_history th
        LEFT JOIN ticket_history_type tht ON th.history_type_id = tht.id
        WHERE th.ticket_id = %s
        ORDER BY th.create_time ASC
        LIMIT 10
    """
}

def preview(value, max_len=80):
    if value is None:
        return "None"
    if isinstance(value, bytes):
        return repr(value[:max_len])
    return repr(str(value)[:max_len])

with conn.cursor() as cur:
    for ticket_id in ticket_ids:
        print("\n" + "=" * 80)
        print(f"TICKET {ticket_id}")
        print("=" * 80)

        for name, sql in queries.items():
            print(f"\n--- {name.upper()} ---")
            cur.execute(sql, (ticket_id,))
            rows = cur.fetchall()

            if not rows:
                print("No rows")
                continue

            for i, row in enumerate(rows[:3], start=1):
                print(f"\nRow {i}:")
                for key, value in row.items():
                    print(
                        f"{key:20} type={type(value).__name__:10} "
                        f"value={preview(value)}"
                    )

conn.close()