import os
from src.core.db import get_db_conn

# Keyword search
def fts_search(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:

    print("\n [FTS SEARCH START]")

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content,
            chunk_type,
            page_number,
            section,
            source_file,
            element_type,
            image_path,
            mime_type,
            position,
            metadata,
            ts_rank_cd(
                to_tsvector('english', content),
                plainto_tsquery('english', %(query)s)
            ) AS score
        FROM multimodal_chunks
        WHERE to_tsvector('english', content)
              @@ plainto_tsquery('english', %(query)s)
        {type_clause}
        ORDER BY score DESC
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {
                "query": query,
                "k": k,
                "chunk_type": chunk_type
            })
            rows = cur.fetchall()

    # --- SIMPLIFIED LOOP ---
    results = []
    for row in rows:
        row_dict = dict(row)

        # Ensure image_path stays in the dictionary (NO POPPING!)
        if "image_path" not in row_dict:
            row_dict["image_path"] = None

        # Rename score → similarity (for consistency)
        if "score" in row_dict:
            row_dict["similarity"] = float(row_dict.pop("score"))

        results.append(row_dict)

    print(f" [FTS DONE] Returned {len(results)} results")

    return results