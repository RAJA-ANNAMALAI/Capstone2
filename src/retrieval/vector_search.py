from src.core.db import get_db_conn, _embeddings_model
import os

def vector_search(query: str, k: int = 5, chunk_type: str | None = None) -> list[dict]:
    """Find the k most similar chunks to a natural-language query."""
    query_vec = _embeddings_model.embed_query(query)
    embedding_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    type_clause = "AND chunk_type = %(chunk_type)s" if chunk_type else ""

    sql = f"""
        SELECT
            content, chunk_type, page_number, section,
            source_file, element_type, image_path, mime_type,
            position, metadata,
            1 - (embedding <=> %(vec)s::vector) AS similarity
        FROM multimodal_chunks
        WHERE 1=1 {type_clause}
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
    """

    with get_db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {"vec": embedding_str, "chunk_type": chunk_type, "k": k})
            rows = cur.fetchall()

    # --- SIMPLIFIED LOOP ---
    results = []
    for row in rows:
        row_dict = dict(row)
        # We just keep image_path as it is in the database!
        if "image_path" not in row_dict:
            row_dict["image_path"] = None
        results.append(row_dict)

    return results