from typing import List
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .embedding import Embedder


def init_db(sa_conn_str: str, embedding_dim: int) -> Engine:
    engine = create_engine(sa_conn_str)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS facts (
                id SERIAL PRIMARY KEY,
                thread_id TEXT,
                content TEXT,
                embedding vector({embedding_dim})
            )
        """))
    return engine


class FactStore:
    def __init__(self, engine: Engine, embedder: Embedder):
        self.engine = engine
        self.embedder = embedder

    def store(self, thread_id: str, text_content: str) -> None:
        if not self.embedder.available:
            return
        try:
            emb = self.embedder.embed(text_content)
            if emb is None:
                return
            vec = Embedder.to_pgvector_literal(emb)
            with self.engine.begin() as conn:
                conn.execute(
                    text("INSERT INTO facts (thread_id, content, embedding) VALUES (:tid, :c, CAST(:e AS vector))"),
                    {"tid": thread_id, "c": text_content, "e": vec},
                )
        except Exception as e:
            print(f"[WARN] 写入长期记忆失败（已跳过）：{e}")

    def retrieve(self, thread_id: str, query: str, k: int = 3) -> List[str]:
        if not self.embedder.available:
            return []
        try:
            q_vec = self.embedder.embed(query)
            if q_vec is None:
                return []
            vec = Embedder.to_pgvector_literal(q_vec)
            with self.engine.begin() as conn:
                rows = conn.execute(
                    text(
                        """
                        SELECT content
                        FROM facts
                        WHERE thread_id = :tid
                        ORDER BY embedding <=> CAST(:e AS vector) ASC
                        LIMIT :k
                        """
                    ),
                    {"tid": thread_id, "e": vec, "k": int(k)},
                ).fetchall()
            results = []
            seen = set()
            for r in rows:
                if not r or not r[0]:
                    continue
                c = str(r[0]).strip()
                if c and c not in seen:
                    results.append(c)
                    seen.add(c)
            return results
        except Exception as e:
            print(f"[WARN] 读取长期记忆失败（已跳过）：{e}")
            return []

