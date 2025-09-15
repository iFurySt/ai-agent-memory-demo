from typing import Optional, Sequence
from langchain_openai import OpenAIEmbeddings
from .config import AppConfig


class Embedder:
    def __init__(self, cfg: AppConfig):
        self.dim = cfg.embedding_dim
        self._emb = None
        try:
            self._emb = OpenAIEmbeddings(
                model=cfg.embedding_model,
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_base_url,
                dimensions=cfg.embedding_dim,
                check_embedding_ctx_length=False,
            )
        except Exception as e:
            print(f"[WARN] 初始化 Embeddings 失败，语义记忆将不可用: {e}")
            self._emb = None

    @property
    def available(self) -> bool:
        return self._emb is not None

    def embed(self, text: str) -> Optional[Sequence[float]]:
        if not self._emb:
            return None
        return self._emb.embed_query(text)

    @staticmethod
    def to_pgvector_literal(values: Sequence[float]) -> str:
        return "[" + ", ".join(f"{v:.8f}" for v in values) + "]"

