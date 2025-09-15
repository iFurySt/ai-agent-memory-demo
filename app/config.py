import os
import re
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


def _normalize_pg_uri(uri: str):
    """Return SQLAlchemy and psycopg styles: (sa_conn, psy_conn)."""
    if not uri:
        return uri, uri
    if uri.startswith("postgres://"):
        psy_conn = "postgresql://" + uri[len("postgres://"):]
    elif uri.startswith("postgresql://"):
        psy_conn = uri
    else:
        psy_conn = uri

    if psy_conn.startswith("postgresql://"):
        sa_conn = "postgresql+psycopg://" + psy_conn[len("postgresql://"):]
    else:
        sa_conn = psy_conn
    return sa_conn, psy_conn


def _mask_conn_str(uri: str) -> str:
    """Mask password in connection string for logs."""
    if not uri:
        return uri
    try:
        return re.sub(r"(\w+://[^:\s/]+):[^@\s]+@", r"\1:***@", uri)
    except Exception:
        return uri


@dataclass
class AppConfig:
    openai_api_key: str
    openai_base_url: str
    postgres_uri: str
    chat_model: str
    embedding_model: str
    embedding_dim: int
    fact_prompt_path: str
    system_prompt_path: str
    sa_conn_str: str
    pg_conn_str: str

    def print_startup(self):
        print("-- 配置信息 --")
        print(f"Base URL       : {self.openai_base_url}")
        print(f"Chat Model     : {self.chat_model or '(未设置)'}")
        print(f"Embed Model    : {self.embedding_model or '(未设置)'}")
        print(f"Embed Dim      : {self.embedding_dim}")
        print(f"Postgres URI   : {_mask_conn_str(self.postgres_uri)}")
        print(f"Fact Prompt    : {self.fact_prompt_path}")
        print(f"System Prompt  : {self.system_prompt_path}")
        print("----------------")


def load_config() -> AppConfig:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_base_url = os.getenv("OPENAI_BASE_URL")
    postgres_uri = os.getenv("POSTGRES_URI")
    chat_model = os.getenv("CHAT_MODEL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    fact_prompt_path = os.getenv("FACT_PROMPT_PATH", "prompts/fact_extraction.prompt")
    system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "prompts/system.prompt")

    if not openai_api_key:
        raise ValueError("请先设置 OPENAI_API_KEY")
    if not openai_base_url:
        raise ValueError("请先设置 OPENAI_BASE_URL")
    if not postgres_uri:
        raise ValueError("请先设置 POSTGRES_URI")

    sa_conn_str, pg_conn_str = _normalize_pg_uri(postgres_uri)

    return AppConfig(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        postgres_uri=postgres_uri,
        chat_model=chat_model,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        fact_prompt_path=fact_prompt_path,
        system_prompt_path=system_prompt_path,
        sa_conn_str=sa_conn_str,
        pg_conn_str=pg_conn_str,
    )
