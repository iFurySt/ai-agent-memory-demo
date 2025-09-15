from langgraph.checkpoint.postgres import PostgresSaver

from .config import load_config
from .embedding import Embedder
from .db import init_db, FactStore
from .llm_node import LLMService, build_graph


def run():
    print(
        ">>> LangGraph Long-term Memory Demo (Postgres + pgvector, v1.0.x)"
    )
    cfg = load_config()
    cfg.print_startup()

    engine = init_db(cfg.sa_conn_str, cfg.embedding_dim)
    embedder = Embedder(cfg)
    fact_store = FactStore(engine, embedder)

    service = LLMService(cfg, fact_store)
    builder = build_graph(service)

    with PostgresSaver.from_conn_string(cfg.pg_conn_str) as checkpointer:
        checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "demo-thread"}}
        while True:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            for event in graph.stream({"messages": [("human", user_input)]}, config=config):
                for value in event.values():
                    print("AI:", value["messages"][-1].content)

