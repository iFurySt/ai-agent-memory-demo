from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

from .config import AppConfig
from .db import FactStore
from .facts import extract_facts_via_llm
from .prompts import load_text


class LLMService:
    def __init__(self, cfg: AppConfig, fact_store: FactStore):
        self.cfg = cfg
        self.fact_store = fact_store

    def call_llm(self, state: MessagesState) -> Dict[str, Any]:
        llm = ChatOpenAI(
            model=self.cfg.chat_model,
            api_key=self.cfg.openai_api_key,
            base_url=self.cfg.openai_base_url,
        )

        thread_id = state.get("configurable", {}).get("thread_id", "default")
        last_msg = state["messages"][-1]

        txt = last_msg.content
        facts_extracted = extract_facts_via_llm(txt, llm, self.cfg)
        for f in facts_extracted:
            self.fact_store.store(thread_id, f)

        facts = self.fact_store.retrieve(thread_id, txt)

        prompt: List[Tuple[str, str]] = []
        # System persona prompt
        system_prompt = load_text(self.cfg.system_prompt_path)
        if system_prompt:
            prompt.append(("system", system_prompt))
        if facts:
            prompt.append(("system", f"以下是我记住的一些相关信息：{facts}"))
        prompt.append((last_msg.type, last_msg.content))

        print("\n--- 本轮实际发送给 LLM 的上下文 ---")
        for role, content in prompt:
            print(role.upper(), ":", content)
        print("---------------------\n")

        resp = llm.invoke(prompt)
        return {"messages": [resp]}


def build_graph(service: LLMService) -> StateGraph:
    builder = StateGraph(MessagesState)
    builder.add_node("llm", service.call_llm)
    builder.add_edge(START, "llm")
    builder.add_edge("llm", END)
    return builder
