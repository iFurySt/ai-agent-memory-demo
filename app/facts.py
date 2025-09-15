import json
import re
from functools import lru_cache
from typing import List, Optional
from langchain_openai import ChatOpenAI

from .config import AppConfig


@lru_cache(maxsize=1)
def load_fact_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[WARN] 读取事实抽取提示词失败，使用内置默认：{e}")
        return (
            "你是一个中文信息抽取器。给定用户本轮消息，"
            "请抽取可以长期记忆的事实（如姓名、偏好、习惯、重要偏好设置等）。"
            "严格输出 JSON，键为 facts，值为字符串数组。例如：\n"
            "{\"facts\":[\"用户的名字是 小王\",\"用户的兴趣爱好是 篮球\"]}"
        )


def _extract_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def extract_facts_via_llm(last_user_text: str, llm: ChatOpenAI, cfg: AppConfig) -> List[str]:
    prompt_text = load_fact_prompt(cfg.fact_prompt_path)
    try:
        sys_prompt = prompt_text.format(text=last_user_text, input=last_user_text)
    except Exception:
        sys_prompt = prompt_text

    messages = [("system", sys_prompt), ("user", last_user_text)]
    try:
        resp = llm.invoke(messages)
        content = getattr(resp, "content", "") or ""
        data = _extract_json(content)
        if isinstance(data, dict) and isinstance(data.get("facts"), list):
            facts = [str(x).strip() for x in data.get("facts") if str(x).strip()]
            seen = set()
            deduped = []
            for f in facts:
                if f not in seen:
                    deduped.append(f)
                    seen.add(f)
            return deduped
    except Exception as e:
        print(f"[WARN] 事实抽取调用失败（已跳过）：{e}")
    return []

