from functools import lru_cache


@lru_cache(maxsize=4)
def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"[WARN] 读取提示词失败（{path}），返回空字符串：{e}")
        return ""

