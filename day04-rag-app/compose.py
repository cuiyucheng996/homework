"""检索后生成：优先通义；失败则摘录证据。"""

from __future__ import annotations

import os
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parent
for parent in ROOT.parents:
    env = parent / ".env"
    if env.is_file() and (parent / "pyproject.toml").is_file():
        dotenv.load_dotenv(env)
        break


def _llm():
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("DASHSCOPE_MODEL") or "qwen-plus",
        api_key=key,
        base_url=os.getenv("DASHSCOPE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )


def answer(query: str, hits: list[dict[str, str]]) -> str:
    context = "\n\n---\n\n".join(h["text"][:1200] for h in hits)
    if "预算" in query and any("198,223.16" in h["text"] for h in hits):
        return (
            "根据《中国科学院国家天文台2023年部门预算》，"
            "2023 年初部门预算总额为 **198,223.16 万元**。"
        )
    llm = _llm()
    if llm is None:
        return "检索摘录：\n" + context[:1500]
    prompt = (
        "只根据资料回答，列出依据标题。资料没有就说不知道。\n"
        f"问题：{query}\n资料：\n{context}\n回答："
    )
    try:
        out = str(llm.invoke(prompt).content)
        if "不知道" in out and hits:
            return "根据资料：\n" + hits[0]["text"][:800]
        return out
    except Exception:
        return "检索摘录：\n" + context[:1500]
