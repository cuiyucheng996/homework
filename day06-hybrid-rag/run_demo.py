"""作业演示：两句示例分别走 LocalRAG / GraphRAG。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import dotenv
from langchain_core.prompts import ChatPromptTemplate

from information_retrieval import GraphRAG

for parent in Path(__file__).resolve().parents:
    env_file = parent / ".env"
    if env_file.is_file() and (parent / "pyproject.toml").is_file():
        dotenv.load_dotenv(env_file)
        break

QUERIES = [
    "中国科学院2023年部门总预算是多少",
    "推荐一个款苹果品牌的手机",
]


async def main() -> None:
    graph_rag = GraphRAG(embeddings=None)
    graph_rag.connect(neo4j_url="neo4j://localhost", neo4j_auth=("neo4j", "12345678"))
    for query in QUERIES:
        print("=" * 60)
        print("问题:", query)
        docs = await graph_rag.search(query=query)
        print("检索路径:", graph_rag.last_source)
        context = "\n".join(d.page_content for d in docs if d.page_content != "空")[:2500]
        print("检索摘录:\n", context[:800])
        if graph_rag.llm is None:
            continue
        prompt = ChatPromptTemplate.from_template(
            "检索路径已确定为 {source}。根据材料回答。\n问题：{query}\n材料：{context}"
        )
        msg = prompt.format_prompt(source=graph_rag.last_source, query=query, context=context or "无")
        out = await graph_rag.llm.ainvoke(msg.to_messages())
        print("回答:\n", out.content)


if __name__ == "__main__":
    asyncio.run(main())
