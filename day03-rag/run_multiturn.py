"""多轮 RAG 演示：终端打印两轮问答，便于截图。不含 ipynb。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import AIMessage, HumanMessage

from rag_lcel import build_rag_history_chain, get_llm, get_retriever

os.chdir(Path(__file__).resolve().parent)


def main():
    llm = get_llm()
    retriever = get_retriever(k=4)
    rag_history_chain = build_rag_history_chain(llm, retriever)
    chat_history = []

    turns = [
        "孙子说的「五事」分别是什么？",
        "那不知道这五项会怎样？",
        "下一篇《作战》里，用兵最看重什么？",
    ]

    print("=" * 72)
    print("LCEL 多轮 RAG 问答")
    print("=" * 72)

    for i, question in enumerate(turns, 1):
        print(f"\n----- 第 {i} 轮 -----")
        print(f"用户: {question}")
        answer = rag_history_chain.invoke(
            {"question": question, "chat_history": chat_history}
        )
        print(f"助手: {answer}")
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

    print("\n" + "=" * 72)
    print("当前会话历史（共 %d 条消息）" % len(chat_history))
    print("=" * 72)
    for msg in chat_history:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        print(f"{role}: {msg.content[:120]}")
    print("=" * 72)
    print("多轮 RAG 结束。请截取以上终端输出。")


if __name__ == "__main__":
    main()
