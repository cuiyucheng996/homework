"""基于 LCEL 的多轮 RAG：问题重述 → 检索 → 带历史生成。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from indexing import CHROMA_DIR, get_vectorstore

UVGRAPH_ROOT = Path(r"D:\program\uvgraph\uvgraph")
sys.path.insert(0, str(UVGRAPH_ROOT))

from dotenv import load_dotenv

load_dotenv(UVGRAPH_ROOT / ".env")


def get_llm():
    from src.llm.chat import get_chat_model

    llm = get_chat_model()
    if llm is None:
        raise RuntimeError("未配置 DeepSeek / Kimi / OpenAI")
    return llm


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(history) -> str:
    if not history:
        return "（无）"
    lines = []
    for message in history[-6:]:
        if isinstance(message, HumanMessage):
            lines.append(f"用户: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"AI: {message.content}")
        else:
            lines.append(str(message))
    return "\n".join(lines)


def get_retriever(k=4):
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        from indexing import build_index

        build_index()
    store = get_vectorstore()
    return store.as_retriever(search_type="similarity", search_kwargs={"k": k})


def build_rephrase_chain(llm):
    # LCEL：历史 + 当前问题 → 独立可检索问句
    prompt = PromptTemplate.from_template(
        """根据对话历史，把最新用户消息改写成一个独立、具体的检索问题。
只输出改写后的问题。若不需要改写，直接输出原问题。

历史：
{chat_history}

用户：{question}
"""
    )
    return (
        {
            "chat_history": RunnableLambda(lambda x: format_chat_history(x["chat_history"])),
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_history_chain(llm, retriever):
    """LCEL 多轮链：{question, chat_history} → 回答。"""
    rephrase_chain = build_rephrase_chain(llm)

    def retrieve_with_rewrite(payload: dict):
        rewritten = rephrase_chain.invoke(payload)
        print(f"===== 重述后的查询: {rewritten} =====", flush=True)
        return retriever.invoke(rewritten)

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""你是专业的中文问答助手，只根据背景资料和历史消息回答。
找不到答案时回答“我不知道”。不要编造。

背景资料：{context}

历史消息：
{chat_history}

问题：{question}

回答：""",
    )

    # rag_chain = {context, question, chat_history} | prompt | llm | parser
    return (
        {
            "context": RunnableLambda(retrieve_with_rewrite) | format_docs,
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: format_chat_history(x["chat_history"])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
