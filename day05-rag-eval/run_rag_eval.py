"""
作业：搭建 RAG，10 道题，对比 Baseline / Multi Query / HyDE / RRF / Rerank，
按 RAGAS 四项指标打分并导出 5 张评估截图。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
UVGRAPH_ROOT = Path(r"D:\program\uvgraph\uvgraph")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UVGRAPH_ROOT))

from dotenv import load_dotenv

load_dotenv(UVGRAPH_ROOT / ".env")
load_dotenv(ROOT / ".env")

from build_index import BGE_PATH, CHROMA_DIR, build_index
from questions import QUERY_LIST
from rag_strategies import STRATEGY_FNS, STRATEGY_TITLES, load_reranker
from retrieve import format_docs, format_history

SUBMIT = ROOT / "submit"
RERANKER_PATH = Path(r"D:\资料\02.RAG项目资料\bge-reranker-base")

EVAL_PROMPT = PromptTemplate.from_template(
    """
你是 RAG 评估员。根据 RAGAS 常用指标，只输出 JSON，不要其它文字。
分数均为 0 到 1 的小数。

指标说明：
- nv_context_relevance: 检索上下文与问题的相关程度
- answer_relevancy: 回答是否针对问题
- faithfulness: 回答中的事实能否被上下文支撑（胡编则低）
- nv_response_groundedness: 回答是否扎根于检索上下文

问题：{query}
上下文：{context}
回答：{answer}

输出格式：
{{"nv_context_relevance": 0.0, "answer_relevancy": 0.0, "faithfulness": 0.0, "nv_response_groundedness": 0.0}}
"""
)


def get_llm():
    sys.path.insert(0, str(UVGRAPH_ROOT))
    from src.llm.chat import get_chat_model

    llm = get_chat_model()
    if llm is None:
        raise RuntimeError("未配置 DeepSeek / Kimi / OpenAI，请检查 uvgraph .env")
    return llm


def get_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=str(BGE_PATH),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_retriever(embedding_model, k=8):
    vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embedding_model)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def quiet_rag_chain(retrieve_result, llm):
    prompt = PromptTemplate(
        input_variables=["context", "history", "query"],
        template="""你是专业的中文问答助手。请仅根据背景资料回答问题，不要发散。找不到答案时回答“我不知道”。

背景资料：{context}

历史消息：[{history}]

问题：{query}

回答：""",
    )
    return (
        {
            "context": lambda x: format_docs(retrieve_result),
            "history": lambda x: format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def parse_scores(text: str) -> dict:
    defaults = {
        "nv_context_relevance": 0.0,
        "answer_relevancy": 0.0,
        "faithfulness": 0.0,
        "nv_response_groundedness": 0.0,
    }
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        print("评分解析失败:", text[:300], flush=True)
        return defaults
    raw = match.group(0)
    raw = raw.replace("，", ",").replace("：", ":")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("JSON失败:", raw[:300], flush=True)
        return defaults
    out = {}
    for k in defaults:
        try:
            out[k] = float(data.get(k, 0.0))
        except (TypeError, ValueError):
            out[k] = 0.0
    return out


def evaluate_one(llm, query: str, contexts: list[str], answer: str) -> dict:
    raw = (EVAL_PROMPT | llm | StrOutputParser()).invoke(
        {
            "query": query,
            "context": "\n\n".join(contexts)[:4000],
            "answer": answer[:2000],
        }
    )
    return parse_scores(raw)


def save_table_png(df: pd.DataFrame, title: str, path: Path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    show = df.copy()
    numeric = [
        "nv_context_relevance",
        "answer_relevancy",
        "faithfulness",
        "nv_response_groundedness",
    ]
    for col in numeric:
        show[col] = show[col].map(lambda x: f"{x:.3f}")
    fig_h = 0.55 * (len(show) + 4)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=12)
    table = ax.table(
        cellText=show.values,
        colLabels=list(show.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_strategy(name: str, llm, retriever, extra: dict) -> pd.DataFrame:
    rows = []
    fn = STRATEGY_FNS[name]
    for i, query in enumerate(QUERY_LIST, 1):
        print(f"[{name}] ({i}/{len(QUERY_LIST)}) {query}", flush=True)
        docs = fn(query, [], llm, retriever, **extra)
        answer = quiet_rag_chain(docs, llm).invoke({"query": query, "history": []})
        contexts = [d.page_content for d in docs]
        scores = evaluate_one(llm, query, contexts, answer)
        rows.append({"question": query, **scores, "answer": answer[:80].replace("\n", " ")})
    return pd.DataFrame(rows)


def main():
    SUBMIT.mkdir(parents=True, exist_ok=True)
    print("1. 构建索引...")
    build_index()
    llm = get_llm()
    embedding = get_embedding()
    retriever = get_retriever(embedding, k=8)
    extra = {}
    print("2. 加载 reranker...")
    tok, model = load_reranker(str(RERANKER_PATH))
    extra["rerank_tokenizer"] = tok
    extra["rerank_model"] = model

    order = ["baseline", "multi_query", "hyde", "rrf", "rerank"]
    summary_rows = []
    for name in order:
        df = run_strategy(name, llm, retriever, extra)
        csv_path = SUBMIT / f"{name}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        metric_cols = [
            "nv_context_relevance",
            "answer_relevancy",
            "faithfulness",
            "nv_response_groundedness",
        ]
        mean = df[metric_cols].mean()
        summary_rows.append({"strategy": STRATEGY_TITLES[name], **mean.to_dict()})
        shot = df[["question"] + metric_cols].copy()
        shot.loc[len(shot)] = ["平均值"] + [round(mean[c], 4) for c in metric_cols]
        save_table_png(shot, f"RAG 评估 - {STRATEGY_TITLES[name]}", SUBMIT / f"{name}.png")
        print(f"已保存 {csv_path} / {name}.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUBMIT / "summary.csv", index=False, encoding="utf-8-sig")
    save_table_png(summary, "五种策略平均分对比", SUBMIT / "summary.png")
    print(summary.to_string(index=False))
    print("完成。截图目录:", SUBMIT)


if __name__ == "__main__":
    main()
