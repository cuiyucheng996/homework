"""五种检索策略：基线、Multi Query、HyDE、RRF、Rerank Model。"""
from hashlib import sha256
from typing import Callable

import torch
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from retrieve import format_docs, format_history, get_rag_chain, get_retriever, rephrase_retrieve


def get_unique_docs(docs: list[list[Document]]) -> list[Document]:
    seen = set()
    unique_docs = []
    for sublist in docs:
        for doc in sublist:
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)
    return unique_docs


def reciprocal_rank_fusion(
    docs: list[list[Document]], k=60, docs_return_num=10
) -> list[Document]:
    fused_scores = {}
    unique_docs_by_content = {}
    for doc_list in docs:
        for rank, doc in enumerate(doc_list):
            key = sha256(doc.page_content.encode("utf-8")).hexdigest()
            unique_docs_by_content[key] = doc
            fused_scores[key] = fused_scores.get(key, 0) + 1 / (rank + k)
    sorted_content_hashes = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_docs = [unique_docs_by_content[key] for key, _ in sorted_content_hashes]
    return reranked_docs[:docs_return_num]


def model_rerank(query, docs: list[Document], rerank_tokenizer, rerank_model, docs_return_num=5, batch_size=16):
    device = torch.device("cpu")
    rerank_model.to(device)
    scores = []
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        inputs = rerank_tokenizer(
            text=[query] * len(batch_docs),
            text_pair=[doc.page_content for doc in batch_docs],
            padding=True,
            max_length=min(512, rerank_tokenizer.model_max_length or 512),
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = rerank_model(**inputs)
            batch_scores = outputs.logits.squeeze()
        if batch_scores.ndim == 0:
            scores.append(float(batch_scores))
        else:
            scores.extend(batch_scores.tolist())
    reranked_docs = [doc for doc, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    return reranked_docs[:docs_return_num]


def _rephrase(query: str, history: list, llm) -> str:
    rephrase_prompt = PromptTemplate.from_template(
        """
        根据对话历史简要完善最新的用户消息，使其更加具体。只输出完善后的问题。如果问题不需要完善，请直接输出原始问题。
        {history}
        用户：{query}
        """
    )
    chain = (
        {
            "history": lambda x: format_history(x.get("history")),
            "query": lambda x: x.get("query"),
        }
        | rephrase_prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"history": history, "query": query})


def retrieve_baseline(query: str, history: list, llm, retriever, **_kwargs) -> list[Document]:
    return rephrase_retrieve({"query": query, "history": history}, llm, retriever)


def retrieve_multi_query(query: str, history: list, llm, retriever, multi_query_num=3, **_kwargs) -> list[Document]:
    rephrased = _rephrase(query, history, llm)
    multi_query_prompt = PromptTemplate.from_template(
        """
        你是一名AI语言模型助理。你的任务是生成给定问题的{query_num}个不同版本，以从矢量数据库中检索相关文档。
        通过从多个视角生成问题，克服基于距离的相似性搜索的局限。请使用换行符分隔备选问题，不要编号以外的解释。
        原始问题：{query}
        """
    )
    expend_query_chain = (
        multi_query_prompt
        | llm
        | StrOutputParser()
        | (lambda x: [item.strip() for item in x.split("\n") if item.strip()][:multi_query_num])
    )
    queries = expend_query_chain.invoke({"query": rephrased, "query_num": multi_query_num})
    if rephrased not in queries:
        queries = [rephrased] + queries
    doc_lists = [retriever.invoke(q) for q in queries]
    return get_unique_docs(doc_lists)[:10]


def retrieve_hyde(query: str, history: list, llm, retriever, **_kwargs) -> list[Document]:
    rephrased = _rephrase(query, history, llm)
    hyde_prompt = PromptTemplate.from_template(
        """
        请根据常识和推理，为问题编写一段看起来合理且详细的回答性段落，哪怕你不确定真实答案。
        问题：{query}
        """
    )
    hypo = (hyde_prompt | llm | StrOutputParser()).invoke({"query": rephrased})
    return retriever.invoke(hypo)


def retrieve_rrf(query: str, history: list, llm, retriever, multi_query_num=3, **_kwargs) -> list[Document]:
    rephrased = _rephrase(query, history, llm)
    multi_query_prompt = PromptTemplate.from_template(
        """
        你是一名AI语言模型助理。你的任务是生成给定问题的{query_num}个不同版本，以从矢量数据库中检索相关文档。
        请使用换行符分隔备选问题，不要额外解释。
        原始问题：{query}
        """
    )
    expend_query_chain = (
        multi_query_prompt
        | llm
        | StrOutputParser()
        | (lambda x: [item.strip() for item in x.split("\n") if item.strip()][:multi_query_num])
    )
    queries = expend_query_chain.invoke({"query": rephrased, "query_num": multi_query_num})
    if rephrased not in queries:
        queries = [rephrased] + queries
    doc_lists = [retriever.invoke(q) for q in queries]
    return reciprocal_rank_fusion(doc_lists)


def retrieve_rerank(query: str, history: list, llm, retriever, rerank_tokenizer=None, rerank_model=None, **_kwargs) -> list[Document]:
    rephrased = _rephrase(query, history, llm)
    docs = retriever.invoke(rephrased)
    return model_rerank(rephrased, docs, rerank_tokenizer, rerank_model)


def load_reranker(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


STRATEGY_FNS: dict[str, Callable] = {
    "baseline": retrieve_baseline,
    "multi_query": retrieve_multi_query,
    "hyde": retrieve_hyde,
    "rrf": retrieve_rrf,
    "rerank": retrieve_rerank,
}

STRATEGY_TITLES = {
    "baseline": "Baseline 基线检索",
    "multi_query": "Multi Query 多重查询",
    "hyde": "HyDE 假设文档",
    "rrf": "RRF 倒数排序融合",
    "rerank": "Rerank Model 重排序模型",
}
