"""将 day01/day02 讲义写入 Chroma。"""
import os
from pathlib import Path

import torch
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / ".chroma_db"
BGE_PATH = Path(r"D:\资料\02.RAG项目资料\bge-base-zh-v1.5")
DAY02_SUBMIT = Path(r"D:\program\uvgraph\uvgraph\External\homework\day02_rag_chunk\submit")


def ensure_knowledge():
    KB_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("day01.md", "day02.md"):
        src = DAY02_SUBMIT / name
        dst = KB_DIR / name
        if src.exists() and not dst.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def build_index():
    ensure_knowledge()
    docs = []
    for md in sorted(KB_DIR.glob("*.md")):
        docs.extend(TextLoader(str(md), encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。", "\n"],
        chunk_size=400,
        chunk_overlap=40,
    )
    texts = splitter.split_documents(docs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=str(BGE_PATH),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) and not os.environ.get("FORCE_REINDEX"):
        vs = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embedding_model)
        if vs._collection.count() > 0:
            print(f"已有索引 {vs._collection.count()} 条，跳过重建")
            return vs
    vs = Chroma.from_documents(texts, embedding_model, persist_directory=str(CHROMA_DIR))
    print(f"索引完成，共 {len(texts)} 个切片")
    return vs


if __name__ == "__main__":
    os.chdir(ROOT)
    build_index()
