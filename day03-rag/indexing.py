"""文档加载、切分、写入 Chroma（对应 01-test 索引，无 notebook）。"""
from pathlib import Path

import torch
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent
KB_DIR = ROOT / "knowledge_base"
CHROMA_DIR = ROOT / ".chroma_db"
BGE_PATH = Path(r"D:\资料\02.RAG项目资料\bge-base-zh-v1.5")


def get_embedding():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=str(BGE_PATH),
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index():
    docs = []
    for path in sorted(KB_DIR.glob("*.txt")):
        docs.extend(TextLoader(str(path), encoding="utf-8").load())
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。", "\n"],
        chunk_size=400,
        chunk_overlap=40,
    )
    chunks = splitter.split_documents(docs)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vs = Chroma.from_documents(chunks, get_embedding(), persist_directory=str(CHROMA_DIR))
    print(f"索引完成，切片数：{len(chunks)}")
    return vs


def get_vectorstore():
    return Chroma(persist_directory=str(CHROMA_DIR), embedding_function=get_embedding())


if __name__ == "__main__":
    build_index()
