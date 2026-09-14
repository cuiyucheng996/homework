"""独立 RAG 应用：标题切块 + 关键词检索 + 引用，不是课堂 app.py 拷贝。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from compose import answer
from store import build_store

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
store, _chunks = build_store(ROOT / "kb")

app = FastAPI(title="Day04 自研 RAG")
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/ask")
def ask(q: str = Query(..., min_length=1)):
    hits = store.search(q, k=4)
    text = answer(q, hits)
    cites = [{"title": h["title"], "source": h["source"], "preview": h["text"][:220]} for h in hits]
    return JSONResponse({"answer": text, "citations": cites})


if __name__ == "__main__":
    import uvicorn

    print("打开 http://127.0.0.1:8010")
    uvicorn.run(app, host="127.0.0.1", port=8010)
