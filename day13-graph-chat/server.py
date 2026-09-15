"""电商客服 Web：POST /api/chat。"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from chat_service import ChatService

app = FastAPI()
_service = ChatService()


class Question(BaseModel):
    message: str


class Answer(BaseModel):
    message: str


@app.get("/")
def index():
    return FileResponse(HERE / "web" / "index.html")


@app.post("/api/chat")
def chat(question: Question) -> Answer:
    return Answer(message=_service.chat(question.message))


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8013, reload=False)
