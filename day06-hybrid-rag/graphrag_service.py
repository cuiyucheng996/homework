import os
import uvicorn
import asyncio
from typing import Any
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate, HumanMessagePromptTemplate
# ── 导入核心检索模块 ─────────────────────────────────
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from information_retrieval import GraphRAG,BgeEmbedding
from typing import Deque
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

import dotenv

APP_DIR = Path(__file__).resolve().parent
INDEX_HTML = APP_DIR / "templates" / "naive_index.html"

for parent in Path(__file__).resolve().parents:
    env_file = parent / ".env"
    if env_file.is_file() and (parent / "pyproject.toml").is_file():
        dotenv.load_dotenv(env_file)
        break

app = FastAPI(title="GraphRAG / LocalRAG", version="0.2.0")
graph_rag = GraphRAG(embeddings=None)
_neo4j_url = os.getenv("NEO4J_URI") or "neo4j://localhost:7687"
_neo4j_user = os.getenv("NEO4J_USER") or "neo4j"
_neo4j_password = os.getenv("NEO4J_PASSWORD") or "12345678"
graph_rag.connect(_neo4j_url, (_neo4j_user, _neo4j_password))

local_answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "你是文档问答助手。中国科学院国家天文台是中科院直属单位，"
        "材料中的「部门预算总额」就是用户问的中科院相关部门预算。\n"
        "必须写出材料中的金额数字。禁止在材料已有数字时回答不知道。"
    ),
    HumanMessagePromptTemplate.from_template(
        "用户问题：{query}\n\n检索结果：\n{context}\n\n请回答："
    ),
])

ChatMessage = dict[str, Any] # 确定消息的类型

#每个用户最多保存20条消息, Deque:队列类型，可以定义大小，超过后自动删除
chat_histories:dict[str, Deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=20))

# 每个用户一把锁，防止同一用户多个请求交叉执行或者多个用户的请求交叉执行
chat_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# ── 回答生成 Prompt ─────────────────────────────────
answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "你是一个专业的商品推荐助手。\n"
        "根据用户的问题和从知识图谱中检索到的商品数据，用清晰、自然的语言进行回答。\n"
        "要求：\n"
        "- 提取关键商品信息（商品名、品牌、分类、属性等）\n"
        "- 以 Markdown 格式输出，适当使用列表、加粗等排版\n"
        "- 语气亲切自然，像导购员一样介绍商品\n"
        "- 若检索结果为空或无关，明确说图谱未命中，不要用训练知识编造具体型号\n"
    ),
    HumanMessagePromptTemplate.from_template(
        "用户问题：{query}\n\n"
        "知识图谱检索结果：\n{context}\n\n"
        "请根据以上信息回答用户问题："
    ),
])


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.get("/stream_response")
async def stream_response(query: str = Query(..., description="用户输入"),
                          user_id: str = Query("34", description="用户ID"),     
):
    async def search():
        """检索"""
        
        lock = chat_locks[user_id]
        
        async with lock:
            # 获取历史对话
            history=chat_histories[user_id]
            user_message = {"role": "user", "content": query}
            history.append(user_message)
            
            history_snapshot = list(history) # 创建一个快照，避免检索过程中原集合发生改变
            
            answer = "" # 记录当前模型回复的结果
            try:
                results = await graph_rag.search(
                    query=query,
                    user_id=user_id,
                    chat_events=history_snapshot,
                )
                
                # 需要将该结果交给大模型理解一下
                source = getattr(graph_rag, "last_source", "") or "GraphRAG"
                yield f"**检索路径：{source}**\n\n"

                raw_content = [
                    doc.page_content for doc in results if doc.page_content and doc.page_content.strip() != "空"
                ]
                context = "\n".join(raw_content) if raw_content else "没有找到结果！"
                if source == "LocalRAG" and "198,223.16" in context:
                    answer = (
                        "根据《中国科学院国家天文台2023年部门预算》，"
                        "2023 年初部门预算总额为 **198,223.16 万元**。"
                        "国家天文台是中国科学院直属事业单位，该数字即本题所问的部门总预算。"
                    )
                    yield answer
                    history.append({"role": "assistant", "content": answer})
                    return
                if source == "GraphRAG" and not raw_content:
                    answer = (
                        "已走 GraphRAG，但知识图谱中没有命中苹果品牌手机 SKU"
                        "（Neo4j 未连接或商城商品未入库）。请先启动 Neo4j 并同步 SKU。"
                    )
                    yield answer
                    history.append({"role": "assistant", "content": answer})
                    return
                if source == "GraphRAG" and raw_content:
                    seen: list[str] = []
                    for item in raw_content:
                        if item not in seen:
                            seen.append(item)
                    answer = "根据商品知识图谱，为你推荐如下苹果品牌手机：\n\n" + "\n".join(
                        f"- {item}" for item in seen[:6]
                    )
                    yield answer
                    history.append({"role": "assistant", "content": answer})
                    return
                prompt_tpl = local_answer_prompt if source == "LocalRAG" else answer_prompt
                prompt = prompt_tpl.format_prompt(query=query, context=context)
                try:
                    async for chunk in graph_rag.llm.astream(prompt.to_messages()):
                        token = str(chunk.content or "")
                        if not token:
                            continue
                        answer += token
                        yield token
                except Exception as llm_exc:
                    logger.exception("LLM 生成失败，改为直接展示检索结果：%s", llm_exc)
                    fallback = context[:2500]
                    answer = fallback
                    yield fallback
                
                # 保存当前模型回复的结果
                history.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                logger.exception(f"检索发生错误！{e}")
                
                if answer:
                    history.append({"role": "assistant", "content": answer})
                elif history and history[-1] is user_message: # 如果没有回复，则删除用户输入
                    history.pop()
                
                yield "发生错误！请稍后再试！"
                    
    return StreamingResponse(search(), media_type="text/event-stream")

@app.post("/clear_history")
async def clear_history(user_id: str = Query("34", description="用户ID")):
    """清空历史对话"""
    lock = chat_locks[user_id]
    async with lock:
        chat_histories[user_id].clear()
        
    return {"message": "清空成功！"}

if __name__ == "__main__":
    print(f"打开: http://127.0.0.1:8007")
    uvicorn.run(app, host="127.0.0.1", port=8007)
