# Day03：LCEL 多轮 RAG

对应课程 `03-rag`（检索 + Prompt + LLM），用 **Python 脚本**实现，**不提交 ipynb**。

## 做什么

LCEL 管道：

`{question, chat_history}` → 按历史重述问题 → Chroma 检索 → Prompt → LLM → 字符串回答

第二、三轮可以用「那不知道…」「下一篇…」这种指代，重述链会补全成独立问句。

## 运行

```powershell
chcp 65001
cd day03-rag
D:\program\uvgraph\uvgraph\.venv\Scripts\python.exe indexing.py
D:\program\uvgraph\uvgraph\.venv\Scripts\python.exe -u run_multiturn.py
```

终端会出现三轮「用户 / 助手」和「重述后的查询」，**截终端**即可。

知识库：`knowledge_base/sample.txt`（《孙子兵法》节选，课程资料自带）。
