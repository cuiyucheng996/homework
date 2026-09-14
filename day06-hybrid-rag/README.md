# GraphRAG + LocalRAG 作业

仓库：https://github.com/cuiyucheng996/homework.git  
本目录对应提交路径：`day06-hybrid-rag/`

## 完成内容

在 `02-graphrag` 上扩展 LocalRAG，并按问题分流：

| 问题 | 路径 |
| --- | --- |
| 中国科学院2023年部门总预算是多少 | LocalRAG（RAGFlow，失败则读 `knowledge_base/sample.md`） |
| 推荐一个款苹果品牌的手机 | GraphRAG（Neo4j 商品图谱） |

页面流式回复开头会标 `检索路径：LocalRAG` 或 `检索路径：GraphRAG`。

截图放在仓库 `day06-hybrid-rag/submit/`。

## 运行

```powershell
cd External/homework/GraphRAG-完结/02-代码/02-graphrag
# 或 homework-git/day06-hybrid-rag
$env:DASHSCOPE_API_KEY="你的key"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen-plus"
# 可选：RAGFlow
# $env:RAGFLOW_API_KEY="..."
# $env:RAGFLOW_DATASET_ID="3f5414ca9a5611f193cc010101010000"
python test_route.py
python graphrag_service.py
```

打开 http://127.0.0.1:8007 分别问上面两句话。
