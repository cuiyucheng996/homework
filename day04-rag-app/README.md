# Day04 自研 RAG 应用

仓库：https://github.com/cuiyucheng996/homework.git  
目录：`day04-rag-app/`

**不是**课堂 `01-rag/app.py`、`retrieve.py`、`naive_index.html` 的拷贝。

差异：按 Markdown 标题/表格切块；关键词检索（不绑上课 bge 路径和 Chroma 目录名）；接口 `/ask` 返回引用；页面展示来源切片。

知识库来自课程资料：`kb/cmdline.md`、`kb/naoc_budget.md`。

```powershell
cd day04-rag-app
python run_cli.py
python server.py
```

打开 http://127.0.0.1:8010 ，可问：

- 中国科学院2023年部门总预算是多少
- 熟练使用命令行有什么好处
