# 图谱作业：用户行为数据入库

仓库地址：https://github.com/cuiyucheng996/homework.git

## 完成内容

1. 从 MySQL `mall.user_view_log` 读取用户浏览日志，写入 Neo4j：
   - 节点：`(:User {user_id})`
   - 关系：`(User)-[:View {view_time}]->(:SKU)`
2. 图谱 User 节点截图见 `submit/user_nodes.png`、`submit/user_node_detail.png`

## 运行

先保证 MySQL `mall` 库、Neo4j 可用，并已同步商品 SKU（`src/datasync/mysql_data_sync.py`）。

```powershell
$env:MYSQL_PASSWORD="root"
$env:NEO4J_PASSWORD="你的Neo4j密码"
python src/datasync/view_log_sync.py
```

或一键入库并导出截图：

```powershell
python run_ingest.py
```

本次本地入库结果：User 节点 50 个，View 关系 102 条。

---

# Day03 LCEL 多轮 RAG

见 `day03-rag/`（仅 `.py`，无 ipynb）。运行 `run_multiturn.py` 后截取终端三轮问答。

---

# Day05 RAG 评估

见目录 `day05-rag-eval/`，说明与分析：`day05-rag-eval/REPORT.md`。

五种评估截图：`day05-rag-eval/submit/{baseline,multi_query,hyde,rrf,rerank}.png`。
