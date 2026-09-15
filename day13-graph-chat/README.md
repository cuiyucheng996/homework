# Day13：查询某商品的所有单品

对照课堂 `04-graph` 聊天链路，只实现细意图 **查询某商品的所有单品**。

仓库：https://github.com/cuiyucheng996/homework.git

选题报告：`TOPIC.md`

## 流程

1. 意图：命中「都有哪些版本 / 所有单品 / 有哪些款」等 → `查询某商品的所有单品`
2. 纠错：如 `Ultre` → `Ultra`
3. 抽槽：用图谱里的 `SPU.spu_name` 做最长子串对齐（作业环境不提交 UIE 权重）
4. 查询：

```cypher
MATCH (spu:SPU)<-[:Belong]-(s:SKU)
WHERE spu.spu_name IN $spu_names
RETURN s.sku_name AS sku_name
```

## 运行

先同步商品到 Neo4j（`src/datasync/mysql_data_sync.py`），密码用环境变量 `NEO4J_PASSWORD`。

```powershell
cd day13-graph-chat
python test_chat_service.py
python chat_service.py "小米12S Ultre 都有哪些版本"
python server.py
```

浏览器：http://127.0.0.1:8013

课堂示例问句：`小米12S Ultra 都有哪些版本`（纠错演示可用 `Ultre`）。

终端样例：`submit/cli_run.txt`
