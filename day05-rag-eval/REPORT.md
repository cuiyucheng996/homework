# Day05 RAG 评估作业

仓库：https://github.com/cuiyucheng996/homework.git

## 1. RAG 流程

知识库：`day01.md` + `day02.md`（RAG 入门与搭建讲义）  
索引：Markdown 加载 → 400/40 切分 → `bge-base-zh-v1.5` 向量化 → Chroma  
查询：问题重述 → 检索（5 种策略）→ 基于上下文生成

## 2. 十道评测问题

1. 大模型有哪些主要局限性？
2. 什么是 RAG？它的基本思想是什么？
3. RAG 数据索引阶段包含哪些关键步骤？
4. RAG 数据查询阶段包含检索和生成，各自做什么？
5. 检索前处理 Pre-Retrieval 通常完成哪些工作？
6. 检索后处理 Post-Retrieval 通常完成哪些工作？
7. 对比 RAG 与微调：知识更新方式和训练成本有什么不同？
8. 什么是 Naive RAG（朴素 RAG）？
9. 搭建 RAG 应用时，索引过程包括哪些模块？
10. 向量数据库 Chroma、FAISS、Milvus 各有什么特点？

## 3. 五种策略

| 策略 | 作用阶段 | 做法 |
|---|---|---|
| Baseline | 基线 | 重述后相似度检索 |
| Multi Query | 检索前 | 生成多视角子查询，合并去重 |
| HyDE | 检索前 | 先写假设答案，用假设文档检索 |
| RRF | 检索后 | 多路结果按倒数排名融合 |
| Rerank Model | 检索后 | `bge-reranker-base` 交叉编码重排 |

评估指标对齐 RAGAS：`nv_context_relevance`、`answer_relevancy`、`faithfulness`、`nv_response_groundedness`。

## 4. 截图（5 种评估结果）

见 `submit/`：

- `baseline.png`
- `multi_query.png`
- `hyde.png`
- `rrf.png`
- `rerank.png`

总表：`summary.png`

## 5. 平均分

| 策略 | 上下文相关 | 回答相关 | 忠实度 | 证据支撑 |
|---|---:|---:|---:|---:|
| Baseline | 0.950 | 1.000 | 1.000 | 1.000 |
| Multi Query | 0.940 | 1.000 | 1.000 | 1.000 |
| HyDE | 0.930 | 1.000 | 1.000 | 1.000 |
| RRF | 0.950 | 1.000 | 1.000 | 1.000 |
| Rerank | 0.940 | 1.000 | 1.000 | 1.000 |

## 6. 分析结论

1. **知识库与问题高度对齐**（都来自同一份讲义），十道题都能从原文找到对应段落，因此回答相关性、忠实度、证据支撑三项普遍打满 1.0。差异主要体现在 **上下文相关性**。
2. **Baseline 与 RRF 并列最高（0.950）**。讲义问答是短事实题，单路语义检索已经够用；RRF 把多路排名融合后，没有引入明显噪声，也没有拉开差距。
3. **HyDE 略低（0.930）**。假设文档会补进讲义里没有的措辞，向量检索可能带上边缘切片，上下文相关分被拉低；生成阶段仍能靠 prompt 约束写出正确内容，所以后三项仍为 1.0。
4. **Multi Query / Rerank 为 0.940**。多查询扩大召回，也可能并入次相关块；Rerank 把最相关段落提前，但对已经高相关的集合提升有限，平均分与 Multi Query 接近。
5. **适用建议**：封闭、表述稳定的教材型知识库，先把切分和 embedding 做好，基线往往就够；开放域、问法漂移大时再上 Multi Query + RRF/Rerank。HyDE 更适合问题短、文档长、用词不一致的场景，不适合本作业这种“问题几乎等于小标题”的设定。

## 运行

```powershell
python build_index.py
python run_rag_eval.py
```
