from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index,upsert_vectors,create_fulltext_index
import logging
from sentence_transformers import SentenceTransformer
import re
import jieba

# 配置控制台日志
logger = logging.getLogger("indexing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("[%(levelname)s]%(asctime)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
embed_model = SentenceTransformer("./bge-base-zh-v1.5")    
vector_dimensions = 768
embed_bath_size = 64
def drop_constraint(driver):
    """删除所有约束"""
    records = driver.execute_query("show constraints").records
    for record in records:
        driver.execute_query(f"drop constraint {record['name']} if exists")
    logger.info(f"所有约束删除成功")


def drop_index(driver):
    """删除所有索引"""
    records = driver.execute_query("show indexes").records
    for record in records:
        if not record["owningConstraint"]:
            driver.execute_query(f"drop index {record['name']} if exists")
    logger.info(f"所有索引删除成功")
def create_my_vector_index(driver, label_name, property_name):
    """创建向量索引"""
    # 1.给标签创建向量字段
    create_vector_index(
                driver,
                name=f"{label_name.lower()}_vector",
                label=label_name,
                embedding_property="embedding",
                dimensions=vector_dimensions,
                similarity_fn="cosine"
            )
    # 2.填充向量值: 先生成embedding，然后填充
    record_list = driver.execute_query(f"MATCH (n:{label_name}) where n.embedding is null return elementId(n) as id, n.{property_name} as text").records
    record_tuple_list = [(record["id"], record["text"]) for record in record_list]
    ids, texts = zip(*record_tuple_list)
    ids = list(ids)
    texts = list(texts)
    
    # 对texts进行向量化
    embeddings = embed_model.encode(
        texts, 
        batch_size=embed_bath_size, 
        show_progress_bar=True, 
        normalize_embeddings=True
    )
    
    upsert_vectors(
                driver,
                ids=ids,
                embedding_property="embedding",
                embeddings=embeddings # type: ignore
            )
    
    
def create_index(driver, label, property_name):
    """创建全文索引，并添加节点属性"""

    # 创建全文索引
    create_fulltext_index(
        driver,
        name=f"{label.lower()}_fulltext",  # 索引的唯一名称
        label=label,  # 要创建索引的节点标签
        node_properties=["fulltext"],  # 要创建全文索引的节点属性列表
    )

    # 查询 fulltext 为 null 的节点，获取 elementId 和 指定属性
    record_list = driver.execute_query(
        f"""match (n:{label}) where n.fulltext is null
            return elementId(n) as id, n.{property_name} as text""",
    ).records
    record_tuple_list = [(r["id"], r["text"]) for r in record_list]
    if not record_tuple_list:
        logger.info(f"{label} 所有节点皆存在全文索引属性")
        return

    # 文本分词，作为全文索引属性
    logger.info(f"计算 {label} ({len(record_list)}) 的全文索引")
    pattern = re.compile(r"[a-zA-Z0-9\u4e00-\u9fa5]+") #匹配英文字母、数字和中文字符
    fulltext_tuple_list = [
        (
            id_,
            " ".join(
                [
                    word.strip()
                    for word in jieba.lcut(text)
                    if pattern.fullmatch(word.strip())#过滤掉不符合正则表达式的词汇
                ]
            ),
        )
        for id_, text in record_tuple_list
    ]
    ids, fulltexts = zip(*fulltext_tuple_list)
    ids = list(ids)
    fulltexts = list(fulltexts)

    # 按 elementId 添加全文索引属性
    logger.info(f"写入 {label} ({len(fulltexts)}) 的全文索引")
    insert_batch_size = 1000
    for i in range(0, len(record_tuple_list), insert_batch_size):
        batch_rows = [
            {"id": id_, "fulltext": ft}
            for id_, ft in zip(
                ids[i: i + insert_batch_size],
                fulltexts[i: i + insert_batch_size],
            )
        ]

        # UNWIND：将列表数据展开为多行记录
        driver.execute_query(
            "UNWIND $rows AS row " #将传入的rows列表（通过参数传递）展开，每一项作为一行数据，命名为row
            "MATCH (n) "
            "WHERE elementId(n) = row.id "
            "SET n.fulltext = row.fulltext ",
            {"rows": batch_rows},
        )

if __name__ == '__main__':
    with GraphDatabase.driver("neo4j://localhost", auth=("neo4j", "12345678")) as driver:
        logger.info("索引构建开始...")    
        # 1.清空约束和索引
        drop_constraint(driver)
        drop_index(driver)
        logger.info("清空约束和索引完成")

        # 2.创建向量索引
        # 清空所有的向量值
        driver.execute_query("MATCH (n) REMOVE n.embedding")
        create_my_vector_index(driver, "Category3", "category3_name")
        create_my_vector_index(driver, "Category2", "category2_name")
        create_my_vector_index(driver, "Category1", "category1_name")
        create_my_vector_index(driver, "Trademark", "trademark_name")
        create_my_vector_index(driver, "SPU", "spu_name")
        create_my_vector_index(driver, "SKU", "sku_name")
        create_my_vector_index(driver, "Attr", "attr_value")

        # # 3.创建全文索引-倒排索引
        driver.execute_query("MATCH (n) REMOVE n.fulltext")
        create_index(driver, "Category3", "category3_name")
        create_index(driver, "Category2", "category2_name")
        create_index(driver, "Category1", "category1_name")
        create_index(driver, "Trademark", "trademark_name")
        create_index(driver, "SPU", "spu_name")
        create_index(driver, "SKU", "sku_name")
        create_index(driver, "Attr", "attr_value")