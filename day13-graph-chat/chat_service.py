"""查询某商品的所有单品：意图 → 纠错 → 抽商品槽 → Cypher 查 SKU。"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

HOMEWORK_ROOT = Path(__file__).resolve().parents[1]
UVGRAPH_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HOMEWORK_ROOT))
sys.path.insert(0, str(HOMEWORK_ROOT / "src"))

load_dotenv(UVGRAPH_ROOT / ".env")
load_dotenv(HOMEWORK_ROOT / ".env")

from configs import config  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

logger = logging.getLogger(__name__)

INTENT_SKU_LIST = "查询某商品的所有单品"

_SKU_INTENT_HINTS = (
    "都有哪些版本",
    "有哪些版本",
    "哪些版本",
    "所有单品",
    "有哪些单品",
    "哪些单品",
    "有哪些sku",
    "有哪些款",
    "哪些型号",
    "有哪些型号",
)

_SPELL = (
    ("ultre", "Ultra"),
    ("Ultre", "Ultra"),
    ("小迷", "小米"),
)


def predict_intent(question: str) -> str:
    text = (question or "").replace(" ", "").lower()
    if any(hint.replace(" ", "").lower() in text for hint in _SKU_INTENT_HINTS):
        return INTENT_SKU_LIST
    return "未知意图"


def spell_check(question: str) -> str:
    text = question or ""
    for src, dst in _SPELL:
        text = text.replace(src, dst)
    return text


def extract_spu_names(question: str, catalog: list[str]) -> list[str]:
    q = question or ""
    hits = [name for name in catalog if name and name in q]
    hits.sort(key=len, reverse=True)
    kept: list[str] = []
    for name in hits:
        if any(name in other for other in kept):
            continue
        kept.append(name)
    return kept


class ChatService:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            uri=config.NEO4J_CONFIG["uri"],
            auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
        )

    def close(self) -> None:
        self._driver.close()

    def load_spu_catalog(self) -> list[str]:
        records, _, _ = self._driver.execute_query(
            "MATCH (s:SPU) RETURN s.spu_name AS n"
        )
        return [str(row["n"]) for row in records if row.get("n")]

    def query_skus(self, spu_names: list[str]) -> list[str]:
        cypher = """
            MATCH (spu:SPU)<-[:Belong]-(s:SKU)
            WHERE spu.spu_name IN $spu_names
            RETURN s.sku_name AS sku_name
        """
        records, _, _ = self._driver.execute_query(cypher, {"spu_names": spu_names})
        return [str(row["sku_name"]) for row in records if row.get("sku_name")]

    def chat(self, question: str) -> str:
        raw = (question or "").strip()
        if not raw:
            return "请输入问题。"

        intent = predict_intent(raw)
        logger.info("1. 意图识别结果：%s", intent)
        corrected = spell_check(raw)
        logger.info("2. 拼写纠错结果：%s", corrected)

        if intent != INTENT_SKU_LIST:
            return f"意图为：{intent}，本作业只实现「{INTENT_SKU_LIST}」，请换种方式试一试。"

        catalog = self.load_spu_catalog()
        names = extract_spu_names(corrected, catalog)
        logger.info("3. 实体抽取结果：%s", {"商品": names})
        if not names:
            return "没有抽出商品名，请说出具体商品，例如：小米12S Ultra 都有哪些版本"

        skus = self.query_skus(names)
        if not skus:
            return f"图谱里没有查到「{'、'.join(names)}」的单品。"
        listing = "\n".join(skus)
        shown = "、".join(names)
        return f"{shown}的所有单品有：\n{listing}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    service = ChatService()
    try:
        q = " ".join(sys.argv[1:]).strip() or "小米12S Ultre 都有哪些版本"
        print("问：", q)
        print("答：")
        print(service.chat(q))
    finally:
        service.close()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
