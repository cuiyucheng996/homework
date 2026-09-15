from __future__ import annotations

import unittest
from unittest import mock

import chat_service as cs


class ChatPipelineTest(unittest.TestCase):
    def test_intent(self) -> None:
        self.assertEqual(cs.predict_intent("小米12S Ultra 都有哪些版本"), cs.INTENT_SKU_LIST)
        self.assertEqual(cs.predict_intent("今天天气如何"), "未知意图")

    def test_spell(self) -> None:
        self.assertIn("Ultra", cs.spell_check("小米12S Ultre 都有哪些版本"))

    def test_extract_longest(self) -> None:
        names = cs.extract_spu_names(
            "小米12S Ultra 都有哪些版本",
            ["小米", "小米12S Ultra", "Redmi 10X"],
        )
        self.assertEqual(names, ["小米12S Ultra"])

    def test_chat_happy_path(self) -> None:
        svc = cs.ChatService.__new__(cs.ChatService)
        svc._driver = mock.Mock()
        svc.load_spu_catalog = mock.Mock(return_value=["小米12S Ultra"])
        svc.query_skus = mock.Mock(return_value=["小米12S Ultra 骁龙8+128GB 冷杉绿 5G手机"])
        text = svc.chat("小米12S Ultre 都有哪些版本")
        self.assertIn("所有单品", text)
        svc.query_skus.assert_called_once_with(["小米12S Ultra"])


if __name__ == "__main__":
    unittest.main()
