from information_retrieval import heuristic_route


def test_local_budget():
    row = heuristic_route("中国科学院2023年部门总预算是多少")
    assert row is not None
    assert row.need_local_search is True
    assert row.need_graph_search is False


def test_graph_phone():
    row = heuristic_route("推荐一个款苹果品牌的手机")
    assert row is not None
    assert row.need_graph_search is True
    assert row.need_local_search is False


if __name__ == "__main__":
    test_local_budget()
    test_graph_phone()
    print("route ok")
