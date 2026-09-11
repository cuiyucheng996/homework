"""本地执行：商品入图 + 用户浏览日志入图，并导出 User 节点截图。"""
import os
import sys
from pathlib import Path

HOMEWORK_ROOT = Path(__file__).resolve().parent
UVGRAPH_ROOT = Path(r"D:\program\uvgraph\uvgraph")

if UVGRAPH_ROOT.exists():
    sys.path.insert(0, str(UVGRAPH_ROOT))
    os.chdir(str(UVGRAPH_ROOT))
    try:
        from src.configs.config_loader import get_app_settings

        pwd = get_app_settings().neo4j_password
        if pwd:
            os.environ["NEO4J_PASSWORD"] = pwd
    except Exception:
        pass

sys.path.insert(0, str(HOMEWORK_ROOT))
sys.path.insert(0, str(HOMEWORK_ROOT / "src"))

from pymysql.cursors import DictCursor
import pymysql
from neo4j import GraphDatabase

from configs import config
from datasync.mysql_data_sync import (
    read_sku_attr_info,
    read_sku_base_info,
    write_sku_attr_info,
    write_sku_base_info,
)
from datasync.view_log_sync import read_user_view_log, write_user_view_log


def export_user_screenshot(png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("未安装 matplotlib/networkx，跳过截图，仅打印 User 节点")
        return

    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG["uri"],
        auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
    ) as driver:
        records, _, _ = driver.execute_query(
            """
            MATCH (u:User)-[v:View]->(s:SKU)
            RETURN u.user_id AS user_id, s.sku_id AS sku_id, s.sku_name AS sku_name
            LIMIT 80
            """
        )

    graph = nx.DiGraph()
    for rec in records:
        user = f"User {rec['user_id']}"
        sku = f"SKU {rec['sku_id']}"
        graph.add_node(user, kind="user")
        graph.add_node(sku, kind="sku")
        graph.add_edge(user, sku)

    plt.figure(figsize=(16, 10))
    pos = nx.spring_layout(graph, k=0.6, seed=42)
    users = [n for n, d in graph.nodes(data=True) if d.get("kind") == "user"]
    skus = [n for n, d in graph.nodes(data=True) if d.get("kind") == "sku"]
    nx.draw_networkx_nodes(graph, pos, nodelist=users, node_color="#4C78A8", node_size=700)
    nx.draw_networkx_nodes(graph, pos, nodelist=skus, node_color="#F58518", node_size=500)
    nx.draw_networkx_edges(graph, pos, arrows=True, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, font_size=8)
    plt.title("Neo4j User nodes (User -View-> SKU)")
    plt.axis("off")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"User 节点截图已保存: {png_path}")

    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG["uri"],
        auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
    ) as driver:
        detail, _, _ = driver.execute_query(
            """
            MATCH (u:User)-[v:View]->(s:SKU)
            WITH u, count(v) AS cnt
            ORDER BY cnt DESC
            LIMIT 1
            MATCH (u)-[v2:View]->(s2:SKU)
            RETURN u.user_id AS user_id, s2.sku_id AS sku_id, s2.sku_name AS sku_name, v2.view_time AS view_time
            """
        )

    g2 = nx.DiGraph()
    if detail:
        user = f"User {detail[0]['user_id']}"
        g2.add_node(user, kind="user")
        for rec in detail:
            sku = f"SKU {rec['sku_id']}"
            g2.add_node(sku, kind="sku")
            g2.add_edge(user, sku)

        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(g2, seed=1)
        users = [n for n, d in g2.nodes(data=True) if d.get("kind") == "user"]
        skus = [n for n, d in g2.nodes(data=True) if d.get("kind") == "sku"]
        nx.draw_networkx_nodes(g2, pos, nodelist=users, node_color="#4C78A8", node_size=1800)
        nx.draw_networkx_nodes(g2, pos, nodelist=skus, node_color="#F58518", node_size=1200)
        nx.draw_networkx_edges(g2, pos, arrows=True, width=2)
        nx.draw_networkx_labels(g2, pos, font_size=11)
        plt.title(f"Neo4j User node detail: {user} -[:View]-> SKU")
        plt.axis("off")
        detail_path = png_path.with_name("user_node_detail.png")
        plt.tight_layout()
        plt.savefig(detail_path, dpi=150)
        plt.close()
        print(f"User 节点明细截图已保存: {detail_path}")


if __name__ == "__main__":
    with pymysql.connect(**config.MYSQL_CONFIG) as conn:
        with conn.cursor(cursor=DictCursor) as cursor:
            sku_base_info = read_sku_base_info(cursor)
            sku_attr_info = read_sku_attr_info(cursor)

    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG["uri"],
        auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
    ) as driver:
        write_sku_base_info(driver, sku_base_info)
        write_sku_attr_info(driver, sku_attr_info)

    user_view_log = read_user_view_log()
    print(f"读取到 {len(user_view_log)} 条用户浏览记录")
    write_user_view_log(user_view_log)

    out = HOMEWORK_ROOT / "submit" / "user_nodes.png"
    export_user_screenshot(out)
    print("作业入库流程完成")
