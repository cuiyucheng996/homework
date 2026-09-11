#用户访问历史同步
import pymysql
from neo4j import GraphDatabase
from pymysql.cursors import DictCursor
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from configs import config


def read_user_view_log():
    with pymysql.connect(**config.MYSQL_CONFIG) as connection:
        with connection.cursor(cursor=DictCursor) as cursor:
            cursor.execute("""
                select
                    user_id,
                    sku_id,
                    view_time
                from user_view_log
            """)
            return cursor.fetchall()


def write_user_view_log(user_view_log):
    with GraphDatabase.driver(
        uri=config.NEO4J_CONFIG["uri"],
        auth=(config.NEO4J_CONFIG["user"], config.NEO4J_CONFIG["password"]),
    ) as driver:
        for item in user_view_log:
            row = dict(item)
            view_time = row.get("view_time")
            if view_time is not None:
                row["view_time"] = str(view_time)
            driver.execute_query(
                """
                MATCH (sku:SKU{sku_id:$sku_id})
                MERGE (user:User{user_id:$user_id})
                MERGE (user)-[:View{view_time:$view_time}]->(sku)
                """,
                row,
            )
    print("用户行为日志写入成功！")


if __name__ == "__main__":
    # 1.读取用户行为日志
    user_view_log = read_user_view_log()
    print(f"读取到 {len(user_view_log)} 条用户浏览记录")

    # 2.将日志数据写入neo4j
    write_user_view_log(user_view_log)
    print("用户行为数据入库完成！")
