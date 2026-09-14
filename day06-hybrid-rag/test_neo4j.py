from neo4j import GraphDatabase
# 连接配置
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678")

# 执行查询
# with GraphDatabase.driver(URI, auth=AUTH) as driver:
#     # 查询某个导演在某个年份之后拍过的电影
#     records, _, _ = driver.execute_query(
#         """
#             MATCH (p:Person{name:$name})-[r:DIRECTED]->(m:Movie)
#             WHERE m.year > $year
#             RETURN p.name AS director,m.year AS year, m.title AS movie
#         """,
#         parameters_={"name": "张艺谋", "year": 1990},
#         database_="neo4j"
#     )
    
#     # 处理结果
#     print(f"查询返回了 {len(records)} 条记录")
#     for record in records:
#         print(f"{record['director']}在{record['year']}年拍摄了《{record['movie']}》")
        
        
    # 请显示系统中每个导演及其代表作的电影名，显示导演名称和代表作名称
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # 查询某个导演在某个年份之后拍过的电影
    records, _, _ = driver.execute_query(
        """
            MATCH (p:Person)-[r:DIRECTED]->(m:Movie)
            order by r.rating desc RETURN p.name as director, collect(m.title)[0] as rep
        """
    )
    
    # 处理结果
    print(f"查询返回了 {len(records)} 条记录")
    for record in records:
        print(f"{record['director']}的代表作是《{record['rep']}》")