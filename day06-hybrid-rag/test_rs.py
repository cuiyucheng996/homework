from langchain_community.graphs import Neo4jGraph

neo4j_graph = Neo4jGraph("neo4j://localhost", "neo4j","12345678", enhanced_schema=True)
#print(neo4j_graph.structured_schema.keys())
print("relationships:", neo4j_graph.structured_schema.get("relationships",""))

"""
relationships: [
    {'start': 'User', 'type': 'View', 'end': 'SKU'}, 
    {'start': 'SKU', 'type': 'Belong', 'end': 'SPU'}, 
    {'start': 'SKU', 'type': 'Have', 'end': 'Attr'}, 
    {'start': 'SPU', 'type': 'Belong', 'end': 'Trademark'}, 
    {'start': 'SPU', 'type': 'Belong', 'end': 'Category3'}, 
    {'start': 'Category3', 'type': 'Belong', 'end': 'Category2'}, 
    {'start': 'Category2', 'type': 'Belong', 'end': 'Category1'}
]

"""
