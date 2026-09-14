import os
import requests

resp = requests.post(
    "http://localhost:9380/api/v1/datasets/search",
    headers={
        "Authorization": f"Bearer {os.getenv('RAGFLOW_API_KEY', '')}",
        "Content-Type": "application/json",
    },
    json={
        "question": "中国科学院2023年总预算是多少？",
        "dataset_ids": [os.getenv("RAGFLOW_DATASET_ID", "3f5414ca9a5611f193cc010101010000")],
        "top_k": 5,
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
    },
)
print(resp.status_code)
print(resp.json())
