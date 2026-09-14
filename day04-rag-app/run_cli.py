from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compose import answer
from store import build_store

store, chunks = build_store(Path(__file__).resolve().parent / "kb")
print(f"切片数 {len(chunks)}")
for q in (
    "中国科学院2023年部门总预算是多少",
    "熟练使用命令行有什么好处",
):
    hits = store.search(q)
    print("=" * 50)
    print("问:", q)
    print("引用:", [h["title"] for h in hits])
    print("答:", answer(q, hits))
