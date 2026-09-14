"""按 Markdown 标题切块：表格单独成块，超长章再切。"""

from __future__ import annotations

import re
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.+)$")


def chunk_markdown(text: str, source: str, max_len: int = 700) -> list[dict[str, str]]:
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    chunks: list[dict[str, str]] = []
    title = Path(source).stem
    buf: list[str] = []
    in_fence = False
    in_table = False
    table_buf: list[str] = []

    def flush_body() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        if len(body) <= max_len:
            chunks.append({"source": source, "title": title, "text": f"{title}\n{body}"})
            return
        for i in range(0, len(body), max_len - 40):
            part = body[i : i + max_len]
            chunks.append({"source": source, "title": title, "text": f"{title}\n{part}"})

    def flush_table() -> None:
        raw = "\n".join(table_buf).strip()
        table_buf.clear()
        if raw:
            chunks.append({"source": source, "title": f"{title} / 表格", "text": f"{title}\n{raw[:max_len]}"})

    for line in lines:
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not in_fence and line.lstrip().startswith("<table"):
            flush_body()
            in_table = True
            table_buf.append(line)
            if "</table>" in line:
                in_table = False
                flush_table()
            continue
        if in_table:
            table_buf.append(line)
            if "</table>" in line:
                in_table = False
                flush_table()
            continue
        m = HEADING.match(line) if not in_fence else None
        if m:
            flush_body()
            title = m.group(2).strip()
            continue
        buf.append(line)
    if in_table:
        flush_table()
    flush_body()
    return chunks


def load_kb(kb_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(kb_dir.glob("*.md")):
        rows.extend(chunk_markdown(path.read_text(encoding="utf-8", errors="ignore"), path.name))
    return rows
