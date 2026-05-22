"""人工评估集管理。

低分 bad case 会先进入候选集，人工补充 expected_answer / label 后可作为
离线回归评测集，降低完全依赖 LLM-as-Judge 的偏差。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

_GOLDEN_PATH = Path("data/golden_cases.jsonl")


def add_candidate(case: dict) -> None:
    """将坏例追加为待人工标注样本。"""
    _GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": f"case_{int(time.time() * 1000)}",
        "tenant_id": case.get("tenant_id", "default"),
        "query": case.get("query", ""),
        "answer": case.get("answer", ""),
        "context_nodes": case.get("context_nodes", []),
        "expected_answer": "",
        "label": "pending",
        "source": "bad_case",
        "created_at": time.time(),
    }
    with _GOLDEN_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_cases(limit: int = 50, label: str | None = None) -> list[dict]:
    """读取人工评估集样本。"""
    if not _GOLDEN_PATH.exists():
        return []
    rows = []
    for line in _GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if label and row.get("label") != label:
            continue
        rows.append(row)
    return rows[-limit:]


def stats() -> dict:
    rows = list_cases(limit=100000)
    labeled = [r for r in rows if r.get("label") != "pending"]
    return {
        "path": str(_GOLDEN_PATH),
        "total": len(rows),
        "pending": len(rows) - len(labeled),
        "labeled": len(labeled),
    }
