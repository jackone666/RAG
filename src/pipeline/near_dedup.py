"""
近重复文档检测 — 从 SHA256 完全去重升级到 embedding 相似度去重

原方案：SHA256(content) 完全匹配才视为重复（精确去重）
升级后：新增 embedding 余弦相似度检测，发现改写版、转载版、翻译版文档

流程：
  1. 新文档 → SHA256 精确匹配（快速路径）
  2. 精确未命中 → 计算文档级 embedding
  3. embedding 与已有文档的 embedding 计算余弦相似度
  4. 超过阈值 (near_duplicate_threshold) → 标记为疑似重复

参考字节 RAG 实践 §3.2「文档理解与预处理」— 去重不只是 hash，
需要对改写、摘抄、翻译等变体有感知能力。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.config.rag_params import rag_params

# 持久化已有文档 embedding 的文件路径
_INDEX_PATH = Path("data/near_dedup_index.json")


def _compute_doc_embedding(text: str, embed_model) -> tuple[list[float], bytes]:
    """计算文档级嵌入向量和对应的哈希。

    策略：取文本前 N 个字符（代表文档主题），调用嵌入模型。
    短文本直接使用原文，长文本截取开头+结尾（摘要/结论通常在此）。
    """
    max_chars = rag_params.embedding_max_input_chars
    if len(text) > max_chars:
        half = max_chars // 2
        sample = text[:half] + text[-half:]
    else:
        sample = text

    emb = embed_model._get_text_embedding(sample[:max_chars])
    h = hashlib.sha256(sample[:max_chars].encode("utf-8", errors="replace")).hexdigest()[:16]
    return emb, h.encode()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_index() -> dict[str, dict[str, Any]]:
    """加载已有文档的 (content_hash → embedding 元信息) 索引。"""
    if not _INDEX_PATH.exists():
        return {}
    try:
        return json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"近重复索引文件损坏，重建: {e}")
        return {}


def _save_index(index: dict) -> None:
    """持久化近重复索引。"""
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def register_doc_embedding(
        tenant_id: str,
        doc_id: str,
        content_hash: str,
        embedding: list[float],
) -> None:
    """将文档的语义向量注册到近重复索引，供后续比对。

    Args:
        tenant_id: 租户标识
        doc_id: 文档标识
        content_hash: 原始内容 SHA256
        embedding: 文档级 embedding 向量
    """
    index = _load_index()
    index[content_hash] = {
        "tenant_id": tenant_id,
        "doc_id": doc_id,
        "embedding": embedding,
        "registered_at": time.time(),
    }
    _save_index(index)
    logger.info(f"文档语义指纹已注册: doc_id={doc_id}")


def detect_near_duplicate(
        tenant_id: str,
        text: str,
        embed_model,
) -> dict[str, Any] | None:
    """检测新文档是否与已入库文档存在语义级重复。

    流程：
    1. SHA256 精确匹配（快速路径）
    2. 计算文档 embedding
    3. 与同租户下已有文档 embedding 做余弦相似度比较
    4. 超过阈值返回最相似文档信息

    Args:
        tenant_id: 租户标识（仅在同租户范围内比较）
        text: 新文档文本
        embed_model: 嵌入模型实例（需有 _get_text_embedding 方法）

    Returns:
        None 表示未发现重复，否则返回:
        {"existing_doc_id": "...", "similarity": 0.97, "match_type": "exact"|"near"}
    """
    threshold = rag_params.near_duplicate_threshold

    # 1. SHA256 精确匹配
    content_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    index = _load_index()

    if content_hash in index:
        existing = index[content_hash]
        if existing.get("tenant_id") == tenant_id:
            return {
                "existing_doc_id": existing["doc_id"],
                "similarity": 1.0,
                "match_type": "exact",
            }

    # 2. 计算 embedding
    query_emb, _ = _compute_doc_embedding(text, embed_model)

    # 3. 余弦相似度比较（按租户过滤）
    best_match = None
    best_similarity = 0.0
    for ch, entry in index.items():
        if entry.get("tenant_id") != tenant_id:
            continue
        stored_emb = entry.get("embedding")
        if not stored_emb:
            continue
        sim = _cosine_similarity(query_emb, stored_emb)
        if sim > best_similarity:
            best_similarity = sim
            best_match = entry

    # 4. 阈值判断
    if best_match and best_similarity >= threshold:
        return {
            "existing_doc_id": best_match["doc_id"],
            "similarity": round(best_similarity, 4),
            "match_type": "near",
        }

    # 未匹配：注册当前文档的语义指纹
    register_doc_embedding(tenant_id, "pending", content_hash, query_emb)
    return None


def remove_doc_from_index(content_hash: str) -> None:
    """文档删除后清理索引（防止死引用）。"""
    index = _load_index()
    if content_hash in index:
        del index[content_hash]
        _save_index(index)


def index_stats() -> dict:
    """返回近重复索引统计。"""
    index = _load_index()
    return {
        "path": str(_INDEX_PATH),
        "total": len(index),
        "tenants": len({e.get("tenant_id") for e in index.values()}),
    }
