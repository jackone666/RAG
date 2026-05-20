"""
本地缓存层 — 基于 cachetools 实现 Caffeine 风格的高性能缓存。

cachetools.TTLCache 提供接近 Caffeine 的 LRU + TTL 双策略：
- 容量满时自动淘汰最久未使用条目（LRU）
- 超时条目自动过期（TTL）
- 线程安全，无需外部加锁

参考: Caffeine (Java) / cachetools (Python)
"""
from cachetools import TTLCache

from src.config.rag_params import rag_params


class RetrievalCache:
    """热点检索缓存 —— 参考字节 §4.3.1「请求缓存策略」。

    缓存粒度：tenant_id + query → 检索结果列表
    淘汰策略：LRU（超过 maxsize）+ TTL 过期
    """

    def __init__(self):
        self._cache: TTLCache = TTLCache(
            maxsize=rag_params.cache_maxsize,
            ttl=rag_params.cache_ttl,
        )

    def get(self, key: str) -> list | None:
        return self._cache.get(key)

    def set(self, key: str, nodes: list):
        self._cache[key] = nodes

    def __len__(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()


class EvalCache:
    """近期评估结果缓存 —— 供前端轮询 /v1/evaluation/{query_id}。

    淘汰策略：LRU（超过 maxsize），无 TTL（前端主动轮询消费）。
    """

    def __init__(self):
        self._cache: TTLCache = TTLCache(
            maxsize=rag_params.eval_cache_maxsize,
            ttl=rag_params.eval_cache_ttl,
        )

    def get(self, query_id: str) -> dict | None:
        return self._cache.get(query_id)

    def set(self, query_id: str, result: dict):
        self._cache[query_id] = result

    def __contains__(self, query_id: str) -> bool:
        return query_id in self._cache

    def __len__(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()


# 全局单例
retrieval_cache = RetrievalCache()
eval_cache = EvalCache()
