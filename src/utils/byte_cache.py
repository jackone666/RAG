"""
ByteCache — Redis 向量检索热点缓存

参考字节跳动 ByteCache 方案：将高频查询的向量检索结果缓存在 Redis，
命中时直接返回，绕过 Milvus，将 P99 延迟从 ~50ms 降至 ~15ms。

架构：
  Query → embedding hash → Redis GET → hit? → return cached
                           ↓ miss
                    Milvus search → Redis SET (TTL 7d) → return

热点追踪：
- Redis Sorted Set 记录每个 cache key 的访问频次
- 定期或按阈值筛选 top-K 热点 key，保持缓存有效性
"""
import hashlib
import json
import time
from typing import Optional

import redis
from loguru import logger

from src.config.rag_params import rag_params
from src.config.settings import settings


class ByteCache:
    """Redis 向量检索热点缓存。

    缓存粒度：query_embedding_hash → top-K 检索结果的序列化快照
    TTL：默认 7 天（604800 秒），LRU 由 Redis 内置 maxmemory-policy 保证
    """

    # Redis key 前缀
    CACHE_PREFIX = "bytecache:v"
    HOT_PREFIX = "bytecache:hot"

    def __init__(self):
        self._client: Optional[redis.Redis] = None
        self._ttl = rag_params.byte_cache_ttl
        self._enabled = rag_params.byte_cache_enabled

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(
                settings.redis_url,
                socket_timeout=settings.redis_socket_timeout,
                decode_responses=True,
            )
        return self._client

    # ── cache key ────────────────────────────────────────────

    @staticmethod
    def build_key(tenant_id: str, embedding_bytes: bytes, top_k: int) -> str:
        """根据嵌入向量哈希生成缓存键。"""
        h = hashlib.sha256(embedding_bytes).hexdigest()[:20]
        return f"{ByteCache.CACHE_PREFIX}:{tenant_id}:{top_k}:{h}"

    @staticmethod
    def embedding_hash(embedding: list[float]) -> bytes:
        """将嵌入向量转为字节哈希输入。"""
        return hashlib.sha256(
            ",".join(f"{v:.6f}" for v in embedding[:128]).encode()
        ).digest()

    # ── public API ───────────────────────────────────────────

    def get(self, tenant_id: str, embedding: list[float], top_k: int) -> Optional[list[dict]]:
        """查询缓存。

        Returns:
            缓存命中时返回 [{"node_id": ..., "score": ..., "text": ..., "metadata": ...}, ...]
            未命中返回 None
        """
        if not self._enabled:
            return None

        try:
            key = self.build_key(tenant_id, self.embedding_hash(embedding), top_k)
            raw = self.client.get(key)
            if raw:
                self._touch_hot(key)
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"ByteCache 读取失败: {e}")

        return None

    def set(
            self,
            tenant_id: str,
            embedding: list[float],
            top_k: int,
            results: list[dict],
    ) -> None:
        """写入缓存。

        Args:
            results: 序列化后的检索结果列表
        """
        if not self._enabled:
            return

        try:
            key = self.build_key(tenant_id, self.embedding_hash(embedding), top_k)
            self.client.setex(key, self._ttl, json.dumps(results, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"ByteCache 写入失败: {e}")

    # ── hot-tracking ─────────────────────────────────────────

    def _touch_hot(self, key: str):
        """记录一次热点访问（异步计数）。"""
        try:
            today = time.strftime("%Y%m%d")
            hot_key = f"{self.HOT_PREFIX}:{today}"
            self.client.zincrby(hot_key, 1, key)
            self.client.expire(hot_key, 86400 * 8)  # 8 天过期
        except Exception:
            pass

    def top_hot_keys(self, n: int = 20) -> list[tuple[str, float]]:
        """获取今日访问次数最多的缓存键（用于监控）。"""
        try:
            today = time.strftime("%Y%m%d")
            hot_key = f"{self.HOT_PREFIX}:{today}"
            return self.client.zrevrange(hot_key, 0, n - 1, withscores=True) or []
        except Exception:
            return []

    # ── fine-grained invalidation ─────────────────────────────

    # Redis key 前缀：反向索引 doc_id/tenant_id → 受影响的 cache keys
    DOC_INDEX_PREFIX = "bytecache:doc"
    TENANT_INDEX_PREFIX = "bytecache:tenant"

    def _track_doc_keys(self, tenant_id: str, doc_id: str, cache_key: str):
        """记录 doc_id → cache_key 反向索引，用于文档更新后精确清理。"""
        try:
            idx_key = f"{self.DOC_INDEX_PREFIX}:{tenant_id}:{doc_id}"
            self.client.sadd(idx_key, cache_key)
            self.client.expire(idx_key, self._ttl)
        except Exception:
            pass

    def set_with_tracking(
            self,
            tenant_id: str,
            embedding: list[float],
            top_k: int,
            results: list[dict],
            doc_ids: list[str] | None = None,
    ) -> None:
        """写入缓存并建立 doc_id 反向索引，支持文档更新后精确失效。

        相比 set()，额外记录哪些 doc_id 触发了此缓存条目，
        后续 delete_document 时可据此清理受影响的热点缓存。
        """
        if not self._enabled:
            return

        try:
            key = self.build_key(tenant_id, self.embedding_hash(embedding), top_k)
            self.client.setex(key, self._ttl, json.dumps(results, ensure_ascii=False))
            if doc_ids:
                for doc_id in doc_ids:
                    self._track_doc_keys(tenant_id, doc_id, key)
        except Exception as e:
            logger.warning(f"ByteCache 写入失败: {e}")

    def invalidate_by_doc(self, tenant_id: str, doc_id: str) -> int:
        """文档删除/更新后，精确清理该文档关联的所有缓存条目。

        通过反向索引找到受影响的 cache keys 并批量删除。

        Returns:
            清理的缓存条目数
        """
        if not self._enabled:
            return 0

        try:
            idx_key = f"{self.DOC_INDEX_PREFIX}:{tenant_id}:{doc_id}"
            cache_keys = self.client.smembers(idx_key)
            if not cache_keys:
                return 0
            deleted = self.client.delete(*cache_keys)
            self.client.delete(idx_key)
            logger.info(f"ByteCache 已清理 doc_id={doc_id} 的 {deleted} 条缓存")
            return deleted
        except Exception as e:
            logger.warning(f"ByteCache 按文档清理失败: {e}")
            return 0

    def invalidate_by_tenant(self, tenant_id: str) -> int:
        """租户注销时，批量清理该租户所有缓存条目。

        扫描 cache 前缀下的所有 key 并批量删除。
        """
        if not self._enabled:
            return 0

        try:
            pattern = f"{self.CACHE_PREFIX}:{tenant_id}:*"
            keys = []
            cursor = 0
            while True:
                cursor, batch = self.client.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                self.client.delete(*keys)
            # 同时清理反向索引
            idx_pattern = f"{self.DOC_INDEX_PREFIX}:{tenant_id}:*"
            cursor = 0
            while True:
                cursor, batch = self.client.scan(cursor, match=idx_pattern, count=100)
                if batch:
                    self.client.delete(*batch)
                if cursor == 0:
                    break
            logger.info(f"ByteCache 已清理租户 {tenant_id} 的 {len(keys)} 条缓存")
            return len(keys)
        except Exception as e:
            logger.warning(f"ByteCache 按租户清理失败: {e}")
            return 0

    # ── stats ────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回缓存统计（用于 /health 或监控面板）。"""
        return {
            "enabled": self._enabled,
            "ttl_seconds": self._ttl,
            "ttl_days": self._ttl // 86400,
        }


# 全局单例
byte_cache = ByteCache()
