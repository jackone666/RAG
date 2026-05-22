"""
数据治理管道 - 文档生命周期同步与向量清理模块

功能说明：
- 当上游业务系统删除/更新文档时，负责清理向量库中的对应数据
- 支持按 doc_id 精确删除和按 tenant_id 批量清理
- 防止已删除文档的历史 Chunk 产生"幽灵幻觉"

生产场景：
- 合同到期、产品下架、知识库归档等场景
- GDPR 用户数据删除请求（"被遗忘权"）
- 文档内容更新时的"先删后写"策略
"""
from loguru import logger
from pymilvus import Collection, connections, utility

from src.config.settings import settings
from src.utils.helpers import _escape_milvus_expr


class SyncManager:
    """文档生命周期同步管理器。

    核心职责：
    - delete_document: 按 doc_id 精确删除文档所有 Chunk
    - delete_tenant: 按 tenant_id 批量清理（用于租户注销/GDPR）

    实现细节：
    - 懒连接模式：首次操作时才建立 Milvus 连接
    - 先查询后删除：确保删除前已定位到具体实体
    - 空集合保护：集合不存在时直接返回，避免异常中断
    """

    def __init__(self):
        """初始化同步管理器（延迟连接模式）。"""
        self._connected = False

    def _ensure_connection(self):
        """确保与 Milvus 的连接已建立。

        使用懒连接模式避免模块导入时的副作用，
        仅在首次数据操作时才建立实际网络连接。
        """
        if not self._connected:
            connections.connect(
                alias="default",
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )
            self._connected = True

    def delete_document(self, doc_id: str, tenant_id: str | None = None) -> int:
        """根据文档 ID 删除向量库中对应的所有 Chunk，并同步清理 ES 索引。

        删除策略：
        1. 构建过滤表达式: doc_id == "xxx" [&& tenant_id == "yyy"]
        2. 查询匹配实体的主键 ID
        3. 按主键批量删除
        4. 同步删除 Elasticsearch 中的对应文档

        提供 tenant_id 可加速查询（缩小扫描范围），同时防止误删其他租户
        的同名文档。

        Args:
            doc_id: 上游业务系统的文档唯一标识
            tenant_id: 可选，租户标识（提供则同时校验租户归属）

        Returns:
            实际删除的 Chunk 数量
        """
        self._ensure_connection()
        collection_name = settings.milvus_collection_name

        milvus_deleted = 0
        result = []  # 初始化 result，防止集合不存在时 UnboundLocalError
        if not utility.has_collection(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在，跳过 Milvus 删除")
        else:
            collection = Collection(collection_name)
            collection.load()

            expr = f'doc_id == "{_escape_milvus_expr(doc_id)}"'
            if tenant_id:
                expr += f' && tenant_id == "{_escape_milvus_expr(tenant_id)}"'

            result = collection.query(expr=expr, output_fields=["id", "content_hash"])
            ids_to_delete = [r["id"] for r in result]

            if ids_to_delete:
                collection.delete(expr=expr)
                milvus_deleted = len(ids_to_delete)
                logger.info(f"Milvus 已删除 {milvus_deleted} 个 Chunk: doc_id={doc_id}, tenant={tenant_id}")
            else:
                logger.info(f"Milvus 未找到需删除的数据: doc_id={doc_id}, tenant={tenant_id}")

        # 同步清理 Elasticsearch
        es_deleted = 0
        try:
            from src.engine.retrievers import ElasticsearchKeywordRetriever

            es = ElasticsearchKeywordRetriever()
            es_deleted = es.delete_document(doc_id, tenant_id or "")
        except Exception as e:
            logger.warning(f"ES 文档删除失败（非阻塞）: {e}")

        # 清理相关缓存（TTLCache + ByteCache + near_dedup），防止查到已删除的旧知识
        total = max(milvus_deleted, es_deleted)
        if total > 0 and tenant_id:
            try:
                from src.utils.cache import retrieval_cache
                retrieval_cache.clear()
            except Exception as e:
                logger.warning(f"TTLCache 清理失败: {e}")
            try:
                from src.utils.byte_cache import byte_cache
                byte_cache.invalidate_by_doc(tenant_id, doc_id)
            except Exception as e:
                logger.warning(f"ByteCache 按文档清理失败: {e}")
            try:
                # 从 Milvus 查询该文档的 content_hash 以清理近重复索引
                from src.pipeline.near_dedup import remove_doc_from_index
                if milvus_deleted > 0:
                    cleanup_hashes = set()
                    for r in result:
                        ch = r.get("content_hash", "")
                        if ch:
                            cleanup_hashes.add(ch)
                    for ch in cleanup_hashes:
                        remove_doc_from_index(ch)
            except Exception as e:
                logger.warning(f"近重复索引清理失败: {e}")

        return total

    def delete_tenant(self, tenant_id: str) -> int:
        """按租户 ID 批量清理所有数据，同时清理 ES（GDPR / 租户注销场景）。

        警告：此操作不可逆，生产环境应加二次确认机制。

        Args:
            tenant_id: 待清理的租户标识

        Returns:
            删除的 Chunk 总数
        """
        self._ensure_connection()
        collection_name = settings.milvus_collection_name

        milvus_deleted = 0
        if not utility.has_collection(collection_name):
            logger.warning(f"集合 '{collection_name}' 不存在，跳过 Milvus 清理")
        else:
            collection = Collection(collection_name)
            collection.load()

            escaped_tenant = _escape_milvus_expr(tenant_id)
            result = collection.query(
                expr=f'tenant_id == "{escaped_tenant}"',
                output_fields=["id"],
            )
            if result:
                collection.delete(expr=f'tenant_id == "{escaped_tenant}"')
                milvus_deleted = len(result)
                logger.info(f"Milvus 已清理租户 {tenant_id} 的全部 {milvus_deleted} 条数据")

        # 同步清理 Elasticsearch
        es_deleted = 0
        try:
            from src.engine.retrievers import ElasticsearchKeywordRetriever

            es = ElasticsearchKeywordRetriever()
            es_deleted = es.delete_tenant(tenant_id)
        except Exception as e:
            logger.warning(f"ES 租户清理失败（非阻塞）: {e}")

        # 清理该租户的所有缓存
        total = max(milvus_deleted, es_deleted)
        if total > 0:
            try:
                from src.utils.cache import retrieval_cache
                retrieval_cache.clear()
            except Exception as e:
                logger.warning(f"TTLCache 清理失败: {e}")
            try:
                from src.utils.byte_cache import byte_cache
                byte_cache.invalidate_by_tenant(tenant_id)
            except Exception as e:
                logger.warning(f"ByteCache 按租户清理失败: {e}")

        return total
