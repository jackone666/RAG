"""
检索引擎 - 多租户混合检索模块（向量 + Elasticsearch 关键词融合）

实现向量检索 + Elasticsearch 关键词检索的融合查询，所有检索路径强制注入
tenant_id 元数据过滤，确保存储引擎层面完成租户隔离。

并发安全设计：
- 每个请求独立构建 VectorStoreQuery 对象，多协程间零共享可变状态
- Elasticsearch 查询天然无状态，按 tenant_id 过滤保证隔离
"""
import time
from dataclasses import dataclass, field

from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.config.rag_params import rag_params
from src.utils.byte_cache import byte_cache
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.milvus import MilvusVectorStore
from elasticsearch import Elasticsearch
from loguru import logger

from src.config.settings import settings


@dataclass
class RetrievalTrace:
    """混合检索全链路追踪数据，用于前端可视化。"""

    vector_count: int = 0
    vector_latency_ms: float = 0.0
    keyword_count: int = 0
    keyword_latency_ms: float = 0.0
    fusion_count: int = 0
    fusion_mode: str = "relative_score"
    top_scores: list[float] = field(default_factory=list)
    # 召回文档预览（用于 Langfuse / 前端展示）
    doc_previews: list[dict] = field(default_factory=list)

# 全局共享的嵌入模型实例
_shared_embed_model = None


class SiliconFlowEmbedding(BaseEmbedding):
    """远程嵌入模型 — 兼容 LlamaIndex BaseEmbedding，可用于 SemanticSplitterNodeParser。

    BAAI/bge-m3: 1024 维，多语言，输入最长 8192 token。
    """

    _model: str = "BAAI/bge-m3"
    _api_key: str = ""
    _api_url: str = ""
    _client: object = None

    def __init__(self, model: str = "BAAI/bge-m3", api_key: str = "", api_url: str = ""):
        super().__init__(model_name=model)
        from openai import OpenAI as SyncOpenAI

        self._model = model
        self._api_key = api_key or settings.openai_api_key
        self._api_url = api_url or settings.embedding_api_url
        self._client = SyncOpenAI(api_key=self._api_key, base_url=self._api_url)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        """调用远程 API 获取嵌入向量。"""
        texts = [t[:8000] for t in texts]
        result = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            result.extend([d.embedding for d in resp.data])
        return result

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._call_api([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        return self._call_api(texts)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._call_api([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        import asyncio
        return await asyncio.to_thread(self._get_query_embedding, query)

    def get_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self._call_api(texts)

    async def aget_text_embedding_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self._call_api, texts)


def get_shared_embed_model() -> SiliconFlowEmbedding:
    global _shared_embed_model
    if _shared_embed_model is None:
        _shared_embed_model = SiliconFlowEmbedding(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key or settings.openai_api_key,
            api_url=settings.embedding_api_url,
        )
        logger.info(f"远程嵌入: {settings.embedding_model} @ {settings.embedding_api_url}")
    return _shared_embed_model


def _build_tenant_filter(tenant_id: str) -> MetadataFilters:
    """构建租户级别的元数据过滤器，用于向量存储层的前置硬过滤。"""
    return MetadataFilters(
        filters=[MetadataFilter(key="tenant_id", value=tenant_id, operator="==")]
    )


class ElasticsearchKeywordRetriever:
    """基于 Elasticsearch 的关键词检索器。

    替代原先的内存 BM25 实现，提供：
    - 持久化的文档索引存储
    - 基于 ES match 查询的全文关键词匹配
    - tenant_id 维度的数据隔离
    - 支持文档/租户级别的增量删除
    """

    def __init__(self):
        self._client: Elasticsearch | None = None
        self._index_ready: bool = False

    @property
    def client(self) -> Elasticsearch:
        """延迟初始化 ES 客户端。"""
        if self._client is None:
            self._client = Elasticsearch(
                settings.elasticsearch_url,
                request_timeout=10,
            )
        return self._client

    def _ensure_index(self):
        """确保索引存在并按配置的分词器创建 mapping，处理 IK 插件缺失的降级。"""
        if self._index_ready:
            return

        index = settings.elasticsearch_index
        analyzer = settings.elasticsearch_analyzer

        if self.client.indices.exists(index=index):
            self._index_ready = True
            return

        # 尝试按配置的 analyzer 创建索引
        def _make_body(analyzer_name: str) -> dict:
            return {
                "settings": {
                    "index": {"number_of_shards": 1, "number_of_replicas": 0},
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": analyzer_name},
                        "tenant_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                    }
                },
            }

        try:
            self.client.indices.create(index=index, body=_make_body(analyzer))
            logger.info(f"ES 索引已创建: {index}, analyzer={analyzer}")
        except Exception:
            logger.warning(
                f"ES 索引创建失败（analyzer={analyzer}），降级为 standard"
            )
            self.client.indices.create(index=index, body=_make_body("standard"))
            logger.info(f"ES 索引已创建（standard analyzer）: {index}")

        self._index_ready = True

    def index_chunks(self, tenant_id: str, doc_id: str, nodes: list) -> int:
        """将入库后的 Chunk 批量索引到 Elasticsearch。

        Args:
            tenant_id: 租户标识
            doc_id: 文档标识
            nodes: 已嵌入的 Node 列表

        Returns:
            成功索引的文档数
        """
        from elasticsearch.helpers import bulk

        self._ensure_index()

        actions = []
        for i, node in enumerate(nodes):
            chunk_id = node.node_id or f"{doc_id}_{i}"
            actions.append({
                "_index": settings.elasticsearch_index,
                "_id": chunk_id,
                "_source": {
                    "text": node.get_content(),
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                },
            })

        if not actions:
            return 0

        success, errors = bulk(self.client, actions, refresh=True)
        if errors:
            logger.warning(f"ES 索引部分失败: {len(errors)} errors")
        logger.info(f"ES 已索引 {success} chunks: doc_id={doc_id}, tenant={tenant_id}")
        return success

    def search(self, tenant_id: str, query_str: str, top_k: int | None = None) -> list[NodeWithScore]:
        """Elasticsearch 关键词检索（match 查询 + tenant_id 过滤）。

        Args:
            tenant_id: 租户标识
            query_str: 查询文本
            top_k: 返回的最大节点数

        Returns:
            按 ES _score 降序排列的 NodeWithScore 列表
        """
        top_k = top_k or rag_params.retrieval_top_k
        try:
            self._ensure_index()
            resp = self.client.search(
                index=settings.elasticsearch_index,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"text": query_str}},
                            ],
                            "filter": [
                                {"term": {"tenant_id": tenant_id}},
                            ],
                        }
                    },
                    "size": top_k,
                },
            )
        except Exception as e:
            logger.warning(f"ES 检索失败: {e}")
            return []

        hits = resp.get("hits", {}).get("hits", [])
        nodes = []
        from llama_index.core.schema import TextNode

        for hit in hits:
            src = hit["_source"]
            node = TextNode(text=src.get("text", ""), id_=hit["_id"])
            node.metadata["tenant_id"] = src.get("tenant_id", "")
            node.metadata["doc_id"] = src.get("doc_id", "")
            score = hit.get("_score", 0)
            max_score = resp.get("hits", {}).get("max_score")
            if max_score is None or max_score == 0:
                max_score = 1
            nodes.append(NodeWithScore(node=node, score=score / max_score))

        return nodes

    def delete_document(self, doc_id: str, tenant_id: str) -> int:
        """从 ES 中删除指定文档的所有 Chunk。

        Args:
            doc_id: 文档标识
            tenant_id: 租户标识（用于二次校验）

        Returns:
            删除的文档数
        """
        try:
            resp = self.client.delete_by_query(
                index=settings.elasticsearch_index,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"doc_id": doc_id}},
                                {"term": {"tenant_id": tenant_id}},
                            ]
                        }
                    }
                },
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            logger.info(f"ES 已删除 {deleted} chunks: doc_id={doc_id}, tenant={tenant_id}")
            return deleted
        except Exception as e:
            logger.warning(f"ES 删除文档失败: {e}")
            return 0

    def delete_tenant(self, tenant_id: str) -> int:
        """从 ES 中删除指定租户的所有数据（GDPR / 租户注销场景）。

        Args:
            tenant_id: 租户标识

        Returns:
            删除的文档数
        """
        try:
            resp = self.client.delete_by_query(
                index=settings.elasticsearch_index,
                body={"query": {"term": {"tenant_id": tenant_id}}},
                refresh=True,
            )
            deleted = resp.get("deleted", 0)
            logger.info(f"ES 已清理租户 {tenant_id} 的全部 {deleted} 条数据")
            return deleted
        except Exception as e:
            logger.warning(f"ES 清理租户失败: {e}")
            return 0


def _ensure_node_with_score(nodes: list) -> list[NodeWithScore]:
    """确保所有节点都包装为 NodeWithScore，兼容不同 LlamaIndex 版本。"""
    result = []
    for n in nodes:
        if isinstance(n, NodeWithScore):
            result.append(n)
        else:
            result.append(NodeWithScore(node=n, score=0.0))
    return result


class _FusionAdapter(BaseRetriever):
    """内部适配器：确保所有节点都包装为 NodeWithScore，兼容不同 LlamaIndex 版本。"""

    def __init__(self, nodes: list[NodeWithScore]):
        super().__init__()
        self._cached = []
        for n in nodes:
            if hasattr(n, "node"):
                self._cached.append(n)  # 已是 NodeWithScore
            else:
                self._cached.append(NodeWithScore(node=n))  # 裸 TextNode → 包装

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._cached

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return self._cached


def _build_doc_previews(nodes: list[NodeWithScore], limit: int = 10) -> list[dict]:
    """从节点列表提取文档预览，用于 Langfuse span 和前端展示。"""
    return [
        {
            "rank": i + 1,
            "score": round(n.score or 0.0, 4),
            "text": (n.node.get_content() or "")[:rag_params.doc_preview_chars].replace("\n", " "),
            "doc_id": n.node.metadata.get("doc_id", ""),
            "file_name": n.node.metadata.get("file_name", ""),
        }
        for i, n in enumerate(nodes[:limit])
    ]


def _serialize_node(n) -> dict:
    """将 NodeWithScore 或 TextNode 序列化为可 JSON 存储的 dict（ByteCache 用）。"""
    # 兼容两种输入：NodeWithScore（含 .node）或裸 TextNode
    if hasattr(n, 'node') and hasattr(n, 'score'):
        inner, score = n.node, n.score or 0.0
    else:
        inner, score = n, 0.0

    text = ""
    if hasattr(inner, 'get_content'):
        text = inner.get_content() or ""
    elif hasattr(inner, 'text'):
        text = inner.text or ""
    text = text[:rag_params.byte_cache_max_chars]

    return {
        "node_id": getattr(inner, 'node_id', ''),
        "score": round(score, 6),
        "text": text,
        "metadata": dict(getattr(inner, 'metadata', {}) or {}),
    }


def _deserialize_node(data: dict) -> NodeWithScore:
    """从 ByteCache 序列化数据还原 NodeWithScore。"""
    from llama_index.core.schema import TextNode
    node = TextNode(
        id_=data.get("node_id"),
        text=data.get("text", ""),
        metadata=data.get("metadata", {}),
    )
    return NodeWithScore(node=node, score=data.get("score", 0.0))


def _weighted_rrf_fusion(
    vector_nodes: list[NodeWithScore],
    keyword_nodes: list[NodeWithScore],
    vw: float = 0.6,
    kw: float = 0.4,
    k: int = 60,
    top_k: int = 10,
) -> list[NodeWithScore]:
    """加权倒数排名融合 (Weighted Reciprocal Rank Fusion)。

    参考字节 §4.3.2「融合排序优化」— 向量语义理解 + 关键词精确匹配，
    通过可配置权重平衡两条通路的贡献。

    公式：score(d) = vw / (k + rank_v(d)) + kw / (k + rank_kw(d))
    """
    # 按原始分数降序建立排名
    vec_sorted = sorted(vector_nodes, key=lambda n: n.score or 0, reverse=True)
    kw_sorted = sorted(keyword_nodes, key=lambda n: n.score or 0, reverse=True)

    vec_rank: dict[str, int] = {}
    for rank, node in enumerate(vec_sorted, start=1):
        vec_rank[node.node.node_id] = rank

    kw_rank: dict[str, int] = {}
    for rank, node in enumerate(kw_sorted, start=1):
        kw_rank[node.node.node_id] = rank

    # 合并所有节点，计算加权 RRF 分数
    seen: dict[str, NodeWithScore] = {}
    rrf_scores: dict[str, float] = {}

    for node in vector_nodes + keyword_nodes:
        nid = node.node.node_id
        if nid not in seen:
            seen[nid] = node
            r = vec_rank.get(nid, len(vec_sorted) + 1)
            rrf_scores[nid] = vw / (k + r)
        # 累加关键词通路分数
        r = kw_rank.get(nid, len(kw_sorted) + 1)
        rrf_scores[nid] = rrf_scores.get(nid, 0.0) + kw / (k + r)

    # 按 RRF 分数降序排列
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    result = []
    for nid in sorted_ids:
        node = seen[nid]
        node.score = rrf_scores[nid]
        result.append(node)

    from loguru import logger
    logger.info(
        f"加权RRF融合 (向量:{vw}, 关键词:{kw}, k={k}) → {len(result)} 节点"
    )
    return result


class TenantAwareQueryFusionRetriever:
    """多租户混合检索引擎：融合向量相似度检索与 ES 关键词检索。

    核心安全机制：
    - 向量检索在 Milvus 存储层强制注入 tenant_id 元数据过滤
    - 关键词检索在 ES 查询层强制追加 tenant_id term filter
    - 两条检索通路独立执行，最后加权 RRF 或 LlamaIndex Fusion 融合排序

    并发安全：
    - 直接构造 VectorStoreQuery + aquery()，无全局状态
    - ES 检索无状态，天然并发安全
    """

    def __init__(self, embed_model=None):
        self._vector_store: MilvusVectorStore | None = None
        self._keyword_retriever = ElasticsearchKeywordRetriever()
        self._embed_model_override = embed_model

    def _get_embed_model(self):
        """懒加载嵌入模型。"""
        if self._embed_model_override is not None:
            return self._embed_model_override
        return get_shared_embed_model()

    def _get_vector_store(self) -> MilvusVectorStore:
        if self._vector_store is None:
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
                collection_name=settings.milvus_collection_name,
                dim=settings.embedding_dim,
                embedding_field="embedding",
            )
        return self._vector_store

    async def _vector_search(
        self, query_str: str, tenant_filter: MetadataFilters, top_k: int
    ) -> list[NodeWithScore]:
        """带租户过滤的向量检索 + ByteCache 热点缓存。

        ByteCache 策略（参考字节 §4.3.1）：
        - 命中 → 直接返回序列化快照（~15ms），跳过 Milvus
        - 未命中 → Milvus aquery（~50ms）→ 写回 ByteCache
        """
        query_embedding = await self._get_embed_model().aget_query_embedding(query_str)

        # ByteCache: 提取 tenant_id 尝试命中
        tenant_id = ""
        if tenant_filter and tenant_filter.filters:
            for f in tenant_filter.filters:
                if f.key == "tenant_id":
                    tenant_id = str(f.value)
                    break

        if tenant_id:
            cached = byte_cache.get(tenant_id, query_embedding, top_k)
            if cached is not None:
                from llama_index.core.schema import TextNode
                return [_deserialize_node(item) for item in cached]

        # 缓存未命中 → Milvus
        query = VectorStoreQuery(
            query_str=query_str,
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            filters=tenant_filter,
        )
        result = await self._get_vector_store().aquery(query)
        nodes = result.nodes or []

        # 写回 ByteCache
        if tenant_id and nodes:
            byte_cache.set(tenant_id, query_embedding, top_k,
                           [_serialize_node(n) for n in nodes])

        return nodes

    async def retrieve(self, query: str, tenant_id: str, top_k: int | None = None) -> list[NodeWithScore]:
        """执行混合检索：向量检索（Milvus）+ 关键词检索（ES）→ 融合排序。

        流程：
        1. 向量检索（Milvus aquery + MetadataFilters）
        2. 关键词检索（ES match + tenant_id filter）
        3. QueryFusionRetriever 融合两条通路结果
        """
        top_k = top_k or rag_params.retrieval_top_k
        tenant_filter = _build_tenant_filter(tenant_id)
        query_bundle = QueryBundle(query_str=query)

        # 阶段 1：向量检索（Milvus）
        vector_nodes = await self._vector_search(query, tenant_filter, top_k)

        # 阶段 2：关键词检索（Elasticsearch，独立于向量结果）
        keyword_nodes = self._keyword_retriever.search(tenant_id, query, top_k)

        if not vector_nodes and not keyword_nodes:
            logger.info(f"向量+关键词检索均无结果: tenant={tenant_id}")
            return []

        if not vector_nodes:
            logger.info(f"向量检索无结果，仅返回关键词结果: {len(keyword_nodes)}")
            return keyword_nodes

        if not keyword_nodes:
            logger.info(f"关键词检索无结果，仅返回向量结果: {len(vector_nodes)}")
            return vector_nodes

        # 阶段 3：融合排序（RELATIVE_SCORE）
        fusion_retriever = QueryFusionRetriever(
            retrievers=[_FusionAdapter(vector_nodes), _FusionAdapter(keyword_nodes)],
            similarity_top_k=top_k,
            num_queries=1,
            mode=getattr(FUSION_MODES, rag_params.fusion_mode.upper(), FUSION_MODES.RELATIVE_SCORE),
            use_async=True,
        )

        nodes = await fusion_retriever.aretrieve(query_bundle)
        logger.info(f"混合检索完成 → {len(nodes)} 节点 (向量+ES融合), tenant={tenant_id}")
        return nodes

    async def aretrieve_with_trace(
        self, query: str, tenant_id: str, top_k: int | None = None
    ) -> tuple[list[NodeWithScore], RetrievalTrace]:
        """执行混合检索并返回全链路追踪数据，用于前端可视化。

        Returns:
            (融合后的节点列表, RetrievalTrace 追踪数据)
        """
        trace = RetrievalTrace()
        top_k = top_k or rag_params.retrieval_top_k
        tenant_filter = _build_tenant_filter(tenant_id)
        query_bundle = QueryBundle(query_str=query)

        # 使用独立 top_k 分别检索
        vec_top_k = rag_params.retrieval_vector_top_k
        kw_top_k = rag_params.retrieval_keyword_top_k

        # 阶段 1：向量检索（Milvus）
        t0 = time.monotonic()
        vector_nodes = _ensure_node_with_score(await self._vector_search(query, tenant_filter, vec_top_k))
        trace.vector_latency_ms = (time.monotonic() - t0) * 1000
        trace.vector_count = len(vector_nodes)

        # 阶段 2：关键词检索（Elasticsearch）
        t0 = time.monotonic()
        keyword_nodes = _ensure_node_with_score(self._keyword_retriever.search(tenant_id, query, kw_top_k))
        trace.keyword_latency_ms = (time.monotonic() - t0) * 1000
        trace.keyword_count = len(keyword_nodes)

        if not vector_nodes and not keyword_nodes:
            logger.info(f"向量+关键词检索均无结果: tenant={tenant_id}")
            return [], trace

        if not vector_nodes:
            logger.info(f"向量检索无结果，仅返回关键词结果: {len(keyword_nodes)}")
            trace.fusion_count = len(keyword_nodes)
            trace.top_scores = [n.score or 0.0 for n in keyword_nodes[:5]]
            trace.doc_previews = _build_doc_previews(keyword_nodes)
            return keyword_nodes, trace

        if not keyword_nodes:
            logger.info(f"关键词检索无结果，仅返回向量结果: {len(vector_nodes)}")
            trace.fusion_count = len(vector_nodes)
            trace.top_scores = [n.score or 0.0 for n in vector_nodes[:5]]
            trace.doc_previews = _build_doc_previews(vector_nodes)
            return vector_nodes, trace

        # 阶段 3：融合排序
        if rag_params.fusion_mode == "weighted_rrf":
            nodes = _weighted_rrf_fusion(
                vector_nodes, keyword_nodes,
                vw=rag_params.vector_weight, kw=rag_params.keyword_weight,
                k=rag_params.rrf_k, top_k=top_k,
            )
        else:
            fusion_retriever = QueryFusionRetriever(
                retrievers=[_FusionAdapter(vector_nodes), _FusionAdapter(keyword_nodes)],
                similarity_top_k=top_k,
                num_queries=1,
                mode=getattr(FUSION_MODES, rag_params.fusion_mode.upper(), FUSION_MODES.RELATIVE_SCORE),
                use_async=True,
            )
            nodes = _ensure_node_with_score(await fusion_retriever.aretrieve(query_bundle))

        trace.fusion_count = len(nodes)
        trace.top_scores = [n.score or 0.0 for n in nodes[:5]]
        trace.doc_previews = _build_doc_previews(nodes)
        logger.info(f"混合检索完成 → {len(nodes)} 节点 (向量+ES融合), tenant={tenant_id}")
        return nodes, trace


# 模块级共享实例，避免重复加载嵌入模型
shared_retriever = TenantAwareQueryFusionRetriever()
