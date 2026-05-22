"""
数据治理管道 - 文档入库与多租户元数据注入模块

功能说明：
- 使用语义分块器将原始文档切分为语义完整的 Node
- 强制在每个 Node 的 metadata 中注入 tenant_id、doc_id、content_hash
- 基于内容哈希检测重复文档，避免同一文档重复入库
- 下游检索引擎通过 metadata 字段实现租户数据隔离

安全要求（规范 4.1）：
- 入库时必须为每个 Chunk 标记所属租户
- metadata 中的 tenant_id 是后续 RBAC 过滤的唯一依据
"""
import hashlib
from pathlib import Path

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, NodeRelationship, RelatedNodeInfo
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from loguru import logger

from src.config.rag_params import rag_params
from src.config.settings import settings
from src.engine.retrievers import get_shared_embed_model

# docstore 持久化路径
_DOCSTORE_PATH = Path("data/docstore.json")


def _index_es_background(tenant_id: str, doc_id: str, nodes: list) -> None:
    """后台线程写入 ES，不阻塞主流程。"""
    try:
        from src.engine.retrievers import ElasticsearchKeywordRetriever
        es_retriever = ElasticsearchKeywordRetriever()
        es_retriever.index_chunks(tenant_id, doc_id, nodes)
    except Exception as e:
        logger.warning(f"ES 索引失败（非阻塞）: {e}")


def _compute_content_hash(text: str) -> str:
    """计算文档内容的 SHA256 哈希值，用于重复文档检测。"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# Milvus VARCHAR 字段最大 65535 字符，预留安全余量
_MAX_CHUNK_CHARS = rag_params.pre_split_chars  # 小块策略：超过该字符数即预拆分


def _force_split_oversized(nodes: list[BaseNode], max_chars: int | None = None) -> list[BaseNode]:
    """递归强制截断超过 Milvus VARCHAR(65535) 限制的 chunk。"""
    if max_chars is None:
        max_chars = rag_params.max_chunk_chars
    result = []
    for node in nodes:
        text = node.get_content()
        if len(text) <= max_chars:
            result.append(node)
            continue
        logger.warning(f"Chunk 超长 ({len(text)} chars)，递归强制截断")
        from llama_index.core.schema import TextNode

        mid = len(text) // 2
        # 尽量在句号处断开
        for sep in ["。", ". ", "；", "\n\n", "\n", "，"]:
            pos = text.rfind(sep, max(mid - max_chars // 2, 0), min(mid + max_chars // 2, len(text)))
            if pos > 0:
                mid = pos + 1
                break

        left = TextNode(text=text[:mid])
        right = TextNode(text=text[mid:])
        for child in (left, right):
            child.metadata = dict(node.metadata)
            child.excluded_embed_metadata_keys = list(node.excluded_embed_metadata_keys) if hasattr(node,
                                                                                                    "excluded_embed_metadata_keys") else []
            child.excluded_llm_metadata_keys = list(node.excluded_llm_metadata_keys) if hasattr(node,
                                                                                                "excluded_llm_metadata_keys") else []
        # 递归处理，确保最终所有 chunk 都在限制内
        result.extend(_force_split_oversized([left, right], max_chars))
    return result


def _pre_split_large_documents(documents: list[Document]) -> list[Document]:
    """管道前预拆分：将超大文档按字符数切分，防止 chunk 超过 Milvus VARCHAR 限制。"""
    result = []
    for doc in documents:
        text = doc.get_content()
        if len(text) <= _MAX_CHUNK_CHARS:
            result.append(doc)
            continue
        logger.warning(
            f"文档过大 ({len(text)} chars)，预拆分为 {_MAX_CHUNK_CHARS} 字符段"
        )
        for i in range(0, len(text), _MAX_CHUNK_CHARS):
            sub_doc = Document(
                text=text[i:i + _MAX_CHUNK_CHARS],
                metadata=dict(doc.metadata),
            )
            sub_doc.metadata["_pre_split_part"] = i // _MAX_CHUNK_CHARS
            result.append(sub_doc)
    return result


class TenantAwareIngestionPipeline:
    """带租户感知的文档入库管道，内置重复检测与 docstore 缓存。

    管道流程：
    1. 预处理：计算内容哈希 → 检查是否重复
    2. Document 输入 → SemanticSplitterNodeParser 语义分块
    3. 每个 Node 注入 metadata={"tenant_id": ..., "doc_id": ..., "content_hash": ...}
    4. 本地 HuggingFace Embedding 向量化
    5. 写入 Milvus 向量库（自动创建集合）
    6. 写入 docstore 缓存（避免重复处理）

    去重策略：
    - 内容哈希（SHA256）：相同内容 → 同一哈希值
    - 向量存储层：查询 Milvus 中是否存在相同 content_hash 的 chunk
    - docstore：LlamaIndex SimpleDocumentStore 持久化缓存文档哈希
    """

    def __init__(self):
        self._embed_model = None  # 延迟加载
        self._splitter: SentenceSplitter | SemanticSplitterNodeParser | None = None
        self._vector_store: MilvusVectorStore | None = None
        self._pipeline: IngestionPipeline | None = None
        self._docstore: SimpleDocumentStore | None = None

    def _get_embed_model(self):
        if self._embed_model is None:
            self._embed_model = get_shared_embed_model()
        return self._embed_model

    def _get_splitter(self):
        if self._splitter is None:
            emb = self._get_embed_model()
            self._splitter = SemanticSplitterNodeParser(
                embed_model=emb,
                breakpoint_percentile_threshold=rag_params.semantic_breakpoint_percentile,
                buffer_size=rag_params.semantic_buffer_size,
            )
            logger.info("使用 SemanticSplitterNodeParser（语义分块）")
        return self._splitter

    def _get_docstore(self) -> SimpleDocumentStore:
        """懒初始化持久化 docstore，用于缓存已处理文档的哈希。"""
        if self._docstore is None:
            _DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if _DOCSTORE_PATH.exists():
                self._docstore = SimpleDocumentStore.from_persist_path(str(_DOCSTORE_PATH))
            else:
                self._docstore = SimpleDocumentStore()
            self._docstore.persist(str(_DOCSTORE_PATH))
        return self._docstore

    def _get_pipeline(self) -> IngestionPipeline:
        """懒初始化向量库与 IngestionPipeline（首次调用时触发 Milvus 连接）。"""
        if self._pipeline is None:
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
                collection_name=settings.milvus_collection_name,
                dim=settings.embedding_dim,
                embedding_field="embedding",
            )
            self._pipeline = IngestionPipeline(
                transformations=[self._get_splitter(), self._get_embed_model()],
                vector_store=self._vector_store,
                docstore=self._get_docstore(),
                docstore_strategy=DocstoreStrategy.UPSERTS,
            )
        return self._pipeline

    def _inject_metadata(self, nodes: list[BaseNode], tenant_id: str, doc_id: str, content_hash: str) -> list[BaseNode]:
        """向所有 Node 强制注入租户、文档标识和内容哈希元数据。

        入库前清除 SemanticSplitter 产生的冗余元数据（_node_content、
        relationships 等），仅保留 RBAC 必需的字段，防止序列化后超过
        Milvus VARCHAR(65535) 限制。
        """
        for node in nodes:
            # 统一 doc_id：MilvusVectorStore 通过 relationships 取 ref_doc_id
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
                node_id=doc_id, metadata={"tenant_id": tenant_id}
            )
            # 仅保留多租户隔离必需的字段
            node.metadata = {
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "content_hash": content_hash,
            }
            # 禁止 LlamaIndex 自动序列化内部字段
            node.excluded_embed_metadata_keys = ["_node_content", "_node_type", "document_id", "ref_doc_id",
                                                 "relationships"]
            node.excluded_llm_metadata_keys = ["_node_content", "_node_type", "document_id", "ref_doc_id",
                                               "relationships"]
        return nodes

    def check_duplicate(self, tenant_id: str, content_hash: str, text: str = "") -> dict | None:
        """检查同一租户下是否已存在相同/相似内容的文档。

        先 SHA256 精确匹配（快速路径），再近重复检测（语义相似度）。
        查询 Milvus 集合中匹配 tenant_id + content_hash 的 chunk，
        若存在则返回已入库文档的信息，否则返回 None。
        使用 MilvusClient API 与 LlamaIndex 共用同一连接。
        """
        # 精确匹配（Milvus metadata 字段）
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
            )

            collection_name = settings.milvus_collection_name
            if client.has_collection(collection_name):
                from src.utils.helpers import _escape_milvus_expr

                safe_tenant = _escape_milvus_expr(tenant_id)
                safe_hash = _escape_milvus_expr(content_hash)
                filter_expr = f'tenant_id == "{safe_tenant}" && content_hash == "{safe_hash}"'
                results = client.query(
                    collection_name=collection_name,
                    filter=filter_expr,
                    output_fields=["doc_id"],
                    limit=1,
                )

                if results:
                    existing_doc_id = results[0].get("doc_id", "unknown")
                    safe_doc_id = _escape_milvus_expr(existing_doc_id)
                    count_expr = f'tenant_id == "{safe_tenant}" && doc_id == "{safe_doc_id}"'
                    all_chunks = client.query(
                        collection_name=collection_name,
                        filter=count_expr,
                        output_fields=["doc_id"],
                        limit=10000,
                    )
                    return {
                        "doc_id": existing_doc_id,
                        "chunk_count": len(all_chunks),
                        "content_hash": content_hash,
                        "match_type": "exact",
                    }
        except Exception as e:
            logger.warning(f"精确重复检查失败（降级放行）: {e}")

        # 近重复检测（embedding 余弦相似度）
        if text:
            try:
                from src.pipeline.near_dedup import detect_near_duplicate

                emb_model = self._get_embed_model()
                result = detect_near_duplicate(tenant_id, text, emb_model)
                if result:
                    return {
                        "doc_id": result["existing_doc_id"],
                        "chunk_count": 0,
                        "content_hash": content_hash,
                        "match_type": result["match_type"],
                        "similarity": result["similarity"],
                    }
            except Exception as e:
                logger.warning(f"近重复检测失败（降级放行）: {e}")

        return None

    async def _embed_and_split(self, text: str, tenant_id: str, doc_id: str, content_hash: str) -> list[BaseNode]:
        """Step 1+2: 预拆分 → 分块 → 注入元数据（不含嵌入）。"""
        doc = Document(text=text, metadata={"tenant_id": tenant_id, "doc_id": doc_id})
        documents = _pre_split_large_documents([doc])
        splitter = self._get_splitter()
        nodes = splitter.get_nodes_from_documents(documents)
        nodes = _force_split_oversized(nodes)
        nodes = self._inject_metadata(nodes, tenant_id, doc_id, content_hash)
        logger.info(f"分块完成: {len(nodes)} chunks")
        return nodes

    async def _do_embed(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Step 3: 远程嵌入。"""
        if not nodes:
            return nodes
        embed_model = self._get_embed_model()
        texts = [n.get_content() for n in nodes]
        embeddings = await embed_model.aget_text_embedding_batch(texts, show_progress=False)
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb
        logger.info(f"远程嵌入完成: {len(nodes)} vectors")
        return nodes

    async def _do_milvus_insert(self, nodes: list[BaseNode]) -> list[BaseNode]:
        """Step 4: 写入 Milvus。"""
        if not nodes:
            return nodes
        if self._vector_store is None:
            self._vector_store = MilvusVectorStore(
                uri=settings.milvus_uri,
                token=settings.milvus_token or None,
                collection_name=settings.milvus_collection_name,
                dim=settings.embedding_dim,
                embedding_field="embedding",
            )
        self._vector_store.add(nodes)
        logger.info(f"Milvus 入库完成: {len(nodes)} rows")
        return nodes

    async def ingest(self, documents: list[Document], tenant_id: str, doc_id: str, content_hash: str = "") -> list[
        BaseNode]:
        """完整入库管道（兼容旧调用）。"""
        documents = _pre_split_large_documents(documents)
        docstore = self._get_docstore()
        splitter = self._get_splitter()
        nodes = splitter.get_nodes_from_documents(documents)
        nodes = _force_split_oversized(nodes)
        nodes = self._inject_metadata(nodes, tenant_id, doc_id, content_hash)
        nodes = await self._do_embed(nodes)
        nodes = await self._do_milvus_insert(nodes)
        if content_hash:
            try:
                from src.storage.pg_store import set_doc_hash as pg_set_doc_hash
                pg_set_doc_hash(doc_id, content_hash)
            except Exception:
                pass
        loop = __import__("asyncio").get_event_loop()
        loop.run_in_executor(None, _index_es_background, tenant_id, doc_id, nodes)
        return nodes

    async def ingest_text(self, text: str, tenant_id: str, doc_id: str) -> list[BaseNode]:
        """便捷方法：从纯文本创建 Document 后走完整入库管道。

        Args:
            text: 纯文本内容
            tenant_id: 租户标识
            doc_id: 文档标识

        Returns:
            完成处理的 Node 列表
        """
        content_hash = _compute_content_hash(text)
        doc = Document(
            text=text,
            metadata={
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "content_hash": content_hash,
            },
        )
        nodes = await self.ingest([doc], tenant_id, doc_id, content_hash)

        # 注册文档语义指纹到近重复索引
        try:
            from src.pipeline.near_dedup import register_doc_embedding

            embed_model = self._get_embed_model()
            # 取前几条 chunk 的 embedding 均值作为文档级语义指纹
            sample_nodes = nodes[:3] if len(nodes) > 3 else nodes
            if sample_nodes and sample_nodes[0].embedding is not None:
                dim = len(sample_nodes[0].embedding)
                avg_embedding = [0.0] * dim
                for n in sample_nodes:
                    if n.embedding:
                        for i, v in enumerate(n.embedding):
                            avg_embedding[i] += v
                for i in range(dim):
                    avg_embedding[i] /= len(sample_nodes)
                register_doc_embedding(tenant_id, doc_id, content_hash, avg_embedding)
        except Exception as e:
            logger.warning(f"语义指纹注册失败（非阻塞）: {e}")

        return nodes
