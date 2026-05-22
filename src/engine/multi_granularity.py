"""
多粒度索引检索 — 参考字节跳动 RAG 实践 §3.3「多粒度索引」

三级检索漏斗：
  L1 文档级粗筛 → 按 doc_id metadata 快速过滤不相关文档
  L2 段落级中筛 → 现有 Milvus chunk 级语义匹配
  L3 句子级精筛 → 拆句并二次精排，最小化生成层噪声

收益：
- 减少进入 rerank 的候选量（L1 粗筛）
- 提高上下文信息密度（L3 句子级精排）
- token 成本更可控
"""
from __future__ import annotations

import re

from llama_index.core.schema import NodeWithScore
from loguru import logger

from src.config.rag_params import rag_params


class MultiGranularityRetriever:
    """多粒度检索引擎：三层漏斗式检索。

    设计思路（参考字节 §3.3）：
    - 文档级：利用元数据过滤，只检索特定文档范围内的 chunk
    - 段落级：标准 chunk 级语义检索（由 TenantAwareQueryFusionRetriever 承担）
    - 句子级：对 Top-K chunk 拆句，计算 query-sentence 相关性，取 top 句

    句子级精排不依赖额外模型，使用 query term 命中 + 位置加权计分。
    """

    # 句子级精排参数
    MIN_SENTENCE_CHARS: int = 6
    POSITION_BONUS_DECAY: float = 0.05  # 靠前句子的位置加分衰减率

    def filter_by_doc_ids(
            self, nodes: list[NodeWithScore], doc_ids: set[str]
    ) -> list[NodeWithScore]:
        """L1 文档级粗筛：仅保留属于指定 doc_id 的 chunk。

        Args:
            nodes: 待筛选的节点列表
            doc_ids: 允许通过的 doc_id 集合

        Returns:
            过滤后的节点列表
        """
        if not doc_ids:
            return nodes
        result = [
            n for n in nodes
            if n.node.metadata.get("doc_id", "") in doc_ids
        ]
        logger.info(f"L1 文档级粗筛: {len(nodes)} → {len(result)} (doc_ids={len(doc_ids)})")
        return result

    def sentence_level_rerank(
            self,
            nodes: list[NodeWithScore],
            query: str,
            top_n: int | None = None,
            max_chars_per_doc: int | None = None,
    ) -> list[NodeWithScore]:
        """L3 句子级精筛：将 chunk 拆成句子，取与 query 最相关的 top 句。

        对每个 chunk：
        1. 按中英文标点拆句
        2. 对每个句子计算 query term 命中分 + 位置加权
        3. 选取得分最高的 top-k 句重新拼接
        4. 如果原 chunk 很短（≤ max_chars），保留原样

        Args:
            nodes: 已检索并 rerank 的 chunk 列表
            query: 用户查询
            top_n: 每个 chunk 最多保留的句子数（默认 5）
            max_chars_per_doc: 每个 chunk 最终最大字符数

        Returns:
            句子级精排后的节点列表
        """
        if top_n is None:
            top_n = rag_params.rerank_top_n
        if max_chars_per_doc is None:
            max_chars_per_doc = rag_params.context_max_chars_per_doc

        if not nodes:
            return nodes

        terms = self._extract_terms(query)
        if not terms:
            return nodes

        refined: list[NodeWithScore] = []
        for node in nodes:
            text = node.node.get_content()
            if len(text) <= max_chars_per_doc:
                refined.append(node)
                continue

            sentences = self._split_sentences(text)
            if len(sentences) <= 1:
                refined.append(node)
                continue

            scored = self._score_sentences(sentences, terms)
            scored.sort(key=lambda x: -x[0])

            selected: list[str] = []
            total_chars = 0
            for score, sentence in scored:
                if total_chars + len(sentence) > max_chars_per_doc and selected:
                    break
                selected.append(sentence)
                total_chars += len(sentence)

            if not selected:
                refined.append(node)
                continue

            from llama_index.core.schema import TextNode

            compressed = "\n".join(selected)
            new_node = TextNode(
                text=compressed,
                id_=node.node.node_id,
                metadata=dict(node.node.metadata or {}),
            )
            new_node.metadata["_granularity"] = "sentence"
            new_node.metadata["_sentence_count"] = len(selected)
            refined.append(NodeWithScore(node=new_node, score=node.score))

        logger.info(
            f"L3 句子级精筛完成: {len(nodes)} chunks → {len(refined)} 节点 "
            f"(query_terms={len(terms)})"
        )
        return refined

    # ── internal helpers ───────────────────────────────────────

    @staticmethod
    def _extract_terms(query: str) -> set[str]:
        """从 query 提取有效搜索词。"""
        terms: set[str] = set()
        for token in re.findall(r"[a-zA-Z0-9_]{2,}", query.lower()):
            terms.add(token)
        for chunk in re.findall(r"[一-鿿]{1,6}", query):
            if len(chunk) <= 6:
                terms.add(chunk)
            for i in range(max(0, len(chunk) - 1)):
                terms.add(chunk[i:i + 2])
        return {t for t in terms if len(t) > 1}

    @classmethod
    def _split_sentences(cls, text: str) -> list[str]:
        """按中英文标点拆句，过滤过短片段。"""
        parts = re.findall(r"[^。！？；.!?;\n]+[。！？；.!?;]?", text)
        return [
            p.strip()
            for p in parts
            if len(p.strip()) >= cls.MIN_SENTENCE_CHARS
        ]

    @classmethod
    def _score_sentences(
            cls, sentences: list[str], terms: set[str]
    ) -> list[tuple[float, str]]:
        """对句子按 term 命中分 + 位置加权打分。"""
        total = len(sentences)
        scored: list[tuple[float, str]] = []
        for idx, sentence in enumerate(sentences):
            lowered = sentence.lower()
            term_score = sum(
                1.0 + (0.5 if len(term) >= 3 else 0)
                for term in terms
                if term.lower() in lowered
            )
            position_bonus = 1.0 / (1.0 + idx * cls.POSITION_BONUS_DECAY)
            final_score = term_score * position_bonus
            if term_score > 0:
                scored.append((final_score, sentence))
        return scored


shared_multi_granularity = MultiGranularityRetriever()
