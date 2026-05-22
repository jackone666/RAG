"""
查询改写与增强模块 — 参考字节跳动 RAG 实践 §5.1「检索触发与查询理解」

字节实践要点：
- 查询分解：将复杂多跳问题拆分为原子子查询，分别检索后合并
- 查询扩展：生成同义/近义表述，扩大检索覆盖面
- HyDE（假设文档嵌入）：先生成假设性答案再以其向量检索，提升语义对齐

本模块使用原生 openai SDK（非 LlamaIndex 封装），以兼容 DeepSeek 等
非 OpenAI 模型端点。
"""
from loguru import logger
from openai import AsyncOpenAI

from src.config.rag_params import rag_params
from src.config.settings import settings
from src.utils.terminology import normalize_query_terms


class QueryRewriter:
    """查询改写器：分解复杂查询、生成多路改写、提升检索召回率。

    设计原则（参考字节 §5.1.2）：
    - 改写不改变原意，只做展开和分解
    - 子查询之间独立检索，避免级联误差
    - 所有改写均异步非阻塞，失败回退至原始查询
    """

    def __init__(self):
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._client

    async def rewrite(self, query: str) -> list[str]:
        """将原始查询改写为一组增强查询（含原始查询）。

        返回至少包含原始查询的列表，失败时仅返回原始查询。
        """
        normalized = normalize_query_terms(query)
        queries = [query]  # 始终保留原始查询
        if normalized != query:
            queries.append(normalized)

        try:
            expanded = await self._expand(query)
            queries.extend(expanded)
        except Exception as e:
            logger.warning(f"查询扩展失败，回退至原始查询: {e}")

        if rag_params.query_decomposition_enabled:
            queries.extend(self._rule_decompose(query))

        if rag_params.hyde_enabled:
            try:
                hyde = await self._hyde(query)
                if hyde:
                    queries.append(hyde)
            except Exception as e:
                logger.warning(f"HyDE 生成失败，跳过假设文档检索: {e}")

        # 去重但保持顺序
        seen = set()
        unique = []
        for q in queries:
            q_stripped = q.strip()
            if q_stripped and q_stripped not in seen:
                seen.add(q_stripped)
                unique.append(q_stripped)

        logger.info(f"查询改写: '{query[:50]}...' → {len(unique)} 个变体")
        return unique[:rag_params.query_rewrite_max_variants]

    def classify(self, query: str) -> str:
        """轻量意图分类：semantic / keyword / mixed。"""
        import re

        q = query.strip()
        has_code = bool(re.search(r"[A-Z]{2,}[-_]?\d+|\d{3,}|v\d+(\.\d+)+", q))
        has_explain = any(w in q for w in ["为什么", "如何", "原理", "影响", "区别", "总结", "分析"])
        has_exact = any(w in q for w in ["编号", "金额", "日期", "版本", "条款", "错误码", "ID", "PE-TTM", "ROE"])
        if has_code or (has_exact and not has_explain):
            return "keyword"
        if has_explain and not has_exact:
            return "semantic"
        return "mixed"

    def _rule_decompose(self, query: str) -> list[str]:
        """规则级 query decomposition，失败成本低，作为 LLM 改写的补充。"""
        import re

        parts = re.split(r"[？?；;。]|\s+和\s+|\s+以及\s+|\s+并且\s+", query)
        return [p.strip() for p in parts if 4 <= len(p.strip()) < len(query)]

    async def _expand(self, query: str) -> list[str]:
        """使用 LLM 生成查询的语义等价变体。"""
        client = self._get_client()
        prompt = (
            "你是一个查询改写助手。请为以下问题生成 2-3 个语义等价的改写版本，"
            "用于改进检索召回率。\n"
            "规则：\n"
            "1. 保持原意不变\n"
            "2. 分解复合问题为原子问题\n"
            "3. 使用不同的表达方式\n"
            "4. 每行一个改写，不要编号\n\n"
            f"原始问题：{query}\n\n改写："
        )
        response = await client.chat.completions.create(
            model=settings.fallback_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rag_params.rewrite_temperature,
            max_tokens=rag_params.rewrite_max_tokens,
        )
        text = response.choices[0].message.content or ""
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        # 过滤掉可能的编号前缀
        cleaned = []
        for line in lines:
            for prefix in ["- ", "• ", "1. ", "2. ", "3. ", "4. ", "5. "]:
                if line.startswith(prefix):
                    line = line[len(prefix):]
            cleaned.append(line.strip())
        return [l for l in cleaned if l and len(l) > 3]

    async def _hyde(self, query: str) -> str:
        """生成 HyDE 假设性答案，用于语义检索对齐。"""
        client = self._get_client()
        prompt = (
            "请基于问题写一段可能出现在企业知识库中的假设性答案，"
            "只写事实风格段落，不要编造具体数字；用于向量检索，不是最终回答。\n\n"
            f"问题：{query}\n\n假设性文档："
        )
        response = await client.chat.completions.create(
            model=settings.fallback_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=rag_params.rewrite_max_tokens,
        )
        return (response.choices[0].message.content or "").strip()


# 模块级单例
shared_rewriter = QueryRewriter()
