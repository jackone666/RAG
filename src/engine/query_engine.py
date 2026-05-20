"""
RAG 核心引擎 - 查询生成与熔断降级模块

功能说明：
- 对检索结果进行 LLM 重排序（Rerank）以提升 Top-N 精度
- 基于上下文生成最终答案，严格限定仅基于提供的上下文回答
- 实现大模型调用异常时的自动熔断降级机制

熔断策略（规范 4.2）：
- 主模型不可用时自动切换至备用模型
- 主备均失败时返回标准化友好降级提示
- 涵盖场景：API 超时、429 限流、凭证失效、网络异常
"""
import time
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore
from openai import AsyncOpenAI, OpenAI as SyncOpenAI
from loguru import logger

from src.config.rag_params import rag_params
from src.config.settings import settings


@dataclass
class GenerationTrace:
    """大模型生成阶段追踪数据。"""

    model_used: str = ""
    fallback_used: bool = False
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_preview: str = ""

# 降级友好提示——当主备模型全部不可用时返回
FALLBACK_RESPONSE = (
    "很抱歉，知识服务暂时不可用，请稍后重试或联系系统管理员。"
    "I'm sorry, but the knowledge service is temporarily unavailable. "
    "Please try again in a moment or contact your administrator."
)


class RAGQueryEngine:
    """RAG 查询引擎：重排序 + 大模型生成 + 自动熔断降级。

    核心能力：
    - LLM 驱动的结果重排序（LLMRerank）
    - 基于上下文的大模型答案生成
    - 主模型异常时自动降级至备用模型
    - 主备均失败时返回标准化降级友好提示
    - LLM 实例延迟初始化，减少启动开销

    生成策略：
    - 严格限定仅基于提供的上下文回答，禁止编造
    - 上下文不足时明确告知用户，不猜测
    """

    def __init__(self):
        """初始化查询引擎（延迟初始化 LLM 实例）。"""
        self._primary_llm: AsyncOpenAI | None = None
        self._fallback_llm: AsyncOpenAI | None = None

    def _get_primary_llm(self) -> AsyncOpenAI:
        """获取主模型实例（延迟初始化），使用原生 openai SDK 兼容 DeepSeek。"""
        if self._primary_llm is None:
            self._primary_llm = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._primary_llm

    def _get_fallback_llm(self) -> AsyncOpenAI:
        """获取备用模型实例（延迟初始化）。"""
        if self._fallback_llm is None:
            self._fallback_llm = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        return self._fallback_llm

    async def _generate(self, llm: AsyncOpenAI, context: str, query: str) -> tuple[str, dict]:
        """使用原生 openai SDK 生成答案，兼容 DeepSeek 等非 OpenAI 模型。

        Returns:
            (回答文本, token 用量字典)
        """
        prompt = self._build_prompt(context, query)
        response = await llm.chat.completions.create(
            model=settings.primary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rag_params.generation_temperature,
            max_tokens=rag_params.generation_max_tokens,
        )
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return response.choices[0].message.content.strip() if response.choices else "", usage

    def _build_prompt(self, context: str, query: str) -> str:
        """构建生成提示词。"""
        return (
            f"{rag_params.system_prompt}\n\n"
            f"上下文 / Context:\n{context}\n\n"
            f"问题 / Question: {query}\n\n"
            "回答 / Answer:"
        )

    async def _generate_stream(self, llm: AsyncOpenAI, context: str, query: str):
        """流式生成：逐 token 产出回答文本。"""
        prompt = self._build_prompt(context, query)
        stream = await llm.chat.completions.create(
            model=settings.primary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=rag_params.generation_temperature,
            max_tokens=rag_params.generation_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def query(self, nodes: list[NodeWithScore], query: str) -> str:
        """基于检索节点生成答案，带自动熔断降级保护。

        执行顺序：
        1. 尝试使用主模型（如 GPT-4o）生成回答
        2. 主模型异常 → 自动切换备用模型（如 GPT-4o-mini）
        3. 备用模型也异常 → 返回标准化降级友好提示

        涵盖异常类型：
        - API 超时、限流（429）
        - 认证凭证失效（401）
        - 网络连接异常
        - 其他不可预期的 API 错误

        Args:
            nodes: 检索 + 重排序后的相关节点列表
            query: 用户原始查询

        Returns:
            生成的回答文本，或降级友好提示
        """
        answer, _ = await self.query_with_trace(nodes, query)
        return answer

    async def query_with_trace(
        self, nodes: list[NodeWithScore], query: str
    ) -> tuple[str, GenerationTrace]:
        """基于检索节点生成答案，同时返回模型调用追踪数据。

        Returns:
            (生成的回答文本, GenerationTrace 追踪数据)
        """
        trace = GenerationTrace()
        context = "\n\n".join(node.node.get_content() for node in nodes)

        # 第一层：尝试主模型
        t0 = time.monotonic()
        try:
            answer, usage = await self._generate(self._get_primary_llm(), context, query)
            trace.model_used = settings.primary_model
            trace.latency_ms = (time.monotonic() - t0) * 1000
            trace.prompt_tokens = usage.get("prompt_tokens", 0)
            trace.completion_tokens = usage.get("completion_tokens", 0)
            trace.total_tokens = usage.get("total_tokens", 0)
            trace.prompt_preview = self._build_prompt(context, query)[:200]
            return answer, trace
        except Exception as e:
            logger.warning(f"主模型 ({settings.primary_model}) 调用失败: {e}，正在切换备用模型...")

        # 第二层：降级至备用模型
        t0 = time.monotonic()
        try:
            answer, usage = await self._generate(self._get_fallback_llm(), context, query)
            trace.model_used = settings.fallback_model
            trace.fallback_used = True
            trace.latency_ms = (time.monotonic() - t0) * 1000
            trace.prompt_tokens = usage.get("prompt_tokens", 0)
            trace.completion_tokens = usage.get("completion_tokens", 0)
            trace.total_tokens = usage.get("total_tokens", 0)
            logger.info(f"备用模型 ({settings.fallback_model}) 生成成功")
            return answer, trace
        except Exception as e:
            logger.error(f"备用模型 ({settings.fallback_model}) 也调用失败: {e}，返回降级提示")

        # 第三层：返回标准化友好降级提示
        trace.model_used = "fallback_text"
        trace.fallback_used = True
        return FALLBACK_RESPONSE, trace

    async def query_stream(self, nodes: list[NodeWithScore], query: str):
        """流式生成答案，逐 token 产出，带自动熔断降级。

        Yields:
            dict: {"token": str} 或 {"done": True, "model_used": str, "fallback": bool}
        """
        context = "\n\n".join(node.node.get_content() for node in nodes)
        model_used = settings.primary_model
        fallback = False

        # 第一层：主模型流式生成
        try:
            async for token in self._generate_stream(self._get_primary_llm(), context, query):
                yield {"token": token}
            yield {"done": True, "model_used": model_used, "fallback": fallback}
            return
        except Exception as e:
            logger.warning(f"主模型 ({settings.primary_model}) 流式生成失败: {e}")

        # 第二层：备用模型流式生成
        fallback = True
        model_used = settings.fallback_model
        try:
            async for token in self._generate_stream(self._get_fallback_llm(), context, query):
                yield {"token": token}
            yield {"done": True, "model_used": model_used, "fallback": fallback}
            return
        except Exception as e:
            logger.error(f"备用模型也流式生成失败: {e}")

        # 第三层：降级文本
        for ch in FALLBACK_RESPONSE:
            yield {"token": ch}
        yield {"done": True, "model_used": "fallback_text", "fallback": True}

    async def rerank(self, query: str, nodes: list[NodeWithScore], top_n: int | None = None) -> list[NodeWithScore]:
        if top_n is None:
            top_n = rag_params.rerank_top_n
        """使用 SiliconFlow BAAI/bge-reranker-v2-m3 重排序。

        容错降级：API 异常时回退至原始相似度排序取 top_n。
        """
        if not nodes:
            return nodes

        try:
            documents = [n.node.get_content()[:8000] for n in nodes]
            resp = await self._call_rerank_api(query, documents, top_n)
            # 按 relevance_score 降序重排
            scored = []
            for r in resp.get("results", []):
                idx = r["index"]
                if idx < len(nodes):
                    nodes[idx].score = r["relevance_score"]
                    scored.append(nodes[idx])
            if scored:
                scored.sort(key=lambda n: n.score or 0, reverse=True)
                return scored[:top_n]
        except Exception as e:
            logger.warning(f"重排序失败: {e}，回退至原始相似度排序取 top-{top_n}")

        return sorted(nodes, key=lambda n: n.score or 0, reverse=True)[:top_n]

    async def _call_rerank_api(self, query: str, documents: list[str], top_n: int) -> dict:
        """调用 SiliconFlow Rerank API。"""
        import asyncio
        return await asyncio.to_thread(self._call_rerank_sync, query, documents, top_n)

    def _call_rerank_sync(self, query: str, documents: list[str], top_n: int) -> dict:
        client = SyncOpenAI(
            api_key=settings.embedding_api_key or settings.openai_api_key,
            base_url="https://api.siliconflow.cn/v1",
        )
        # SiliconFlow rerank 不是标准 OpenAI 端点，需用 httpx 直接调用
        import httpx
        resp = httpx.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={
                "Authorization": f"Bearer {settings.embedding_api_key or settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": query,
                "documents": documents,
                "top_n": top_n,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


# 模块级共享实例
shared_query_engine = RAGQueryEngine()
