"""
IntelliLens-MCP 企业级智能数据治理与 Agentic RAG 系统 — 主入口

本文件是 FastAPI 应用的主入口，负责：
1. 挂载所有 HTTP 中间件（鉴权、限流、可观测性）
2. 串联完整的 RAG 请求生命周期
3. 挂载 MCP 协议子应用供外部大模型客户端连接
4. 启动异步评估裁判的任务投递

完整请求闭环流程：
  接收问题 → 鉴权 → 限流 → 混合检索(带 RBAC) → LLM 重排
  → 大模型生成(带降级熔断) → 异步评估打分 → 坏例沉淀
"""
import json
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config.rag_params import rag_params
from src.config.settings import settings
from src.engine.query_engine import FALLBACK_RESPONSE, GenerationTrace, shared_query_engine
from src.engine.retrievers import RetrievalTrace, shared_retriever
from src.evaluation.evaluator import evaluator as async_evaluator
from src.middleware.auth import extract_tenant_context
from src.middleware.rate_limiter import rate_limit_dependency
from loguru import logger

from src.observability.tracer import init_observability, trace_rag_query, trace_span, get_langfuse_client

# 模块级单例引擎实例，所有请求复用
retriever = shared_retriever
query_engine = shared_query_engine

from src.engine.query_rewriter import shared_rewriter  # noqa: E402

# 缓存层 — 基于 cachetools.TTLCache（Caffeine 风格 LRU + TTL 双策略）
from src.utils.cache import retrieval_cache, eval_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化可观测性与数据存储。

    在应用启动阶段（所有请求处理之前），完成 Langfuse 追踪器的初始化。
    关闭阶段暂无需特殊清理操作。
    """
    init_observability()
    try:
        from src.storage.pg_store import init_db
        init_db()
    except Exception:
        pass
    yield


# ==================== FastAPI 应用实例 ====================
app = FastAPI(
    title="IntelliLens-MCP",
    version="2.1.0",
    lifespan=lifespan,
    description="企业级智能数据治理与 Agentic RAG 系统",
)


# ==================== HTTP 中间件 ====================


@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    """租户上下文注入中间件——在每个请求处理前提取并注入租户身份。

    执行时机：所有路由处理之前。
    功能：
    1. 从 JWT Bearer Token 或 X-* Header 提取 tenant_id、role
    2. 注入到 request.state.tenant_context 供下游模块使用
    3. 后续中间件（如限流）和路由处理器均从此处读取租户信息
    """
    ctx = await extract_tenant_context(request)
    request.state.tenant_context = ctx
    response = await call_next(request)
    return response


# ==================== 请求/响应模型 ====================


class QueryRequest(BaseModel):
    """RAG 查询请求体。"""

    query: str  # 用户自然语言查询


class StepDetail(BaseModel):
    """管线步骤详情。"""

    name: str  # 步骤名称
    count: int = 0  # 节点数量
    latency_ms: float = 0.0  # 耗时（毫秒）
    details: dict = {}  # 额外细节


class PipelineTrace(BaseModel):
    """RAG 管线全链路追踪。"""

    vector_search: StepDetail
    keyword_search: StepDetail
    fusion: StepDetail
    rerank: StepDetail
    generation: StepDetail


class QueryResponse(BaseModel):
    """RAG 查询响应体——包含完整评估指标。"""

    answer: str  # 生成的回答文本
    tenant_id: str  # 当前请求的租户标识
    node_count: int  # 最终用于生成的上下文节点数
    query_id: str  # 本次查询唯一标识（用于轮询评估结果）
    pipeline_trace: PipelineTrace  # 全链路追踪详情
    # 评估指标（异步评估完成后填充）
    precision: float | None = None
    recall: float | None = None
    faithfulness: float | None = None
    relevance: float | None = None
    eval_passing: bool | None = None
    eval_overall: float | None = None


class EvalStatsResponse(BaseModel):
    """评估统计响应。"""

    total_queries: int
    bad_cases: int
    pass_rate: float
    avg_score: float
    avg_precision: float = 0.0
    avg_recall: float = 0.0
    avg_mrr: float = 0.0
    avg_hit_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_relevance: float = 0.0
    recent_bad_cases: list[dict] = []


# ==================== 核心 RAG 查询端点 ====================


def _cache_key(tenant_id: str, query: str) -> str:
    """生成检索缓存键。"""
    return f"{tenant_id}::{query.strip().lower()}"


async def _retrieve_with_rewrite_and_cache(
    query: str, tenant_id: str
) -> tuple[list, RetrievalTrace, int]:
    """执行检索：查询改写 → 多路检索去重 → 热点缓存。

    参考字节跳动 RAG 实践：
    - §5.1 查询改写提升召回率
    - §4.3.1 热点请求缓存降低延迟

    Returns:
        (节点列表, RetrievalTrace, 改写查询数)
    """
    import hashlib

    # 检查热点缓存
    ck = _cache_key(tenant_id, query)
    cache_hash = hashlib.md5(ck.encode()).hexdigest()
    nodes = retrieval_cache.get(cache_hash)
    if nodes is not None:
        logger.info(f"命中检索缓存: {query[:50]}...")
        trace = RetrievalTrace(
            vector_count=len(nodes),
            fusion_count=len(nodes),
            top_scores=[n.score or 0.0 for n in nodes[:5]],
        )
        return nodes, trace, 0

    # 查询改写
    rewritten = await shared_rewriter.rewrite(query)
    rewrite_count = len(rewritten) - 1

    if rewrite_count > 0:
        # 多路检索 + 去重合并
        all_nodes: dict[str, tuple] = {}  # node_id → (node, max_score)
        merged_trace = RetrievalTrace()

        for q in rewritten:
            nodes, trace = await retriever.aretrieve_with_trace(q, tenant_id)
            merged_trace.vector_count += trace.vector_count
            merged_trace.keyword_count += trace.keyword_count
            merged_trace.vector_latency_ms += trace.vector_latency_ms
            merged_trace.keyword_latency_ms += trace.keyword_latency_ms
            for n in nodes:
                nid = n.node.node_id or n.node.get_content()[:64]
                if nid not in all_nodes or (n.score or 0) > (all_nodes[nid][1] or 0):
                    all_nodes[nid] = (n, n.score)

        nodes = sorted(
            [v[0] for v in all_nodes.values()],
            key=lambda x: x.score or 0,
            reverse=True,
        )[:10]
        merged_trace.fusion_count = len(nodes)
        merged_trace.top_scores = [n.score or 0.0 for n in nodes[:5]]
        merged_trace.fusion_mode = "rewrite_fusion"
    else:
        nodes, merged_trace = await retriever.aretrieve_with_trace(query, tenant_id)

    retrieval_cache.set(cache_hash, nodes)

    return nodes, merged_trace, rewrite_count


@app.post(
    "/v1/query",
    response_model=QueryResponse,
    dependencies=[Depends(rate_limit_dependency)],
)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
):
    """企业 RAG 查询端点——完整的六阶段请求处理管道。

    请求生命周期（严格按顺序执行）：
    ┌─────────────┐
    │ 1. 鉴权      │ ← tenant_context_middleware 已完成
    ├─────────────┤
    │ 2. 限流      │ ← rate_limit_dependency (Depends)
    ├─────────────┤
    │ 3. 查询改写  │ ← QueryRewriter (多路语义扩展，参考字节 §5.1)
    ├─────────────┤
    │ 4. 混合检索  │ ← 多路检索去重 + 热点缓存 (参考字节 §4.3.1)
    ├─────────────┤
    │ 5. 重排+生成  │ ← RAGQueryEngine.rerank + query (含熔断降级)
    ├─────────────┤
    │ 6. 异步评估  │ ← BackgroundTasks → AsyncEvaluator.evaluate
    └─────────────┘
    """
    import uuid

    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]
    query_id = uuid.uuid4().hex[:12]

    # Langfuse trace
    lf = get_langfuse_client()
    langfuse_trace = lf.trace(name="RAG Query", input={"query": body.query, "tenant_id": tenant_id}) if lf else None

    # 阶段 3+4：查询改写 + 混合检索（含热点缓存）
    nodes, retrieval_trace, rewrite_count = await _retrieve_with_rewrite_and_cache(
        query=body.query, tenant_id=tenant_id
    )

    # Langfuse: retrieval span
    with trace_span(langfuse_trace, "retrieval", input_data={"query": body.query},
                    output_data={
                        "vector_count": retrieval_trace.vector_count,
                        "keyword_count": retrieval_trace.keyword_count,
                        "fusion_count": retrieval_trace.fusion_count,
                        "documents": retrieval_trace.doc_previews,
                    }):
        pass

    if not nodes:
        return QueryResponse(
            answer=f"在租户 '{tenant_id}' 的知识库中未找到相关文档。",
            tenant_id=tenant_id,
            node_count=0,
            query_id=query_id,
            pipeline_trace=PipelineTrace(
                vector_search=StepDetail(
                    name="向量检索 (Milvus)",
                    count=retrieval_trace.vector_count,
                    latency_ms=round(retrieval_trace.vector_latency_ms, 1),
                ),
                keyword_search=StepDetail(
                    name="关键词检索 (Elasticsearch)",
                    count=retrieval_trace.keyword_count,
                    latency_ms=round(retrieval_trace.keyword_latency_ms, 1),
                ),
                fusion=StepDetail(name="融合排序", count=0),
                rerank=StepDetail(name="LLM 重排序", count=0),
                generation=StepDetail(name="大模型生成", count=0),
            ),
            precision=None, recall=None, faithfulness=None, relevance=None,
            eval_passing=None, eval_overall=None,
        )

    # 阶段 5a：LLM 重排序
    t0 = time.monotonic()
    reranked = await query_engine.rerank(body.query, nodes)
    rerank_latency = (time.monotonic() - t0) * 1000

    # Langfuse: rerank span
    rerank_previews = [
        {"rank": i + 1, "score": round(n.score or 0.0, 4),
         "text": (n.node.get_content() or "")[:rag_params.doc_preview_chars].replace("\n", " "),
         "doc_id": n.node.metadata.get("doc_id", "")}
        for i, n in enumerate(reranked[:10])
    ]
    with trace_span(langfuse_trace, "rerank", input_data={"query": body.query, "candidates": len(nodes)},
                    output_data={"top_n": len(reranked), "latency_ms": round(rerank_latency, 1),
                                 "documents": rerank_previews}):
        pass

    # 阶段 5b：大模型生成（含主/备模型自动降级熔断）— 带追踪
    answer, gen_trace = await query_engine.query_with_trace(reranked, body.query)

    # Langfuse: generation span
    with trace_span(langfuse_trace, "generation", input_data={"context_chunks": len(reranked)},
                    output_data={"model": gen_trace.model_used, "fallback": gen_trace.fallback_used,
                                 "latency_ms": round(gen_trace.latency_ms, 1),
                                 "completion_tokens": gen_trace.completion_tokens,
                                 "answer_preview": answer[:500]}):
        pass

    # Langfuse: finalize trace
    if langfuse_trace:
        langfuse_trace.update(output={"answer": answer[:500], "status": "completed"})

    # 阶段 6：裁判评估（带超时，优先返回结果）
    import asyncio as _asyncio
    context_texts = [node.node.get_content() for node in reranked]
    eval_precision = eval_recall = eval_faithfulness = eval_relevance = None
    eval_passing = eval_overall = None
    try:
        await _asyncio.wait_for(
            _evaluate_and_cache(
                query_id=query_id, query=body.query,
                context_nodes=context_texts, answer=answer, tenant_id=tenant_id,
            ),
            timeout=rag_params.eval_timeout_seconds,
        )
        if query_id in eval_cache:
            ev = eval_cache.get(query_id)
            eval_precision = ev.get("precision")
            eval_recall = ev.get("recall")
            eval_faithfulness = ev.get("faithfulness")
            eval_relevance = ev.get("relevance")
            eval_passing = ev.get("eval_passing")
            eval_overall = ev.get("overall")
    except _asyncio.TimeoutError:
        background_tasks.add_task(
            _evaluate_and_cache,
            query_id=query_id, query=body.query,
            context_nodes=context_texts, answer=answer, tenant_id=tenant_id,
        )

    trace = PipelineTrace(
        vector_search=StepDetail(
            name="向量检索 (Milvus)",
            count=retrieval_trace.vector_count,
            latency_ms=round(retrieval_trace.vector_latency_ms, 1),
            details={
                "top_scores": [round(s, 4) for s in retrieval_trace.top_scores[:5]],
                "rewrite_queries": rewrite_count,
            },
        ),
        keyword_search=StepDetail(
            name="关键词检索 (Elasticsearch)",
            count=retrieval_trace.keyword_count,
            latency_ms=round(retrieval_trace.keyword_latency_ms, 1),
        ),
        fusion=StepDetail(
            name="融合排序 (RELATIVE_SCORE)",
            count=retrieval_trace.fusion_count,
            details={"mode": retrieval_trace.fusion_mode},
        ),
        rerank=StepDetail(
            name="LLM 重排序",
            count=len(reranked),
            latency_ms=round(rerank_latency, 1),
            details={"top_n": len(reranked), "model": settings.primary_model},
        ),
        generation=StepDetail(
            name="大模型生成",
            count=len(reranked),
            latency_ms=round(gen_trace.latency_ms, 1),
            details={
                "model": gen_trace.model_used,
                "fallback": gen_trace.fallback_used,
                "context_chunks": len(reranked),
                "prompt_tokens": gen_trace.prompt_tokens,
                "completion_tokens": gen_trace.completion_tokens,
                "total_tokens": gen_trace.total_tokens,
                "prompt_preview": gen_trace.prompt_preview,
            },
        ),
    )

    return QueryResponse(
        answer=answer,
        tenant_id=tenant_id,
        node_count=len(reranked),
        query_id=query_id,
        pipeline_trace=trace,
        precision=eval_precision,
        recall=eval_recall,
        faithfulness=eval_faithfulness,
        relevance=eval_relevance,
        eval_passing=eval_passing,
        eval_overall=eval_overall,
    )


async def _evaluate_and_cache(
    query_id: str, query: str, context_nodes: list[str], answer: str, tenant_id: str
) -> None:
    """执行评估并将结果写入内存缓存，供前端轮询。"""
    try:
        result = await async_evaluator.evaluate(query, context_nodes, answer, tenant_id)
    except Exception as e:
        from loguru import logger
        logger.error(f"评估任务异常 (query_id={query_id}): {e}")
        eval_cache.set(query_id, {
            "query": query, "answer": answer[:500],
            "eval_score": None, "eval_passing": None,
            "error": str(e)[:200], "evaluated_at": time.time(),
        })
        return
    if result:
        eval_cache.set(query_id, {
            "query": query,
            "answer": answer[:500],
            "overall": result.get("overall", 0),
            "eval_score": result.get("overall", 0),
            "eval_passing": result.get("passing", False),
            "precision": result.get("precision", 0),
            "recall": result.get("recall", 0),
            "mrr": result.get("mrr", 0),
            "hit_rate": result.get("hit_rate", 0),
            "faithfulness": result.get("faithfulness", 0),
            "faithfulness_reason": result.get("faithfulness_reason", ""),
            "relevance": result.get("relevance", 0),
            "relevance_reason": result.get("relevance_reason", ""),
            "evaluated_at": time.time(),
        })
    else:
        eval_cache.set(query_id, {
            "query": query,
            "answer": answer[:500],
            "eval_score": None,
            "eval_passing": None,
            "evaluated_at": time.time(),
        })


# ==================== 流式 RAG 查询端点 (SSE) ====================


@app.post(
    "/v1/query/stream",
    dependencies=[Depends(rate_limit_dependency)],
)
async def query_stream_endpoint(
    request: Request,
    body: QueryRequest,
    background_tasks: BackgroundTasks,
):
    """流式 RAG 查询端点 —— Server-Sent Events 逐 token 推送回答。

    参考字节 RAG 实践（§6.4 生成效率与成本优化）：
    - 流式生成降低首 token 延迟，提升用户感知速度
    - 全链路追踪数据作为最终事件发送
    """
    import uuid
    from fastapi.responses import StreamingResponse

    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]
    query_id = uuid.uuid4().hex[:12]

    # Langfuse trace (手动管理，因流式响应无法用 context manager)
    lf = get_langfuse_client()
    lf_trace = lf.trace(name="RAG Query (stream)", input={"query": body.query, "tenant_id": tenant_id}) if lf else None

    # 阶段 3+4：查询改写 + 混合检索（含热点缓存）
    nodes, retrieval_trace, rewrite_count = await _retrieve_with_rewrite_and_cache(
        query=body.query, tenant_id=tenant_id
    )

    if lf_trace:
        with trace_span(lf_trace, "retrieval", input_data={"query": body.query},
                        output_data={
                            "vector_count": retrieval_trace.vector_count,
                            "keyword_count": retrieval_trace.keyword_count,
                            "fusion_count": retrieval_trace.fusion_count,
                            "documents": retrieval_trace.doc_previews,
                        }):
            pass

    if not nodes:
        if lf_trace:
            lf_trace.update(output={"status": "no_results"})
        async def empty_stream():
            yield f"data: {json.dumps({'answer': f'在租户 {tenant_id} 的知识库中未找到相关文档。', 'query_id': query_id, 'node_count': 0, 'done': True})}\n\n"
        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    # 阶段 4a：LLM 重排序
    t0 = time.monotonic()
    reranked = await query_engine.rerank(body.query, nodes)
    rerank_latency = (time.monotonic() - t0) * 1000

    if lf_trace:
        _rerank_previews = [
            {"rank": i + 1, "score": round(n.score or 0.0, 4),
             "text": (n.node.get_content() or "")[:rag_params.doc_preview_chars].replace("\n", " "),
             "doc_id": n.node.metadata.get("doc_id", "")}
            for i, n in enumerate(reranked[:10])
        ]
        with trace_span(lf_trace, "rerank", input_data={"query": body.query, "candidates": len(nodes)},
                        output_data={"top_n": len(reranked), "latency_ms": round(rerank_latency, 1),
                                     "documents": _rerank_previews}):
            pass

    # 评估延迟到流式结束后（获取完整 answer 后）

    async def event_stream():
        collected_tokens = []
        gen_model = ""
        gen_fallback = False
        gen_start = time.monotonic()

        # 先发送管线元数据
        meta = {
            "type": "meta",
            "query_id": query_id,
            "tenant_id": tenant_id,
            "rewrite_count": rewrite_count,
            "pipeline": {
                "vector_count": retrieval_trace.vector_count,
                "keyword_count": retrieval_trace.keyword_count,
                "fusion_count": retrieval_trace.fusion_count,
                "rerank_count": len(reranked),
                "rerank_latency_ms": round(rerank_latency, 1),
            },
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 流式生成
        async for event in query_engine.query_stream(reranked, body.query):
            if "token" in event:
                collected_tokens.append(event["token"])
                yield f"data: {json.dumps({'type': 'token', 'token': event['token']}, ensure_ascii=False)}\n\n"
            elif event.get("done"):
                gen_model = event.get("model_used", "")
                gen_fallback = event.get("fallback", False)

        gen_latency = (time.monotonic() - gen_start) * 1000

        # 发送最终管线追踪
        trace = {
            "type": "trace",
            "query_id": query_id,
            "node_count": len(reranked),
            "pipeline_trace": {
                "vector_search": {
                    "name": "向量检索 (Milvus)",
                    "count": retrieval_trace.vector_count,
                    "latency_ms": round(retrieval_trace.vector_latency_ms, 1),
                },
                "keyword_search": {
                    "name": "关键词检索 (Elasticsearch)",
                    "count": retrieval_trace.keyword_count,
                    "latency_ms": round(retrieval_trace.keyword_latency_ms, 1),
                },
                "fusion": {
                    "name": "融合排序 (RELATIVE_SCORE)",
                    "count": retrieval_trace.fusion_count,
                    "details": {"mode": retrieval_trace.fusion_mode},
                },
                "rerank": {
                    "name": "LLM 重排序",
                    "count": len(reranked),
                    "latency_ms": round(rerank_latency, 1),
                },
                "generation": {
                    "name": "大模型生成 (流式)",
                    "count": len(reranked),
                    "latency_ms": round(gen_latency, 1),
                    "details": {
                        "model": gen_model,
                        "fallback": gen_fallback,
                        "completion_tokens": len(collected_tokens),
                    },
                },
            },
        }
        yield f"data: {json.dumps(trace, ensure_ascii=False)}\n\n"

        # Langfuse: generation span + trace complete
        if lf_trace:
            with trace_span(lf_trace, "generation", input_data={"context_chunks": len(reranked)},
                            output_data={"model": gen_model, "fallback": gen_fallback,
                                         "latency_ms": round(gen_latency, 1),
                                         "completion_tokens": len(collected_tokens),
                                         "answer_preview": "".join(collected_tokens)[:500]}):
                pass
            lf_trace.update(output={"answer": "".join(collected_tokens)[:500], "status": "completed"})

        # 异步评估 + 等待结果（短超时）推送到 SSE
        full_answer = "".join(collected_tokens)
        context_texts = [node.node.get_content() for node in reranked]
        if full_answer.strip() and full_answer.strip() != FALLBACK_RESPONSE.strip():
            import asyncio as _asyncio
            eval_task = _asyncio.create_task(_evaluate_and_cache(
                query_id=query_id, query=body.query,
                context_nodes=context_texts, answer=full_answer, tenant_id=tenant_id,
            ))
            try:
                await _asyncio.wait_for(eval_task, timeout=12.0)
                if query_id in eval_cache and eval_cache.get(query_id).get("eval_score") is not None:
                    ev = eval_cache.get(query_id)
                    yield f"data: {json.dumps({'type': 'eval', 'precision': ev.get('precision'), 'recall': ev.get('recall'), 'faithfulness': ev.get('faithfulness'), 'relevance': ev.get('relevance'), 'overall': ev.get('overall'), 'passing': ev.get('eval_passing')}, ensure_ascii=False)}\n\n"
            except _asyncio.TimeoutError:
                pass

        yield "data: [DONE]\n\n"

        if (not full_answer.strip() or full_answer.strip() == FALLBACK_RESPONSE.strip()) and query_id in eval_cache:
            eval_cache.set(query_id, {
                "query": body.query,
                "answer": full_answer[:500],
                "eval_score": None, "eval_passing": None,
                "evaluated_at": time.time(),
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ==================== MCP 协议子应用挂载 ====================

# 导入 tools 模块触发 @mcp.tool() 装饰器注册
import src.mcp_server.tools  # noqa: E402
from src.mcp_server.server import mcp  # noqa: E402

# 注册文档管理 API 路由
from src.api.documents import router as documents_router  # noqa: E402

app.include_router(documents_router)

# 注册认证 API 路由
from src.api.auth import router as auth_router  # noqa: E402

app.include_router(auth_router)

# 将 FastMCP 的 HTTP 子应用挂载到 /mcp 路径
# 大模型客户端（如 Claude Desktop）通过 ws://host:port/mcp 连接
mcp_app = mcp.streamable_http_app()
app.mount("/mcp", mcp_app)

# 挂载前端静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== 评估结果查询端点 ====================


@app.get("/v1/evaluation/stats", response_model=EvalStatsResponse)
async def evaluation_stats():
    """返回全局评估统计数据（从 PostgreSQL 读取聚合）。"""
    from src.storage.pg_store import get_bad_case_stats

    data = get_bad_case_stats()
    return EvalStatsResponse(
        total_queries=data["total_queries"],
        bad_cases=data["bad_cases"],
        pass_rate=data["pass_rate"],
        avg_score=data["avg_score"],
        avg_precision=data.get("avg_precision", 0),
        avg_recall=data.get("avg_recall", 0),
        avg_mrr=data.get("avg_mrr", 0),
        avg_hit_rate=data.get("avg_hit_rate", 0),
        avg_faithfulness=data.get("avg_faithfulness", 0),
        avg_relevance=data.get("avg_relevance", 0),
        recent_bad_cases=data.get("recent_bad_cases", []),
    )


@app.get("/v1/evaluation/{query_id}")
async def query_evaluation(query_id: str):
    """根据 query_id 查询某次查询的评估结果。"""
    if query_id not in eval_cache:
        raise HTTPException(404, "评估结果尚未生成或 query_id 无效")
    return eval_cache.get(query_id)


# ==================== 健康检查与根路由 ====================


@app.get("/health")
async def health():
    """Kubernetes / 负载均衡器健康检查端点。"""
    return {"status": "ok", "service": "IntelliLens-MCP"}


@app.get("/v1/tenants/me")
async def tenant_info(request: Request):
    """返回当前租户的基本信息及文档数量。"""
    from src.api.documents import _query_documents_for_tenant

    ctx: dict = request.state.tenant_context
    tenant_id = ctx["tenant_id"]
    role = ctx.get("role", "viewer")
    docs = _query_documents_for_tenant(tenant_id)
    return {"tenant_id": tenant_id, "role": role, "document_count": len(docs)}


@app.get("/")
async def root():
    """根路径——重定向到登录页。"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/login")


@app.get("/login")
async def login_page():
    """登录页面。"""
    from fastapi.responses import FileResponse

    return FileResponse("static/login.html")


@app.get("/app")
async def app_page():
    """前端应用主页。"""
    from fastapi.responses import FileResponse

    return FileResponse("static/index.html")


# ==================== 开发模式启动 ====================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,  # 开发热重载
    )
