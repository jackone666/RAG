"""
可观测性配置 - 全链路 LLMOps 追踪模块 (Langfuse v4)

功能说明：
- 通过 Langfuse @observe 装饰器 + trace context 手动追踪 RAG 管线
- 静默记录检索耗时、重排耗时、LLM 调用 Prompt 及 Token 消耗
- 启动时 patch LlamaIndex 模型名白名单，兼容 DeepSeek 等非 OpenAI 模型

配置要点：
- 仅当 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 均配置后才启用
- Langfuse v4 移除了 LlamaIndex 集成，改用原生 trace API
"""
from contextlib import contextmanager
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

from src.config.settings import settings

_langfuse_client = None


def _patch_llama_index_models():
    """将自定义模型名注入 LlamaIndex 白名单。"""
    try:
        from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS, CHAT_MODELS

        for model in [settings.primary_model, settings.fallback_model, settings.judge_model]:
            if model and model not in ALL_AVAILABLE_MODELS:
                ALL_AVAILABLE_MODELS[model] = 128000
                CHAT_MODELS[model] = 128000

        from loguru import logger
        logger.info(f"LlamaIndex 模型白名单已扩展: {settings.primary_model}")
    except ImportError:
        pass


def init_observability():
    """初始化 Langfuse 可观测性。"""
    global _langfuse_client

    _patch_llama_index_models()

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return

    try:
        import langfuse
        _langfuse_client = langfuse.Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        from loguru import logger
        logger.info(f"Langfuse 已连接: {settings.langfuse_host} (v{langfuse.__version__})")
    except Exception as e:
        from loguru import logger
        logger.warning(f"Langfuse 初始化失败: {e}")


def get_langfuse_client():
    """获取 Langfuse 客户端（可能为 None）。"""
    return _langfuse_client


@contextmanager
def trace_rag_query(query: str, tenant_id: str):
    """为一次 RAG 查询创建 Langfuse trace context。"""
    if _langfuse_client is None:
        yield None
        return

    trace = _langfuse_client.trace(
        name="RAG Query",
        input={"query": query, "tenant_id": tenant_id},
    )
    try:
        yield trace
    finally:
        trace.update(output={"status": "completed"})


def trace_span(trace, name: str, input_data: dict = None, output_data: dict = None):
    """在 trace 内创建一个 span 用于追踪子步骤，可作为 context manager 使用。"""
    if trace is None:
        return _SpanWrapper(None)

    span = trace.span(
        name=name,
        input=input_data,
        output=output_data,
    )
    return _SpanWrapper(span)


class _SpanWrapper:
    """统一 span 包装器，兼容 context manager 协议。"""
    def __init__(self, span):
        self._span = span

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._span is not None:
            self._span.end()

    def update(self, **kwargs):
        if self._span is not None:
            self._span.update(**kwargs)

    def end(self):
        if self._span is not None:
            self._span.end()
