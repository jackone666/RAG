"""
RAG 效果调参中心 — 所有影响 RAG 管线效果的参数集中管理。

修改此文件即可调整检索/生成/评估质量，无需改动业务代码。
参考: 字节跳动 RAG 最佳实践 (rag.pdf)
"""
from dataclasses import dataclass, field


@dataclass
class RAGParams:
    """RAG 管线全部可调参数。"""

    # =========================================================================
    # §A. 文档预处理 (Preprocessing)
    # =========================================================================

    # 大文档预拆分：超过此字符数先切为子文档再语义分块
    pre_split_chars: int = 20_000

    # 语义分块 — 小数块策略（参考字节 §3.2.2「分块策略」）
    semantic_breakpoint_percentile: int = 50
    semantic_buffer_size: int = 1

    # 超大节点二次拆分（防止单个节点超过 LLM 上下文窗口）
    max_chunk_chars: int = 60_000

    # 标题检测启发式阈值（字符数）
    heading_max_len: int = 60
    heading_min_len: int = 2

    # 文档预览截取长度（Langfuse span / 前端展示用）
    doc_preview_chars: int = 300

    # =========================================================================
    # §B. 混合检索 (Retrieval) — 参考字节 §4.3「混合检索策略」
    # =========================================================================

    # 通用检索召回数（未单独指定 vector/keyword 时的默认值）
    retrieval_top_k: int = 10

    # 向量检索召回候选数（ByteCache + weighted_rrf 模式使用）
    retrieval_vector_top_k: int = 10

    # 关键词检索召回候选数
    retrieval_keyword_top_k: int = 10

    # 融合排序模式：weighted_rrf | RELATIVE_SCORE | RECIPROCAL_RANK | DIST_BASED_SCORE
    # - weighted_rrf: 加权倒数排名融合，通过 vector_weight/keyword_weight 控制比例
    # - RELATIVE_SCORE: LlamaIndex 相对分数融合
    # - RECIPROCAL_RANK: RRF 等权融合
    fusion_mode: str = "weighted_rrf"

    # 加权融合比例（仅 weighted_rrf 模式生效）：向量权重 vs 关键词权重
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # RRF 平滑常数 k（越大排名影响越弱，推荐 60）
    rrf_k: int = 60

    # =========================================================================
    # §C. 重排序 (Rerank)
    # =========================================================================

    # LLM 重排序最终保留的文档数（送进大模型生成上下文）
    rerank_top_n: int = 5

    # 重排序 API 调用的最大内容长度（字符数）
    rerank_max_chars_per_doc: int = 3000

    # =========================================================================
    # §D. 查询改写 (Query Rewriting)
    # =========================================================================

    rewrite_temperature: float = 0.0
    rewrite_max_tokens: int = 512

    # =========================================================================
    # §E. 大模型生成 (Generation)
    # =========================================================================

    generation_temperature: float = 0.1
    generation_max_tokens: int = 1024

    # 系统提示词模板
    system_prompt: str = (
        "你是企业智能知识助手，擅长基于上下文回答问题。\n"
        "要求：\n"
        "1. 严格基于下方给出的【参考资料】生成答案，不得臆造信息\n"
        "2. 如果参考资料不足以回答问题，请如实说明\n"
        "3. 回答末尾标注引用的文档编号（如 [文档1]、[文档3]）\n"
        "4. 使用中文回答，条理清晰"
    )

    # =========================================================================
    # §F. 裁判评估 (Evaluation)
    # =========================================================================

    judge_temperature: float = 0.0
    judge_max_tokens: int = 2048  # 裁判需要输出 JSON，token 需求更大
    eval_score_threshold: float = 0.8  # 低于此分数视为坏例
    eval_timeout_seconds: float = 15.0  # 评估等待超时（秒）

    # =========================================================================
    # §G. 缓存与限流
    # =========================================================================

    # 热点检索缓存 TTL（秒）
    cache_ttl: int = 300
    # 热点检索缓存最大条目数
    cache_maxsize: int = 256

    # 评估结果缓存
    eval_cache_maxsize: int = 200
    eval_cache_ttl: int = 600  # 兜底过期时间（秒）

    # ByteCache: Redis 向量检索缓存（参考字节 §4.3.1）
    byte_cache_enabled: bool = True
    byte_cache_ttl: int = 604800  # 7 天
    byte_cache_max_chars: int = 2000  # 单个节点序列化截断长度

    # 限流
    rate_limit_per_minute: int = 60
    rate_limit_window_seconds: int = 60

    # =========================================================================
    # §H. 嵌入模型
    # =========================================================================

    embedding_batch_size: int = 32
    embedding_dim: int = 1024
    embedding_max_input_chars: int = 8000


# 全局单例
rag_params = RAGParams()
