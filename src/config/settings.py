"""
全局配置模块 - 基于 Pydantic Settings 的环境变量解析

设计原则：
- 所有配置项从环境变量/.env 文件读取，禁止硬编码
- 使用 pydantic-settings 提供类型校验和自动补全
- 通过模块级单例 settings 供全项目引用
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，自动从 .env 文件和环境变量中加载。

    配置分类：
    - OpenAI: 大模型 API 密钥、Base URL、模型选型
    - Milvus: 向量数据库连接参数
    - Redis: 分布式限流与缓存
    - JWT: 身份认证与租户上下文解析
    - Langfuse: 全链路 LLMOps 可观测性
    - App: 服务运行参数
    - Evaluation: 异步评估裁判配置
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ========== OpenAI 大模型配置 ==========
    openai_api_key: str  # OpenAI API 密钥（必填）
    openai_base_url: str = "https://api.openai.com/v1"  # 兼容 Azure / 代理

    # 模型选型：主模型用于线上回答，备用模型用于降级熔断，裁判模型用于评估
    primary_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    # 嵌入模型：支持本地路径或远程 API（SiliconFlow / OpenAI 兼容）
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""  # 为空则用 openai_api_key
    embedding_dim: int = 1024  # bge-m3 → 1024 维
    judge_model: str = "gpt-4o"

    # ========== Milvus 向量数据库 ==========
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""  # Milvus 集群认证令牌（可选）
    milvus_collection_name: str = "intellilens_knowledge"

    # ========== MinIO 对象存储（文档原始文件） ==========
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "intellilens-documents"
    minio_secure: bool = False

    # ========== Elasticsearch 关键词检索引擎 ==========
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "intellilens_knowledge"
    elasticsearch_analyzer: str = "smartcn"  # 中文分词器（ES 官方），也可用 ik_max_word（需装 IK 插件）

    # ========== Redis 配置 ==========
    redis_url: str = "redis://localhost:6379/0"
    redis_pool_size: int = 20  # 连接池大小，建议 = worker_num * 2
    redis_socket_timeout: float = 5.0  # Redis 操作超时秒数

    # ========== JWT 认证配置 ==========
    jwt_secret_key: str = "dev-secret-change-in-production"
    jwt_algorithm: str = "HS256"

    # ========== Langfuse 可观测性 ==========
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ========== 应用服务配置 ==========
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    rate_limit_per_minute: int = 60  # 每个租户每分钟最大请求数

    # ========== 评估裁判配置 ==========
    bad_cases_path: str = "data/bad_cases.jsonl"  # 坏例沉淀文件路径
    eval_score_threshold: float = 0.8  # 低于此分数的回答视为"幻觉/低质量"


# 模块级单例，全项目通过 from src.config.settings import settings 引用
settings = Settings()
