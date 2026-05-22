# 🚀 IntelliLens-MCP v2.1

企业级智能数据治理与 Agentic RAG 系统。基于 LlamaIndex + FastAPI + MCP 协议，集成五大生产防线。

## 快速开始

```bash
# 1. 安装依赖
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 OPENAI_API_KEY 等必填项

# 3. 启动基础设施
docker compose up -d

# 4. 启动服务
python main.py
# API 文档: http://localhost:8000/docs
# Langfuse: http://localhost:3000
```

## 架构

```
HTTP Request
  → tenant_context_middleware (JWT / Header → tenant_id)
  → rate_limit_dependency (Redis 滑动窗口)
  → POST /v1/query
    → QueryRewriter (多路改写)
    → ByteCache (Redis 向量缓存，命中 ~15ms)
    → TenantAwareQueryFusionRetriever (向量 + ES 关键词，加权 RRF 融合)
    → RAGQueryEngine.rerank (LLM 重排序 → top-N)
    → RAGQueryEngine.query (主模型 → 备用模型 → 熔断降级)
    → AsyncEvaluator.evaluate (并行裁判评估，沉淀 bad cases)
    → Langfuse 全链路 trace (retrieval → rerank → generation)
```

## 五大防线

| 防线               | 模块                                                    | 说明                                 |
|------------------|-------------------------------------------------------|------------------------------------|
| **多租户隔离 (RBAC)** | `middleware/auth.py`                                  | JWT 鉴权，Milvus/ES 强制注入 tenant_id 过滤 |
| **文档生命周期同步**     | `pipeline/sync_manager.py`                            | 删除文档时同步清理 Milvus + ES              |
| **限流 + 熔断降级**    | `middleware/rate_limiter.py` `engine/query_engine.py` | 滑动窗口限流，主备模型自动切换                    |
| **LLMOps 可观测性**  | `observability/tracer.py`                             | Langfuse 全链路 trace + span          |
| **异步 LLM 裁判评估**  | `evaluation/evaluator.py`                             | 并行评估检索+生成质量，自动沉淀 bad cases         |

## 评估指标

每此查询返回 6 项质量指标：

| 指标           | 类型 | 说明            |
|--------------|----|---------------|
| Precision    | 检索 | 检索文档中真正相关的比例  |
| Recall       | 检索 | 相关文档被检索到的比例   |
| MRR          | 检索 | 首个相关文档排名的倒数   |
| Hit Rate     | 检索 | Top-K 中存在相关文档 |
| Faithfulness | 生成 | 回答是否严格基于上下文   |
| Relevance    | 生成 | 回答与问题的相关程度    |

## 混合检索配置

所有 RAG 参数集中在 `src/config/rag_params.py`：

- `fusion_mode`: 加权 RRF / RELATIVE_SCORE
- `vector_weight` / `keyword_weight`: 向量/关键词通路权重
- `retrieval_top_k`: 检索召回数
- `rerank_top_n`: LLM 重排序保留数
- `byte_cache_enabled` / `byte_cache_ttl`: Redis 向量缓存开关

## API

| 端点                           | 方法     | 说明                |
|------------------------------|--------|-------------------|
| `/v1/query`                  | POST   | 非流式 RAG 查询（含评估指标） |
| `/v1/query/stream`           | POST   | SSE 流式 RAG 查询     |
| `/v1/evaluation/{query_id}`  | GET    | 轮询评估结果            |
| `/v1/documents/upload`       | POST   | 单文件上传             |
| `/v1/documents/upload-batch` | POST   | 批量/文件夹上传          |
| `/v1/documents`              | GET    | 文档列表              |
| `/v1/documents/{doc_id}`     | DELETE | 删除文档              |
| `/health`                    | GET    | 健康检查              |

## 项目结构

```
├── main.py                      # FastAPI 入口
├── src/
│   ├── config/                  # 配置 (settings + rag_params)
│   ├── middleware/               # 鉴权 + 限流
│   ├── engine/                   # 检索 + 生成 + 改写
│   ├── pipeline/                 # 入库 + 预处理 + 同步
│   ├── evaluation/              # LLM 裁判评估
│   ├── observability/           # Langfuse 追踪
│   ├── mcp_server/              # MCP 工具服务
│   ├── api/                     # 文档管理 API
│   ├── utils/                   # 缓存 (ByteCache + 本地)
│   └── storage/                 # PostgreSQL 存储
├── static/                      # 前端页面
├── tests/                       # 测试用例
└── docker-compose.yml           # 基础设施编排
```

## License

MIT
