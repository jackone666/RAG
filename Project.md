---

# 🚀 IntelliLens-MCP 企业级生产版项目说明书 (v2.1)

**(致 Claude Code / Cursor 的执行上下文)**

## 1. 项目概述 (Project Overview)

* **项目名称:** IntelliLens-MCP (Production Edition)
* **定位:** 企业级智能数据治理与 Agentic RAG 系统。
* **核心目标:** 在 Python 3.10+ 环境下，基于 LlamaIndex 框架构建高并发、抗幻觉的 RAG 核心；通过 Model Context Protocol (MCP) 对接企业数据源实现动态工具调用。同时，项目必须严格实装生产级五大防线：**多租户数据权限隔离 (RBAC)**、**文档增量同步与清理**、**网关限流与大模型熔断降级**、**LLMOps 全链路可观测性追踪**以及**完全异步的大模型裁判评估中枢**。

## 2. 核心技术栈 (Tech Stack)

请在初始化项目时，确保 `requirements.txt` 包含以下依赖：

* **RAG 基础:** `llama-index-core`, `llama-index-llms-openai`, `llama-index-embeddings-openai`
* **向量库支持:** `llama-index-vector-stores-milvus`
* **MCP 协议:** `mcp` (官方 Python SDK)
* **Web 与并发防线:** `fastapi`, `uvicorn`, `redis`, `slowapi`, `pyjwt`
* **可观测性 (LLMOps):** `langfuse` (或 `arize-phoenix`)
* **健壮性与鲁棒性:** `tenacity` (用于大模型 API 指数退避重试)
* **基础工具:** `pydantic-settings`, `loguru`, `python-dotenv`

## 3. 项目目录结构 (Directory Structure)

请严格按照以下目录结构生成完整的项目骨架与文件：

```text
intellilens-mcp/
├── .env.example                 # 环境变量模板 (包含 OPENAI_API_KEY, MILVUS_URI 等)
├── requirements.txt             # 核心依赖清单
├── main.py                      # FastAPI/MCP 启动入口 (挂载所有 Middleware 与路由)
├── src/
│   ├── __init__.py
│   ├── config/                  # 全局配置模块
│   │   └── settings.py          # 基于 Pydantic Settings 的环境变量解析
│   ├── observability/           # 可观测性配置层
│   │   ├── __init__.py
│   │   └── tracer.py            # 配置 Langfuse/Phoenix 全链路追踪拦截器
│   ├── middleware/              # 网关安全防线层
│   │   ├── __init__.py
│   │   ├── auth.py              # JWT/Header 解析 Token，提取 tenant_id 和角色权限
│   │   └── rate_limiter.py      # 基于 Redis/内存 的限流拦截器
│   ├── pipeline/                # 数据治理与流水线
│   │   ├── __init__.py
│   │   ├── ingestion.py         # 文档入库 (强制在 Node 级别注入 tenant_id 元数据)
│   │   └── sync_manager.py      # 处理文档增量更新与物理/软删除同步逻辑
│   ├── engine/                  # RAG 核心检索引擎
│   │   ├── __init__.py
│   │   ├── retrievers.py        # QueryFusionRetriever (双路混合检索 + 强制权限过滤)
│   │   └── query_engine.py      # Rerank 打分排序与【大模型异常降级 Fallback 机制】
│   ├── mcp_server/              # MCP 协议服务端层
│   │   ├── __init__.py
│   │   ├── server.py            # FastMCP Server 实例与生命周期管理
│   │   └── tools.py             # 暴露给 Agent 的动态工具 (要求绑定身份上下文)
│   └── evaluation/              # 异步质量监控中心
│       ├── __init__.py
│       └── evaluator.py         # 独立的大模型裁判 (Faithfulness 防幻觉评估)
└── data/                        # 物理数据与测试文件存放目录

```

## 4. 生产级模块实现要求 (Module Specifications)

### 模块 1: 数据管道与多租户隔离 (`src/pipeline/ingestion.py` & `sync_manager.py`)

* **RBAC 隔离入库:** 在 `IngestionPipeline` 使用 `SemanticSplitterNodeParser` 进行语义分块时，切出的每个 Node 必须强制注入 `metadata={"tenant_id": "xxx", "doc_id": "yyy"}`。
* **文档生命周期同步:** 在 `sync_manager.py` 中实现 `delete_document(doc_id)` 接口。当上游业务系统删除文档时，能够根据 `doc_id` 从 Milvus 向量库和 Elasticsearch 中精准清理相关的 Chunks，防止历史数据产生幽灵幻觉。

### 模块 2: 带权限过滤与降级的检索引擎 (`src/engine/retrievers.py` & `query_engine.py`)

* **前置硬过滤 (Pre-filtering):** `QueryFusionRetriever` 必须融合向量检索与关键词检索。检索时**必须**接收当前请求上下文的 `tenant_id`，并使用 LlamaIndex 的 `MetadataFilters` 拼装到检索参数中，确保底层的存储引擎在检索阶段就完成租户隔离。
* **大模型降级 (Fallback):** 在 `query_engine.py` 的大模型调用处，使用 `try-except` 捕获大模型 API 超时、限流（429 错误）或凭证失效等异常。当主模型（如 GPT-4o）不可用时，系统需自动平滑降级切换至备用模型（如 GPT-4o-mini 或本地开源模型），或返回标准的降级友好提示。

### 模块 3: 网关防线与监控 (`src/middleware/` & `src/observability/`)

* **流量控制 (`rate_limiter.py`):** 结合 FastAPI 的依赖注入，实现基础的限流逻辑（可用内存 Dict 模拟 Redis 行为），防止恶意的接口刷量和大模型 Token 爆刷。
* **全链路追踪 (`tracer.py`):** 使用 LlamaIndex 官方集成（如 `llama_index.core.set_global_handler("langfuse")`），确保每一次检索耗时、重排耗时、最终生成的 Prompt 以及 Token 消耗都被静默记录，便于线上 Bad Case 溯源。

### 模块 4: MCP 服务端 (`src/mcp_server/tools.py`)

* 使用 FastMCP 构建 Server 实例。通过 `@mcp.tool()` 暴露的核心问答工具 `search_enterprise_knowledge(query: str, tenant_id: str)` 必须强制要求传入 `tenant_id` 身份上下文，使得通过大模型客户端（如 Claude 桌面端）调用该工具时，依然受到严格的多租户权限管控。

### 模块 5: 异步评估与坏例治理 (`src/evaluation/evaluator.py`)

* **绝对异步解耦:** 评估逻辑严禁阻塞 C 端用户的响应。在 FastAPI 的请求生命周期中，必须使用 `FastAPI.BackgroundTasks` 异步投递 `[Query, Context_Nodes, Answer]` 日志供评估器消费。
* **抗限流重试:** 裁判模型（如 GPT-4o）在调用 LlamaIndex 自带的 `FaithfulnessEvaluator` 时，必须使用 `tenacity` 库包一层自动重试与指数退避（Exponential Backoff）逻辑，以应对高并发下的裁判模型限流。
* **坏例沉淀闭环:** 如果裁判模型判定 `passing == False` 或 `score < 0.8`（意味着存在幻觉或质量极低），系统必须自动将该三元组数据及 `tenant_id` 序列化保存到 `data/bad_cases.jsonl` 中，自动沉淀为“待治理坏例集”。

## 5. 给 Claude Code 的分步执行指令 (Execution Steps)

请严格按照以下顺序逐步执行

* **Phase 1 - 基建、安全防线与可观测性:** 创建目录与 `requirements.txt`；实现 `settings.py` 环境变量解析；实现 `middleware/` 下的鉴权与限流框架；在 `main.py` 中挂载 `tracer.py` 的可观测性拦截器。
* **Phase 2 - 多租户数据流水线:** 实现 `src/pipeline/ingestion.py`，编写带有 `tenant_id` 元数据注入的文档处理流水线；在 `sync_manager.py` 中提供基于 `doc_id` 删除向量的清理接口。
* **Phase 3 - 核心引擎与熔断机制:** 实现 `src/engine/` 下的查询引擎，**强制要求**在检索参数中加入基于 `tenant_id` 的 `MetadataFilters` 权限硬隔离；在大模型生成外层编写 Fallback 降级策略代码。
* **Phase 4 - MCP 工具链与异步评估:** 构建 `mcp_server/`，封装带身份验证的 RAG 工具；实现 `evaluation/evaluator.py` 的 `BackgroundTasks` 异步大模型打分与坏例落库逻辑。
* **Phase 5 - 全链路联调测试:** 完善 `main.py` 启动入口，模拟一次完整闭环请求：接收问题 -> 鉴权限流 -> 混合检索(带权限过滤) -> 大模型生成(带降级保护) -> 异步评估打分与坏例沉淀。

---

*(End of Specification)*