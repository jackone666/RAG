# IntelliLens-MCP 项目说明文档（最新版）

## 0. 快速信息卡

| 项目 | 说明 |
|------|------|
| **名称** | IntelliLens-MCP v2.1.0 |
| **类型** | 企业级 RAG + MCP 系统 |
| **代码规模** | 36 个 Python 模块，~5000 行核心代码 |
| **技术栈** | FastAPI + LlamaIndex + Milvus + Elasticsearch + Redis + MinIO |
| **部署** | Docker Compose，支持本地开发和云端部署 |
| **核心能力** | 多租户隔离、混合检索、查询理解、质量评估、Agent 工具化 |

## 1. 项目定位

IntelliLens-MCP 是一个面向企业知识库的智能数据治理与 Agentic RAG 系统。它解决的不是”让大模型读几个文档回答问题”这么简单，而是把企业
RAG 在真实生产里会遇到的安全、效果、性能、评估、可观测和 Agent 接入问题做成一条完整闭环。

一句话介绍：

> IntelliLens-MCP 是一个带多租户隔离、混合检索、查询理解、上下文压缩、主备模型降级、LLMOps 追踪、异步评估和 MCP 工具暴露的企业级
> RAG 中台。

便于记忆：

> 不是”向量库 + LLM”，而是”数据治理 + 检索增强 + 质量闭环 + Agent 工具化”。

## 2. 核心目标

| 目标 | 实现方式 | 面试要点 |
|------|--------|--------|
| **安全** | 多租户隔离下推到存储层（Milvus MetadataFilters + ES term filter） | 纵深防御，不在应用层设防 |
| **准确** | 查询改写 + 术语归一化 + 混合检索 + RRF 融合 + rerank + 上下文压缩 | 检索前理解问题，生成前压缩噪声 |
| **稳定** | Redis 限流 + 主备模型降级 + 异步评估 + 缓存失效机制 | 单点故障不影响用户体验 |
| **可治理** | 文档生命周期同步 + 坏例沉淀 + 人工评估集候选 + Langfuse 追踪 | 从线上问题到离线改进的闭环 |
| **可接入 Agent** | MCP 工具暴露 + 企业知识搜索接口 | 让 Claude Desktop 等 Agent 调用 |

## 3. 最新请求链路

```text
HTTP / MCP Request
  -> tenant_context_middleware
     JWT / Header 解析 tenant_id、role、user_id
  -> rate_limit_dependency
     Redis ZSET + Lua 滑动窗口限流（fail-open 策略）
  -> QueryRewriter
     术语归一化 + 查询改写 + 规则分解 + HyDE 假设文档
  -> TenantAwareQueryFusionRetriever
     Milvus 向量检索 + Elasticsearch 关键词检索
     两路都强制 tenant_id 过滤（纵深防御）
  -> Fusion
     weighted RRF / LlamaIndex fusion（可并行优化）
  -> Rerank
     BAAI/bge-reranker-v2-m3 精排 Top-N
  -> Context Compression
     按 query 关键字抽取关键句，生成 [文档1] 引用上下文
  -> Generation
     primary_model -> fallback_model -> FALLBACK_RESPONSE
     支持 SSE 流式输出
  -> AsyncEvaluator
     检索指标（precision/recall/MRR/hit_rate）
     生成指标（faithfulness/relevance）
     坏例写入 bad_cases.jsonl
  -> Langfuse
     retrieval / rerank / generation trace
```

**面试记忆**：先理解问题 → 多路召回 → 精排压缩 → 生成评估 → 异步沉淀

## 4. 文档入库链路

```text
Upload Document
  -> 文件类型/大小校验（txt/md/pdf/docx，最大 50MB）
  -> 文本抽取：PyMuPDF（PDF）、python-docx（DOCX）
  -> 文本清洗：空白、控制字符、标点规范化
  -> 结构解析：标题层级、章节、字符数
  -> 内容哈希：SHA256 完全去重（先清洗再 hash）
  -> MinIO 保存原始文件（支持重入库）
  -> 大文档预拆分：超过阈值先按字符段切开
  -> SemanticSplitter 语义分块（保持段落完整性）
  -> 超长 chunk 保护：递归二分（避免超过 Milvus VARCHAR 限制）
  -> 注入 tenant_id / doc_id / content_hash（强制）
  -> BAAI/bge-m3 嵌入（多语言，1024 维）
  -> Milvus 向量入库（带标量索引）
  -> Elasticsearch 关键词索引后台写入
```

**关键设计**：
- 预处理决定 RAG 下限，不是简单切块
- 元数据强制注入，后续检索/删除/去重都依赖
- 语义分块优先于固定长度，减少信息散落
- 双重保护超长 chunk，避免存储和上下文污染

**面试记忆**：入库不是”切块写向量库”，而是”抽取、清洗、结构化、去重、存原文、分块、打标签、建双索引”

## 5. 核心模块与技术栈

### 5.1 技术栈总览

| 层级 | 技术选型 | 为什么选它 |
|------|--------|---------|
| **Web 框架** | FastAPI + Uvicorn | 异步原生，自动 OpenAPI 文档，性能好 |
| **RAG 框架** | LlamaIndex | MetadataFilters 原生支持，Langfuse 官方集成 |
| **向量库** | Milvus | 支持标量过滤，多租户隔离友好 |
| **关键词索引** | Elasticsearch | 企业级搜索，支持复杂查询 |
| **缓存** | Redis | 分布式限流、ByteCache、热点查询加速 |
| **文件存储** | MinIO | S3 兼容，支持原文保存和重入库 |
| **嵌入模型** | BAAI/bge-m3 | 多语言，1024 维，中文友好 |
| **重排模型** | BAAI/bge-reranker-v2-m3 | 专门做 query-document 相关性 |
| **生成模型** | gpt-4o（主）+ gpt-4o-mini（备） | 效果稳定 + 成本平衡 |
| **评估模型** | gpt-4o | 强推理能力，异步执行不影响首响 |
| **可观测性** | Langfuse | 全链路 trace，支持自定义 span |
| **Agent 接入** | MCP（Model Context Protocol） | 标准化工具暴露，Claude Desktop 原生支持 |

### 5.2 核心模块分层

| 层级 | 模块 | 职责 | 关键文件 |
|------|------|------|--------|
| **API 层** | 查询、文档、认证 | HTTP 接口、MCP 工具暴露 | `main.py`, `src/api/documents.py` |
| **安全层** | 认证、限流 | JWT 验证、RBAC、Redis 限流 | `src/middleware/auth.py`, `rate_limiter.py` |
| **查询理解** | 查询改写、术语归一化 | 多路等价查询、HyDE、规则分解 | `src/engine/query_rewriter.py`, `terminology.py` |
| **检索层** | 混合检索、融合 | Milvus + ES、RRF 融合、tenant 过滤 | `src/engine/retrievers.py` |
| **生成层** | 重排、压缩、生成 | rerank、关键句压缩、主备降级、SSE | `src/engine/query_engine.py` |
| **数据治理** | 入库、同步、清理 | 文档生命周期、删除同步、租户清理 | `src/pipeline/ingestion.py`, `sync_manager.py` |
| **质量评估** | LLM judge、坏例沉淀 | 检索/生成指标、golden set 候选 | `src/evaluation/evaluator.py`, `golden_set.py` |
| **缓存** | TTLCache、ByteCache | 热点查询加速、向量检索缓存 | `src/utils/byte_cache.py` |
| **可观测性** | Langfuse 追踪 | 全链路 span、性能指标 | `src/observability/tracer.py` |
| **Agent 接入** | MCP 工具 | 企业知识搜索工具暴露 | `src/mcp_server/tools.py` |

## 6. 已实现的关键优化

### 6.1 多租户隔离（纵深防御）

**实现方式**：
1. JWT/Header 注入 `request.state.tenant_context`
2. 文档入库时每个 chunk 写入 `tenant_id`
3. **Milvus 检索通过 `MetadataFilters(tenant_id == 当前租户)` 前置过滤**（存储层硬隔离）
4. Elasticsearch 查询使用 `term tenant_id` 过滤
5. 删除文档和删除租户时同样带 tenant 条件
6. MCP tool 强制要求 `tenant_id` 参数

**面试要点**：
- 不在应用层设防，而是在存储引擎层设防
- 即使上层代码有 bug，Milvus 也不会返回其他租户的数据
- 这是**纵深防御**思路，不是单点防御

**已知问题**：
- 当前 `TenantAwareQueryFusionRetriever` 中有 monkey-patch 问题（并发安全）
- 应该改用 `VectorStoreQuery` 的 `filters` 参数，而不是全局修改 `vector_store.query`

### 6.2 混合检索（语义 + 精确）

**实现方式**：
1. Milvus 负责语义召回（向量相似度）
2. Elasticsearch 负责关键词/编号/条款召回（BM25）
3. weighted RRF 或 LlamaIndex fusion 合并两路结果
4. reranker 再做精排

**面试要点**：
- 企业知识里有大量编号、条款、缩写，纯向量会漏
- 纯关键词又不懂语义，所以必须混合检索
- 当前实现是串行（向量 → 关键词），可优化为并行（asyncio.gather）

**记忆**：向量负责”意思像”，关键词负责”字面准”

### 6.3 查询理解增强

**当前能力**：
1. **LLM query rewrite**：生成语义等价问法
2. **query decomposition**：用规则把复杂问题拆成子问题
3. **HyDE**：生成假设性文档用于语义检索
4. **领域术语归一化**：把”动态市盈率”等别名扩展为 `PE-TTM`

**面试要点**：
- 参考字节 RAG 实践，检索前要先理解用户到底在问什么
- 改写负责同义扩展，分解负责多跳问题，HyDE 负责语义对齐，术语表负责企业黑话
- 这是轻量版实现，生产可扩展为更复杂的 query 理解

**记忆**：用户问一句，检索前先翻译成”知识库听得懂的几句话”

### 6.4 生成前上下文压缩

**实现方式**：
1. rerank 后的文档片段按中英文标点切句
2. 从 query 抽取关键词和中文 bigram
3. 对句子按命中数打分
4. 每个文档只保留最相关关键句
5. 上下文加 `[文档1] source=xxx` 编号

**面试要点**：
- 参考字节 RAG 的检索结果摘要思想
- 不是把所有 chunk 全文塞给大模型，而是先抽关键句
- 减少 token 成本和噪声，提升引用可解释性

**记忆**：不是把整本书塞给模型，而是先给它划重点

### 6.5 主备模型降级

**实现方式**：
```
primary_model 失败
  → fallback_model
    → 标准化降级文案
```

**面试要点**：
- LLM API 会超时、限流、凭证异常
- 线上系统不能把这些错误直接暴露给用户
- 生成层做主备模型切换，支持多 Provider 配置

**记忆**：主力不在线，替补上场；替补也不行，给用户体面提示

### 6.6 异步质量评估 + 人工评估集候选

**实现方式**：
1. 查询完成后触发异步 evaluator（BackgroundTasks）
2. 检索侧评估 precision、recall、MRR、hit rate
3. 生成侧评估 faithfulness、relevance
4. 低分样本写入 `bad_cases.jsonl`
5. 新增 `golden_set.py`，可把坏例沉淀为人工标注候选集

**面试要点**：
- LLM judge 适合自动发现问题，但不能完全替代人工标注
- 坏例会继续沉淀成人工评估集，后续做离线回归
- 当前是”尽力而为”（best-effort），可优化为 DLQ + 定时重试

**记忆**：机器先筛错题，人再整理成题库

### 6.7 缓存与失效机制

**当前能力**：
1. `TTLCache` 缓存热点 query 的检索结果
2. `ByteCache` 按 embedding hash 缓存向量检索结果
3. 文档删除/租户删除时可扩展主动清理相关缓存
4. 测试环境关闭分布式 ByteCache，避免 Redis 跨测试污染

**面试要点**：
- 缓存能显著降低热点查询延迟
- 文档更新后必须考虑缓存失效，否则可能查到旧知识
- 当前缓存失效是被动的（TTL 过期），可优化为主动清理

**记忆**：缓存是加速器，但删文档时要记得清旧账

### 6.8 Redis 限流（Lua 脚本 + 滑动窗口）

**实现方式**：
- ZSET 记录每个请求的精确时间戳
- Lua 脚本保证原子性，消除 TOCTOU 竞态
- fail-open 策略：Redis 挂了放行请求，不中断服务

**面试要点**：
- 为什么用 Lua？消除竞态窗口，三条命令在 Redis 服务端单线程执行
- 为什么用 ZSET？避免固定窗口的边界突发问题
- 为什么 fail-open？限流是保护性功能，故障不应导致服务中断

**记忆**：限流是保护，不是业务路径

### 6.9 LLMOps 可观测性

**实现方式**：
- Langfuse `CallbackManager` 全链路追踪
- 记录 retrieval / rerank / generation span
- 每个 span 包含 latency、token 数、tenant_id、user_id

**面试要点**：
- 选 LlamaIndex 的一个重要原因是官方 Langfuse 集成
- 一行代码完成全链路追踪，不需要自己写拦截器
- 可用于性能分析、成本统计、异常告警

**记忆**：可观测性是生产系统的必需品

## 7. 模型选型说明

### 7.1 生成主模型：`primary_model`

默认：`gpt-4o`

选择理由：

1. 复杂问题理解和长上下文综合能力强。
2. 中文表达、结构化回答和推理稳定。
3. 适合面向用户的最终答案生成。

面试说法：

> 主模型承担“最后对用户说话”的职责，所以优先选效果稳定、推理和表达能力强的模型。

记忆点：

> 主模型负责“答得好”。

### 7.2 备用模型：`fallback_model`

默认：`gpt-4o-mini`

选择理由：

1. 成本低、延迟低。
2. 主模型异常时能兜底。
3. 适合 query rewrite、HyDE 这类中间任务。

面试说法：

> 备用模型不是为了追求最强效果，而是为了可用性和成本平衡。

记忆点：

> 备用模型负责“别挂掉”。

### 7.3 嵌入模型：`BAAI/bge-m3`

选择理由：

1. 多语言能力好，适合中英混合企业文档。
2. 1024 维语义表达较强。
3. 支持长输入，适合企业文档 chunk。
4. 可通过 SiliconFlow/OpenAI 兼容接口调用，部署成本低。

面试说法：

> embedding 决定召回上限，bge-m3 在中文、多语言和通用语义检索上比较均衡。

记忆点：

> 嵌入模型负责“找得到”。

### 7.4 重排模型：`BAAI/bge-reranker-v2-m3`

选择理由：

1. 专门做 query-document 相关性判断。
2. 比单纯向量相似度更适合精排。
3. 可把“多召回”的候选压到 Top-5，降低生成噪声。

面试说法：

> 检索阶段宁可多召回，reranker 负责最后把真正相关的文档排到前面。

记忆点：

> reranker 负责“排得准”。

### 7.5 裁判模型：`judge_model`

默认：`gpt-4o`

选择理由：

1. 评估 faithfulness/relevance 需要更强推理和判断能力。
2. 裁判调用异步执行，不直接影响用户首响。
3. 用强模型做评估能减少误判。

面试说法：

> 裁判模型负责发现幻觉和低质量回答，宁可慢一点，也要判断更稳。

记忆点：

> 裁判模型负责“判得准”。

## 8. 结合字节 RAG 实践的优化对照

| 字节 RAG 实践 | 本项目落地                                              | 面试解释                   |
|-----------|----------------------------------------------------|------------------------|
| 查询理解      | query rewrite + query decomposition + HyDE + 术语归一化 | 检索前先把问题变成知识库更容易命中的表达   |
| 混合检索      | Milvus + Elasticsearch + weighted RRF              | 同时兼顾语义召回和精确匹配          |
| 检索结果处理    | rerank + 关键句压缩 + 文档编号引用                            | 减少 token、降低噪声、提升引用可解释性 |
| ByteCache | Redis ByteCache + TTLCache                         | 热点向量检索命中缓存，降低延迟        |
| 数据治理      | MinIO 原文 + Milvus/ES 索引 + 删除同步                     | 支持重入库、模型切换和知识过期清理      |
| 质量评估      | LLM judge + bad case + golden set candidate        | 从线上坏例沉淀到离线评估集          |

## 9. 仍可继续演进的方向

以下是面试中可以主动说明的“下一步优化”，体现你知道当前方案边界：

1. **动态融合权重**：根据 query 类型动态调整向量/关键词权重。语义型更偏向向量，编号/金额/条款型更偏向关键词。
2. **多粒度索引**：文档级粗筛、段落级中筛、句子级精筛。目前主要是 chunk 级索引，后续可扩展多 collection 或 metadata level。
3. **近重复检测**：当前有 SHA256 完全去重，后续可用 embedding similarity >= 0.95 检测转载版、轻微改写版文档。
4. **更细缓存失效**：文档更新后根据 doc_id/tenant_id 精确清理 query cache 和 ByteCache。
5. **表格/OCR 增强**：PDF 表格、图片 OCR、页码和 bbox 引用定位可以继续增强。
6. **人工评估闭环**：把 pending golden cases 做成标注后台，定期跑离线回归。

便于记忆：

> 下一步就是“权重更聪明、索引更细、去重更像人、缓存更懂删除、文档理解更接近版面、评估更接近人工标准”。

## 10. 面试 1 分钟版本

IntelliLens-MCP 是一个企业级 RAG + MCP 系统。我主要解决三类问题：第一是安全，通过 JWT、RBAC 和 tenant_id 下推到 Milvus/ES
实现多租户隔离；第二是效果，通过术语归一化、查询改写、HyDE、向量 + 关键词混合检索、RRF 融合、reranker
和上下文压缩提升召回与生成质量；第三是生产可用性，通过 Redis 限流、ByteCache、主备模型降级、Langfuse 追踪、异步 LLM judge
和人工评估集候选沉淀，把 RAG 从 demo 做成可运营的系统。同时通过 MCP 把企业知识搜索暴露给外部 Agent 调用。

## 11. 面试 3 分钟版本

文档上传后，系统会先做类型和大小校验，然后抽取 txt、md、pdf、docx 文本，做清洗、标题结构解析和 hash 去重。原始文件存 MinIO，文本通过
SemanticSplitter 做语义分块，每个 chunk 注入 tenant_id、doc_id、content_hash，再用 bge-m3 做 embedding 写入 Milvus，同时后台写入
Elasticsearch 做关键词索引。

查询时，网关先解析 JWT 或 Header 得到 tenant_context，并用 Redis Lua 滑动窗口限流。进入 RAG 主链路后，QueryRewriter
会做术语归一化、查询改写、规则分解和 HyDE 假设文档，多路检索后合并去重。检索层同时走 Milvus 向量召回和 ES 关键词召回，两条路径都带
tenant_id filter，然后用 weighted RRF 融合，再用 bge-reranker-v2-m3 精排 Top-5。

生成前，我参考字节 RAG 的检索结果摘要思想，对 rerank 后的长 chunk 做关键句压缩，并加 `[文档1]` 这样的编号，减少 token
和上下文噪声。生成层先调主模型，失败切备用模型，主备都失败返回友好降级文案。响应会返回 pipeline_trace，Langfuse 记录检索、重排和生成
span。后台还会异步评估 precision、recall、MRR、hit rate、faithfulness 和 relevance，低质量样本进入 bad cases，并可沉淀为人工
golden set 候选。

## 12. 高频问题回答

### Q1：为什么要混合检索？

纯向量适合语义相似，但对编号、条款、金额、错误码不稳定；关键词检索适合精确匹配，但不懂同义表达。企业知识库两类问题都有，所以用向量 +
关键词，再融合和重排。

记忆：向量找意思，关键词找字面。

### Q2：为什么要 query rewrite、HyDE 和术语归一化？

用户表达和企业文档表达经常不一致。改写扩展同义问法，分解处理复杂问题，HyDE 生成假设答案帮助语义对齐，术语归一化解决企业缩写和指标别名。

记忆：把用户话翻译成知识库语言。

### Q3：为什么要上下文压缩？

rerank 后的 chunk 仍然可能很长，直接塞给大模型会增加 token 成本和噪声。关键句压缩只保留和 query 最相关的句子，降低幻觉风险，也让引用更稳定。

记忆：先划重点，再交卷。

### Q4：为什么选择 bge-m3 做 embedding？

bge-m3 多语言能力强，适合中文和中英混合企业文档；1024 维表达能力较好；接口调用部署成本低。

记忆：embedding 负责找得到。

### Q5：为什么选择 reranker？

召回阶段为了避免漏召会取更多候选，但候选里有噪声。reranker 用 query-document 相关性重新排序，保证进入生成层的是最相关的
Top-N。

记忆：召回多捞鱼，重排挑好鱼。

### Q6：LLM judge 为什么还要人工评估集？

LLM judge 可以自动发现坏例，但也会有偏差。把 bad cases 沉淀成 golden set candidate 后，人工补 expected answer 和
label，就能做稳定的离线回归。

记忆：机器批改快，人工定标准。

### Q7：缓存为什么要失效？

ByteCache 和 TTLCache 能降低热点查询延迟，但文档删除或更新后，如果不清缓存，可能查到旧知识。缓存失效是 RAG 数据治理的一部分。

记忆：缓存提速，失效保真。

