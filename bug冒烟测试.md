# Bug 冒烟测试记录

> 更新时间：2026-05-20

---

## BUG-001: Embedding 批处理 OOM — Invalid buffer size: 40.00 GiB

**发现时间**：2026-05-20

**错误日志**：

```
RuntimeError: Invalid buffer size: 40.00 GiB
```

**调用链**：

```
upload → ingestion_pipeline.ingest_text() → SemanticSplitterNodeParser
→ HuggingFaceEmbedding.get_text_embedding_batch()
→ sentence_transformers.encode() → torch.scaled_dot_product_attention
→ RuntimeError: Invalid buffer size: 40.00 GiB
```

**根因**：

- `HuggingFaceEmbedding` 创建时未设置 `embed_batch_size` 和 `max_length`
- `SemanticSplitterNodeParser` 将所有句子一次性传入 `get_text_embedding_batch()`
- XLM-RoBERTa 的自注意力矩阵为 O(n²)，大文档导致 40GB 内存分配失败

**修复**：

- `src/engine/retrievers.py`：`HuggingFaceEmbedding` 增加 `embed_batch_size=16` 和 `max_length=512`

**状态**：✅ 已修复

---

## BUG-002: pymilvus 弃用 API 警告

**发现时间**：2026-05-20

**错误日志**：

```
PyMilvusDeprecationWarning: `connections.get_connection_addr` is an ORM-style PyMilvus API
  and will be removed in PyMilvus 3.1. Use `MilvusClient` instead.
PyMilvusDeprecationWarning: `utility.has_collection` is an ORM-style PyMilvus API
  and will be removed in PyMilvus 3.1. Use `MilvusClient` instead.
```

**根因**：

- `_query_documents_for_tenant()` 和 `check_duplicate()` 使用旧版 ORM API
- LlamaIndex 已切换至 `MilvusClient`，旧 API 将在 PyMilvus 3.1 移除

**修复**：

- `src/api/documents.py`：`_query_documents_for_tenant` 改用 `MilvusClient`
- `src/pipeline/ingestion.py`：`check_duplicate` 改用 `MilvusClient`

**状态**：✅ 已修复

---

## BUG-003: embedding 模型启动时加载两次

**发现时间**：2026-05-20

**现象**：

```
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 61787.84it/s]
Loading weights: 100%|██████████| 391/391 [00:00<00:00, 90009.49it/s]
```

**根因**：

- `TenantAwareQueryFusionRetriever.__init__` 和 `TenantAwareIngestionPipeline.__init__` 在 import 时立即调用
  `get_shared_embed_model()`
- uvicorn reload 模式有 reloader + worker 两个进程，各触发一次

**修复**：

- `retrievers.py`：`__init__` 改为存储 `_embed_model_override`，新增 `_get_embed_model()` 懒加载
- `ingestion.py`：`__init__` 改为懒加载 `_get_embed_model()` / `_get_splitter()`

**状态**：✅ 已修复

---

## BUG-004: DocstoreStrategy.UPSERT 不存在

**发现时间**：2026-05-20

**错误日志**：

```
AttributeError: type object 'DocstoreStrategy' has no attribute 'UPSERT'. Did you mean: 'UPSERTS'?
```

**根因**：枚举名拼写错误，LlamaIndex 中正确名称是 `UPSERTS`

**修复**：`ingestion.py` 中 `DocstoreStrategy.UPSERT` → `DocstoreStrategy.UPSERTS`

**状态**：✅ 已修复

---

## BUG-005: MinIO metadata 不支持中文文件名

**发现时间**：2026-05-20

**错误日志**：

```
对象存储写入失败: unsupported metadata value 从零构建大模型.pdf; only US-ASCII encoded characters are supported
```

**根因**：MinIO 的 user metadata 仅支持 US-ASCII 编码

**修复**：`doc_store.py` 移除 metadata 中的 `original_filename` 字段（文件名已在 S3 路径中保留）

**状态**：✅ 已修复

---

## BUG-006: 查询改写 deepseek-chat 模型名被拒绝

**发现时间**：2026-05-20

**错误日志**：

```
查询扩展失败，回退至原始查询: Unknown model 'deepseek-chat'. Please provide a valid OpenAI model name...
```

**根因**：LlamaIndex 的 `OpenAI` LLM 在客户端校验模型名，仅允许 OpenAI 官方模型列表

**修复**：`query_rewriter.py` 改用原生 `openai.AsyncOpenAI` SDK（非 LlamaIndex 封装），自定义 base_url 时不校验模型名

**状态**：✅ 已修复

---

## BUG-007: buildPipelineSVG 函数内重复 const 声明导致前端无法点击

**发现时间**：2026-05-20

**现象**：前端页面所有按钮无响应

**根因**：`index.html` 中 `buildPipelineSVG()` 函数内 `const genDetail` 声明两次（行 720 和 749），JS SyntaxError 阻断所有脚本

**修复**：删除重复声明

**状态**：✅ 已修复

---

## BUG-008: Milvus chunk 超过 VARCHAR 65535 限制

**发现时间**：2026-05-20

**错误日志**：

```
MilvusException: the length (207981) of 4th string exceeds max length (65535)
```

**根因**：

- `SemanticSplitterNodeParser` 对缺少自然断点的长段落产生超大 chunk（>200k 字符）
- Milvus VARCHAR 字段最大 65535 字符，插入时被拒

**修复**：

- `ingestion.py` 新增 `_split_oversized_nodes()`：超过 60000 字符的 chunk 自动按字符数强制拆分

**状态**：✅ 已修复
