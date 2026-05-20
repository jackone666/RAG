# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project

- **IntelliLens-MCP** (v2.1.0) — enterprise-grade intelligent data governance & Agentic RAG system
- **Language**: Python 3.10+
- **Virtual environment**: `.venv/`

## Build / Test / Run

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py           # starts FastAPI + MCP server on 0.0.0.0:8000
```

API docs available at `http://localhost:8000/docs` after startup.

## Architecture Overview

```
HTTP Request
  → tenant_context_middleware (JWT / Header auth → request.state.tenant_context)
  → rate_limit_dependency (sliding-window per tenant)
  → POST /v1/query
    → TenantAwareQueryFusionRetriever (vector + keyword, MetadataFilters on tenant_id)
    → RAGQueryEngine.rerank (LLMRerank → top-5)
    → RAGQueryEngine.query (primary_model → fallback_model → FALLBACK_RESPONSE)
    → BackgroundTasks → AsyncEvaluator.evaluate (tenacity retry → bad_cases.jsonl)
```

### Five Production Defense Lines

1. **RBAC (multi-tenant isolation)**: `middleware/auth.py` extracts tenant context; `engine/retrievers.py` pushes `MetadataFilters(tenant_id=...)` to the storage layer
2. **Document lifecycle sync**: `pipeline/sync_manager.py` — `delete_document(doc_id)` purges chunks from Milvus; `delete_tenant(tenant_id)` handles GDPR offboarding
3. **Rate limiting + LLM circuit breaker**: `middleware/rate_limiter.py` (sliding window); `engine/query_engine.py` (primary → fallback → graceful degradation)
4. **LLMOps observability**: `observability/tracer.py` — Langfuse global CallbackManager for retrieval latency, rerank, prompt, and token tracing
5. **Async LLM judge**: `evaluation/evaluator.py` — `FaithfulnessEvaluator` via `BackgroundTasks`, with `tenacity` exponential backoff; `passing=False` or `score < 0.8` auto-saved to `data/bad_cases.jsonl`

### Key Modules

| Module | Purpose |
|---|---|
| `src/config/settings.py` | Pydantic Settings — all env vars parsed here |
| `src/middleware/auth.py` | JWT decode + `RBACGuard` dependency |
| `src/middleware/rate_limiter.py` | `InMemoryRateLimiter` with periodic GC |
| `src/observability/tracer.py` | Langfuse `CallbackManager` init |
| `src/pipeline/ingestion.py` | `SemanticSplitterNodeParser` + `tenant_id`/`doc_id` metadata injection |
| `src/pipeline/sync_manager.py` | `delete_document()` / `delete_tenant()` via pymilvus |
| `src/engine/retrievers.py` | `TenantAwareQueryFusionRetriever` (vector + keyword, RBAC filters) |
| `src/engine/query_engine.py` | `RAGQueryEngine` with rerank + primary/fallback LLM chain |
| `src/mcp_server/server.py` | `FastMCP` singleton |
| `src/mcp_server/tools.py` | `search_enterprise_knowledge(query, tenant_id)` — MCP tool |
| `src/evaluation/evaluator.py` | `AsyncEvaluator` — `BackgroundTasks`-driven faithfulness judge |

### Environment Variables

Copy `.env.example` to `.env` and fill in required values. Critical: `OPENAI_API_KEY` is mandatory.
