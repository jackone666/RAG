# Bug Fix & Test Report — IntelliLens-MCP v2.1.0

> Generated: 2026-05-20 | Tests: 132 passed / 0 failed

---

## Bugs Found & Fixed

### 1. Duplicate `_escape_milvus_expr` function (Code Quality)

**File**: `src/api/documents.py` + `src/pipeline/sync_manager.py`

Both files independently defined the same `_escape_milvus_expr()` helper function.
Any fix applied to one copy would leave the other stale — a maintenance risk.

**Fix**: Extracted to `src/utils/helpers.py` as a shared utility. Both modules now import
from `src.utils.helpers import _escape_milvus_expr`.

**New files**:
- `src/utils/__init__.py` — package init, re-exports the helper
- `src/utils/helpers.py` — canonical implementation

---

### 2. Import-time Milvus Connection Blocking Startup (Critical)

**File**: `src/pipeline/ingestion.py` — `TenantAwareIngestionPipeline.__init__()`

The constructor called `MilvusVectorStore(...)` which immediately opens a gRPC connection
to Milvus. Since `documents.py` creates a module-level `ingestion_pipeline = TenantAwareIngestionPipeline()`,
merely importing the application triggers a network connection. If Milvus is down or
unreachable, the entire application fails to start.

**Stack trace before fix**:
```
pymilvus.exceptions.MilvusException: Fail connecting to server on localhost:19530
```

**Fix**: Refactored to lazy-initialization pattern — `MilvusVectorStore` and
`IngestionPipeline` are now created on first `ingest()` call via `_get_pipeline()`,
not at `__init__` time. The embedding model and splitter are still created at init
(local, no network), while the Milvus connection is deferred.

**Change**:
```python
# Before (connects at init)
def __init__(self):
    self.vector_store = MilvusVectorStore(...)
    self.pipeline = IngestionPipeline(...)

# After (connects on first use)
def __init__(self):
    self._vector_store = None
    self._pipeline = None

def _get_pipeline(self):
    if self._pipeline is None:
        self._vector_store = MilvusVectorStore(...)
        self._pipeline = IngestionPipeline(...)
    return self._pipeline
```

---

### 3. ES Score Normalization — Falsy `max_score` Check (Minor)

**File**: `src/engine/retrievers.py` — `ElasticsearchKeywordRetriever.search()`

The expression `resp.get("hits", {}).get("max_score") or 1` evaluates to `1` when
`max_score` is `0` or `None`. While `max_score == 0` is extremely rare in practice
(all hits would have score 0), the logic was semantically incorrect.

**Fix**: Replaced with explicit `None`/`0` check:
```python
# Before
max_score = resp.get("hits", {}).get("max_score") or 1

# After
max_score = resp.get("hits", {}).get("max_score")
if max_score is None or max_score == 0:
    max_score = 1
```

Also removed the now-redundant `if max_score else 0` guard on the division line.

---

## New Test Coverage

### Test files added (5 new files, 54 new tests)

| File | Tests | Coverage |
|---|---|---|
| `tests/test_helpers.py` | 9 | `_escape_milvus_expr` — normal chars, quotes, SQL injection, Unicode, empty strings |
| `tests/test_config.py` | 24 | `Settings` defaults, env overrides, required-field validation |
| `tests/test_main_api.py` | 13 | All FastAPI endpoints: `GET /`, `/health`, `/app`, `/v1/tenants/me`, `POST /v1/query`, `/docs`, `/openapi.json` |
| `tests/test_mcp_tools.py` | 3 | MCP tool `search_enterprise_knowledge` — happy path, no results, pipeline args |
| `tests/test_tracer.py` | 5 | `init_observability` — public/secret key skip, ImportError handling, successful init |

### Test file changes

| File | Change |
|---|---|
| `tests/test_config.py` | All `Settings()` calls now pass `_env_file=None` to prevent `.env` file interference |

---

## Full Test Suite Results

```
132 passed, 27 warnings in 45.90s
```

### Coverage by module

| Module | Test File | Tests |
|---|---|---|
| `src/middleware/auth.py` | `test_auth.py` | 16 |
| `src/config/settings.py` | `test_config.py` | 24 |
| `src/api/documents.py` | `test_documents.py` | 14 |
| `src/evaluation/evaluator.py` | `test_evaluator.py` | 6 |
| `src/utils/helpers.py` | `test_helpers.py` | 9 |
| `src/pipeline/ingestion.py` | `test_ingestion.py` | 3 |
| `main.py` (all endpoints) | `test_main_api.py` | 13 |
| `src/mcp_server/tools.py` | `test_mcp_tools.py` | 3 |
| `src/engine/query_engine.py` | `test_query_engine.py` | 9 |
| `src/middleware/rate_limiter.py` | `test_rate_limiter.py` | 9 |
| `src/engine/retrievers.py` | `test_retrievers.py` | 12 |
| `src/pipeline/sync_manager.py` | `test_sync_manager.py` | 9 |
| `src/observability/tracer.py` | `test_tracer.py` | 5 |
| **Total** | | **132** |

---

## API Endpoint Test Coverage

Every endpoint in the application is covered by at least one test:

| Method | Endpoint | Tests |
|---|---|---|
| `GET` | `/` | Root info, JSON content-type |
| `GET` | `/health` | Health check returns `{"status": "ok"}` |
| `GET` | `/app` | Redirects to `/static/index.html` |
| `GET` | `/v1/tenants/me` | Header auth, JWT auth, default role |
| `POST` | `/v1/query` | Happy path, no results, background eval, tenant context passthrough, default tenant |
| `POST` | `/v1/documents/upload` | File validation, text extraction (covered via unit tests) |
| `GET` | `/v1/documents` | Document listing (covered via unit tests) |
| `DELETE` | `/v1/documents/{doc_id}` | Document deletion (covered via unit tests) |
| `GET` | `/docs` | OpenAPI docs accessible |
| `GET` | `/openapi.json` | Schema includes all expected paths |
