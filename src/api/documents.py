"""
文档管理 API 路由 — 上传、列表、删除

提供文档全生命周期管理的 REST 端点：
- POST /v1/documents/upload — 上传并入库文档（含重复检测）
- GET  /v1/documents        — 列出当前租户的所有文档
- DELETE /v1/documents/{id} — 删除指定文档及其所有 Chunk
"""
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from src.config.settings import settings
from src.middleware.auth import RBACGuard
from src.middleware.rate_limiter import rate_limit_dependency
from src.pipeline.doc_store import doc_store
from src.pipeline.ingestion import TenantAwareIngestionPipeline, _compute_content_hash
from src.pipeline.preprocessing import preprocess_document
from src.pipeline.sync_manager import SyncManager
from src.utils.helpers import _escape_milvus_expr  # noqa: F811 — shared, re-exported for backward compat

# pymilvus MilvusClient 在函数内按需导入，避免启动时连接

router = APIRouter(prefix="/v1/documents", tags=["documents"])

ingestion_pipeline = TenantAwareIngestionPipeline()
sync_manager = SyncManager()

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


# ── Pydantic models ──────────────────────────────────────────────


class PipelineStep(BaseModel):
    step: str
    status: str  # "running" | "done" | "error"
    detail: str = ""
    elapsed_ms: float = 0.0


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    status: str
    duplicate: bool = False
    existing_doc_id: str | None = None
    steps: list[PipelineStep] = []


class DocumentItem(BaseModel):
    doc_id: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total: int


class BatchUploadResponse(BaseModel):
    """批量上传响应——每个文件一条结果。"""

    items: list[UploadResponse]
    total: int
    success: int
    failed: int
    duplicate: int
    folder: str = ""  # 上传来源文件夹（若有）


class DeleteResponse(BaseModel):
    doc_id: str
    deleted_chunks: int


# ── Text extraction helpers ──────────────────────────────────────


def _extract_text_from_pdf(content: bytes) -> str:
    doc = fitz.open(stream=content, filetype="pdf")
    parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(parts)


def _extract_text_from_docx(content: bytes) -> str:
    import io

    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_text(filename: str, content: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="replace")
    elif ext == ".pdf":
        return _extract_text_from_pdf(content)
    elif ext == ".docx":
        return _extract_text_from_docx(content)
    raise HTTPException(400, f"Unsupported file type: {ext}")


# ── Milvus doc listing helper ────────────────────────────────────


def _query_documents_for_tenant(tenant_id: str) -> list[DocumentItem]:
    """查询 Milvus 中某租户的所有唯一文档及其 Chunk 数量。
    使用 MilvusClient API（新版，与 LlamaIndex 共用连接）。"""
    from pymilvus import MilvusClient

    collection_name = settings.milvus_collection_name
    client = MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token or None,
    )

    try:
        if not client.has_collection(collection_name):
            return []
        results = client.query(
            collection_name=collection_name,
            filter=f'tenant_id == "{_escape_milvus_expr(tenant_id)}"',
            output_fields=["doc_id"],
            limit=10000,
        )
    except Exception:
        return []

    counts: dict[str, int] = {}
    for r in results:
        doc_id = r.get("doc_id", "unknown")
        counts[doc_id] = counts.get(doc_id, 0) + 1

    return [DocumentItem(doc_id=k, chunk_count=v) for k, v in counts.items()]


# ── Endpoints ────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=UploadResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(RBACGuard("editor"))],
)
async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        doc_id: str | None = None,
):
    """上传文档并执行入库管道（重复检测 → 分块 → 元数据注入 → 向量化 → 写入 Milvus）。

    重复检测机制：
    1. 对提取的文本内容计算 SHA256 哈希
    2. 查询 Milvus 中是否已有相同 content_hash 的 chunk
    3. 存在 → 返回 duplicate=true 及已存在文档的 doc_id
    4. 不存在 → 正常入库，并记录 content_hash 供后续去重

    需要 editor 及以上角色。
    """
    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext}，允许: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"文件过大（最大 50MB），当前: {len(content) / 1024 / 1024:.1f}MB")

    try:
        text = _extract_text(file.filename, content)
    except Exception as e:
        raise HTTPException(400, f"文件解析失败: {e}")

    if not text.strip():
        raise HTTPException(400, "提取到的文本为空")

    # ── 文档预处理（清洗 + 结构解析）──
    text, doc_meta = preprocess_document(text)
    if not text.strip():
        raise HTTPException(400, "预处理后文本为空")

    # ── 重复文档检测 ──
    content_hash = _compute_content_hash(text)
    duplicate = ingestion_pipeline.check_duplicate(tenant_id, content_hash)

    if duplicate:
        return UploadResponse(
            doc_id=duplicate["doc_id"],
            filename=file.filename,
            chunk_count=duplicate["chunk_count"],
            status="duplicate",
            duplicate=True,
            existing_doc_id=duplicate["doc_id"],
        )

    # ── 正常入库（分阶段计时） ──
    import time as _time
    steps = []
    doc_id = doc_id or uuid.uuid4().hex

    # Step 1: MinIO 对象存储
    t0 = _time.monotonic()
    try:
        doc_store.store_document(tenant_id, doc_id, file.filename, content)
        steps.append(PipelineStep(step="对象存储 (MinIO)", status="done",
                                  detail=f"{len(content) / 1024:.0f} KB",
                                  elapsed_ms=round((_time.monotonic() - t0) * 1000)))
    except Exception as e:
        steps.append(PipelineStep(step="对象存储 (MinIO)", status="error",
                                  detail=str(e)[:100], elapsed_ms=round((_time.monotonic() - t0) * 1000)))
        raise HTTPException(500, f"对象存储写入失败: {e}")

    # Step 2: 文本分块
    t0 = _time.monotonic()
    nodes = await ingestion_pipeline._embed_and_split(text, tenant_id, doc_id, content_hash)
    steps.append(PipelineStep(step="文本分块 (SentenceSplitter)", status="done",
                              detail=f"{len(nodes)} chunks (语义)", elapsed_ms=round((_time.monotonic() - t0) * 1000)))

    # Step 3: 远程嵌入
    t0 = _time.monotonic()
    nodes = await ingestion_pipeline._do_embed(nodes)
    steps.append(PipelineStep(step="远程嵌入 (BAAI/bge-m3)", status="done",
                              detail=f"{len(nodes)} vectors × 1024d",
                              elapsed_ms=round((_time.monotonic() - t0) * 1000)))

    # Step 4: 写入 Milvus
    t0 = _time.monotonic()
    nodes = await ingestion_pipeline._do_milvus_insert(nodes)
    steps.append(PipelineStep(step="向量入库 (Milvus)", status="done",
                              detail=f"{len(nodes)} rows", elapsed_ms=round((_time.monotonic() - t0) * 1000)))

    # Step 5: 写入 ES（后台）
    t0 = _time.monotonic()
    from src.pipeline.ingestion import _index_es_background
    import asyncio as _asyncio
    loop = _asyncio.get_event_loop()
    loop.run_in_executor(None, _index_es_background, tenant_id, doc_id, nodes)
    steps.append(PipelineStep(step="关键词索引 (ES)", status="done",
                              detail="后台写入", elapsed_ms=round((_time.monotonic() - t0) * 1000)))

    return UploadResponse(
        doc_id=doc_id, filename=file.filename, chunk_count=len(nodes),
        status="success", steps=steps,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(request: Request):
    """列出当前租户下的所有文档（按 doc_id 去重，统计 Chunk 数）。"""
    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]

    documents = _query_documents_for_tenant(tenant_id)
    return DocumentListResponse(documents=documents, total=len(documents))


@router.post(
    "/upload-batch",
    response_model=BatchUploadResponse,
    dependencies=[Depends(rate_limit_dependency), Depends(RBACGuard("editor"))],
)
async def upload_documents_batch(
        request: Request,
        files: list[UploadFile] = File(...),
        folder: str = "",
):
    """批量上传文档——支持多文件、递归文件夹导入。

    前端通过文件选择器多选文件或拖入文件夹后，将文件列表
    一次性发送至本端点。可选 `folder` 参数标记来源目录。

    返回每个文件的处理结果汇总。
    """
    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]

    import asyncio as _asyncio
    import uuid as _uuid

    async def _process_one(f: UploadFile) -> UploadResponse:
        ext = Path(f.filename).suffix.lower() if f.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            return UploadResponse(
                doc_id="", filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="文件类型检查", status="error",
                                                    detail=f"不支持: {ext}")],
            )
        content = await f.read()
        if len(content) > MAX_FILE_SIZE:
            return UploadResponse(
                doc_id="", filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="文件大小检查", status="error",
                                                    detail=f"过大: {len(content) / 1024 / 1024:.1f}MB")],
            )
        try:
            text = _extract_text(f.filename, content)
        except Exception as e:
            return UploadResponse(
                doc_id="", filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="文件解析", status="error",
                                                    detail=str(e)[:100])],
            )
        if not text.strip():
            return UploadResponse(
                doc_id="", filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="文本提取", status="error",
                                                    detail="提取到的文本为空")],
            )

        # 预处理
        text, _doc_meta = preprocess_document(text)
        if not text.strip():
            return UploadResponse(
                doc_id="", filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="预处理", status="error",
                                                    detail="预处理后文本为空")],
            )

        # 重复检测
        content_hash = _compute_content_hash(text)
        duplicate = ingestion_pipeline.check_duplicate(tenant_id, content_hash)
        if duplicate:
            return UploadResponse(
                doc_id=duplicate["doc_id"], filename=f.filename or "unknown",
                chunk_count=duplicate["chunk_count"], status="duplicate",
                duplicate=True, existing_doc_id=duplicate["doc_id"],
            )

        doc_id = _uuid.uuid4().hex
        try:
            doc_store.store_document(tenant_id, doc_id, f.filename, content)
        except Exception as e:
            return UploadResponse(
                doc_id=doc_id, filename=f.filename or "unknown", chunk_count=0,
                status="error", steps=[PipelineStep(step="对象存储", status="error",
                                                    detail=str(e)[:100])],
            )

        nodes = await ingestion_pipeline._embed_and_split(text, tenant_id, doc_id, content_hash)
        nodes = await ingestion_pipeline._do_embed(nodes)
        nodes = await ingestion_pipeline._do_milvus_insert(nodes)

        # ES 后台写入
        loop = _asyncio.get_event_loop()
        from src.pipeline.ingestion import _index_es_background
        loop.run_in_executor(None, _index_es_background, tenant_id, doc_id, nodes)

        return UploadResponse(
            doc_id=doc_id, filename=f.filename or "unknown",
            chunk_count=len(nodes), status="success",
        )

    tasks = [_process_one(f) for f in files]
    results = await _asyncio.gather(*tasks, return_exceptions=True)

    items: list[UploadResponse] = []
    success = failed = duplicate = 0
    for r in results:
        if isinstance(r, Exception):
            items.append(UploadResponse(
                doc_id="", filename="", chunk_count=0,
                status="error", steps=[PipelineStep(step="处理异常", status="error",
                                                    detail=str(r)[:100])],
            ))
            failed += 1
        else:
            items.append(r)
            if r.status == "success":
                success += 1
            elif r.status == "duplicate":
                duplicate += 1
            else:
                failed += 1

    return BatchUploadResponse(
        items=items, total=len(files),
        success=success, failed=failed, duplicate=duplicate,
        folder=folder,
    )


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(RBACGuard("editor"))],
)
async def delete_document(request: Request, doc_id: str):
    """删除指定文档及其在向量库中的所有 Chunk。

    需要 editor 及以上角色。
    """
    tenant_ctx: dict = request.state.tenant_context
    tenant_id = tenant_ctx["tenant_id"]

    deleted = sync_manager.delete_document(doc_id, tenant_id)
    return DeleteResponse(doc_id=doc_id, deleted_chunks=deleted)

# Tenant info is registered at /v1/tenants/me via main.py
