"""Tests for src/pipeline/ingestion.py — metadata injection, pipeline flow."""

from unittest.mock import patch

import pytest
from llama_index.core.schema import Document, TextNode

from src.pipeline.ingestion import TenantAwareIngestionPipeline, _compute_content_hash


class TestMetadataInjection:
    def test_inject_metadata_adds_tenant_and_doc_id(self):
        pipeline = TenantAwareIngestionPipeline.__new__(TenantAwareIngestionPipeline)
        nodes = [
            TextNode(text="chunk 1", metadata={}),
            TextNode(text="chunk 2", metadata={"existing": "val"}),
        ]

        result = pipeline._inject_metadata(nodes, "tenant_x", "doc_123", "abc123hash")

        for node in result:
            assert node.metadata["tenant_id"] == "tenant_x"
            assert node.metadata["doc_id"] == "doc_123"
            assert node.metadata["content_hash"] == "abc123hash"

    def test_inject_metadata_replaces_metadata_to_avoid_milvus_overflow(self):
        """Metadata 会被完全替换以防止 SemanticSplitter 内部字段溢出 Milvus VARCHAR 限制。"""
        pipeline = TenantAwareIngestionPipeline.__new__(TenantAwareIngestionPipeline)
        node = TextNode(text="chunk", metadata={"custom_key": "custom_val"})
        result = pipeline._inject_metadata([node], "tenant_x", "doc_1", "hash456")

        # 旧字段被清除，仅保留多租户必需的字段
        assert "custom_key" not in result[0].metadata
        assert result[0].metadata["tenant_id"] == "tenant_x"
        assert result[0].metadata["doc_id"] == "doc_1"
        assert result[0].metadata["content_hash"] == "hash456"
        assert len(result[0].metadata) == 3

    def test_ingest_text_creates_document_with_metadata(self):
        with patch("src.pipeline.ingestion.TenantAwareIngestionPipeline.ingest") as mock_ingest:
            mock_ingest.return_value = [TextNode(text="chunk")]
            pipeline = TenantAwareIngestionPipeline.__new__(TenantAwareIngestionPipeline)

            import asyncio

            async def run():
                return await pipeline.ingest_text("hello world", "tenant_x", "doc_1")

            result = asyncio.run(run())
            assert len(result) == 1

            called_docs = mock_ingest.call_args[0][0]
            expected_hash = _compute_content_hash("hello world")
            assert called_docs[0].text == "hello world"
            assert called_docs[0].metadata["tenant_id"] == "tenant_x"
            assert called_docs[0].metadata["doc_id"] == "doc_1"
            assert called_docs[0].metadata["content_hash"] == expected_hash

    def test_compute_content_hash_deterministic(self):
        h1 = _compute_content_hash("hello world")
        h2 = _compute_content_hash("hello world")
        h3 = _compute_content_hash("different text")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64  # SHA256 hex
