"""Tests for src/api/documents.py — file validation, text extraction, API responses."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.documents import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE,
    _extract_text,
    _query_documents_for_tenant,
)


class TestFileValidation:
    def test_allowed_extensions_are_lowercase(self):
        for ext in ALLOWED_EXTENSIONS:
            assert ext == ext.lower()
            assert ext.startswith(".")

    def test_pdf_is_allowed(self):
        assert ".pdf" in ALLOWED_EXTENSIONS

    def test_docx_is_allowed(self):
        assert ".docx" in ALLOWED_EXTENSIONS

    def test_txt_is_allowed(self):
        assert ".txt" in ALLOWED_EXTENSIONS

    def test_max_file_size_is_50_mb(self):
        assert MAX_FILE_SIZE == 50 * 1024 * 1024


class TestTextExtraction:
    def test_extract_txt(self):
        content = "hello world".encode("utf-8")
        result = _extract_text("test.txt", content)
        assert result == "hello world"

    def test_extract_markdown(self):
        content = "# Title\n\nBody".encode("utf-8")
        result = _extract_text("readme.md", content)
        assert result == "# Title\n\nBody"

    def test_extract_unsupported_extension(self):
        with pytest.raises(HTTPException) as exc:
            _extract_text("image.png", b"fake")
        assert exc.value.status_code == 400

    def test_extract_pdf(self):
        pdf_content = (
            b"%PDF-1.4\n"
            b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
            b"xref\n"
            b"0 4\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"trailer<</Size 4/Root 1 0 R>>\n"
            b"startxref\n"
            b"190\n"
            b"%%EOF"
        )
        result = _extract_text("doc.pdf", pdf_content)
        assert isinstance(result, str)

    def test_extract_text_with_invalid_utf8(self):
        content = b"\xff\xfeinvalid utf8"
        result = _extract_text("file.txt", content)
        assert isinstance(result, str)


class TestQueryDocumentsForTenant:
    """Tests for _query_documents_for_tenant using MilvusClient API."""

    def test_returns_empty_list_when_collection_missing(self):
        with patch("pymilvus.MilvusClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.has_collection.return_value = False
            mock_client_cls.return_value = mock_client
            result = _query_documents_for_tenant("tenant_x")
            assert result == []

    def test_returns_documents_when_collection_exists(self):
        with patch("pymilvus.MilvusClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_client.query.return_value = [
                {"doc_id": "doc_a"},
                {"doc_id": "doc_a"},
                {"doc_id": "doc_b"},
            ]
            mock_client_cls.return_value = mock_client
            result = _query_documents_for_tenant("tenant_x")

            assert len(result) == 2
            doc_ids = {d.doc_id for d in result}
            assert doc_ids == {"doc_a", "doc_b"}
            doc_a = next(d for d in result if d.doc_id == "doc_a")
            assert doc_a.chunk_count == 2

    def test_handles_collection_exception_gracefully(self):
        with patch("pymilvus.MilvusClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.has_collection.side_effect = Exception("connection error")
            mock_client_cls.return_value = mock_client
            result = _query_documents_for_tenant("tenant_x")
            assert result == []

    def test_missing_doc_id_defaults_to_unknown(self):
        with patch("pymilvus.MilvusClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.has_collection.return_value = True
            mock_client.query.return_value = [{}]
            mock_client_cls.return_value = mock_client
            result = _query_documents_for_tenant("tenant_x")
            assert len(result) == 1
            assert result[0].doc_id == "unknown"
            assert result[0].chunk_count == 1
