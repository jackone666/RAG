"""Tests for src/pipeline/sync_manager.py — document/tenant deletion, Milvus expressions."""

from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.sync_manager import SyncManager


class TestSyncManagerDeleteDocument:
    def test_delete_expr_includes_doc_id_only(self):
        sm = SyncManager()
        mock_collection = MagicMock()
        mock_collection.query.return_value = [{"id": 123}]

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = True
                with patch("src.pipeline.sync_manager.Collection", return_value=mock_collection):
                    # Patch the lazy import inside delete_document
                    with patch(
                            "src.engine.retrievers.ElasticsearchKeywordRetriever"
                    ) as mock_es_cls:
                        mock_es = MagicMock()
                        mock_es.delete_document.return_value = 0
                        mock_es_cls.return_value = mock_es
                        sm.delete_document("doc_123")

            call_expr = mock_collection.query.call_args[1]["expr"]
            assert 'doc_id == "doc_123"' in call_expr

    def test_delete_expr_includes_tenant_id_when_provided(self):
        sm = SyncManager()
        mock_collection = MagicMock()
        mock_collection.query.return_value = [{"id": 456}]

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = True
                with patch("src.pipeline.sync_manager.Collection", return_value=mock_collection):
                    with patch(
                            "src.engine.retrievers.ElasticsearchKeywordRetriever"
                    ) as mock_es_cls:
                        mock_es = MagicMock()
                        mock_es.delete_document.return_value = 0
                        mock_es_cls.return_value = mock_es
                        sm.delete_document("doc_123", tenant_id="tenant_x")

            call_expr = mock_collection.query.call_args[1]["expr"]
            assert 'doc_id == "doc_123"' in call_expr
            assert 'tenant_id == "tenant_x"' in call_expr

    def test_delete_skips_when_collection_missing(self):
        sm = SyncManager()

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = False
                with patch(
                        "src.engine.retrievers.ElasticsearchKeywordRetriever"
                ) as mock_es_cls:
                    mock_es = MagicMock()
                    mock_es.delete_document.return_value = 0
                    mock_es_cls.return_value = mock_es
                    result = sm.delete_document("doc_123")

            assert result == 0

    def test_delete_document_with_special_characters_in_id(self):
        """BUG TEST: doc_id with double quotes breaks Milvus expression parsing.

        The current code uses f-string interpolation:
            expr = f'doc_id == "{doc_id}"'

        If doc_id contains double quotes, the resulting expression is malformed:
            doc_id == "doc_with_"quote""  ← broken
        """
        sm = SyncManager()
        mock_collection = MagicMock()

        problematic_doc_id = 'doc_with_"quote"'

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = True
                with patch("src.pipeline.sync_manager.Collection", return_value=mock_collection):
                    with patch(
                            "src.engine.retrievers.ElasticsearchKeywordRetriever"
                    ) as mock_es_cls:
                        mock_es = MagicMock()
                        mock_es.delete_document.return_value = 0
                        mock_es_cls.return_value = mock_es
                        sm.delete_document(problematic_doc_id)

            call_expr = mock_collection.query.call_args[1]["expr"]
            # BUG: double-quotes in doc_id are NOT escaped in the Milvus expression.
            # The f-string `f'doc_id == "{doc_id}"'` produces a malformed expression
            # like: doc_id == "doc_with_"quote""
            # A correct implementation would escape: doc_id == "doc_with_\"quote\""
            assert '\\"' not in call_expr  # No escaping
            assert '"quote"' in call_expr  # Raw quotes embedded in expression

    def test_lazy_connection_established_once(self):
        sm = SyncManager()
        assert sm._connected is False

        with patch("src.pipeline.sync_manager.connections") as mock_conn:
            sm._ensure_connection()
            assert sm._connected is True
            mock_conn.connect.assert_called_once()

            sm._ensure_connection()
            mock_conn.connect.assert_called_once()


class TestSyncManagerDeleteTenant:
    def test_delete_tenant_cleans_both_milvus_and_es(self):
        sm = SyncManager()
        mock_collection = MagicMock()
        mock_collection.query.return_value = [{"id": 1}, {"id": 2}, {"id": 3}]

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = True
                with patch("src.pipeline.sync_manager.Collection", return_value=mock_collection):
                    with patch(
                            "src.engine.retrievers.ElasticsearchKeywordRetriever"
                    ) as mock_es_cls:
                        mock_es = MagicMock()
                        mock_es.delete_tenant.return_value = 5
                        mock_es_cls.return_value = mock_es
                        result = sm.delete_tenant("tenant_to_delete")

            assert result == 5

    def test_delete_tenant_empty_collection(self):
        sm = SyncManager()
        mock_collection = MagicMock()
        mock_collection.query.return_value = []

        with patch.object(sm, "_ensure_connection"):
            with patch("src.pipeline.sync_manager.utility") as mock_utility:
                mock_utility.has_collection.return_value = True
                with patch("src.pipeline.sync_manager.Collection", return_value=mock_collection):
                    with patch(
                            "src.engine.retrievers.ElasticsearchKeywordRetriever"
                    ) as mock_es_cls:
                        mock_es = MagicMock()
                        mock_es.delete_tenant.return_value = 0
                        mock_es_cls.return_value = mock_es
                        result = sm.delete_tenant("empty_tenant")

            assert result == 0
