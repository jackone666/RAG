"""Tests for src/engine/retrievers.py — tenant filter, ES search, fusion logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.engine.retrievers import (
    ElasticsearchKeywordRetriever,
    TenantAwareQueryFusionRetriever,
    _FusionAdapter,
    _build_tenant_filter,
)


class TestBuildTenantFilter:
    def test_creates_metadata_filters_with_tenant_id(self):
        filters = _build_tenant_filter("tenant_xyz")
        assert len(filters.filters) == 1
        assert filters.filters[0].key == "tenant_id"
        assert filters.filters[0].value == "tenant_xyz"
        assert filters.filters[0].operator == "=="


class TestFusionAdapter:
    @pytest.mark.asyncio
    async def test_aretrieve_returns_cached_nodes(self):
        nodes = [NodeWithScore(node=TextNode(text="test"), score=0.9)]
        adapter = _FusionAdapter(nodes)
        from llama_index.core.schema import QueryBundle

        result = await adapter._aretrieve(QueryBundle(query_str="test"))
        assert result == nodes

    def test_retrieve_returns_cached_nodes(self):
        nodes = [NodeWithScore(node=TextNode(text="test"), score=0.9)]
        adapter = _FusionAdapter(nodes)
        from llama_index.core.schema import QueryBundle

        result = adapter._retrieve(QueryBundle(query_str="test"))
        assert result == nodes


class TestElasticsearchKeywordRetrieverSearch:
    def test_search_returns_normalized_scores(self):
        retriever = ElasticsearchKeywordRetriever()
        mock_es = MagicMock()
        mock_response = {
            "hits": {
                "max_score": 2.0,
                "hits": [
                    {
                        "_id": "chunk_1",
                        "_score": 1.5,
                        "_source": {"text": "hello world", "tenant_id": "t1", "doc_id": "d1"},
                    },
                    {
                        "_id": "chunk_2",
                        "_score": 2.0,
                        "_source": {"text": "hello again", "tenant_id": "t1", "doc_id": "d1"},
                    },
                ],
            }
        }
        mock_es.search.return_value = mock_response
        retriever._client = mock_es

        results = retriever.search("t1", "hello", top_k=5)
        assert len(results) == 2
        assert results[0].score == pytest.approx(0.75)
        assert results[1].score == pytest.approx(1.0)

    def test_search_normalizes_when_max_score_is_none(self):
        retriever = ElasticsearchKeywordRetriever()
        mock_es = MagicMock()
        mock_response = {
            "hits": {
                "max_score": None,
                "hits": [
                    {
                        "_id": "chunk_1",
                        "_score": 5.0,
                        "_source": {"text": "hello", "tenant_id": "t1", "doc_id": "d1"},
                    },
                ],
            }
        }
        mock_es.search.return_value = mock_response
        retriever._client = mock_es

        results = retriever.search("t1", "hello")
        assert len(results) == 1
        assert results[0].score == pytest.approx(5.0)

    def test_search_handles_elasticsearch_error_gracefully(self):
        retriever = ElasticsearchKeywordRetriever()
        mock_es = MagicMock()
        mock_es.search.side_effect = Exception("connection refused")
        retriever._client = mock_es

        results = retriever.search("t1", "hello")
        assert results == []

    def test_search_handles_empty_hits(self):
        retriever = ElasticsearchKeywordRetriever()
        mock_es = MagicMock()
        mock_response = {"hits": {"hits": []}}
        mock_es.search.return_value = mock_response
        retriever._client = mock_es

        results = retriever.search("t1", "nonexistent")
        assert results == []


class TestTenantAwareQueryFusionRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_no_results_from_either_source(self):
        retriever = TenantAwareQueryFusionRetriever()
        with patch.object(retriever._keyword_retriever, "search", return_value=[]):
            with patch.object(retriever, "_get_vector_store") as mock_vs:
                mock_store = MagicMock()
                mock_store.aquery = AsyncMock(return_value=MagicMock(nodes=[]))
                mock_vs.return_value = mock_store

                results = await retriever.retrieve("query", "tenant_x")
                assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_only_vector_results(self):
        retriever = TenantAwareQueryFusionRetriever()
        vec_nodes = [NodeWithScore(node=TextNode(text="vec"), score=0.9)]

        with patch.object(retriever._keyword_retriever, "search", return_value=[]):
            with patch.object(retriever, "_get_vector_store") as mock_vs:
                mock_store = MagicMock()
                mock_store.aquery = AsyncMock(return_value=MagicMock(nodes=vec_nodes))
                mock_vs.return_value = mock_store

                results = await retriever.retrieve("query", "tenant_x")
                assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_only_keyword_results(self):
        retriever = TenantAwareQueryFusionRetriever()
        kw_nodes = [NodeWithScore(node=TextNode(text="kw"), score=0.8)]

        with patch.object(retriever._keyword_retriever, "search", return_value=kw_nodes):
            with patch.object(retriever, "_get_vector_store") as mock_vs:
                mock_store = MagicMock()
                mock_store.aquery = AsyncMock(return_value=MagicMock(nodes=[]))
                mock_vs.return_value = mock_store

                results = await retriever.retrieve("query", "tenant_x")
                assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_fusion_of_both_sources(self):
        retriever = TenantAwareQueryFusionRetriever()
        vec_nodes = [NodeWithScore(node=TextNode(text="vec"), score=0.9)]
        kw_nodes = [NodeWithScore(node=TextNode(text="kw"), score=0.8)]

        with patch.object(retriever._keyword_retriever, "search", return_value=kw_nodes):
            with patch.object(retriever, "_get_vector_store") as mock_vs:
                mock_store = MagicMock()
                mock_store.aquery = AsyncMock(return_value=MagicMock(nodes=vec_nodes))
                mock_vs.return_value = mock_store

                results = await retriever.retrieve("query", "tenant_x")
                assert len(results) > 0
