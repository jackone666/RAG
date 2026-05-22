"""Tests for src/mcp_server/tools.py — MCP tool search_enterprise_knowledge."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import BEFORE any mcp mocking — the @mcp.tool() decorator runs at import time
from src.mcp_server.tools import search_enterprise_knowledge


class TestSearchEnterpriseKnowledge:
    @pytest.mark.asyncio
    async def test_returns_not_found_when_no_nodes(self):
        """Test the tool function directly by mocking its dependencies."""
        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=[])

        mock_engine = MagicMock()
        mock_engine.query = AsyncMock()

        with patch(
                "src.mcp_server.tools._retriever", mock_retriever
        ), patch(
            "src.mcp_server.tools._query_engine", mock_engine
        ):
            result = await search_enterprise_knowledge(
                query="missing", tenant_id="tenant_xyz"
            )

            assert "未找到相关文档" in result
            assert "tenant_xyz" in result
            mock_engine.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_answer_when_nodes_found(self):
        """Test full pipeline: retrieve → rerank → generate."""
        nodes = [MagicMock()]
        reranked = [MagicMock()]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=nodes)

        mock_engine = MagicMock()
        mock_engine.rerank = AsyncMock(return_value=reranked)
        mock_engine.query = AsyncMock(return_value="Answer from MCP")

        with patch(
                "src.mcp_server.tools._retriever", mock_retriever
        ), patch(
            "src.mcp_server.tools._query_engine", mock_engine
        ):
            result = await search_enterprise_knowledge(
                query="test query", tenant_id="tenant_abc"
            )

            assert result == "Answer from MCP"
            mock_retriever.retrieve.assert_called_once_with(
                query="test query", tenant_id="tenant_abc"
            )
            mock_engine.rerank.assert_called_once_with("test query", nodes)
            mock_engine.query.assert_called_once_with(reranked, "test query")

    @pytest.mark.asyncio
    async def test_rerank_and_generate_called_with_correct_args(self):
        """Verify the data flow through each pipeline stage."""
        nodes = [MagicMock(), MagicMock()]
        reranked = [MagicMock()]

        mock_retriever = MagicMock()
        mock_retriever.retrieve = AsyncMock(return_value=nodes)

        mock_engine = MagicMock()
        mock_engine.rerank = AsyncMock(return_value=reranked)
        mock_engine.query = AsyncMock(return_value="Generated response")

        with patch(
                "src.mcp_server.tools._retriever", mock_retriever
        ), patch(
            "src.mcp_server.tools._query_engine", mock_engine
        ):
            result = await search_enterprise_knowledge(query="q", tenant_id="t1")

            mock_engine.rerank.assert_called_once_with("q", nodes)
            mock_engine.query.assert_called_once_with(reranked, "q")
            assert result == "Generated response"
