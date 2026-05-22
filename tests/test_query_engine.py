"""Tests for src/engine/query_engine.py — generation, fallback, rerank, edge cases."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.engine.query_engine import FALLBACK_RESPONSE, RAGQueryEngine


def make_node(text: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=text), score=0.9)


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
    total_tokens = 150


class FakeChoice:
    def __init__(self, content):
        self.message = MagicMock(content=content)


class FakeResponse:
    def __init__(self, text):
        self.choices = [FakeChoice(text)]
        self.usage = FakeUsage()


class TestRAGQueryEngineFallback:
    @pytest.mark.asyncio
    async def test_primary_model_success(self):
        engine = RAGQueryEngine()
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Answer from primary")
        )

        with patch.object(engine, "_get_primary_llm", return_value=mock_client):
            result = await engine.query([make_node("context")], "question?")
            assert result == "Answer from primary"

    @pytest.mark.asyncio
    async def test_fallback_when_primary_fails(self):
        engine = RAGQueryEngine()
        mock_primary = MagicMock()
        mock_primary.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        mock_fallback = MagicMock()
        mock_fallback.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Answer from fallback")
        )

        with patch.object(engine, "_get_primary_llm", return_value=mock_primary):
            with patch.object(engine, "_get_fallback_llm", return_value=mock_fallback):
                result = await engine.query([make_node("context")], "question?")
                assert result == "Answer from fallback"

    @pytest.mark.asyncio
    async def test_fallback_uses_fallback_model_name(self):
        engine = RAGQueryEngine()
        mock_primary = MagicMock()
        mock_primary.chat.completions.create = AsyncMock(side_effect=Exception("API error"))
        mock_fallback = MagicMock()
        mock_fallback.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Answer from fallback")
        )

        with patch.object(engine, "_get_primary_llm", return_value=mock_primary):
            with patch.object(engine, "_get_fallback_llm", return_value=mock_fallback):
                with patch("src.engine.query_engine.settings") as mock_settings:
                    mock_settings.primary_model = "primary-model"
                    mock_settings.fallback_model = "fallback-model"
                    await engine.query([make_node("context")], "question?")

        assert mock_fallback.chat.completions.create.call_args.kwargs["model"] == "fallback-model"

    @pytest.mark.asyncio
    async def test_degraded_response_when_both_fail(self):
        engine = RAGQueryEngine()
        mock_primary = MagicMock()
        mock_primary.chat.completions.create = AsyncMock(side_effect=Exception("Primary down"))
        mock_fallback = MagicMock()
        mock_fallback.chat.completions.create = AsyncMock(side_effect=Exception("Fallback down"))

        with patch.object(engine, "_get_primary_llm", return_value=mock_primary):
            with patch.object(engine, "_get_fallback_llm", return_value=mock_fallback):
                result = await engine.query([make_node("context")], "question?")
                assert result == FALLBACK_RESPONSE

    @pytest.mark.asyncio
    async def test_fallback_not_called_when_primary_succeeds(self):
        engine = RAGQueryEngine()
        mock_primary = MagicMock()
        mock_primary.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Success")
        )
        mock_fallback = MagicMock()

        with patch.object(engine, "_get_primary_llm", return_value=mock_primary):
            with patch.object(engine, "_get_fallback_llm", return_value=mock_fallback):
                await engine.query([make_node("context")], "question?")
                mock_fallback.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_lazy_initialization(self):
        engine = RAGQueryEngine()
        assert engine._primary_llm is None
        assert engine._fallback_llm is None

    @pytest.mark.asyncio
    async def test_generate_prompt_contains_context_and_query(self):
        engine = RAGQueryEngine()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Answer")
        )

        with patch.object(engine, "_get_primary_llm", return_value=mock_client):
            await engine.query([make_node("contextual info")], "what is this?")
            prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
            assert "contextual info" in prompt
            assert "what is this?" in prompt
            assert "企业智能知识助手" in prompt

    @pytest.mark.asyncio
    async def test_context_is_numbered_and_compressed_for_generation(self):
        engine = RAGQueryEngine()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=FakeResponse("Answer")
        )
        noisy_text = (
                "无关介绍内容。" * 300
                + "退款政策规定用户可以在七天内申请退款。"
                + "更多无关内容。" * 300
        )

        with patch.object(engine, "_get_primary_llm", return_value=mock_client):
            await engine.query([make_node(noisy_text)], "退款政策是什么？")

        prompt = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "[文档1]" in prompt
        assert "退款政策规定用户可以在七天内申请退款" in prompt
        assert len(prompt) < len(noisy_text)


class TestRAGQueryEngineRerank:
    @pytest.mark.asyncio
    async def test_rerank_returns_top_n(self):
        engine = RAGQueryEngine()
        nodes = [make_node(f"document {i}") for i in range(10)]
        mock_resp = {"results": [{"index": 3, "relevance_score": 0.95},
                                 {"index": 0, "relevance_score": 0.9},
                                 {"index": 7, "relevance_score": 0.85},
                                 {"index": 1, "relevance_score": 0.8},
                                 {"index": 5, "relevance_score": 0.75}]}

        with patch.object(engine, "_call_rerank_sync", return_value=mock_resp):
            result = await engine.rerank("query", nodes, top_n=5)
            assert len(result) == 5
            assert result[0].score == 0.95

    @pytest.mark.asyncio
    async def test_rerank_fallback_on_error(self):
        engine = RAGQueryEngine()
        nodes = [make_node(f"doc {i}") for i in range(10)]

        with patch.object(engine, "_call_rerank_api", side_effect=Exception("API error")):
            result = await engine.rerank("query", nodes, top_n=5)
            assert len(result) == 5

    @pytest.mark.asyncio
    async def test_rerank_empty_nodes(self):
        engine = RAGQueryEngine()
        result = await engine.rerank("query", [], top_n=5)
        assert result == []
