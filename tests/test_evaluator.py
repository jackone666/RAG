"""Tests for src/evaluation/evaluator.py — 6 evaluation metrics."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src.evaluation.evaluator import AsyncEvaluator


@pytest.fixture
def evaluator():
    return AsyncEvaluator()


class TestAsyncEvaluator:
    @pytest.mark.asyncio
    async def test_all_scores_good_passes(self, evaluator):
        """高分通过时不存坏例。"""
        with patch.object(evaluator, "_evaluate_retrieval", AsyncMock(return_value={
            "precision": 0.9, "recall": 0.85, "mrr": 0.8, "hit_rate": 1.0,
        })):
            with patch.object(evaluator, "_evaluate_generation", AsyncMock(return_value={
                "faithfulness": 0.95, "faithfulness_reason": "ok",
                "relevance": 0.9, "relevance_reason": "ok",
            })):
                with patch("src.evaluation.evaluator.settings") as ms:
                    ms.eval_score_threshold = 0.8
                    result = await evaluator.evaluate("q", ["c"], "answer", "t1")

        assert result is not None
        assert result["passing"] is True
        assert result["precision"] == 0.9
        assert result["faithfulness"] == 0.95

    @pytest.mark.asyncio
    async def test_low_scores_fails(self, evaluator):
        """低分不通过。"""
        with patch.object(evaluator, "_evaluate_retrieval", AsyncMock(return_value={
            "precision": 0.3, "recall": 0.2, "mrr": 0.1, "hit_rate": 0.0,
        })):
            with patch.object(evaluator, "_evaluate_generation", AsyncMock(return_value={
                "faithfulness": 0.4, "faithfulness_reason": "", "relevance": 0.5, "relevance_reason": "",
            })):
                with patch("src.evaluation.evaluator.settings") as ms:
                    ms.eval_score_threshold = 0.8
                    result = await evaluator.evaluate("q", ["c"], "answer", "t1")

        assert result["passing"] is False

    @pytest.mark.asyncio
    async def test_skip_empty_answer(self, evaluator):
        """空回答跳过评估。"""
        result = await evaluator.evaluate("q", ["c"], "", "t1")
        assert result is None

    @pytest.mark.asyncio
    async def test_six_metrics_present(self, evaluator):
        """6 个指标全部存在。"""
        with patch.object(evaluator, "_evaluate_retrieval", AsyncMock(return_value={
            "precision": 0.8, "recall": 0.7, "mrr": 0.6, "hit_rate": 1.0,
        })):
            with patch.object(evaluator, "_evaluate_generation", AsyncMock(return_value={
                "faithfulness": 0.9, "faithfulness_reason": "", "relevance": 0.85, "relevance_reason": "",
            })):
                with patch("src.evaluation.evaluator.settings") as ms:
                    ms.eval_score_threshold = 0.8
                    result = await evaluator.evaluate("q", ["c"], "a", "t1")

        for k in ["precision", "recall", "mrr", "hit_rate", "faithfulness", "relevance"]:
            assert k in result
            assert 0 <= result[k] <= 1.0
