"""Tests for src/observability/tracer.py — Langfuse init, skip conditions."""

from unittest.mock import MagicMock, patch

import pytest


class TestInitObservability:
    def test_skips_when_public_key_missing(self, monkeypatch):
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_public_key", ""
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_secret_key", "secret"
        )
        from src.observability.tracer import init_observability

        result = init_observability()
        assert result is None

    def test_skips_when_secret_key_missing(self, monkeypatch):
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_public_key", "public"
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_secret_key", ""
        )
        from src.observability.tracer import init_observability

        result = init_observability()
        assert result is None

    def test_skips_when_both_keys_missing(self, monkeypatch):
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_public_key", ""
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_secret_key", ""
        )
        from src.observability.tracer import init_observability

        result = init_observability()
        assert result is None

    def test_import_error_handled_gracefully(self, monkeypatch):
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_public_key", "pk"
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_secret_key", "sk"
        )
        # Block the langfuse.llama_index import to trigger ImportError path
        # by making the import itself raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "langfuse.llama_index":
                raise ImportError("Mocked import error")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            from src.observability.tracer import init_observability

            result = init_observability()
            assert result is None

    def test_successfully_initializes_when_keys_present(self, monkeypatch):
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_public_key", "pk"
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_secret_key", "sk"
        )
        monkeypatch.setattr(
            "src.observability.tracer.settings.langfuse_host", "http://localhost:3000"
        )

        mock_handler = MagicMock()
        mock_callback_mgr = MagicMock()

        # The function does `from langfuse.llama_index import LlamaIndexCallbackHandler`
        # We need to make this import succeed with our mock
        import sys

        # Create a mock for langfuse.llama_index module
        mock_llama_index = MagicMock()
        mock_llama_index.LlamaIndexCallbackHandler = MagicMock(
            return_value=mock_handler
        )

        with patch.dict(sys.modules, {"langfuse.llama_index": mock_llama_index}):
            with patch(
                "llama_index.core.callbacks.CallbackManager",
                return_value=mock_callback_mgr,
            ):
                from src.observability.tracer import init_observability

                result = init_observability()
                assert result is None
