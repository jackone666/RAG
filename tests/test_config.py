"""Tests for src/config/settings.py — env parsing, defaults, validation."""

import os
from unittest.mock import patch

import pytest


class TestSettingsDefaults:
    def test_default_openai_base_url(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.openai_base_url == "https://api.openai.com/v1"

    def test_default_primary_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.primary_model == "gpt-4o"

    def test_default_fallback_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.fallback_model == "gpt-4o-mini"

    def test_default_judge_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.judge_model == "gpt-4o"

    def test_default_embedding_dim(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.embedding_dim == 1024

    def test_default_milvus_uri(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.milvus_uri == "http://localhost:19530"

    def test_default_milvus_token_empty(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.milvus_token == ""

    def test_default_milvus_collection_name(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.milvus_collection_name == "intellilens_knowledge"

    def test_default_elasticsearch_url(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.elasticsearch_url == "http://localhost:9200"

    def test_default_redis_url(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.redis_url == "redis://localhost:6379/0"

    def test_default_redis_pool_size(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.redis_pool_size == 20

    def test_default_jwt_algorithm(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.jwt_algorithm == "HS256"

    def test_default_app_host(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.app_host == "0.0.0.0"

    def test_default_app_port(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.app_port == 8000

    def test_default_rate_limit(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.rate_limit_per_minute == 60

    def test_default_bad_cases_path(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.bad_cases_path == "data/bad_cases.jsonl"

    def test_default_eval_score_threshold(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.eval_score_threshold == 0.8

    def test_langfuse_defaults_empty(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.langfuse_public_key == ""
            assert s.langfuse_secret_key == ""


class TestSettingsEnvOverride:
    def test_custom_primary_model_from_env(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "PRIMARY_MODEL": "gpt-4-turbo",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.primary_model == "gpt-4-turbo"

    def test_custom_app_port_from_env(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "APP_PORT": "9999",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.app_port == 9999

    def test_custom_rate_limit_from_env(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "RATE_LIMIT_PER_MINUTE": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.rate_limit_per_minute == 120

    def test_custom_redis_url(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "REDIS_URL": "redis://custom:6380/1",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.redis_url == "redis://custom:6380/1"

    def test_custom_jwt_secret(self):
        env = {
            "OPENAI_API_KEY": "sk-test",
            "JWT_SECRET_KEY": "prod-secret-key",
        }
        with patch.dict(os.environ, env, clear=True):
            from src.config.settings import Settings

            s = Settings(_env_file=None)
            assert s.jwt_secret_key == "prod-secret-key"


class TestSettingsRequired:
    def test_openai_api_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            from src.config.settings import Settings

            with pytest.raises(Exception):
                Settings(_env_file=None)
