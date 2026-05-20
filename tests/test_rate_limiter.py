"""Tests for src/middleware/rate_limiter.py — sliding window, fail-open, key generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, status

from src.middleware.rate_limiter import (
    RedisSlidingWindowRateLimiter,
    rate_limit_dependency,
)


class TestRateLimiterInit:
    def test_default_values_from_settings(self):
        rl = RedisSlidingWindowRateLimiter()
        assert rl.window_seconds == 60
        assert rl.window_ms == 60000

    def test_custom_values(self):
        rl = RedisSlidingWindowRateLimiter(
            redis_url="redis://custom:6379/0",
            max_requests=100,
            window_seconds=30,
        )
        assert rl.window_seconds == 30
        assert rl.window_ms == 30000
        assert rl.max_requests == 100


class TestRateLimiterIsAllowed:
    @pytest.mark.asyncio
    async def test_allowed_when_redis_returns_flag_1(self):
        rl = RedisSlidingWindowRateLimiter(max_requests=10)
        mock_redis = AsyncMock()
        mock_redis.script_load = AsyncMock(return_value="abc123")
        # Lua returns {1, count+1} → Python list [1, 6]
        mock_redis.evalsha = AsyncMock(return_value=[1, 6])

        with patch.object(rl, "_get_redis", AsyncMock(return_value=mock_redis)):
            allowed, current = await rl.is_allowed("rate_limit:tenant:test")
            assert allowed is True
            assert current == 6

    @pytest.mark.asyncio
    async def test_denied_when_redis_returns_flag_0(self):
        rl = RedisSlidingWindowRateLimiter(max_requests=10)
        mock_redis = AsyncMock()
        mock_redis.script_load = AsyncMock(return_value="abc123")
        # Lua returns {0, count} → Python list [0, 5]
        mock_redis.evalsha = AsyncMock(return_value=[0, 5])

        with patch.object(rl, "_get_redis", AsyncMock(return_value=mock_redis)):
            allowed, current = await rl.is_allowed("rate_limit:tenant:test")
            assert allowed is False
            assert current == 5

    @pytest.mark.asyncio
    async def test_fail_open_on_connection_error(self):
        rl = RedisSlidingWindowRateLimiter(max_requests=10)
        mock_redis = AsyncMock()
        mock_redis.script_load = AsyncMock(side_effect=ConnectionError("connection refused"))

        with patch.object(rl, "_get_redis", AsyncMock(return_value=mock_redis)):
            allowed, _ = await rl.is_allowed("rate_limit:tenant:test")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout_error(self):
        import redis.asyncio as aioredis

        rl = RedisSlidingWindowRateLimiter(max_requests=10)

        async def raise_timeout():
            raise aioredis.TimeoutError("timeout")

        with patch.object(rl, "_get_redis", AsyncMock(side_effect=raise_timeout)):
            allowed, _ = await rl.is_allowed("rate_limit:tenant:test")
            assert allowed is True

    @pytest.mark.asyncio
    async def test_lua_script_loaded_only_once(self):
        rl = RedisSlidingWindowRateLimiter(max_requests=10)
        mock_redis = AsyncMock()
        mock_redis.script_load = AsyncMock(return_value="abc123")
        mock_redis.evalsha = AsyncMock(return_value=[1, 1])

        with patch.object(rl, "_get_redis", AsyncMock(return_value=mock_redis)):
            await rl.is_allowed("key1")
            await rl.is_allowed("key2")
            # script_load called only once (cached SHA)
            assert mock_redis.script_load.call_count == 1


class TestRateLimitDependency:
    @pytest.mark.asyncio
    async def test_tenant_key_used_for_known_tenant(self):
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "tenant_abc"}
        request.client.host = "1.2.3.4"

        with patch(
            "src.middleware.rate_limiter.rate_limiter.is_allowed",
            AsyncMock(return_value=(True, 10)),
        ) as mock_check:
            await rate_limit_dependency(request)
            called_key = mock_check.call_args[0][0]
            assert "tenant:tenant_abc" in called_key

    @pytest.mark.asyncio
    async def test_ip_fallback_for_default_tenant(self):
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "default"}
        request.client.host = "10.0.0.1"

        with patch(
            "src.middleware.rate_limiter.rate_limiter.is_allowed",
            AsyncMock(return_value=(True, 10)),
        ) as mock_check:
            await rate_limit_dependency(request)
            called_key = mock_check.call_args[0][0]
            assert "ip:10.0.0.1" in called_key

    @pytest.mark.asyncio
    async def test_returns_429_when_denied(self):
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "tenant_abc"}
        request.client.host = "1.2.3.4"

        with patch(
            "src.middleware.rate_limiter.rate_limiter.is_allowed",
            AsyncMock(return_value=(False, 60)),
        ):
            with pytest.raises(HTTPException) as exc:
                await rate_limit_dependency(request)
            assert exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "Retry-After" in exc.value.headers
