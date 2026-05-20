"""Tests for src/middleware/auth.py — JWT decode, tenant extraction, RBAC guard."""

from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import HTTPException, status

from src.middleware.auth import (
    ROLE_HIERARCHY,
    RBACGuard,
    _decode_token,
    extract_tenant_context,
)


class TestDecodeToken:
    def test_valid_token_returns_payload(self):
        payload = {"tenant_id": "t1", "role": "admin", "sub": "user1"}
        token = jwt.encode(payload, "secret", algorithm="HS256")
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            result = _decode_token(token)
        assert result["tenant_id"] == "t1"
        assert result["role"] == "admin"

    def test_invalid_token_raises_401(self):
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            with pytest.raises(HTTPException) as exc:
                _decode_token("bad.token.here")
            assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_expired_token_raises_401(self):
        from datetime import datetime, timedelta, timezone

        payload = {
            "tenant_id": "t1",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
        }
        token = jwt.encode(payload, "secret", algorithm="HS256")
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            with pytest.raises(HTTPException) as exc:
                _decode_token(token)
            assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    def test_wrong_algorithm_token_raises_401(self):
        payload = {"tenant_id": "t1"}
        token = jwt.encode(payload, "secret", algorithm="HS512")
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            with pytest.raises(HTTPException) as exc:
                _decode_token(token)
            assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestExtractTenantContext:
    @pytest.mark.asyncio
    async def test_bearer_token_extraction(self):
        payload = {"tenant_id": "t2", "role": "editor", "sub": "u2"}
        token = jwt.encode(payload, "secret", algorithm="HS256")
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {token}"}

        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            ctx = await extract_tenant_context(request)

        assert ctx["tenant_id"] == "t2"
        assert ctx["role"] == "editor"
        assert ctx["user_id"] == "u2"

    @pytest.mark.asyncio
    async def test_header_fallback_authentication(self):
        request = MagicMock()
        request.headers = {"X-Tenant-ID": "header_tenant", "X-Role": "viewer"}
        ctx = await extract_tenant_context(request)
        assert ctx["tenant_id"] == "header_tenant"
        assert ctx["role"] == "viewer"
        assert ctx["user_id"] == "anonymous"

    @pytest.mark.asyncio
    async def test_no_auth_defaults_to_default_tenant(self):
        request = MagicMock()
        request.headers = {}
        ctx = await extract_tenant_context(request)
        assert ctx["tenant_id"] == "default"
        assert ctx["role"] == "viewer"

    @pytest.mark.asyncio
    async def test_empty_bearer_token_raises_401(self):
        """Empty Bearer token triggers JWT decode error → 401."""
        request = MagicMock()
        request.headers = {"Authorization": "Bearer "}
        with pytest.raises(HTTPException) as exc:
            await extract_tenant_context(request)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_bearer_with_extra_spaces(self):
        """Edge case: leading/trailing spaces in token."""
        payload = {"tenant_id": "t3", "role": "admin", "sub": "u3"}
        token = jwt.encode(payload, "secret", algorithm="HS256")
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer   {token}  "}
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            ctx = await extract_tenant_context(request)
        assert ctx["tenant_id"] == "t3"


class TestRBACGuard:
    @pytest.mark.asyncio
    async def test_admin_can_access_editor_route(self):
        guard = RBACGuard("editor")
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "t", "role": "admin", "user_id": "u"}
        ctx = await guard(request)
        assert ctx["role"] == "admin"

    @pytest.mark.asyncio
    async def test_viewer_cannot_access_admin_route(self):
        guard = RBACGuard("admin")
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "t", "role": "viewer", "user_id": "u"}
        with pytest.raises(HTTPException) as exc:
            await guard(request)
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_no_tenant_context_raises_401(self):
        guard = RBACGuard("viewer")
        request = MagicMock()
        request.state = MagicMock()
        del request.state.tenant_context
        with pytest.raises(HTTPException) as exc:
            await guard(request)
        assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_unknown_role_treated_as_minimum(self):
        guard = RBACGuard("editor")
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "t", "role": "unknown_role", "user_id": "u"}
        with pytest.raises(HTTPException) as exc:
            await guard(request)
        assert exc.value.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_same_role_can_access(self):
        guard = RBACGuard("editor")
        request = MagicMock()
        request.state.tenant_context = {"tenant_id": "t", "role": "editor", "user_id": "u"}
        ctx = await guard(request)
        assert ctx["role"] == "editor"

    def test_role_hierarchy_values(self):
        assert ROLE_HIERARCHY["admin"] > ROLE_HIERARCHY["editor"]
        assert ROLE_HIERARCHY["editor"] > ROLE_HIERARCHY["viewer"]

    def test_rbac_guard_unknown_required_role(self):
        guard = RBACGuard("nonexistent")
        assert guard._required_level == 0
