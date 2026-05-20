"""Auth API — login endpoint that issues JWT tokens."""

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config.settings import settings
from src.middleware.user_store import authenticate_user, init_default_users

router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    tenant_id: str
    role: str
    display_name: str


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Authenticate with username/password and receive a JWT token.

    The returned token encodes tenant_id, role, and sub claims.
    Use it as: Authorization: Bearer <access_token>
    """
    init_default_users()

    user = authenticate_user(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": user["username"],
        "tenant_id": user["tenant_id"],
        "role": user["role"],
        "display_name": user["display_name"],
        "iat": now,
        "exp": now + timedelta(hours=24),
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    return LoginResponse(
        access_token=token,
        tenant_id=user["tenant_id"],
        role=user["role"],
        display_name=user["display_name"],
    )
