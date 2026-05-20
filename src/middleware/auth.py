"""
网关安全防线 - 身份认证与 RBAC 权限控制模块

功能说明：
1. 从 HTTP 请求中提取租户身份上下文（JWT Bearer Token 或 X-* Header）
2. 解析 tenant_id、角色权限等信息，注入到 request.state 供下游使用
3. 提供 RBACGuard 依赖注入类，用于路由级别的细粒度权限控制

安全设计：
- JWT 验证失败直接返回 401，不放过任何未认证请求
- 角色层级: admin(3) > editor(2) > viewer(1)
- 开发/机机通信场景支持 Header 降级认证（X-Tenant-ID + X-Role）
"""
import jwt
from fastapi import HTTPException, Request, status

from src.config.settings import settings

# 角色层级映射：数值越大权限越高
ROLE_HIERARCHY = {"admin": 3, "editor": 2, "viewer": 1}


def _decode_token(token: str) -> dict:
    """解码并验证 JWT Token。

    Args:
        token: Bearer Token 字符串（不含 "Bearer " 前缀）

    Returns:
        解码后的 JWT payload 字典

    Raises:
        HTTPException 401: Token 无效或已过期
    """
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


async def extract_tenant_context(request: Request) -> dict:
    """从请求中提取租户身份上下文。

    认证策略（按优先级）：
    1. Authorization: Bearer <JWT> → 解码 JWT 获取 tenant_id、role、sub
    2. X-Tenant-ID / X-Role Headers → 开发/机机通信降级方案

    此函数由 HTTP 中间件直接调用（非 FastAPI Depends 方式），
    因此手动解析 Header 而不依赖 FastAPI 的依赖注入系统。

    Args:
        request: FastAPI Request 对象

    Returns:
        包含 tenant_id, role, user_id 的上下文字典
    """
    auth_header = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        payload = _decode_token(token)
        return {
            "tenant_id": payload.get("tenant_id", "default"),
            "role": payload.get("role", "viewer"),
            "user_id": payload.get("sub", "anonymous"),
        }

    # 降级方案：Header 直传认证（适用于开发环境与机器间通信）
    return {
        "tenant_id": request.headers.get("X-Tenant-ID", "default"),
        "role": request.headers.get("X-Role", "viewer"),
        "user_id": "anonymous",
    }


class RBACGuard:
    """基于角色的访问控制守卫（FastAPI 依赖注入）。

    使用方式：
        @app.get("/admin-only")
        async def admin_route(ctx: dict = Depends(RBACGuard("admin"))):
            ...

    权限校验逻辑：
    - 从 request.state.tenant_context 中获取当前用户角色
    - 比对角色层级数值，不足则返回 403
    """

    def __init__(self, required_role: str = "viewer"):
        """初始化权限守卫。

        Args:
            required_role: 访问该路由所需的最低角色（admin/editor/viewer）
        """
        self._required_level = ROLE_HIERARCHY.get(required_role, 0)

    async def __call__(self, request: Request) -> dict:
        """FastAPI 依赖注入入口，执行权限校验。

        Args:
            request: FastAPI Request 对象

        Returns:
            校验通过后返回租户上下文字典

        Raises:
            HTTPException 401: 未注入租户上下文
            HTTPException 403: 角色权限不足
        """
        ctx: dict = getattr(request.state, "tenant_context", None)
        if ctx is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No tenant context")
        user_level = ROLE_HIERARCHY.get(ctx.get("role", ""), 0)
        if user_level < self._required_level:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return ctx
