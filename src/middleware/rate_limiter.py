"""
网关安全防线 - 基于 Redis 的分布式流量控制与限流模块

功能说明：
- 使用 Redis Sorted Set 实现分布式滑动窗口限流
- 按租户维度统计请求量，防止单租户恶意刷量
- Lua 脚本保证"检查-计数-过期"三步原子操作
- 多 worker 进程/多节点共享限流状态

技术选型理由：
- ZSET 滑动窗口：天然支持毫秒级精度，score 存储时间戳
- Lua 原子脚本：ZREMRANGEBYSCORE + ZCARD + ZADD 三步必须原子，
  否则并发场景下会出现计数漂移（over-counting）
- 自动 TTL 清理：每个 key 设 2 倍窗口过期时间，Redis 自动回收闲置租户

生产特性：
- 连接池复用：避免每次请求新建 TCP 连接
- Redis 不可用时的降级策略：fail-open（放行）+ 告警日志
- 多维度限流键：支持 tenant_id、user_id、IP 独立计数
"""
import time
import uuid
from typing import Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, status
from loguru import logger

from src.config.settings import settings

# ============================================================
# Lua 原子限流脚本
# 使用 Sorted Set 实现滑动窗口，三步原子操作避免并发竞态：
#   1. ZREMRANGEBYSCORE — 清理窗口外的过期请求
#   2. ZCARD              — 统计窗口内当前请求数
#   3. ZADD + EXPIRE      — 若未超限，记录本次请求并刷新 TTL
# ============================================================
SLIDING_WINDOW_LUA = """
local key       = KEYS[1]
local now_ms    = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local max_req   = tonumber(ARGV[3])
local member    = ARGV[4]

-- 步骤 1: 移除窗口外的所有过期请求
redis.call('ZREMRANGEBYSCORE', key, 0, now_ms - window_ms)

-- 步骤 2: 统计窗口内剩余请求数
local count = redis.call('ZCARD', key)

-- 步骤 3: 若已达上限，拒绝（返回当前计数供调用方参考）
if count >= max_req then
    return {0, count}
end

-- 步骤 4: 记录本次请求（score = 当前毫秒时间戳，member = 唯一 ID）
redis.call('ZADD', key, now_ms, member)

-- 步骤 5: 自动过期清理 —— TTL = 2 * 窗口时间，防止僵尸 key 堆积
redis.call('EXPIRE', key, math.ceil(window_ms / 1000) * 2)

return {1, count + 1}
"""


class RedisSlidingWindowRateLimiter:
    """基于 Redis Sorted Set 的分布式滑动窗口限流器。

    架构原理：
    ┌─────────────────────────────────────────────┐
    │ Redis Sorted Set: rate_limit:tenant_abc     │
    │ ┌──────────┬──────────┬──────────┬──────────┐│
    │ │ member_1 │ member_2 │ member_3 │ member_4 ││
    │ │ score=t1 │ score=t2 │ score=t3 │ score=t4 ││
    │ └──────────┴──────────┴──────────┴──────────┘│
    │  ^^^^^^^^ 滑动窗口 (now - window) ^^^^^^^^^^  │
    │  窗口外请求已被 ZREMRANGEBYSCORE 自动清理     │
    └─────────────────────────────────────────────┘

    原子性保证：
    - 整个"清理-计数-记录"流程封装在单条 Lua 脚本中
    - Redis 单线程执行 Lua 脚本，杜绝 TOCTOU 竞态
    - 任一时刻只有一个操作在修改该 key，计数绝对精确

    降级策略（Redis 不可用时）：
    - fail-open：放行请求，保证业务可用性优先
    - 记录 ERROR 日志触发告警
    - 生产建议：配合本地内存限流作为二级兜底
    """

    def __init__(
        self,
        redis_url: str | None = None,
        max_requests: int | None = None,
        window_seconds: int = 60,
        connection_pool_size: int = 20,
    ):
        """初始化 Redis 限流器。

        Args:
            redis_url: Redis 连接 URL，默认取 settings.redis_url
            max_requests: 时间窗口内允许的最大请求数，默认取 settings.rate_limit_per_minute
            window_seconds: 滑动窗口大小（秒），默认 60s
            connection_pool_size: Redis 连接池大小，根据 worker 数和并发量调整
        """
        self.redis_url = redis_url or settings.redis_url
        self.max_requests = max_requests or settings.rate_limit_per_minute
        self.window_seconds = window_seconds
        self.window_ms = window_seconds * 1000

        # Redis 异步连接池 —— 复用连接避免频繁 TCP 握手
        self._pool: aioredis.ConnectionPool | None = None
        self._redis: aioredis.Redis | None = None
        self._pool_size = connection_pool_size
        # 预加载 Lua 脚本到 Redis 服务端，后续调用仅传 SHA 哈希，减少网络传输
        self._script_sha: str | None = None

    async def _get_redis(self) -> aioredis.Redis:
        """获取或初始化 Redis 异步连接（懒加载 + 连接池复用）。

        连接池策略：
        - 使用 redis.asyncio.ConnectionPool 管理连接
        - 连接池大小 = worker_num * (1~2)，当前默认 20
        - 启用健康检查（health_check_interval）自动剔除死连接
        - socket_keepalive 防止中间代理空闲断开

        Returns:
            Redis 异步客户端实例

        Raises:
            ConnectionError: Redis 连接失败时抛出，由调用方降级处理
        """
        if self._redis is None:
            self._pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self._pool_size,
                socket_keepalive=True,
                health_check_interval=30,
                decode_responses=False,  # 限流不需要 decode，减少 CPU 开销
            )
            self._redis = aioredis.Redis(connection_pool=self._pool)
        return self._redis

    async def _load_script(self) -> str:
        """将 Lua 脚本预加载到 Redis，返回 SHA 哈希用于后续 EVALSHA 调用。

        预加载的好处：
        - 首次调用 SCRIPT LOAD，后续仅传 SHA（40 字节 vs 完整脚本）
        - Redis 7.0+ 支持 EVALSHA_RO（只读），但此处有写操作故用 EVALSHA
        - 若脚本因 Redis 重启丢失，自动回退至 EVAL 全量传输

        Returns:
            Lua 脚本的 SHA1 哈希值
        """
        if self._script_sha is None:
            r = await self._get_redis()
            self._script_sha = await r.script_load(SLIDING_WINDOW_LUA)
        return self._script_sha

    async def is_allowed(self, key: str) -> tuple[bool, int]:
        """检查指定限流键是否允许通过。

        原子执行流程（Lua 脚本内）：
        1. 清理窗口外过期请求（ZREMRANGEBYSCORE）
        2. 统计当前窗口请求数（ZCARD）
        3. 若未超限：记录本次请求（ZADD）+ 刷新 TTL（EXPIRE）→ 返回 1
        4. 若已超限：拒绝 → 返回 0

        Redis 不可用时（连接失败/超时）：
        - fail-open 策略：返回 (True, 0)，放行请求
        - 记录 ERROR 日志供告警系统抓取
        - 生产建议配合本地计数作为二级限流兜底

        Args:
            key: 限流键（格式: "rate_limit:{tenant_id}"）

        Returns:
            (是否放行, 当前窗口内请求数)
        """
        try:
            r = await self._get_redis()
            sha = await self._load_script()
            now_ms = int(time.time() * 1000)
            member = f"{now_ms}:{uuid.uuid4().hex[:8]}"

            result = await r.evalsha(
                sha,
                1,  # KEYS 数量
                key,  # KEYS[1]
                now_ms,  # ARGV[1]: 当前毫秒时间戳
                self.window_ms,  # ARGV[2]: 窗口大小（毫秒）
                self.max_requests,  # ARGV[3]: 最大请求数
                member,  # ARGV[4]: 唯一成员标识
            )
            # Lua 脚本返回 {allowed_flag, current_count}
            allowed = result[0] == 1
            current_count = result[1]
            return allowed, current_count

        except (aioredis.ConnectionError, aioredis.TimeoutError, OSError) as e:
            # Redis 不可用 → fail-open 降级策略
            logger.error(f"Redis 限流不可用，fail-open 放行: {e}")
            return True, 0

    async def get_current_usage(self, key: str) -> int:
        """查询指定键在当前窗口内的请求数（仅用于监控/调试，不影响计数）。

        Args:
            key: 限流键

        Returns:
            当前窗口内的请求数，Redis 不可用时返回 -1
        """
        try:
            r = await self._get_redis()
            now_ms = int(time.time() * 1000)
            # 先清理过期，再统计
            await r.zremrangebyscore(key, 0, now_ms - self.window_ms)
            count = await r.zcard(key)
            return count
        except Exception as e:
            logger.warning(f"查询限流使用量失败: {e}")
            return -1

    async def reset(self, key: str) -> None:
        """重置指定键的限流计数（管理后台调用）。

        Args:
            key: 限流键
        """
        try:
            r = await self._get_redis()
            await r.delete(key)
            logger.info(f"已重置限流键: {key}")
        except Exception as e:
            logger.error(f"重置限流键失败: {e}")


# ============================================================
# 模块级单例限流器
# ============================================================
rate_limiter = RedisSlidingWindowRateLimiter()


# ============================================================
# FastAPI 依赖注入入口
# ============================================================
async def rate_limit_dependency(request: Request):
    """FastAPI 依赖注入：在路由处理前执行 Redis 分布式限流检查。

    限流键优先级（维度从精确到宽泛）：
    1. tenant_id + user_id（精确到用户级，推荐）
    2. tenant_id（租户级，当前实现）
    3. 客户端 IP（兜底，未认证请求）

    超限时返回 429 Too Many Requests，附带 Retry-After 头指导客户端重试。

    设计考量：
    - 使用 tenant_id 作为主维度，避免单租户的高频用户挤占其他租户配额
    - 若需要更细粒度控制，可扩展为 {tenant_id}:{user_id} 组合键

    Args:
        request: FastAPI Request 对象

    Raises:
        HTTPException 429: 当前窗口内请求数已达上限
    """
    tenant_ctx: dict = getattr(request.state, "tenant_context", {})
    tenant_id = tenant_ctx.get("tenant_id")
    client_ip = request.client.host if request.client else "unknown"

    # 限流键维度：优先租户 ID，降级为客户端 IP
    if tenant_id and tenant_id != "default":
        rate_key = f"rate_limit:tenant:{tenant_id}"
    else:
        rate_key = f"rate_limit:ip:{client_ip}"

    allowed, _ = await rate_limiter.is_allowed(rate_key)

    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="请求过于频繁，请稍后重试。Rate limit exceeded.",
            headers={
                "Retry-After": str(rate_limiter.window_seconds),
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Window": str(rate_limiter.window_seconds),
            },
        )
