"""
MCP 协议服务端 - FastMCP Server 实例与生命周期管理

功能说明：
- 创建 FastMCP Server 实例，作为 MCP 协议的服务端入口
- 工具注册通过 @mcp.tool() 装饰器完成（见 tools.py）
- 通过 FastAPI mount 机制将 MCP 子应用挂载到主服务

架构要点：
- 模块级单例 mcp 实例，tools.py 通过 import mcp 注册工具
- streamable_http_app() 方法创建 ASGI 子应用，挂载到 /mcp 路径
- 客户端（如 Claude Desktop）通过此端点连接并调用企业知识工具
"""
from mcp.server.fastmcp import FastMCP

# 全局 FastMCP 实例，tools.py 中的 @mcp.tool() 装饰器注册在此实例上
mcp = FastMCP("IntelliLens-MCP")


def create_mcp_server() -> FastMCP:
    """工厂函数：返回已初始化并注册工具的 FastMCP Server 实例。

    注意：工具注册发生在 tools.py 模块导入时（通过 @mcp.tool() 装饰器）。
    因此在 create_mcp_server() 调用前，必须确保已 import tools 模块。

    Returns:
        已配置的 FastMCP Server 实例
    """
    return mcp
