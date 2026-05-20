"""
MCP 协议服务端 - 工具注册与身份绑定模块

功能说明：
- 将企业 RAG 查询能力封装为 MCP Tool，供外部大模型客户端调用
- 强制要求传入 tenant_id，确保通过任意客户端访问时依然受多租户权限管控
- 工具内部串联完整的检索 → 重排 → 生成流程

安全设计（规范 4.4）：
- search_enterprise_knowledge 的 tenant_id 参数为必填
- 不提供任何"无租户"或"全局搜索"的变体
- 即使通过 Claude Desktop 等外部 MCP 客户端调用，也无法绕过租户隔离
"""
from src.engine.query_engine import shared_query_engine
from src.engine.retrievers import shared_retriever
from src.mcp_server.server import mcp

# 模块级引擎实例，工具调用时复用
_retriever = shared_retriever
_query_engine = shared_query_engine


@mcp.tool()
async def search_enterprise_knowledge(query: str, tenant_id: str) -> str:
    """在企业知识库中搜索并生成回答——租户级别的访问控制。

    该工具是 IntelliLens-MCP 对外暴露的核心 RAG 接口。
    通过 MCP 协议供任何兼容客户端调用（Claude Desktop、自定义 Agent 等）。

    执行流程：
    1. 多租户混合检索（向量 + 关键词，含 RBAC 过滤）
    2. LLM 重排序 Top-5 节点
    3. 大模型答案生成（含主/备模型自动降级熔断）

    Args:
        query: 用户的自然语言查询或问题
        tenant_id: 租户标识符（必填）——每个查询严格限定在单租户范围内，
                   这是 RBAC 数据隔离的核心保障，不可省略。

    Returns:
        基于租户授权知识库生成的回答文本。
        若无相关文档，返回租户范围内的"未找到"提示。
    """
    # 第一步：混合检索（向量 + 关键词，含租户过滤）
    nodes = await _retriever.retrieve(query=query, tenant_id=tenant_id)
    if not nodes:
        return f"在租户 '{tenant_id}' 的知识库中未找到相关文档。"

    # 第二步：LLM 重排序
    reranked = await _query_engine.rerank(query, nodes)

    # 第三步：大模型生成（带自动降级熔断）
    answer = await _query_engine.query(reranked, query)
    return answer
