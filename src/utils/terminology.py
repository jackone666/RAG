"""
领域术语归一化。

企业知识库里常见缩写、产品代号、指标别名。检索前把这些表达扩展成
标准术语，可以同时提升关键词召回和向量召回。

策略：
1. 别名扩展：在 query 末尾追加标准术语，不替换原表达
2. 双向映射：别名→标准术语、标准术语→别名都支持
3. 动态加载：data/term_aliases.json 文件热加载，无需重启

示例：
  输入："动态市盈率怎么算"
  输出："动态市盈率怎么算 PE-TTM"

  输入："ROE指标在哪份报告里"
  输出："ROE指标在哪份报告里 净资产收益率"
"""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

_DEFAULT_TERMS = {
    # 金融指标
    "动态市盈率": "PE-TTM",
    "市盈率ttm": "PE-TTM",
    "静态市盈率": "PE-LYR",
    "市盈率": "PE",
    "市净率": "PB",
    "净资产收益率": "ROE",
    "总资产收益率": "ROA",
    "投入产出比": "ROI",
    "每股收益": "EPS",
    "税息折旧及摊销前利润": "EBITDA",
    # 技术指标
    "召回率": "recall",
    "准确率": "precision",
    "精确度": "precision",
    "命中率": "hit rate",
    "平均倒数排名": "MRR",
    "忠实度": "faithfulness",
    # 通用缩写
    "人工智能": "AI",
    "机器学习": "ML",
    "自然语言处理": "NLP",
    "大语言模型": "LLM",
    "检索增强生成": "RAG",
}

_TERM_FILE = Path("data/term_aliases.json")
_cached_terms: dict[str, str] | None = None
_cache_mtime: float = 0  # 文件修改时间戳，用于热加载检测


def load_term_aliases() -> dict[str, str]:
    """加载企业术语别名表，支持文件热加载。

    文件 data/term_aliases.json 格式：
    {"别名1": "标准术语1", "别名2": "标准术语2", ...}

    文件不存在时使用内置基础表。
    文件修改后下次调用自动热加载，无需重启服务。
    """
    global _cached_terms, _cache_mtime

    # 检查文件是否更新
    try:
        if _TERM_FILE.exists():
            current_mtime = _TERM_FILE.stat().st_mtime
            if _cached_terms is not None and current_mtime == _cache_mtime:
                return _cached_terms
            _cache_mtime = current_mtime
    except Exception:
        pass

    terms = dict(_DEFAULT_TERMS)
    try:
        if _TERM_FILE.exists():
            loaded = json.loads(_TERM_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                terms.update({str(k): str(v) for k, v in loaded.items()})
                logger.info(f"术语表已加载: {len(terms)} 条（含 {len(loaded)} 条自定义）")
    except Exception as e:
        logger.warning(f"术语表加载失败，使用默认术语表: {e}")

    _cached_terms = terms
    return terms


def build_reverse_aliases() -> dict[str, list[str]]:
    """构建标准术语→别名列表的反向映射。

    用于文档入库时将标准术语扩展为"术语 (别名1, 别名2)"形式，
    提升关键词索引的覆盖面。

    示例：
    >>> build_reverse_aliases()["PE-TTM"]
    ["动态市盈率", "市盈率ttm"]
    """
    reverse: dict[str, list[str]] = {}
    for alias, canonical in load_term_aliases().items():
        reverse.setdefault(canonical, []).append(alias)
    return reverse


def normalize_query_terms(query: str) -> str:
    """将 query 中的别名扩展为"别名 标准术语"形式，保留原始表达。

    不替换原词，只在末尾追加标准术语，避免改变用户原意。
    同时处理反向情况：如果 query 中有标准术语，追加中文别名。

    示例：
    >>> normalize_query_terms("什么是动态市盈率")
    "什么是动态市盈率 PE-TTM"
    >>> normalize_query_terms("ROE高的公司有哪些")
    "ROE高的公司有哪些 净资产收益率"
    """
    normalized = query
    lower_query = query.lower()
    additions = []

    # 正向：别名 → 标准术语
    for alias, canonical in load_term_aliases().items():
        if alias.lower() in lower_query and canonical.lower() not in lower_query:
            additions.append(canonical)

    # 反向：标准术语 → 别名（也追加中文名）
    reverse = build_reverse_aliases()
    for canonical, aliases in reverse.items():
        if canonical.lower() in lower_query:
            for alias in aliases:
                if alias.lower() not in lower_query:
                    additions.append(alias)

    if additions:
        normalized = f"{query} {' '.join(dict.fromkeys(additions))}"
    return normalized


def normalize_doc_terms(text: str) -> str:
    """文档入库时扩展标准术语，提升关键词召回。

    对文档中的标准术语追加别名，使关键词索引同时覆盖
    正式术语和口语表达。

    示例：
    >>> normalize_doc_terms("公司PE-TTM为15倍")
    "公司PE-TTM(动态市盈率, 市盈率ttm)为15倍"
    """
    result = text
    reverse = build_reverse_aliases()
    for canonical, aliases in reverse.items():
        if canonical in result:
            alias_str = ", ".join(aliases)
            result = result.replace(canonical, f"{canonical}({alias_str})")
    return result
