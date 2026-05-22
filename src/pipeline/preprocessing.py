"""
文档预处理管道 — 参考字节跳动 RAG 实践 §3「文档理解与预处理」

功能：
- 文本清洗：规范化空白字符、移除控制字符、统一标点
- 结构解析：检测 Markdown/文档标题层级、表格、图片引用、引文定位
- 元数据增强：提取文档标题、分节信息、表格摘要、引用列表
"""
from __future__ import annotations

import re
from typing import Any

from src.config.rag_params import rag_params


def clean_text(text: str) -> str:
    """文本清洗 —— 规范化空白、移除不可见控制字符、统一格式。

    参考字节 §3.2.1「文档清洗」。
    """
    # 移除不可见控制字符（保留换行、制表符）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # 规范化空白：多个空行 → 双空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 规范化空白：多个空格 → 单空格（不处理行首空白）
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # 移除行首行尾多余空白
    text = '\n'.join(line.strip() for line in text.split('\n'))
    # 统一中文标点
    text = text.replace('「', '"').replace('」', '"')
    text = text.replace('『', '"').replace('』', '"')
    return text.strip()


def extract_sections(text: str) -> list[dict]:
    """检测文档结构 —— 提取标题层级和段落边界。

    识别常见标题模式：
    - Markdown 标题: # Title, ## Section
    - 中文编号: 一、二、三、... / 1. 2. 3.
    - 英文编号: Chapter, Section, Part

    Returns:
        [{"level": 1, "title": "背景介绍", "start": 0, "end": 1500}, ...]
    """
    sections = []
    lines = text.split('\n')

    # 模式1：Markdown 标题 (#, ##, ###, ####)
    md_heading = re.compile(r'^(#{1,4})\s+(.+)$')
    # 模式2：中文编号标题（一、二、三、）
    cn_numbered = re.compile(r'^([一二三四五六七八九十]{1,2})[、，,]\s*(.{2,50})$')
    # 模式3：数字编号标题（1. 1.1 1.1.1）
    num_heading = re.compile(r'^(\d+(?:\.\d+)*)\s+(.{2,50})$')
    # 模式4：英文标题关键词
    en_heading = re.compile(r'^(Chapter|Section|Part|Appendix)\s+\d+', re.IGNORECASE)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        level, title = 0, ""

        m = md_heading.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
        else:
            m = cn_numbered.match(line)
            if m:
                level = 2
                title = m.group(0).strip()
            else:
                m = num_heading.match(line)
                if m:
                    level = 2
                    title = m.group(0).strip()
                elif en_heading.match(line):
                    level = 2
                    title = line
                elif (rag_params.heading_min_len < len(line) <= rag_params.heading_max_len
                      and not line.endswith(('.', '!', '?', '。', '！', '？'))):
                    # 短行可能是标题（启发式判断）
                    if i == 0 or (i > 0 and not lines[i - 1].strip()):
                        level = 2
                        title = line

        if level > 0 and title:
            sections.append({"level": level, "title": title, "line": i})

    return sections


# ── 表格检测 ────────────────────────────────────────────────


def extract_tables(text: str) -> list[dict[str, Any]]:
    """检测文档中的表格（Markdown 表格 + ASCII 表格线）。

    识别模式：
    - Markdown pipe table: | col1 | col2 |\\n |---|---|
    - ASCII grid table: +---+---+ 或 ┌───┬───┐ 线框

    Returns:
        [{"type": "markdown"|"ascii", "headers": [...], "rows": [[...], ...], "raw": "..."}]
    """
    tables: list[dict[str, Any]] = []

    # Markdown pipe table
    md_table = re.compile(
        r'^\|(.+)\|\s*\n\|[:\-\s|]+\|\s*\n((?:^\|.+\|\s*\n?)+)',
        re.MULTILINE,
    )
    for m in md_table.finditer(text):
        header_line = m.group(1)
        body_block = m.group(2)
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        rows = []
        for row_line in body_block.strip().split('\n'):
            cells = [c.strip() for c in row_line.split('|') if c.strip()]
            if cells:
                rows.append(cells)
        if headers or rows:
            tables.append({
                "type": "markdown",
                "headers": headers,
                "rows": rows,
                "raw": m.group(0)[:500],
            })

    # ASCII grid table: lines with +---+---+ or ┌───┬───┐ patterns
    ascii_grid = re.compile(
        r'(?:^[+┌├└┐┬┼┤┴┘─╭│╰]+$\n?)+',
        re.MULTILINE,
    )
    for m in ascii_grid.finditer(text):
        raw = m.group(0).strip()
        if len(raw) > 20 and raw.count('\n') >= 2:
            tables.append({
                "type": "ascii",
                "headers": [],
                "rows": [],
                "raw": raw[:500],
            })

    return tables


# ── 图片引用检测 ────────────────────────────────────────────


def extract_image_refs(text: str) -> list[dict[str, str]]:
    """提取文档中的图片引用，便于后续 OCR/多模态扩展。

    Returns:
        [{"type": "markdown"|"html"|"placeholder", "alt": "...", "src": "..."}]
    """
    refs: list[dict[str, str]] = []

    # Markdown: ![alt](path)
    for m in re.finditer(r'!\[([^\]]*)\]\(([^)]+)\)', text):
        alt = m.group(1) or "图片"
        src = m.group(2)
        refs.append({"type": "markdown", "alt": alt, "src": src})

    # HTML: <img src="..." alt="..." />
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', text, re.IGNORECASE):
        src = m.group(1)
        alt_m = re.search(r'alt=["\']([^"\']*)["\']', m.group(0), re.IGNORECASE)
        alt = alt_m.group(1) if alt_m else "图片"
        refs.append({"type": "html", "alt": alt, "src": src})

    # Placeholder: {图: xxx}, [图片: xxx], (见图 X-X)
    for m in re.finditer(r'[{\[]?(?:图|图片|Figure|Fig\.)\s*[:.]?\s*([^}\])]+)[}\]]?', text):
        refs.append({"type": "placeholder", "alt": m.group(0), "src": ""})

    return refs


# ── 引文/引用定位 ────────────────────────────────────────────


def extract_citations(text: str) -> list[dict[str, str]]:
    """提取学术/报告类文档中的引用标记。

    识别：
    - 数字引用: [1], [2,3], [4-6]
    - 作者年份: (Author, 2020), (Smith et al., 2019)
    - 脚注: [^1], [^note]
    - 条款引用: 第X条, Article X, Section X

    Returns:
        [{"type": "numeric"|"author_year"|"footnote"|"clause", "text": "..."}]
    """
    citations: list[dict[str, str]] = []

    for m in re.finditer(r'\[(\d+(?:[\s,\-]+\d+)*)\]', text):
        citations.append({"type": "numeric", "text": m.group(0)})

    for m in re.finditer(r'\(([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?)\)', text):
        citations.append({"type": "author_year", "text": m.group(0)})

    for m in re.finditer(r'\[\^(\w+)\]', text):
        citations.append({"type": "footnote", "text": m.group(0)})

    for m in re.finditer(
            r'(?:第[一二三四五六七八九十\d]+|[Aa]rticle\s+\d+|[Ss]ection\s+\d+[.\d]*|[第]?\d+[\.]\d+[\.]?\d*\s*条)',
            text):
        citations.append({"type": "clause", "text": m.group(0)})

    return citations


# ── 标题层级树 ───────────────────────────────────────────────


def build_heading_tree(sections: list[dict]) -> list[dict]:
    """将扁平的标题列表构建为层级树，便于导航和 chunk 上下文增强。

    算法：栈式层级追踪。遇到新标题时，按 level 寻找父节点挂载。
    同层级 title 连续出现时，后者追加到前者所在的子节点列表。

    Returns:
        [{"title": "概述", "level": 1, "children": [{"title": "背景", ...}]}]
    """
    if not sections:
        return []

    root: list[dict] = []
    # 每个层级（1-based）当前挂载点
    stack: dict[int, dict] = {}

    for sec in sections:
        level = max(sec.get("level", 1), 1)
        node = {
            "title": sec.get("title", ""),
            "level": level,
            "line": sec.get("line", 0),
            "children": [],
        }

        if level == 1:
            root.append(node)
            stack = {k: v for k, v in stack.items() if k < level}
            stack[1] = node
            for k in list(stack.keys()):
                if k > 1:
                    del stack[k]
        else:
            parent_level = level - 1
            while parent_level > 0 and parent_level not in stack:
                parent_level -= 1
            if parent_level > 0 and parent_level in stack:
                stack[parent_level].setdefault("children", []).append(node)
                stack[level] = node
                for k in list(stack.keys()):
                    if k > level:
                        del stack[k]
            else:
                root.append(node)
                stack = {k: v for k, v in stack.items() if k < level}
                stack[level] = node

    return root


# ── 完整预处理入口 ────────────────────────────────────────────


def preprocess_document(text: str) -> tuple[str, list[dict]]:
    """完整文档预处理：清洗 + 结构解析。

    Returns:
        (cleaned_text, sections_metadata)
    """
    text = clean_text(text)
    sections = extract_sections(text)

    # 尝试提取文档主标题（第一个最高层级标题）
    doc_title = "未命名文档"
    if sections:
        top = min(sections, key=lambda s: s["level"])
        doc_title = top["title"]

    # 表格 / 图片 / 引用 — 统一在预处理阶段提取，下游按需消费
    tables = extract_tables(text)
    images = extract_image_refs(text)
    citations = extract_citations(text)
    heading_tree = build_heading_tree(sections)

    metadata = {
        "title": doc_title,
        "section_count": len(sections),
        "char_count": len(text),
        "sections": sections[:20],
        "heading_tree": heading_tree,
        "table_count": len(tables),
        "tables": tables[:10],
        "image_count": len(images),
        "images": images[:20],
        "citation_count": len(citations),
        "citations": citations[:50],
    }
    return text, metadata
