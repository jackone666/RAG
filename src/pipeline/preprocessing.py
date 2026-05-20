"""
文档预处理管道 — 参考字节跳动 RAG 实践 §3「文档理解与预处理」

功能：
- 文本清洗：规范化空白字符、移除控制字符、统一标点
- 结构解析：检测 Markdown/文档标题层级
- 元数据增强：提取文档标题、分节信息
"""
import re

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

    metadata = {
        "title": doc_title,
        "section_count": len(sections),
        "char_count": len(text),
        "sections": sections[:20],  # 最多保留 20 个节标题
    }
    return text, metadata
