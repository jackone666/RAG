"""
Shared utility helpers used across multiple modules.

Purpose: avoid duplicate definitions of small helpers that are needed
by both the API layer (documents.py) and the pipeline layer (sync_manager.py).
"""


def _escape_milvus_expr(value: str) -> str:
    """Escape string values for Milvus expression syntax.

    Milvus uses double-quotes to delimit string literals in expressions
    (e.g. `tenant_id == "abc"`).  To embed a literal double-quote inside
    such a value it must be doubled (`""`), analogous to SQL escaping rules.

    Without escaping, a doc_id or tenant_id containing `"` would break the
    expression grammar and cause a parse error on the Milvus side.
    """
    return value.replace('"', '""')
