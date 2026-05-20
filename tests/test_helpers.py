"""Tests for src/utils/helpers.py — Milvus expression escaping."""

import pytest

from src.utils.helpers import _escape_milvus_expr


class TestEscapeMilvusExpr:
    def test_no_special_chars_returns_unchanged(self):
        assert _escape_milvus_expr("hello") == "hello"

    def test_double_quote_escaped(self):
        result = _escape_milvus_expr('doc_"with"_quotes')
        assert result == 'doc_""with""_quotes'

    def test_multiple_quotes_escaped(self):
        result = _escape_milvus_expr('a"b"c')
        assert result == 'a""b""c'

    def test_consecutive_quotes(self):
        result = _escape_milvus_expr('""')
        assert result == '""""'

    def test_empty_string(self):
        assert _escape_milvus_expr("") == ""

    def test_only_quote(self):
        assert _escape_milvus_expr('"') == '""'

    def test_sql_injection_attempt(self):
        malicious = '" OR 1=1 --'
        result = _escape_milvus_expr(malicious)
        assert '"" OR 1=1 --' == result

    def test_unicode_characters_preserved(self):
        result = _escape_milvus_expr("中文文档")
        assert result == "中文文档"

    def test_unicode_with_quotes(self):
        result = _escape_milvus_expr('中文"文档"')
        assert result == '中文""文档""'
