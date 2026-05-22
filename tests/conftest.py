import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")


@pytest.fixture
def tenant_context():
    return {"tenant_id": "tenant_abc", "role": "editor", "user_id": "user_1"}


@pytest.fixture(autouse=True)
def disable_distributed_byte_cache(monkeypatch):
    """Unit tests should not depend on cross-test Redis ByteCache state."""
    try:
        from src.utils.byte_cache import byte_cache

        monkeypatch.setattr(byte_cache, "_enabled", False)
    except Exception:
        pass
