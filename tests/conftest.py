import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")


@pytest.fixture
def tenant_context():
    return {"tenant_id": "tenant_abc", "role": "editor", "user_id": "user_1"}
