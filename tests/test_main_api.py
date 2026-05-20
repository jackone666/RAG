"""Tests for main.py — FastAPI endpoint integration tests."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(_mock_mcp_for_main_tests):
    """Create a TestClient with mocked dependencies for offline testing."""
    with patch("src.observability.tracer.init_observability"):
        with patch("src.middleware.rate_limiter.rate_limiter.is_allowed", AsyncMock(return_value=(True, 1))):
            from main import app

            with TestClient(app) as c:
                yield c


@pytest.fixture(autouse=False)
def _mock_mcp_for_main_tests():
    """Prevent MCP module import from loading heavy dependencies in main tests."""
    with patch("src.mcp_server.server.mcp"):
        yield


class TestRootEndpoint:
    def test_root_redirects_to_login(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/login" in response.headers.get("location", "")

    def test_login_page_accessible(self, client):
        response = client.get("/login")
        assert response.status_code == 200


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "IntelliLens-MCP"


class TestAppEndpoint:
    def test_app_returns_html_page(self, client):
        response = client.get("/app")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


class TestTenantInfoEndpoint:
    def test_tenant_info_returns_context(self, client):
        with patch("src.api.documents._query_documents_for_tenant", return_value=[]):
            response = client.get(
                "/v1/tenants/me",
                headers={"X-Tenant-ID": "tenant_abc", "X-Role": "editor"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == "tenant_abc"
            assert data["role"] == "editor"
            assert "document_count" in data

    def test_tenant_info_default_role(self, client):
        with patch("src.api.documents._query_documents_for_tenant", return_value=[]):
            response = client.get(
                "/v1/tenants/me",
                headers={"X-Tenant-ID": "tenant_xyz"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["role"] == "viewer"

    def test_tenant_info_jwt_auth(self, client):
        import jwt

        token = jwt.encode(
            {"tenant_id": "jwt_tenant", "role": "admin", "sub": "user1"},
            "secret",
            algorithm="HS256",
        )
        with patch("src.middleware.auth.settings") as mock_settings:
            mock_settings.jwt_secret_key = "secret"
            mock_settings.jwt_algorithm = "HS256"
            with patch("src.api.documents._query_documents_for_tenant", return_value=[]):
                response = client.get(
                    "/v1/tenants/me",
                    headers={"Authorization": f"Bearer {token}"},
                )
        assert response.status_code == 200
        data = response.json()
        assert data["tenant_id"] == "jwt_tenant"
        assert data["role"] == "admin"


class TestQueryEndpoint:
    def test_query_returns_answer_for_happy_path(self, client):
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                [MagicMock(spec=["node"], node=MagicMock(get_content=lambda: "test content"))],
                MagicMock(vector_count=5, keyword_count=3, fusion_count=8, fusion_mode="relative_score",
                          vector_latency_ms=45.0, keyword_latency_ms=23.0, top_scores=[0.9, 0.8]),
                1,
            )
            with patch("main.query_engine.rerank") as mock_rerank:
                mock_rerank.return_value = mock_retrieve.return_value[0]
                with patch("main.query_engine.query_with_trace") as mock_gen:
                    mock_gen.return_value = ("Test answer", MagicMock(
                        model_used="gpt-4o", fallback_used=False, latency_ms=1200.0,
                        prompt_tokens=500, completion_tokens=200, total_tokens=700,
                        prompt_preview="You are a..."
                    ))
                    with patch("main.async_evaluator.evaluate", AsyncMock()):
                        response = client.post(
                            "/v1/query",
                            json={"query": "test question?"},
                            headers={"X-Tenant-ID": "tenant_abc", "X-Role": "viewer"},
                        )
                        assert response.status_code == 200
                        data = response.json()
                        assert data["answer"] == "Test answer"
                        assert data["tenant_id"] == "tenant_abc"
                        assert data["node_count"] == 1
                        assert "pipeline_trace" in data
                        assert "query_id" in data

    def test_query_no_results_returns_fallback_message(self, client):
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                [],
                MagicMock(vector_count=0, keyword_count=0, fusion_count=0, top_scores=[],
                          vector_latency_ms=0.0, keyword_latency_ms=0.0, fusion_mode=""),
                0,
            )
            response = client.post(
                "/v1/query",
                json={"query": "nonexistent topic"},
                headers={"X-Tenant-ID": "tenant_abc", "X-Role": "viewer"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "未找到相关文档" in data["answer"]
            assert data["tenant_id"] == "tenant_abc"
            assert data["node_count"] == 0

    def test_query_background_evaluation_triggered(self, client):
        nodes = [MagicMock(spec=["node"], node=MagicMock(get_content=lambda: "ctx"))]
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                nodes,
                MagicMock(vector_count=5, keyword_count=3, fusion_count=8, fusion_mode="relative_score",
                          vector_latency_ms=45.0, keyword_latency_ms=23.0, top_scores=[0.9]),
                0,
            )
            with patch("main.query_engine.rerank", AsyncMock(return_value=nodes)):
                with patch("main.query_engine.query_with_trace") as mock_gen:
                    mock_gen.return_value = ("answer", MagicMock(
                        model_used="gpt-4o", fallback_used=False, latency_ms=1000.0,
                        prompt_tokens=100, completion_tokens=50, total_tokens=150,
                        prompt_preview=""
                    ))
                    with patch("main.async_evaluator.evaluate", AsyncMock()) as mock_eval:
                        response = client.post(
                            "/v1/query",
                            json={"query": "q"},
                            headers={"X-Tenant-ID": "t1"},
                        )
                        assert response.status_code == 200
                        mock_eval.assert_called_once()

    def test_query_passes_tenant_context_to_retriever(self, client):
        nodes = [MagicMock(spec=["node"], node=MagicMock(get_content=lambda: "ctx"))]
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                nodes,
                MagicMock(vector_count=5, keyword_count=3, fusion_count=8, fusion_mode="relative_score",
                          vector_latency_ms=45.0, keyword_latency_ms=23.0, top_scores=[0.9]),
                0,
            )
            with patch("main.query_engine.rerank", AsyncMock(return_value=nodes)):
                with patch("main.query_engine.query_with_trace") as mock_gen:
                    mock_gen.return_value = ("ans", MagicMock(
                        model_used="gpt-4o", fallback_used=False, latency_ms=1000.0,
                        prompt_tokens=100, completion_tokens=50, total_tokens=150,
                        prompt_preview=""
                    ))
                    with patch("main.async_evaluator.evaluate", AsyncMock()):
                        client.post(
                            "/v1/query",
                            json={"query": "test"},
                            headers={"X-Tenant-ID": "specific_tenant"},
                        )
                        call_kwargs = mock_retrieve.call_args.kwargs
                        assert call_kwargs["tenant_id"] == "specific_tenant"
                        assert call_kwargs["query"] == "test"

    def test_query_default_tenant_context(self, client):
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = ([], MagicMock(
                vector_count=0, keyword_count=0, fusion_count=0, top_scores=[],
                vector_latency_ms=0.0, keyword_latency_ms=0.0, fusion_mode=""), 0)
            response = client.post(
                "/v1/query",
                json={"query": "test"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["tenant_id"] == "default"
            assert "未找到相关文档" in data["answer"]

    def test_streaming_endpoint_returns_sse(self, client):
        nodes = [MagicMock(spec=["node"], node=MagicMock(get_content=lambda: "ctx"))]
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                nodes,
                MagicMock(vector_count=5, keyword_count=3, fusion_count=8, fusion_mode="relative_score",
                          vector_latency_ms=45.0, keyword_latency_ms=23.0, top_scores=[0.9]),
                1,
            )
            with patch("main.query_engine.rerank", AsyncMock(return_value=nodes)):
                with patch("main.query_engine.query_stream") as mock_stream:
                    async def fake_stream(*args, **kwargs):
                        yield {"token": "Hi"}
                        yield {"done": True, "model_used": "gpt-4o", "fallback": False}
                    mock_stream.return_value = fake_stream()
                    with patch("main._evaluate_and_cache", AsyncMock()):
                        response = client.post(
                            "/v1/query/stream",
                            json={"query": "test"},
                            headers={"X-Tenant-ID": "t1"},
                        )
                        assert response.status_code == 200
                        assert "text/event-stream" in response.headers.get("content-type", "")

    def test_query_response_includes_pipeline_trace(self, client):
        nodes = [MagicMock(spec=["node"], node=MagicMock(get_content=lambda: "ctx"))]
        with patch("main._retrieve_with_rewrite_and_cache") as mock_retrieve:
            mock_retrieve.return_value = (
                nodes,
                MagicMock(vector_count=7, keyword_count=4, fusion_count=10, fusion_mode="relative_score",
                          vector_latency_ms=50.0, keyword_latency_ms=25.0, top_scores=[0.95, 0.88, 0.82]),
                2,
            )
            with patch("main.query_engine.rerank", AsyncMock(return_value=nodes)):
                with patch("main.query_engine.query_with_trace") as mock_gen:
                    mock_gen.return_value = ("answer", MagicMock(
                        model_used="gpt-4o", fallback_used=False, latency_ms=1500.0,
                        prompt_tokens=800, completion_tokens=300, total_tokens=1100,
                        prompt_preview="You are an..."
                    ))
                    with patch("main.async_evaluator.evaluate", AsyncMock()):
                        response = client.post(
                            "/v1/query",
                            json={"query": "test"},
                            headers={"X-Tenant-ID": "t1"},
                        )
                        data = response.json()
                        trace = data["pipeline_trace"]
                        assert trace["vector_search"]["count"] == 7
                        assert trace["keyword_search"]["count"] == 4
                        assert trace["fusion"]["count"] == 10
                        assert trace["generation"]["details"]["total_tokens"] == 1100


class TestOpenAPIDocs:
    def test_docs_endpoint_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_schema_accessible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "IntelliLens-MCP"
        assert "/v1/query" in schema["paths"]
        assert "/health" in schema["paths"]


class TestEvaluationEndpoints:
    def test_eval_stats_returns_data(self, client):
        with patch("src.storage.pg_store.get_bad_case_stats", return_value={
            "total_queries": 5, "bad_cases": 1, "pass_rate": 0.8, "avg_score": 0.85,
            "avg_faithfulness": 0.9, "avg_relevancy": 0.85, "avg_correctness": 0.88,
            "avg_completeness": 0.8, "recent_bad_cases": [],
        }):
            response = client.get("/v1/evaluation/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["total_queries"] == 5
            assert data["bad_cases"] == 1

    def test_eval_query_not_found(self, client):
        response = client.get("/v1/evaluation/nonexistent_id")
        assert response.status_code == 404
