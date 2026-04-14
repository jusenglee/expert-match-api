from __future__ import annotations

import apps.api.main as main_module
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.schemas import RecommendationResponse, ReadinessResponse
from apps.core.config import Settings
from apps.domain.models import RecommendationDecision


class FakeRecommendationService:
    async def recommend(self, *, query, filters_override, include_orgs, exclude_orgs, top_k):
        return {
            "intent_summary": query,
            "applied_filters": filters_override,
            "searched_branches": ["basic", "art", "pat", "pjt"],
            "retrieved_count": 4,
            "recommendations": [
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="Hong Gildong",
                    fit="높음",
                    reasons=["Publication evidence is available."],
                    evidence=[{"type": "paper", "title": "Test paper"}],
                    risks=[],
                )
            ],
            "data_gaps": [],
            "not_selected_reasons": [],
            "trace": {
                "planner": {},
                "branch_queries": {"basic": "basic", "art": "paper", "pat": "patent", "pjt": "project"},
                "exclude_orgs": [],
                "candidate_ids": ["1"],
                "recommendation_evidence_summary": [{"expert_id": "1", "evidence_titles": ["Test paper"]}],
                "query_payload": {},
            },
        }

    async def search_candidates(self, *, query, filters_override, include_orgs, exclude_orgs, top_k):
        class Planner:
            intent_summary = query
            hard_filters = filters_override

            def model_dump(self, mode="json"):
                return {"intent_summary": self.intent_summary, "hard_filters": self.hard_filters}

        Planner.exclude_orgs = exclude_orgs
        Planner.include_orgs = include_orgs

        class Card:
            expert_id = "1"
            name = "Hong Gildong"
            organization = "Test Institute"
            branch_coverage = {"basic": True, "art": True, "pat": False, "pjt": True}
            counts = {"article_cnt": 2, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 3}
            data_gaps = ["Patent evidence is missing."]
            risks = []
            shortlist_score = 17.0

        return {
            "planner": Planner(),
            "retrieved_count": 1,
            "candidates": [Card()],
            "branch_queries": {"basic": "basic", "art": "paper", "pat": "patent", "pjt": "project"},
            "query_payload": {"prefetch": [], "query_filter": None, "query": "rrf"},
        }

    def save_feedback(self, *, query, selected_expert_ids, rejected_expert_ids, notes, metadata):
        return 1

    @staticmethod
    def _serialize_query_payload(payload):
        return payload


class FakeValidator:
    def __init__(self, ready: bool = True):
        self._ready = ready

    def validate(self):
        from apps.search.live_validator import LiveValidationReport

        return LiveValidationReport(
            ready=self._ready,
            checks={"collection_exists": self._ready},
            issues=[] if self._ready else ["Collection is not ready."],
            collection_name="researcher_recommend_proto",
            sample_point_id="1" if self._ready else None,
        )


def test_recommend_endpoint_contract():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post("/recommend", json={"query": "Recommend AI semiconductor reviewers"})

    assert response.status_code == 200
    parsed = RecommendationResponse.model_validate(response.json())
    assert parsed.searched_branches == ["basic", "art", "pat", "pjt"]


def test_playground_route_serves_local_chat_ui():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.get("/playground")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="chatForm"' in response.text
    assert "/recommend" in response.text
    assert "/search/candidates" in response.text
    assert "Readiness details" in response.text


def test_feedback_endpoint_contract():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json={
                "query": "Recommend AI semiconductor reviewers",
                "selected_expert_ids": ["1"],
                "rejected_expert_ids": [],
                "metadata": {"operator": "tester"},
            },
        )

    assert response.status_code == 200
    assert response.json()["feedback_id"] == 1


def test_readiness_endpoint_returns_report_when_ready():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(ready=True),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    parsed = ReadinessResponse.model_validate(response.json())
    assert parsed.ready is True


def test_readiness_endpoint_returns_503_with_same_shape_when_not_ready():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(ready=False),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    parsed = ReadinessResponse.model_validate(response.json())
    assert parsed.ready is False
    assert "detail" not in response.json()


def test_startup_failure_is_reported_by_readiness_and_service_endpoints(monkeypatch):
    original_build_app_runtime = main_module.build_app_runtime

    async def fail_build_app_runtime(settings):
        raise RuntimeError("Qdrant bootstrap failed")

    monkeypatch.setattr(main_module, "build_app_runtime", fail_build_app_runtime)
    try:
        app = create_app(settings=Settings(app_env="test", strict_runtime_validation=True))
        with TestClient(app) as client:
            health_response = client.get("/health")
            ready_response = client.get("/health/ready")
            recommend_response = client.post("/recommend", json={"query": "Test query"})
    finally:
        monkeypatch.setattr(main_module, "build_app_runtime", original_build_app_runtime)

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    ready_payload = ReadinessResponse.model_validate(ready_response.json())
    assert ready_payload.ready is False
    assert ready_payload.checks == {"startup_runtime_initialized": False}
    assert ready_payload.issues == ["Application startup failed: Qdrant bootstrap failed"]

    assert recommend_response.status_code == 503
    assert "Qdrant bootstrap failed" in recommend_response.json()["detail"]
