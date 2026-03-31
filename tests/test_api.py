from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.schemas import RecommendationResponse, ReadinessResponse
from apps.core.config import Settings
from apps.domain.models import RecommendationDecision


class FakeRecommendationService:
    async def recommend(self, *, query, filters_override, exclude_orgs, top_k):
        return {
            "intent_summary": query,
            "applied_filters": filters_override,
            "searched_branches": ["basic", "art", "pat", "pjt"],
            "retrieved_count": 4,
            "recommendations": [
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="홍길동",
                    fit="높음",
                    reasons=["논문 근거가 확인됨"],
                    evidence=[{"type": "paper", "title": "테스트 논문"}],
                    risks=[],
                )
            ],
            "data_gaps": [],
            "not_selected_reasons": [],
            "trace": {
                "planner": {},
                "branch_queries": {"basic": "기본", "art": "논문", "pat": "특허", "pjt": "과제"},
                "exclude_orgs": [],
                "candidate_ids": ["1"],
                "recommendation_evidence_summary": [{"expert_id": "1", "evidence_titles": ["테스트 논문"]}],
                "query_payload": {},
            },
        }

    async def search_candidates(self, *, query, filters_override, exclude_orgs, top_k):
        class Planner:
            intent_summary = query
            hard_filters = filters_override

            def model_dump(self, mode="json"):
                return {"intent_summary": self.intent_summary, "hard_filters": self.hard_filters}

        class Card:
            expert_id = "1"
            name = "홍길동"
            organization = "테스트기관"
            branch_coverage = {"basic": True, "art": True, "pat": False, "pjt": True}
            counts = {"article_cnt": 2, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 3}
            data_gaps = ["특허 근거 부족"]
            risks = []
            shortlist_score = 17.0

        return {
            "planner": Planner(),
            "retrieved_count": 1,
            "candidates": [Card()],
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
            issues=[] if self._ready else ["컬렉션 누락"],
            collection_name="expert_master",
            sample_point_id="1" if self._ready else None,
        )


def test_recommend_endpoint_contract():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post("/recommend", json={"query": "AI 반도체 평가위원 추천"})

    assert response.status_code == 200
    parsed = RecommendationResponse.model_validate(response.json())
    assert parsed.searched_branches == ["basic", "art", "pat", "pjt"]


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
                "query": "AI 반도체 평가위원 추천",
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


def test_readiness_endpoint_returns_503_when_not_ready():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(ready=False),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
