import asyncio
from types import MethodType

from apps.domain.models import CandidateCard, EvidenceItem, JudgeOutput, PlannerOutput, RecommendationDecision
from apps.recommendation.service import RecommendationService


class DummyPlanner:
    async def plan(self, **kwargs):
        raise AssertionError("planner should not be called in this unit test")


class DummyRetriever:
    async def search(self, **kwargs):
        raise AssertionError("retriever should not be called in this unit test")


class DummyFilterCompiler:
    def compile(self, *args, **kwargs):
        raise AssertionError("filter compiler should not be called in this unit test")


class DummyFeedbackStore:
    def save_feedback(self, **kwargs):
        return 1


class DummyCardBuilder:
    def shortlist(self, cards, shortlist_limit):
        return cards[:shortlist_limit]


class RecordingJudge:
    def __init__(self, output: JudgeOutput) -> None:
        self.output = output
        self.called = False

    async def judge(self, *, query, plan, shortlist):
        self.called = True
        return self.output


def _candidate_card() -> CandidateCard:
    return CandidateCard(
        expert_id="1",
        name="Kim Tester",
        organization="Test Institute",
        branch_coverage={"basic": True, "art": True, "pat": False, "pjt": False},
        counts={"article_cnt": 1, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 0},
        shortlist_score=12.0,
    )


def _plan() -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend semiconductor reviewers",
        branch_query_hints={"basic": "basic", "art": "papers", "pat": "patents", "pjt": "projects"},
    )


def _build_service(judge_output: JudgeOutput) -> tuple[RecommendationService, RecordingJudge]:
    judge = RecordingJudge(judge_output)
    service = RecommendationService(
        planner=DummyPlanner(),
        retriever=DummyRetriever(),
        filter_compiler=DummyFilterCompiler(),
        card_builder=DummyCardBuilder(),
        judge=judge,
        feedback_store=DummyFeedbackStore(),
        shortlist_limit=10,
    )
    return service, judge


def _bind_search_result(service: RecommendationService, *, cards: list[CandidateCard], retrieved_count: int = 0) -> None:
    async def fake_search_candidates(self, *, query, filters_override=None, exclude_orgs=None, top_k=None):
        return {
            "planner": _plan(),
            "query_filter": None,
            "retrieved_count": retrieved_count,
            "candidates": cards,
            "query_payload": {"prefetch": [], "query_filter": None, "query": "rrf"},
            "branch_queries": {"basic": "basic", "art": "papers", "pat": "patents", "pjt": "projects"},
        }

    service.search_candidates = MethodType(fake_search_candidates, service)


def test_recommend_skips_judge_when_no_candidates_are_retrieved():
    service, judge = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=0)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["data_gaps"] == []
    assert result["not_selected_reasons"] == ["조건을 만족하는 추천 후보를 찾지 못했습니다."]
    assert result["trace"]["recommendation_evidence_summary"] == []


def test_recommend_skips_judge_when_shortlist_is_empty():
    service, judge = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=2)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == ["조건을 만족하는 추천 후보를 찾지 못했습니다."]


def test_recommend_returns_empty_success_when_judge_returns_no_recommendations():
    service, judge = _build_service(JudgeOutput(recommended=[], not_selected_reasons=[], data_gaps=["paper evidence missing"]))
    _bind_search_result(service, cards=[_candidate_card()], retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert result["recommendations"] == []
    assert result["data_gaps"] == ["paper evidence missing"]
    assert result["not_selected_reasons"] == ["조건을 만족하는 추천 후보를 찾지 못했습니다."]
    assert result["trace"]["recommendation_evidence_summary"] == []


def test_recommend_returns_empty_success_when_all_judge_recommendations_lack_evidence():
    service, judge = _build_service(
        JudgeOutput(
            recommended=[
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="Kim Tester",
                    fit="중간",
                    reasons=["Insufficient evidence"],
                    evidence=[],
                    risks=["Missing paper evidence"],
                )
            ],
            not_selected_reasons=[],
            data_gaps=["paper evidence missing"],
        )
    )
    _bind_search_result(service, cards=[_candidate_card()], retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == ["근거가 충분한 추천 결과를 생성하지 못했습니다."]
    assert result["trace"]["recommendation_evidence_summary"] == []


def test_recommend_filters_evidence_less_items_but_keeps_valid_recommendations():
    service, judge = _build_service(
        JudgeOutput(
            recommended=[
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="Kim Tester",
                    fit="중간",
                    reasons=["Insufficient evidence"],
                    evidence=[],
                    risks=["Missing paper evidence"],
                ),
                RecommendationDecision(
                    rank=2,
                    expert_id="2",
                    name="Lee Valid",
                    fit="높음",
                    reasons=["Strong publication evidence"],
                    evidence=[EvidenceItem(type="paper", title="Semiconductor Paper")],
                    risks=[],
                ),
            ],
            not_selected_reasons=["Existing judge reason"],
            data_gaps=[],
        )
    )
    _bind_search_result(service, cards=[_candidate_card()], retrieved_count=2)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0].expert_id == "2"
    assert result["not_selected_reasons"] == ["Existing judge reason", "근거가 충분한 추천 결과를 생성하지 못했습니다."]
    assert result["trace"]["recommendation_evidence_summary"] == [
        {"expert_id": "2", "evidence_titles": ["Semiconductor Paper"]}
    ]
