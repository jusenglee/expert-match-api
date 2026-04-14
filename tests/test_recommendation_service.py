import asyncio
from types import MethodType

from apps.domain.models import (
    CandidateCard,
    EvidenceItem,
    JudgeOutput,
    PlannerOutput,
    RecommendationDecision,
)
from apps.recommendation.service import (
    EMPTY_RETRIEVAL_KEYWORDS_REASON,
    INSUFFICIENT_EVIDENCE_REASON,
    NO_MATCHING_CANDIDATE_REASON,
    RecommendationService,
)


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
        self.shortlist_sizes: list[int] = []

    async def judge(self, *, query, plan, shortlist):
        self.called = True
        self.shortlist_sizes.append(len(shortlist))
        return self.output


def _candidate_card(expert_id: str = "1") -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=f"Kim Tester {expert_id}",
        organization="Test Institute",
        branch_coverage={"basic": True, "art": True, "pat": False, "pjt": False},
        counts={"article_cnt": 1, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 0},
        shortlist_score=12.0,
        rank_score=12.0,
    )


def _plan() -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend semiconductor reviewers",
        core_keywords=["semiconductor", "review"],
    )


def _build_service(
    judge_output: JudgeOutput, *, use_map_reduce_judging: bool = True
) -> tuple[RecommendationService, RecordingJudge]:
    judge = RecordingJudge(judge_output)
    service = RecommendationService(
        planner=DummyPlanner(),
        retriever=DummyRetriever(),
        filter_compiler=DummyFilterCompiler(),
        card_builder=DummyCardBuilder(),
        judge=judge,
        feedback_store=DummyFeedbackStore(),
        shortlist_limit=40,
        use_map_reduce_judging=use_map_reduce_judging,
    )
    return service, judge


def _bind_search_result(
    service: RecommendationService,
    *,
    cards: list[CandidateCard],
    retrieved_count: int = 0,
) -> None:
    async def fake_search_candidates(
        self,
        *,
        query,
        filters_override=None,
        include_orgs=None,
        exclude_orgs=None,
        top_k=None,
    ):
        return {
            "planner": _plan(),
            "planner_trace": {
                "mode": "test",
                "planner_retry_count": 0,
                "retrieval_keywords": ["semiconductor", "review"],
            },
            "query_filter": None,
            "retrieved_count": retrieved_count,
            "candidates": cards,
            "query_payload": {"prefetch": [], "query_filter": None, "query": "rrf"},
            "branch_queries": {
                "basic": "semiconductor\nreview",
                "art": "semiconductor\nreview",
                "pat": "semiconductor\nreview",
                "pjt": "semiconductor\nreview",
            },
            "retrieval_keywords": ["semiconductor", "review"],
            "raw_query": query,
            "retrieval_skipped_reason": None,
            "timers": {
                "plan_ms": 1.0,
                "search_ms": 2.0,
            },
        }

    service.search_candidates = MethodType(fake_search_candidates, service)


def test_recommend_skips_judge_when_no_candidates_are_retrieved():
    service, judge = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=0)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["data_gaps"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]
    assert result["trace"]["planner_trace"]["mode"] == "test"
    assert result["trace"]["judge_trace"] == {}
    assert result["trace"]["recommendation_evidence_summary"] == []
    assert result["trace"]["retrieval_keywords"] == ["semiconductor", "review"]


def test_recommend_skips_judge_when_shortlist_is_empty():
    service, judge = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=2)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]


def test_recommend_returns_data_gap_when_retrieval_is_skipped():
    service, judge = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=0)

    async def fake_search_candidates(
        self,
        *,
        query,
        filters_override=None,
        include_orgs=None,
        exclude_orgs=None,
        top_k=None,
    ):
        return {
            "planner": _plan(),
            "planner_trace": {
                "mode": "deterministic_fallback",
                "planner_retry_count": 1,
                "retrieval_keywords": [],
            },
            "query_filter": None,
            "retrieved_count": 0,
            "candidates": [],
            "query_payload": {
                "skipped": True,
                "reason": EMPTY_RETRIEVAL_KEYWORDS_REASON,
            },
            "branch_queries": {},
            "retrieval_keywords": [],
            "raw_query": query,
            "retrieval_skipped_reason": EMPTY_RETRIEVAL_KEYWORDS_REASON,
            "timers": {
                "plan_ms": 1.0,
                "search_ms": 0.0,
            },
        }

    service.search_candidates = MethodType(fake_search_candidates, service)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["data_gaps"] == [EMPTY_RETRIEVAL_KEYWORDS_REASON]
    assert result["trace"]["planner_retry_count"] == 1
    assert result["trace"]["retrieval_skipped_reason"] == EMPTY_RETRIEVAL_KEYWORDS_REASON


def test_recommend_returns_empty_success_when_judge_returns_no_recommendations():
    service, judge = _build_service(
        JudgeOutput(
            recommended=[],
            not_selected_reasons=[],
            data_gaps=["paper evidence missing"],
        )
    )
    _bind_search_result(service, cards=[_candidate_card()], retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert result["recommendations"] == []
    assert result["data_gaps"] == ["paper evidence missing"]
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]
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
    assert result["not_selected_reasons"] == [INSUFFICIENT_EVIDENCE_REASON]
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
    assert result["not_selected_reasons"] == [
        "Existing judge reason",
        INSUFFICIENT_EVIDENCE_REASON,
    ]
    assert result["trace"]["recommendation_evidence_summary"] == [
        {"expert_id": "2", "evidence_titles": ["Semiconductor Paper"]}
    ]


def test_recommend_delegates_large_shortlist_to_judge_once():
    judge_output = JudgeOutput(
        recommended=[
            RecommendationDecision(
                rank=1,
                expert_id="1",
                name="Kim Tester 1",
                fit="높음",
                reasons=["Strong publication evidence"],
                evidence=[EvidenceItem(type="paper", title="Semiconductor Paper")],
                risks=[],
            )
        ]
    )
    service, judge = _build_service(judge_output, use_map_reduce_judging=True)
    cards = [_candidate_card(str(index)) for index in range(25)]
    _bind_search_result(service, cards=cards, retrieved_count=25)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert judge.shortlist_sizes == [25]
    assert result["recommendations"][0].expert_id == "1"
