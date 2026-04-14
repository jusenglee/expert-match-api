import asyncio
from types import MethodType

from apps.domain.models import (
    BasicInfo,
    CandidateCard,
    EvidenceItem,
    ExpertPayload,
    JudgeOutput,
    PlannerOutput,
    RecommendationDecision,
    ResearcherProfile,
)
from apps.recommendation.evidence import EvidenceResolutionResult
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


class DummyEvidenceResolver:
    def __init__(
        self,
        evidence_by_expert_id: dict[str, list[EvidenceItem]] | None = None,
    ) -> None:
        self.evidence_by_expert_id = evidence_by_expert_id or {}
        self.calls: list[tuple[str, str, bool]] = []

    async def resolve(self, *, query, plan, recommendation, payload):
        _ = plan
        self.calls.append((query, recommendation.expert_id, payload is not None))
        evidence = list(self.evidence_by_expert_id.get(recommendation.expert_id, []))
        return EvidenceResolutionResult(
            evidence=evidence,
            alignment_passed=bool(evidence),
            source_option_ids=[],
            notes=[],
            status="resolved" if evidence else "unaligned",
            resolver_mode="test",
        )


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


def _payload(expert_id: str = "1", name: str = "Kim Tester") -> ExpertPayload:
    return ExpertPayload(
        basic_info=BasicInfo(
            researcher_id=expert_id,
            researcher_name=name,
            affiliated_organization="Test Institute",
        ),
        researcher_profile=ResearcherProfile(
            highest_degree="PhD",
            major_field="Semiconductor",
        ),
    )


def _plan() -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend semiconductor reviewers",
        core_keywords=["semiconductor", "review"],
    )


def _build_service(
    judge_output: JudgeOutput,
    *,
    use_map_reduce_judging: bool = True,
    evidence_resolver: DummyEvidenceResolver | None = None,
) -> tuple[RecommendationService, RecordingJudge, DummyEvidenceResolver]:
    judge = RecordingJudge(judge_output)
    resolver = evidence_resolver or DummyEvidenceResolver()
    service = RecommendationService(
        planner=DummyPlanner(),
        retriever=DummyRetriever(),
        filter_compiler=DummyFilterCompiler(),
        card_builder=DummyCardBuilder(),
        judge=judge,
        feedback_store=DummyFeedbackStore(),
        evidence_resolver=resolver,
        shortlist_limit=40,
        use_map_reduce_judging=use_map_reduce_judging,
    )
    return service, judge, resolver


def _bind_search_result(
    service: RecommendationService,
    *,
    cards: list[CandidateCard],
    retrieved_count: int = 0,
    payload_by_expert_id: dict[str, ExpertPayload] | None = None,
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
        _ = (filters_override, include_orgs, exclude_orgs, top_k)
        return {
            "planner": _plan(),
            "planner_trace": {
                "mode": "test",
                "planner_retry_count": 0,
                "planner_raw_keywords": ["semiconductor", "review", "reviewer"],
                "verifier_keywords": ["semiconductor", "review"],
                "retrieval_keywords": ["semiconductor", "review"],
                "verifier_applied": True,
            },
            "query_filter": None,
            "retrieved_count": retrieved_count,
            "candidates": cards,
            "payload_by_expert_id": payload_by_expert_id or {},
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
    service, judge, _ = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=0)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["data_gaps"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]
    assert result["trace"]["planner_trace"]["mode"] == "test"
    assert result["trace"]["judge_trace"] == {}
    assert result["trace"]["recommendation_evidence_summary"] == []
    assert result["trace"]["planner_raw_keywords"] == [
        "semiconductor",
        "review",
        "reviewer",
    ]
    assert result["trace"]["verifier_keywords"] == ["semiconductor", "review"]
    assert result["trace"]["retrieval_keywords"] == ["semiconductor", "review"]
    assert result["trace"]["evidence_resolution_trace"] == []


def test_recommend_skips_judge_when_shortlist_is_empty():
    service, judge, _ = _build_service(JudgeOutput(recommended=[]))
    _bind_search_result(service, cards=[], retrieved_count=2)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is False
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]


def test_recommend_returns_data_gap_when_retrieval_is_skipped():
    service, judge, _ = _build_service(JudgeOutput(recommended=[]))
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
        _ = (filters_override, include_orgs, exclude_orgs, top_k)
        return {
            "planner": _plan(),
            "planner_trace": {
                "mode": "deterministic_fallback",
                "planner_retry_count": 1,
                "planner_raw_keywords": ["reviewer", "recommendation"],
                "verifier_keywords": [],
                "retrieval_keywords": [],
                "verifier_applied": True,
            },
            "query_filter": None,
            "retrieved_count": 0,
            "candidates": [],
            "payload_by_expert_id": {},
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
    assert result["trace"]["verifier_applied"] is True
    assert result["trace"]["retrieval_skipped_reason"] == EMPTY_RETRIEVAL_KEYWORDS_REASON


def test_recommend_returns_empty_success_when_judge_returns_no_recommendations():
    service, judge, _ = _build_service(
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
    service, judge, _ = _build_service(
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
    resolver = DummyEvidenceResolver(
        evidence_by_expert_id={
            "2": [EvidenceItem(type="paper", title="Aligned Semiconductor Paper")]
        }
    )
    service, judge, resolver = _build_service(
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
                    evidence=[EvidenceItem(type="paper", title="Judge Paper Evidence")],
                    risks=[],
                ),
            ],
            not_selected_reasons=["Existing judge reason"],
            data_gaps=[],
        ),
        evidence_resolver=resolver,
    )
    _bind_search_result(
        service,
        cards=[_candidate_card(), _candidate_card("2")],
        retrieved_count=2,
        payload_by_expert_id={"2": _payload("2", "Lee Valid")},
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert resolver.calls == [
        ("Recommend reviewers", "1", False),
        ("Recommend reviewers", "2", True),
    ]
    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0].expert_id == "2"
    assert result["recommendations"][0].evidence[0].title == "Aligned Semiconductor Paper"
    assert result["not_selected_reasons"] == [
        "Existing judge reason",
        INSUFFICIENT_EVIDENCE_REASON,
    ]
    assert result["trace"]["recommendation_evidence_summary"] == [
        {"expert_id": "2", "evidence_titles": ["Aligned Semiconductor Paper"]}
    ]
    assert result["trace"]["evidence_resolution_trace"][1]["alignment_passed"] is True


def test_recommend_delegates_large_shortlist_to_judge_once():
    resolver = DummyEvidenceResolver(
        evidence_by_expert_id={
            "1": [EvidenceItem(type="paper", title="Aligned Semiconductor Paper")]
        }
    )
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
    service, judge, _ = _build_service(
        judge_output,
        use_map_reduce_judging=True,
        evidence_resolver=resolver,
    )
    cards = [_candidate_card(str(index)) for index in range(25)]
    _bind_search_result(
        service,
        cards=cards,
        retrieved_count=25,
        payload_by_expert_id={"1": _payload()},
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert judge.called is True
    assert judge.shortlist_sizes == [25]
    assert result["recommendations"][0].expert_id == "1"


def test_recommend_rebuilds_ui_evidence_from_resolver_output():
    resolver = DummyEvidenceResolver(
        evidence_by_expert_id={
            "1": [EvidenceItem(type="project", title="Fire Drone Project")]
        }
    )
    service, judge, _ = _build_service(
        JudgeOutput(
            recommended=[
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="Kim Tester",
                    fit="높음",
                    reasons=["Strong fire-response project evidence"],
                    evidence=[EvidenceItem(type="paper", title="Judge Paper Evidence")],
                    risks=[],
                )
            ]
        ),
        evidence_resolver=resolver,
    )
    _bind_search_result(
        service,
        cards=[_candidate_card()],
        retrieved_count=1,
        payload_by_expert_id={"1": _payload()},
    )

    result = asyncio.run(service.recommend(query="Recommend fire drone reviewers"))

    assert judge.called is True
    assert result["recommendations"][0].evidence[0].title == "Fire Drone Project"
    assert result["trace"]["evidence_resolution_trace"][0]["status"] == "resolved"
