import asyncio
import logging
from types import MethodType

from apps.domain.models import (
    CandidateCard,
    PlannerOutput,
    PublicationEvidence,
    RecommendationDecision,
)
from apps.recommendation.evidence_selector import (
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
)
from apps.recommendation.reasoner import ReasonGenerationOutput, ReasonedCandidate
from apps.recommendation.service import (
    EMPTY_RETRIEVAL_KEYWORDS_REASON,
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
    def build_small_cards(self, hits, plan):
        raise AssertionError("build_small_cards should not be called in this unit test")


class DummyEvidenceSelector:
    def __init__(self) -> None:
        self.received_candidate_ids: list[list[str]] = []
        self.last_trace = {"mode": "test_selector", "candidate_evidence_counts": []}

    def select(self, *, candidates, plan):
        _ = plan
        candidate_ids = [candidate.expert_id for candidate in candidates]
        self.received_candidate_ids.append(candidate_ids)
        self.last_trace = {
            "mode": "test_selector",
            "candidate_evidence_counts": [
                {
                    "expert_id": candidate.expert_id,
                    "papers": 1 if candidate.top_papers else 0,
                    "projects": 0,
                    "patents": 0,
                    "total": 1 if candidate.top_papers else 0,
                }
                for candidate in candidates
            ],
        }
        return {
            candidate.expert_id: RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                papers=[
                    RelevantEvidenceItem(
                        item_id="paper:0",
                        type="paper",
                        title=candidate.top_papers[0].publication_title,
                        date=candidate.top_papers[0].publication_year_month,
                        detail=candidate.top_papers[0].journal_name,
                        match_score=10.0,
                    )
                ]
                if candidate.top_papers
                else [],
            )
            for candidate in candidates
        }


class RecordingReasonGenerator:
    def __init__(self, output: ReasonGenerationOutput | list[ReasonGenerationOutput]) -> None:
        self.outputs = output if isinstance(output, list) else [output]
        self.called = False
        self.call_count = 0
        self.received_candidate_ids: list[list[str]] = []
        self.received_relevant_evidence_ids: list[list[str]] = []
        self.last_trace = {"mode": "test"}

    async def generate(
        self,
        *,
        query,
        plan,
        candidates,
        relevant_evidence_by_expert_id=None,
        retrieval_score_traces_by_expert_id=None,
    ):
        _ = (query, plan, retrieval_score_traces_by_expert_id)
        self.called = True
        output_index = min(self.call_count, len(self.outputs) - 1)
        output = self.outputs[output_index]
        self.call_count += 1
        self.received_candidate_ids.append([candidate.expert_id for candidate in candidates])
        self.received_relevant_evidence_ids.append(
            sorted((relevant_evidence_by_expert_id or {}).keys())
        )
        self.last_trace = {
            "mode": "test",
            "candidate_count": len(candidates),
            "output_count": len(output.items),
            "raw_output_count": len(output.items),
            "returned_ids": [item.expert_id for item in output.items],
            "missing_candidate_ids": [
                candidate.expert_id
                for candidate in candidates
                if candidate.expert_id not in {item.expert_id for item in output.items}
            ],
            "empty_reason_candidate_ids": [
                item.expert_id
                for item in output.items
                if not item.recommendation_reason
            ],
            "empty_selected_evidence_candidate_ids": [
                item.expert_id
                for item in output.items
                if not item.selected_evidence_ids
            ],
        }
        return output


def _candidate_card(expert_id: str, name: str, score: float) -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=name,
        organization="Test Institute",
        branch_presence_flags={"basic": True, "art": True, "pat": False, "pjt": False},
        counts={"article_cnt": 1, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 0},
        top_papers=[
            PublicationEvidence(
                publication_title=f"Paper {expert_id}",
                publication_year_month="2026-01",
                journal_name="Test Journal",
            )
        ],
        shortlist_score=score,
        rank_score=score,
    )


def _plan(top_k: int = 2) -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend semiconductor reviewers",
        core_keywords=["semiconductor", "review"],
        task_terms=["reviewer recommendation"],
        top_k=top_k,
    )


def _build_service(
    reason_output: ReasonGenerationOutput,
) -> tuple[RecommendationService, RecordingReasonGenerator, DummyEvidenceSelector]:
    reason_generator = RecordingReasonGenerator(reason_output)
    evidence_selector = DummyEvidenceSelector()
    service = RecommendationService(
        planner=DummyPlanner(),
        retriever=DummyRetriever(),
        filter_compiler=DummyFilterCompiler(),
        card_builder=DummyCardBuilder(),
        evidence_selector=evidence_selector,
        reason_generator=reason_generator,
        feedback_store=DummyFeedbackStore(),
    )
    return service, reason_generator, evidence_selector


def _bind_search_result(
    service: RecommendationService,
    *,
    cards: list[CandidateCard],
    retrieved_count: int,
    planner_trace: dict | None = None,
    retrieval_skipped_reason: str | None = None,
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
            "planner_trace": planner_trace
            or {
                "mode": "test",
                "planner_retry_count": 0,
                "planner_keywords": ["semiconductor", "review"],
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
            "retrieval_score_traces": [
                {
                    "expert_id": card.expert_id,
                    "point_id": f"{card.expert_id}_basic",
                    "point_branch_hint": "basic",
                    "final_score": card.shortlist_score,
                    "primary_branch": "basic",
                    "branch_matches": [
                        {"branch": "basic", "rank": index + 1, "score": card.shortlist_score}
                    ],
                }
                for index, card in enumerate(cards)
            ],
            "raw_query": query,
            "retrieval_skipped_reason": retrieval_skipped_reason,
            "final_sort_policy": "rrf_score_desc_name_asc",
            "timers": {
                "plan_ms": 1.0,
                "search_ms": 2.0,
            },
        }

    service.search_candidates = MethodType(fake_search_candidates, service)


def test_recommend_returns_empty_when_no_candidates_are_retrieved():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    _bind_search_result(service, cards=[], retrieved_count=0)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert reason_generator.called is False
    assert evidence_selector.received_candidate_ids == []
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]
    assert result["trace"]["reason_generation_trace"] == {}
    assert result["trace"]["planner_keywords"] == ["semiconductor", "review"]


def test_recommend_returns_data_gap_when_retrieval_is_skipped():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    _bind_search_result(
        service,
        cards=[],
        retrieved_count=0,
        planner_trace={
            "mode": "deterministic_fallback",
            "planner_retry_count": 1,
            "planner_keywords": [],
            "retrieval_keywords": [],
        },
        retrieval_skipped_reason=EMPTY_RETRIEVAL_KEYWORDS_REASON,
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert reason_generator.called is False
    assert evidence_selector.received_candidate_ids == []
    assert result["data_gaps"] == [EMPTY_RETRIEVAL_KEYWORDS_REASON]
    assert result["trace"]["retrieval_skipped_reason"] == EMPTY_RETRIEVAL_KEYWORDS_REASON


def test_recommend_sends_only_top_k_to_reason_generator_and_preserves_order():
    service, reason_generator, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="2",
                    fit="중간",
                    recommendation_reason="Reason for second candidate",
                    selected_evidence_ids=["paper:0"],
                ),
                ReasonedCandidate(
                    expert_id="1",
                    fit="높음",
                    recommendation_reason="Reason for first candidate",
                    selected_evidence_ids=["paper:0"],
                ),
            ]
        )
    )
    cards = [
        _candidate_card("1", "Alpha", 98.0),
        _candidate_card("2", "Bravo", 95.0),
        _candidate_card("3", "Charlie", 90.0),
    ]
    _bind_search_result(service, cards=cards, retrieved_count=3)

    result = asyncio.run(service.recommend(query="Recommend reviewers", top_k=2))

    assert reason_generator.called is True
    assert reason_generator.call_count == 1
    assert evidence_selector.received_candidate_ids == [["1", "2"]]
    assert reason_generator.received_candidate_ids == [["1", "2"]]
    assert reason_generator.received_relevant_evidence_ids == [["1", "2"]]
    assert [item.expert_id for item in result["recommendations"]] == ["1", "2"]
    assert result["recommendations"][0].recommendation_reason == "Reason for first candidate"
    assert result["recommendations"][1].recommendation_reason == "Reason for second candidate"
    assert result["recommendations"][0].evidence[0].title == "Paper 1"
    assert result["trace"]["recommendation_ids"] == ["1", "2"]
    assert result["trace"]["retrieval_score_traces"][0]["expert_id"] == "1"
    assert result["trace"]["top_k_used"] == 2
    assert "evidence_selection" in result["trace"]["reason_generation_trace"]
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["resolved_evidence_ids"]
        == ["paper:0"]
    )
    assert result["trace"]["reason_generation_trace"]["batch_count"] == 1


def test_recommend_batches_reason_generation_and_preserves_global_order():
    service, reason_generator, evidence_selector = _build_service(
        [
            ReasonGenerationOutput(
                items=[
                    ReasonedCandidate(
                        expert_id=str(index),
                        fit="보통",
                        recommendation_reason=f"Reason {index}",
                        selected_evidence_ids=["paper:0"],
                    )
                    for index in range(1, 6)
                ]
            ),
            ReasonGenerationOutput(
                items=[
                    ReasonedCandidate(
                        expert_id="6",
                        fit="보통",
                        recommendation_reason="Reason 6",
                        selected_evidence_ids=["paper:0"],
                    )
                ]
            ),
        ]
    )
    cards = [_candidate_card(str(index), f"Candidate {index}", 100.0 - index) for index in range(1, 7)]
    _bind_search_result(service, cards=cards, retrieved_count=6)

    result = asyncio.run(service.recommend(query="Recommend reviewers", top_k=6))

    assert reason_generator.call_count == 2
    assert evidence_selector.received_candidate_ids == [["1", "2", "3", "4", "5", "6"]]
    assert reason_generator.received_candidate_ids == [["1", "2", "3", "4", "5"], ["6"]]
    assert [item.expert_id for item in result["recommendations"]] == [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
    ]
    assert result["trace"]["reason_generation_trace"]["batch_count"] == 2
    assert (
        result["trace"]["reason_generation_trace"]["batches"][0]["candidate_ids"]
        == ["1", "2", "3", "4", "5"]
    )
    assert result["trace"]["reason_generation_trace"]["batches"][1]["candidate_ids"] == ["6"]


def test_recommend_falls_back_to_top_relevant_evidence_when_llm_selection_is_missing():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Strong publication history",
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    _bind_search_result(service, cards=cards, retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    recommendation: RecommendationDecision = result["recommendations"][0]
    assert recommendation.evidence[0].title == "Paper 1"
    assert recommendation.recommendation_reason == "Strong publication history"
    assert result["trace"]["reason_generation_trace"]["selected_evidence"][0]["fallback"] == "top_relevant"
    assert recommendation.fit == "보통"


def test_recommend_ignores_invalid_selected_evidence_ids_and_uses_fallback():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Strong publication history",
                    selected_evidence_ids=["project:99", "paper:999"],
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    _bind_search_result(service, cards=cards, retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    recommendation: RecommendationDecision = result["recommendations"][0]
    assert recommendation.evidence[0].title == "Paper 1"
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["selected_evidence_ids"]
        == ["project:99", "paper:999"]
    )
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["resolved_evidence_ids"]
        == ["paper:0"]
    )
    assert recommendation.fit == "보통"


def test_recommend_logs_empty_reason_and_invalid_evidence_selection(caplog):
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="",
                    selected_evidence_ids=["project:99"],
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    _bind_search_result(service, cards=cards, retrieved_count=1)

    with caplog.at_level(logging.WARNING, logger="apps.recommendation.service"):
        result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert result["recommendations"][0].evidence[0].title == "Paper 1"
    assert result["recommendations"][0].recommendation_reason == (
        "'Paper 1' 논문이 확인되어 질의와 관련된 전문성 근거로 참고할 수 있습니다."
    )
    assert "Selected evidence ids could not be resolved" in caplog.text
    assert "Recommendation reason is empty after reason generation" in caplog.text
    assert "Recommendation reason fallback generated" in caplog.text


def test_recommend_generates_fallback_reason_for_omitted_candidate():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Reason for first candidate",
                    selected_evidence_ids=["paper:0"],
                )
            ]
        )
    )
    cards = [
        _candidate_card("1", "Alpha", 98.0),
        _candidate_card("2", "Bravo", 95.0),
    ]
    _bind_search_result(service, cards=cards, retrieved_count=2)

    result = asyncio.run(service.recommend(query="Recommend reviewers", top_k=2))

    assert result["recommendations"][0].recommendation_reason == "Reason for first candidate"
    assert result["recommendations"][1].recommendation_reason == (
        "'Paper 2' 논문이 확인되어 질의와 관련된 전문성 근거로 참고할 수 있습니다."
    )
    assert result["trace"]["reason_generation_trace"]["server_fallback_reasons"] == [
        {
            "expert_id": "2",
            "source": "top_relevant",
            "resolved_evidence_ids": ["paper:0"],
        }
    ]
