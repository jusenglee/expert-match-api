import asyncio
import logging
from types import MethodType

from apps.domain.models import (
    CandidateCard,
    PlannerOutput,
    PublicationEvidence,
)
from apps.recommendation.evidence_selector import (
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
)
from apps.recommendation.reasoner import (
    FIT_HIGH,
    FIT_NORMAL,
    ReasonGenerationOutput,
    ReasonedCandidate,
)
from apps.recommendation.service import (
    EMPTY_RETRIEVAL_KEYWORDS_REASON,
    NO_GATE_PASSED_CANDIDATE_REASON,
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
        raise AssertionError("card builder should not be called in this unit test")


class ConfigurableEvidenceSelector:
    def __init__(self) -> None:
        self.bundles_by_id: dict[str, RelevantEvidenceBundle] = {}
        self.received_candidate_ids: list[list[str]] = []
        self.last_trace: dict[str, object] = {
            "mode": "test_selector",
            "candidate_evidence_counts": [],
            "empty_candidate_ids": [],
        }

    def select(self, *, candidates, plan):
        _ = plan
        candidate_ids = [candidate.expert_id for candidate in candidates]
        self.received_candidate_ids.append(candidate_ids)

        diagnostics = []
        empty_candidate_ids = []
        bundles: dict[str, RelevantEvidenceBundle] = {}
        for candidate in candidates:
            bundle = self.bundles_by_id.get(
                candidate.expert_id,
                RelevantEvidenceBundle(expert_id=candidate.expert_id),
            )
            bundles[candidate.expert_id] = bundle
            if not bundle.all_items():
                empty_candidate_ids.append(candidate.expert_id)
            diagnostics.append(
                {
                    "expert_id": candidate.expert_id,
                    "selected_evidence_ids": [item.item_id for item in bundle.all_items()],
                    "direct_match_count": bundle.direct_match_count,
                    "aspect_coverage": bundle.aspect_coverage,
                    "generic_only": bundle.generic_only,
                    "matched_aspects": list(bundle.matched_aspects),
                    "matched_generic_terms": list(bundle.matched_generic_terms),
                    "dedup_dropped_count": bundle.dedup_dropped_count,
                    "future_selected_evidence_ids": list(
                        bundle.future_selected_evidence_ids
                    ),
                    "papers": len(bundle.papers),
                    "projects": len(bundle.projects),
                    "patents": len(bundle.patents),
                    "total": len(bundle.all_items()),
                }
            )
        self.last_trace = {
            "mode": "test_selector",
            "candidate_evidence_counts": diagnostics,
            "empty_candidate_ids": empty_candidate_ids,
        }
        return bundles


class RecordingReasonGenerator:
    def __init__(self, output: ReasonGenerationOutput | list[ReasonGenerationOutput]) -> None:
        self.outputs = output if isinstance(output, list) else [output]
        self.called = False
        self.call_count = 0
        self.received_candidate_ids: list[list[str]] = []
        self.received_relevant_evidence_ids: list[list[str]] = []
        self.last_trace: dict[str, object] = {"mode": "test_reasoner"}

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
        candidate_ids = [candidate.expert_id for candidate in candidates]
        self.received_candidate_ids.append(candidate_ids)
        self.received_relevant_evidence_ids.append(
            sorted((relevant_evidence_by_expert_id or {}).keys())
        )

        returned_ids = [item.expert_id for item in output.items]
        self.last_trace = {
            "mode": "test_reasoner",
            "candidate_count": len(candidates),
            "output_count": len(output.items),
            "raw_output_count": len(output.items),
            "returned_ids": returned_ids,
            "missing_candidate_ids": [
                candidate_id for candidate_id in candidate_ids if candidate_id not in returned_ids
            ],
            "empty_reason_candidate_ids": [
                item.expert_id for item in output.items if not item.recommendation_reason
            ],
            "empty_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_candidate_ids": [],
            "invalid_selected_evidence_ids_by_candidate": {},
            "retry_count": 0,
            "returned_ratio": round(len(returned_ids) / len(candidate_ids), 3)
            if candidate_ids
            else 0.0,
            "prompt_budget_mode": "primary",
            "trim_applied": False,
            "payload_token_estimate": 0,
            "selected_evidence_count": sum(
                len(bundle.all_items())
                for bundle in (relevant_evidence_by_expert_id or {}).values()
            ),
            "retry_triggered": False,
            "retry_trigger": None,
            "retry_reason": None,
            "attempts": [],
        }
        return output


def _candidate_card(expert_id: str, name: str, score: float) -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=name,
        organization="Test Institute",
        degree="PhD",
        major="Semiconductor",
        branch_presence_flags={"basic": True, "art": True, "pat": False, "pjt": False},
        counts={"article_cnt": 1, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 0},
        top_papers=[
            PublicationEvidence(
                publication_title=f"Paper {expert_id}",
                publication_year_month="2025-01",
                journal_name="Test Journal",
                abstract="Semiconductor research summary",
            )
        ],
        shortlist_score=score,
        rank_score=score,
    )


def _plan(*, top_k: int = 3, must_aspects: list[str] | None = None) -> PlannerOutput:
    aspects = must_aspects or ["semiconductor"]
    return PlannerOutput(
        intent_summary="Semiconductor recommendation",
        core_keywords=list(aspects),
        retrieval_core=list(aspects),
        must_aspects=list(aspects),
        top_k=top_k,
    )


def _bundle(
    expert_id: str,
    *items: RelevantEvidenceItem,
    direct_match_count: int | None = None,
    aspect_coverage: int | None = None,
    matched_aspects: list[str] | None = None,
    generic_only: bool = False,
    matched_generic_terms: list[str] | None = None,
    dedup_dropped_count: int = 0,
) -> RelevantEvidenceBundle:
    papers = [item for item in items if item.type == "paper"]
    projects = [item for item in items if item.type == "project"]
    patents = [item for item in items if item.type == "patent"]
    aspect_values = matched_aspects or [
        match for item in items for match in getattr(item, "aspect_matches", [])
    ]
    return RelevantEvidenceBundle(
        expert_id=expert_id,
        papers=papers,
        projects=projects,
        patents=patents,
        matched_aspects=list(dict.fromkeys(aspect_values)),
        matched_generic_terms=list(matched_generic_terms or []),
        direct_match_count=direct_match_count if direct_match_count is not None else len(items),
        aspect_coverage=aspect_coverage if aspect_coverage is not None else len(set(aspect_values)),
        generic_only=generic_only,
        dedup_dropped_count=dedup_dropped_count,
        future_selected_evidence_ids=[
            item.item_id for item in items if getattr(item, "is_future_item", False)
        ],
    )


def _paper_item(
    evidence_id: str,
    title: str,
    *,
    aspect_matches: list[str] | None = None,
    direct_match: bool = True,
) -> RelevantEvidenceItem:
    return RelevantEvidenceItem(
        item_id=evidence_id,
        type="paper",
        title=title,
        date="2025-01",
        detail="Test Journal",
        snippet="Semiconductor evidence snippet",
        matched_keywords=list(aspect_matches or []),
        aspect_matches=list(aspect_matches or []),
        generic_matches=[],
        direct_match=direct_match,
        match_score=10.0,
    )


def _build_service(
    reason_output: ReasonGenerationOutput | list[ReasonGenerationOutput],
) -> tuple[RecommendationService, RecordingReasonGenerator, ConfigurableEvidenceSelector]:
    reason_generator = RecordingReasonGenerator(reason_output)
    evidence_selector = ConfigurableEvidenceSelector()
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
    plan: PlannerOutput | None = None,
    planner_trace: dict | None = None,
    retrieval_skipped_reason: str | None = None,
) -> None:
    bound_plan = plan or _plan()

    async def fake_search_candidates(
        self,
        *,
        query,
        filters_override=None,
        include_orgs=None,
        exclude_orgs=None,
        top_k=None,
        limit_candidates=True,
    ):
        _ = (filters_override, include_orgs, exclude_orgs, top_k, limit_candidates)
        return {
            "planner": bound_plan,
            "planner_trace": planner_trace
            or {
                "mode": "test",
                "planner_retry_count": 0,
                "planner_keywords": list(bound_plan.core_keywords),
                "retrieval_keywords": list(bound_plan.retrieval_core),
                "removed_meta_terms": ["전문가", "추천"],
                "must_aspects": list(bound_plan.must_aspects),
                "generic_terms": list(bound_plan.generic_terms),
            },
            "query_filter": None,
            "retrieved_count": retrieved_count,
            "candidates": cards,
            "query_payload": {"prefetch": [], "query_filter": None, "query": "rrf"},
            "branch_queries": {
                "basic": "semiconductor",
                "art": "semiconductor",
                "pat": "semiconductor",
                "pjt": "semiconductor",
            },
            "retrieval_keywords": list(bound_plan.retrieval_core),
            "retrieval_score_traces": [
                {
                    "expert_id": card.expert_id,
                    "point_id": f"{card.expert_id}_basic",
                    "point_branch_hint": "basic",
                    "final_score": card.shortlist_score,
                    "primary_branch": "basic",
                    "branch_matches": [
                        {
                            "branch": "basic",
                            "rank": index + 1,
                            "score": card.shortlist_score,
                        }
                    ],
                }
                for index, card in enumerate(cards)
            ],
            "expanded_shadow_hits": [],
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

    result = asyncio.run(service.recommend(query="반도체 전문가 추천"))

    assert reason_generator.called is False
    assert evidence_selector.received_candidate_ids == []
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == [NO_MATCHING_CANDIDATE_REASON]
    assert result["trace"]["reason_generation_trace"] == {}
    assert result["trace"]["planner_trace"]["removed_meta_terms"] == ["전문가", "추천"]


def test_recommend_returns_data_gap_when_retrieval_is_skipped():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    _bind_search_result(
        service,
        cards=[],
        retrieved_count=0,
        plan=_plan(must_aspects=[]),
        planner_trace={
            "mode": "deterministic_fallback",
            "planner_retry_count": 1,
            "planner_keywords": [],
            "retrieval_keywords": [],
            "removed_meta_terms": ["평가위원", "추천"],
            "fallback_terms": [],
        },
        retrieval_skipped_reason=EMPTY_RETRIEVAL_KEYWORDS_REASON,
    )

    result = asyncio.run(service.recommend(query="평가위원 추천"))

    assert reason_generator.called is False
    assert evidence_selector.received_candidate_ids == []
    assert result["data_gaps"] == [EMPTY_RETRIEVAL_KEYWORDS_REASON]
    assert result["trace"]["retrieval_skipped_reason"] == EMPTY_RETRIEVAL_KEYWORDS_REASON
    assert result["trace"]["planner_trace"]["fallback_terms"] == []


def test_recommend_applies_gates_before_top_k_and_reason_generation():
    service, reason_generator, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit=FIT_HIGH,
                    recommendation_reason="Candidate 1 summary",
                ),
                ReasonedCandidate(
                    expert_id="2",
                    fit=FIT_NORMAL,
                    recommendation_reason="Candidate 2 summary",
                ),
            ]
        )
    )
    cards = [
        _candidate_card("1", "Alpha", 98.0),
        _candidate_card("2", "Bravo", 95.0),
        _candidate_card("3", "Charlie", 94.0),
        _candidate_card("4", "Delta", 93.0),
    ]
    evidence_selector.bundles_by_id = {
        "1": _bundle(
            "1",
            _paper_item("paper:0", "AI semiconductor design", aspect_matches=["ai semiconductor"]),
            _paper_item("paper:1", "Advanced chip reliability", aspect_matches=["chip reliability"]),
            direct_match_count=2,
            aspect_coverage=2,
            matched_aspects=["ai semiconductor", "chip reliability"],
        ),
        "2": _bundle(
            "2",
            _paper_item("paper:0", "AI semiconductor process", aspect_matches=["ai semiconductor"]),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["ai semiconductor"],
        ),
        "3": _bundle(
            "3",
            _paper_item("paper:0", "Chip reliability survey", aspect_matches=["chip reliability"]),
            direct_match_count=1,
            aspect_coverage=2,
            matched_aspects=["ai semiconductor", "chip reliability"],
            generic_only=True,
            matched_generic_terms=["experience"],
        ),
        "4": _bundle(
            "4",
            direct_match_count=0,
            aspect_coverage=0,
            matched_aspects=[],
        ),
    }
    _bind_search_result(
        service,
        cards=cards,
        retrieved_count=4,
        plan=_plan(top_k=3, must_aspects=["AI semiconductor", "chip reliability"]),
    )

    result = asyncio.run(service.recommend(query="AI semiconductor 평가위원 추천", top_k=2))

    assert evidence_selector.received_candidate_ids == [["1", "2", "3", "4"]]
    assert reason_generator.received_candidate_ids == [["1", "2"]]
    assert [item.expert_id for item in result["recommendations"]] == ["1", "2"]
    shortlist_gate = result["trace"]["reason_generation_trace"]["shortlist_gate"]
    assert shortlist_gate["kept_candidate_ids"] == ["1"]
    assert shortlist_gate["low_coverage_candidate_ids"] == ["2"]
    assert shortlist_gate["generic_only_candidate_ids"] == ["3"]
    assert shortlist_gate["dropped_candidate_ids"] == ["4"]
    assert result["trace"]["top_k_used"] == 2


def test_recommend_batches_reason_generation_after_gating():
    batch_outputs = [
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id=str(index),
                    fit=FIT_NORMAL,
                    recommendation_reason=f"Reason {index}",
                )
                for index in range(1, 6)
            ]
        ),
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="6",
                    fit=FIT_NORMAL,
                    recommendation_reason="Reason 6",
                )
            ]
        ),
    ]
    service, reason_generator, evidence_selector = _build_service(batch_outputs)
    cards = [_candidate_card(str(index), f"Candidate {index}", 100.0 - index) for index in range(1, 7)]
    evidence_selector.bundles_by_id = {
        card.expert_id: _bundle(
            card.expert_id,
            _paper_item("paper:0", f"Semiconductor paper {card.expert_id}", aspect_matches=["semiconductor"]),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["semiconductor"],
        )
        for card in cards
    }
    _bind_search_result(service, cards=cards, retrieved_count=6, plan=_plan(top_k=6))

    result = asyncio.run(service.recommend(query="반도체 추천", top_k=6))

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
    assert result["trace"]["reason_generation_trace"]["batches"][0]["candidate_ids"] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert result["trace"]["reason_generation_trace"]["batches"][1]["candidate_ids"] == ["6"]
    assert result["trace"]["reason_generation_trace"]["batches"][0]["retry_triggered"] is False


def test_recommend_generates_selected_evidence_fallback_for_empty_reason(caplog):
    service, _, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit=FIT_NORMAL,
                    recommendation_reason="",
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    evidence_selector.bundles_by_id = {
        "1": _bundle(
            "1",
            _paper_item("paper:0", "Semiconductor process platform", aspect_matches=["semiconductor"]),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["semiconductor"],
        )
    }
    _bind_search_result(service, cards=cards, retrieved_count=1)

    with caplog.at_level(logging.WARNING, logger="apps.recommendation.service"):
        result = asyncio.run(service.recommend(query="반도체 전문가 추천"))

    recommendation = result["recommendations"][0]
    assert "Semiconductor process platform" in recommendation.recommendation_reason
    assert result["trace"]["reason_generation_trace"]["server_fallback_reasons"] == [
        {
            "expert_id": "1",
            "source": "selected_evidence",
            "resolved_evidence_ids": ["paper:0"],
        }
    ]
    assert "Recommendation reason fallback generated" in caplog.text


def test_recommend_validator_replaces_reason_when_other_candidate_name_leaks(caplog):
    service, _, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit=FIT_HIGH,
                    recommendation_reason="Bravo has the strongest semiconductor record.",
                )
            ]
        )
    )
    cards = [
        _candidate_card("1", "Alpha", 98.0),
        _candidate_card("2", "Bravo", 95.0),
    ]
    evidence_selector.bundles_by_id = {
        "1": _bundle(
            "1",
            _paper_item("paper:0", "Semiconductor design evidence", aspect_matches=["semiconductor"]),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["semiconductor"],
        ),
        "2": _bundle(
            "2",
            _paper_item("paper:0", "Semiconductor backup evidence", aspect_matches=["semiconductor"]),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["semiconductor"],
        ),
    }
    _bind_search_result(service, cards=cards, retrieved_count=2, plan=_plan(top_k=2))

    with caplog.at_level(logging.WARNING, logger="apps.recommendation.service"):
        result = asyncio.run(service.recommend(query="반도체 추천", top_k=2))

    recommendation = result["recommendations"][0]
    assert recommendation.expert_id == "1"
    assert "Bravo has the strongest" not in recommendation.recommendation_reason
    assert "Semiconductor design evidence" in recommendation.recommendation_reason
    validator_trace = result["trace"]["reason_generation_trace"]["reason_sync_validator"]
    assert validator_trace["fallback_count"] == 1
    assert validator_trace["violations"][0]["violations"] == ["other_candidate_name"]
    assert result["trace"]["reason_generation_trace"]["server_fallback_reasons"][0]["source"] == (
        "reason_sync_validator"
    )
    assert "validator fallback generated" in caplog.text


def test_recommend_returns_gated_empty_when_no_candidate_passes():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    cards = [
        _candidate_card("1", "Alpha", 98.0),
        _candidate_card("2", "Bravo", 95.0),
    ]
    evidence_selector.bundles_by_id = {
        "1": _bundle("1", direct_match_count=0, aspect_coverage=0),
        "2": _bundle("2", direct_match_count=0, aspect_coverage=0),
    }
    _bind_search_result(service, cards=cards, retrieved_count=2)

    result = asyncio.run(service.recommend(query="전문가 추천", top_k=2))

    assert reason_generator.called is False
    assert result["recommendations"] == []
    assert result["not_selected_reasons"] == [NO_GATE_PASSED_CANDIDATE_REASON]
    assert result["trace"]["reason_generation_trace"]["mode"] == "gated_empty"
    assert result["trace"]["reason_generation_trace"]["shortlist_gate"]["dropped_candidate_ids"] == [
        "1",
        "2",
    ]


def test_required_aspect_coverage_uses_phrase_count_not_token_count():
    plan = _plan(top_k=3, must_aspects=["medical imaging analysis"])

    assert RecommendationService._required_aspect_coverage(plan) == 1


def test_recommend_validator_uses_bundle_scope_not_title_only():
    service, _, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit=FIT_HIGH,
                    recommendation_reason="의료영상 분석 경험이 직접 확인됩니다.",
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    evidence_selector.bundles_by_id = {
        "1": _bundle(
            "1",
            RelevantEvidenceItem(
                item_id="paper:0",
                type="paper",
                title="General AI study",
                date="2025-01",
                detail="Clinical imaging collaboration",
                snippet="medical imaging analysis pipeline and validation",
                matched_keywords=["medical imaging analysis"],
                aspect_matches=["medical imaging analysis"],
                generic_matches=[],
                direct_match=True,
                match_score=10.0,
            ),
            direct_match_count=1,
            aspect_coverage=1,
            matched_aspects=["medical imaging analysis"],
        )
    }
    _bind_search_result(
        service,
        cards=cards,
        retrieved_count=1,
        plan=_plan(top_k=1, must_aspects=["medical imaging analysis"]),
    )

    result = asyncio.run(service.recommend(query="medical imaging analysis", top_k=1))

    recommendation = result["recommendations"][0]
    assert recommendation.recommendation_reason == "의료영상 분석 경험이 직접 확인됩니다."
    validator_trace = result["trace"]["reason_generation_trace"]["reason_sync_validator"]
    assert validator_trace["fallback_count"] == 0
