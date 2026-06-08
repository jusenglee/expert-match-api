import asyncio
import logging
from types import MethodType

from apps.domain.models import (
    CandidateCard,
    ChunkEvidence,
    ConceptSpec,
    PlannerOutput,
    RecommendationDecision,
    ResearcherCandidate,
)
from apps.recommendation.evidence_selector import (
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
)
from apps.recommendation.reasoner import ReasonGenerationOutput, ReasonedCandidate
from apps.search.doc_types import DOC_TYPES
from apps.recommendation.service import (
    EMPTY_RETRIEVAL_KEYWORDS_REASON,
    MAX_USER_FACING_RESULTS,
    NO_MATCHING_CANDIDATE_REASON,
    RecommendationService,
)
from apps.search.query_builder import CompiledQueries
from apps.search.retriever import RetrievalResult


def _chunk_id(expert_id: str) -> str:
    """Deterministic flat chunk_id (codec: <doc_type>_<numeric_doc_id>_c<NNN>)."""
    return f"paper_{expert_id}_c000"


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


class LoggingPlanner:
    last_trace = {
        "mode": "test_planner",
        "planner_retry_count": 0,
        "planner_keywords": ["semiconductor"],
        "retrieval_keywords": ["semiconductor"],
    }

    async def plan(self, **kwargs):
        _ = kwargs
        return PlannerOutput(
            intent_summary="semiconductor experts",
            retrieval_core=["semiconductor"],
            core_keywords=["semiconductor"],
            semantic_query="semiconductor research expert",
            bundle_ids=["semiconductor"],
            hard_filters={"highest_degree": "PhD"},
            top_k=2,
        )


class LoggingFilterCompiler:
    def compile(self, *args, **kwargs):
        _ = (args, kwargs)
        return None


class LoggingRetriever:
    async def search(self, **kwargs):
        _ = kwargs
        return RetrievalResult(
            hits=[
                ResearcherCandidate(
                    researcher_id="1",
                    researcher_name="Alpha",
                    group_score=1.0,
                )
            ],
            query_payload={
                "retrieval_mode": "grouped_hybrid_rrf",
                "search_query_plan": {
                    "raw_query": "Recommend reviewers",
                    "dense_query": "Recommend reviewers",
                    "sparse_joint_query": "반도체",
                    "sparse_concept_queries": {"semiconductor": "반도체 시스템반도체 반도체소자 반도체설계"},
                    "required_concepts": ["semiconductor"],
                    "optional_concepts": [],
                },
                "group_count": 3,
                "aggregated_candidate_count": 2,
                "relevance_filtered_candidate_count": 1,
            },
            queries=CompiledQueries(stable="semiconductor", expanded="semiconductor"),
            retrieval_keywords=["semiconductor"],
            retrieval_score_traces=[],
        )


class LoggingCardBuilder:
    def build_small_cards(self, hits, plan):
        _ = (hits, plan)
        return [_candidate_card("1", "Alpha", 1.0)]


class DummyEvidenceSelector:
    """flat selector: reads CandidateCard.evidence_by_type, emits chunk_id-keyed items."""

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
                    "total": len(candidate.evidence_of("paper")),
                    "by_doc_type": {
                        "paper": len(candidate.evidence_of("paper")),
                    },
                }
                for candidate in candidates
            ],
        }
        bundles: dict[str, RelevantEvidenceBundle] = {}
        for candidate in candidates:
            papers = candidate.evidence_of("paper")
            by_doc_type: dict[str, list[RelevantEvidenceItem]] = {}
            if papers:
                ev = papers[0]
                by_doc_type["paper"] = [
                    RelevantEvidenceItem(
                        item_id=ev.chunk_id,
                        type="paper",
                        title=ev.title or "",
                        date=ev.date,
                        detail=(ev.doc_attrs or {}).get("journal_name"),
                        match_score=10.0,
                    )
                ]
            bundles[candidate.expert_id] = RelevantEvidenceBundle(
                expert_id=candidate.expert_id,
                by_doc_type=by_doc_type,
            )
        return bundles


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


def _candidate_card(
    expert_id: str,
    name: str,
    score: float,
    *,
    matched_concepts: list[str] | None = None,
    missing_concepts: list[str] | None = None,
    coverage_type: str = "",
    score_breakdown: dict[str, float] | None = None,
) -> CandidateCard:
    """flat CandidateCard with a single paper ChunkEvidence (item_id == chunk_id)."""
    return CandidateCard(
        expert_id=expert_id,
        name=name,
        organization="Test Institute",
        degree="PhD",
        counts={
            "article_cnt": 1,
            "scie_cnt": 1,
            "patent_cnt": 0,
            "project_cnt": 0,
            "assessor_cnt": 0,
        },
        evidence_by_type={
            "paper": [
                ChunkEvidence(
                    chunk_id=_chunk_id(expert_id),
                    doc_type="paper",
                    title=f"Paper {expert_id}",
                    date="2026-01",
                    snippet=f"Paper {expert_id} abstract",
                    doc_attrs={"journal_name": "Test Journal"},
                    score=score,
                )
            ]
        },
        shortlist_score=score,
        rank_score=score,
        raw_score=score / 100,
        matched_concepts=matched_concepts or [],
        missing_concepts=missing_concepts or [],
        coverage_type=coverage_type,
        score_breakdown=score_breakdown or {},
        top_chunks=[
            {
                "chunk_id": _chunk_id(expert_id),
                "doc_type": "paper",
                "title": f"Paper {expert_id}",
                "concepts": matched_concepts or [],
                "sources": ["dense_full"],
                "score": score / 100,
            }
        ],
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


def test_search_candidates_logs_pipeline_stages(caplog):
    caplog.set_level(logging.INFO)
    service = RecommendationService(
        planner=LoggingPlanner(),
        retriever=LoggingRetriever(),
        filter_compiler=LoggingFilterCompiler(),
        card_builder=LoggingCardBuilder(),
        evidence_selector=DummyEvidenceSelector(),
        reason_generator=RecordingReasonGenerator(ReasonGenerationOutput()),
        feedback_store=DummyFeedbackStore(),
    )

    result = asyncio.run(service.search_candidates(query="Recommend reviewers"))

    assert result["retrieved_count"] == 1
    assert "검색 파이프라인 시작" in caplog.text
    assert "플래너 단계 시작" in caplog.text
    assert "플래너 단계 완료" in caplog.text
    assert "retrieval_core=['semiconductor']" in caplog.text
    assert "retrieval_keywords=['semiconductor']" in caplog.text
    assert "semantic_query='semiconductor research expert'" in caplog.text
    assert "검색 필터 컴파일 완료" in caplog.text
    assert "검색 단계 시작" in caplog.text
    assert "검색 단계 완료" in caplog.text
    assert "후보 카드 생성 완료" in caplog.text


def test_search_candidates_clamps_default_result_count_to_user_facing_maximum():
    class LimitPlanner:
        last_trace = {"mode": "test", "planner_retry_count": 0}

        async def plan(self, **kwargs):
            _ = kwargs
            return PlannerOutput(
                intent_summary="semiconductor experts",
                retrieval_core=["semiconductor"],
                core_keywords=["semiconductor"],
                top_k=30,
            )

    class LimitRetriever:
        async def search(self, **kwargs):
            _ = kwargs
            return RetrievalResult(
                hits=[
                    ResearcherCandidate(
                        researcher_id=str(index),
                        researcher_name=f"Candidate {index}",
                        group_score=100.0 - index,
                    )
                    for index in range(1, 21)
                ],
                query_payload={"retrieval_mode": "multiview_flat_relevance"},
                queries=CompiledQueries(stable="semiconductor", expanded="semiconductor"),
                retrieval_keywords=["semiconductor"],
                retrieval_score_traces=[],
            )

    class LimitCardBuilder:
        def build_small_cards(self, hits, plan):
            _ = plan
            return [
                _candidate_card(hit.researcher_id, hit.researcher_name or "", hit.group_score)
                for hit in hits
            ]

    service = RecommendationService(
        planner=LimitPlanner(),
        retriever=LimitRetriever(),
        filter_compiler=LoggingFilterCompiler(),
        card_builder=LimitCardBuilder(),
        evidence_selector=DummyEvidenceSelector(),
        reason_generator=RecordingReasonGenerator(ReasonGenerationOutput()),
        feedback_store=DummyFeedbackStore(),
    )

    result = asyncio.run(service.search_candidates(query="Recommend reviewers"))

    assert result["top_k_used"] == MAX_USER_FACING_RESULTS
    assert len(result["candidates"]) == MAX_USER_FACING_RESULTS
    assert len(result["hits_with_support"]) == MAX_USER_FACING_RESULTS


def _bind_search_result(
    service: RecommendationService,
    *,
    cards: list[CandidateCard],
    retrieved_count: int,
    planner_top_k: int = 2,
    planner_output: PlannerOutput | None = None,
    planner_trace: dict | None = None,
    query_payload: dict | None = None,
    retrieval_score_traces: list[dict] | None = None,
    filtered_out_candidates: list[dict] | None = None,
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
        default_query_payload = {"prefetch": [], "query_filter": None, "query": "rrf"}
        return {
            "planner": planner_output or _plan(top_k=planner_top_k),
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
            "query_payload": query_payload or default_query_payload,
            "branch_queries": {
                "stable": "semiconductor\nreview",
                "expanded": "semiconductor\nreview",
            },
            "retrieval_keywords": ["semiconductor", "review"],
            "retrieval_score_traces": (
                retrieval_score_traces
                if retrieval_score_traces is not None
                else [
                    {
                        "expert_id": card.expert_id,
                        "point_id": _chunk_id(card.expert_id),
                        "final_score": card.shortlist_score,
                    }
                    for card in cards
                ]
            ),
            "filtered_out_candidates": filtered_out_candidates or [],
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
    assert result["searched_branches"] == list(DOC_TYPES)


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
                    selected_evidence_ids=[_chunk_id("2")],
                ),
                ReasonedCandidate(
                    expert_id="1",
                    fit="높음",
                    recommendation_reason="Reason for first candidate",
                    selected_evidence_ids=[_chunk_id("1")],
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
    # EvidenceItem carries the flat chunk_id and doc_type as evidence.type.
    first_evidence = result["recommendations"][0].evidence[0]
    assert first_evidence.title == "Paper 1"
    assert first_evidence.type == "paper"
    assert first_evidence.chunk_id == _chunk_id("1")
    assert result["trace"]["recommendation_ids"] == ["1", "2"]
    assert result["trace"]["retrieval_score_traces"][0]["expert_id"] == "1"
    assert result["trace"]["top_k_used"] == 2
    assert "evidence_selection" in result["trace"]["reason_generation_trace"]
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["resolved_evidence_ids"]
        == [_chunk_id("1")]
    )
    assert result["trace"]["reason_generation_trace"]["batch_count"] == 1
    assert result["searched_branches"] == list(DOC_TYPES)


def test_recommend_populates_match_score_and_evidence_summaries():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="높음",
                    recommendation_reason="Strong AI semiconductor evidence",
                    selected_evidence_ids=[_chunk_id("1")],
                )
            ]
        )
    )
    plan = PlannerOutput(
        intent_summary="Recommend AI semiconductor reviewers",
        core_keywords=["AI", "semiconductor"],
        required_concepts=["ai", "semiconductor"],
        concept_specs=[
            ConceptSpec(id="ai", label="AI", role="required"),
            ConceptSpec(id="semiconductor", label="반도체", role="required"),
        ],
        top_k=1,
    )
    cards = [
        _candidate_card(
            "1",
            "Alpha",
            98.0,
            matched_concepts=["ai", "semiconductor"],
            coverage_type="joint",
            score_breakdown={"joint": 0.5, "concept": 0.2},
        )
    ]
    _bind_search_result(
        service,
        cards=cards,
        retrieved_count=1,
        planner_output=plan,
        retrieval_score_traces=[
            {
                "expert_id": "1",
                "final_score": 0.77,
                "score_breakdown": {"joint": 0.5, "concept": 0.2},
                "matches": [
                    {
                        "doc_type": "paper",
                        "chunk_id": _chunk_id("1"),
                        "fused_score": 0.42,
                        "concepts": ["ai", "semiconductor"],
                        "sources": ["dense_full", "concept:ai"],
                    }
                ],
            }
        ],
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers", top_k=1))

    recommendation: RecommendationDecision = result["recommendations"][0]
    assert recommendation.match_badges == [
        "AI 충족",
        "반도체 충족",
        "복합 조건 동시 근거",
        "직접 수행 근거 있음",
    ]
    assert recommendation.match_summary == "AI, 반도체 근거가 확인되었습니다."
    assert recommendation.match_details == {
        "matched_concepts": ["ai", "semiconductor"],
        "missing_concepts": [],
        "coverage_type": "joint",
        "matched_doc_types": ["paper"],
        "direct_evidence_count": 1,
    }
    assert recommendation.score_explanation["final_score"] == 0.77
    assert recommendation.score_explanation["rank_score"] == 98.0
    assert recommendation.score_explanation["score_breakdown"] == {
        "joint": 0.5,
        "concept": 0.2,
    }
    assert recommendation.score_explanation["top_chunks"][0] == {
        "doc_type": "paper",
        "title": "Paper 1",
        "concepts": ["ai", "semiconductor"],
        "sources": ["dense_full", "concept:ai"],
        "score": 0.42,
    }
    assert recommendation.evidence_summary == {
        "total_profile_counts": cards[0].counts,
        "matched_evidence_count": 1,
        "shown_evidence_count": 1,
    }


def test_recommend_trace_includes_strict_filter_exclusions():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    _bind_search_result(
        service,
        cards=[],
        retrieved_count=0,
        query_payload={
            "prefetch": [],
            "query_filter": None,
            "query": "rrf",
            "relevance_gate_enabled": True,
            "relevance_gate_active_concepts": ["ai", "semiconductor"],
            "search_query_plan": {"required_concepts": ["ai", "semiconductor"]},
        },
        filtered_out_candidates=[
            {
                "expert_id": "2",
                "name": "Partial",
                "reason": "relevance_concepts_missing",
                "matched_concepts": ["ai"],
                "missing_concepts": ["semiconductor"],
            },
            {
                "expert_id": "3",
                "name": "Org filtered",
                "reason": "excluded_org",
            },
        ],
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    assert reason_generator.called is False
    assert evidence_selector.received_candidate_ids == []
    assert result["trace"]["filtered_out_count"] == 2
    assert result["trace"]["strict_filter"] == {
        "enabled": True,
        "required_concepts": ["ai", "semiconductor"],
        "excluded_candidate_count": 1,
        "excluded_reasons": [
            {
                "expert_id": "2",
                "name": "Partial",
                "reason": "relevance_concepts_missing",
                "matched_concepts": ["ai"],
                "missing_concepts": ["semiconductor"],
            }
        ],
    }


def test_recommend_omits_missing_concept_candidates_from_default_list():
    service, reason_generator, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="2",
                    fit="높음",
                    recommendation_reason="Full concept match",
                    selected_evidence_ids=[_chunk_id("2")],
                )
            ]
        )
    )
    plan = PlannerOutput(
        intent_summary="Recommend AI semiconductor reviewers",
        core_keywords=["AI", "semiconductor"],
        required_concepts=["ai", "semiconductor"],
        concept_specs=[
            ConceptSpec(id="ai", label="AI", role="required"),
            ConceptSpec(id="semiconductor", label="반도체", role="required"),
        ],
        top_k=2,
    )
    cards = [
        _candidate_card(
            "1",
            "Partial",
            99.0,
            matched_concepts=["ai"],
            missing_concepts=["semiconductor"],
            coverage_type="partial",
        ),
        _candidate_card(
            "2",
            "Full",
            90.0,
            matched_concepts=["ai", "semiconductor"],
            coverage_type="separate",
        ),
    ]
    _bind_search_result(
        service,
        cards=cards,
        retrieved_count=2,
        planner_output=plan,
        query_payload={
            "prefetch": [],
            "query_filter": None,
            "query": "rrf",
            "relevance_gate_enabled": True,
            "relevance_gate_active_concepts": ["ai", "semiconductor"],
            "search_query_plan": {"required_concepts": ["ai", "semiconductor"]},
        },
    )

    result = asyncio.run(service.recommend(query="Recommend reviewers", top_k=2))

    assert evidence_selector.received_candidate_ids == [["2"]]
    assert reason_generator.received_candidate_ids == [["2"]]
    assert [item.expert_id for item in result["recommendations"]] == ["2"]
    assert result["trace"]["candidate_ids"] == ["1", "2"]
    assert result["trace"]["strict_filter"]["excluded_candidate_count"] == 1
    assert result["trace"]["strict_filter"]["excluded_reasons"][0] == {
        "expert_id": "1",
        "name": "Partial",
        "reason": "relevance_concepts_missing",
        "matched_concepts": ["ai"],
        "missing_concepts": ["semiconductor"],
    }


def test_recommend_clamps_planner_top_k_to_user_facing_maximum():
    service, reason_generator, evidence_selector = _build_service(ReasonGenerationOutput())
    cards = [
        _candidate_card(str(index), f"Candidate {index}", 100.0 - index)
        for index in range(1, 21)
    ]
    _bind_search_result(service, cards=cards, retrieved_count=20, planner_top_k=30)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    expected_ids = [str(index) for index in range(1, MAX_USER_FACING_RESULTS + 1)]
    assert result["trace"]["top_k_used"] == MAX_USER_FACING_RESULTS
    assert [item.expert_id for item in result["recommendations"]] == expected_ids
    assert evidence_selector.received_candidate_ids == [expected_ids]
    assert reason_generator.received_candidate_ids == [
        expected_ids[0:5],
        expected_ids[5:10],
        expected_ids[10:15],
    ]


def test_recommend_batches_reason_generation_and_preserves_global_order():
    service, reason_generator, evidence_selector = _build_service(
        [
            ReasonGenerationOutput(
                items=[
                    ReasonedCandidate(
                        expert_id=str(index),
                        fit="보통",
                        recommendation_reason=f"Reason {index}",
                        selected_evidence_ids=[_chunk_id(str(index))],
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
                        selected_evidence_ids=[_chunk_id("6")],
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
    assert recommendation.evidence[0].chunk_id == _chunk_id("1")
    assert recommendation.recommendation_reason == "Strong publication history"
    assert recommendation.model_dump(mode="json")["reasons"] == [
        "Strong publication history"
    ]
    # 구조 리팩토링 후에는 별도의 fallback 라벨 없이 항상 EvidenceSelector의 결과가 사용됨
    assert result["trace"]["reason_generation_trace"]["selected_evidence"][0]["fallback"] == "none"
    assert recommendation.fit == "보통"


def test_recommend_ignores_invalid_selected_evidence_ids_and_uses_fallback():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Strong publication history",
                    # LLM이 존재하지 않는 chunk_id를 선택해도 selector가 확정한 증거가 사용됨
                    selected_evidence_ids=["project_99_c000", "paper_999_c000"],
                )
            ]
        )
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    _bind_search_result(service, cards=cards, retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    recommendation: RecommendationDecision = result["recommendations"][0]
    assert recommendation.evidence[0].title == "Paper 1"
    # LLM이 잘못된 ID를 선택했더라도 EvidenceSelector가 확정한 paper chunk가 최종적으로 사용됨
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["selected_evidence_ids"]
        == ["project_99_c000", "paper_999_c000"]
    )
    assert (
        result["trace"]["reason_generation_trace"]["selected_evidence"][0]["resolved_evidence_ids"]
        == [_chunk_id("1")]
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
                    selected_evidence_ids=["project_99_c000"],
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
        "질의와 매칭된 직접 근거 1건이 확인되었습니다. 구체적으로 'Paper 1' 논문 등에서 "
        "관련 내용이 확인되었으며, 직접 근거 수가 제한적이라 추가 검토가 권장됩니다."
    )
    assert "Recommendation reason is empty after reason generation" in caplog.text
    assert "Recommendation reason fallback generated" in caplog.text


def test_recommendation_decision_serializes_empty_legacy_reasons_array():
    recommendation = RecommendationDecision(
        rank=1,
        expert_id="1",
        name="Alpha",
        fit="보통",
        recommendation_reason="",
        evidence=[],
        risks=[],
    )

    assert recommendation.model_dump(mode="json")["reasons"] == []
    assert recommendation.match_badges == []
    assert recommendation.match_summary == ""
    assert recommendation.match_details == {}
    assert recommendation.score_explanation == {}
    assert recommendation.evidence_summary == {}


def test_recommend_profile_fallback_trace_exposes_empty_relevant_bundle():
    service, _, evidence_selector = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Profile-based reason",
                    selected_evidence_ids=["patent_2009_c000"],
                )
            ]
        )
    )
    evidence_selector.select = MethodType(
        lambda self, *, candidates, plan: {
            candidate.expert_id: RelevantEvidenceBundle(expert_id=candidate.expert_id)
            for candidate in candidates
        },
        evidence_selector,
    )
    cards = [_candidate_card("1", "Alpha", 98.0)]
    _bind_search_result(service, cards=cards, retrieved_count=1)

    result = asyncio.run(service.recommend(query="Recommend reviewers"))

    selected_trace = result["trace"]["reason_generation_trace"]["selected_evidence"][0]
    recommendation: RecommendationDecision = result["recommendations"][0]
    assert selected_trace["provided_evidence_ids"] == []
    assert selected_trace["selected_evidence_ids"] == ["patent_2009_c000"]
    assert selected_trace["fallback"] == "profile"
    # profile evidence는 합성 type="profile"이며 chunk_id는 없다.
    assert recommendation.evidence[0].type == "profile"
    assert recommendation.evidence[0].chunk_id is None


def test_recommend_generates_fallback_reason_for_omitted_candidate():
    service, _, _ = _build_service(
        ReasonGenerationOutput(
            items=[
                ReasonedCandidate(
                    expert_id="1",
                    fit="보통",
                    recommendation_reason="Reason for first candidate",
                    selected_evidence_ids=[_chunk_id("1")],
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
        "질의와 매칭된 직접 근거 1건이 확인되었습니다. 구체적으로 'Paper 2' 논문 등에서 "
        "관련 내용이 확인되었으며, 직접 근거 수가 제한적이라 추가 검토가 권장됩니다."
    )
    assert result["trace"]["reason_generation_trace"]["server_fallback_reasons"] == [
        {
            "expert_id": "2",
            "source": "selected_evidence",
            "resolved_evidence_ids": [_chunk_id("2")],
        }
    ]
