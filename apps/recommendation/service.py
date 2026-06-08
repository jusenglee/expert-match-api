"""
사용자의 요청(추천 질의)을 받아 전체 RAG(Retrieval-Augmented Generation) 기반 추천 파이프라인을 지휘하는 오케스트레이터입니다.

[Architecture Overview]
1. Planner (의도 분석): 사용자의 자연어 질의를 분석하여 검색 키워드, 의미 검색 문장, 사전 필터 등을 추출합니다.
2. Retriever (검색): Qdrant 하이브리드 엔진을 통해 수만 명의 후보자 중 가장 관련성 높은 후보자 풀을 확보합니다.
3. Card Builder (정보 수합): 검색된 후보자의 논문, 특허, 과제 실적을 모아 '후보자 카드(CandidateCard)'를 만듭니다.
4. Evidence Selector (증거 선별): 후보자의 수많은 실적 중, 사용자 질의와 가장 매칭되는 실적 상위 N개를 추려냅니다.
5. Reasoner (추천 사유 및 심사): 추려진 핵심 증거를 바탕으로 LLM이 "왜 이 사람이 전문가인지" 심사하고 추천 사유를 작성합니다.
"""
from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.core.timer import Timer
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import CandidateCard, EvidenceItem, PlannerOutput, RecommendationDecision
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.evidence_selector import (
    EvidenceSelector,
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
)
from apps.recommendation.planner import Planner
from apps.recommendation.reasoner import (
    ReasonGenerationOutput,
    ReasonGenerator,
    VALID_EVIDENCE_ID_PATTERN,
)
from apps.search.doc_types import DOC_TYPES
from apps.search.filters import QdrantFilterCompiler
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.domain.models import ResearcherCandidate

logger = logging.getLogger(__name__)

NO_MATCHING_CANDIDATE_REASON = "No matching candidates were found."
EMPTY_RETRIEVAL_KEYWORDS_REASON = (
    "Retrieval skipped because planner core_keywords were empty after retry."
)
FINAL_SORT_POLICY = "rrf_score_desc_name_asc"
REASON_GENERATION_BATCH_SIZE = 5
MAX_USER_FACING_RESULTS = 15


def _sorted_filter_keys(filters: dict[str, Any] | None) -> list[str]:
    return sorted((filters or {}).keys())


def _filter_summary(filters: dict[str, Any] | None) -> dict[str, Any]:
    return {key: (filters or {}).get(key) for key in _sorted_filter_keys(filters)}


def _count_relevant_evidence_items(
    bundles: dict[str, RelevantEvidenceBundle],
) -> int:
    return sum(len(bundle.all_items()) for bundle in bundles.values())


def _clamp_result_limit(value: int | None, *, default: int | None = None) -> int:
    selected = value if value is not None else default
    if selected is None:
        return MAX_USER_FACING_RESULTS
    return max(1, min(int(selected), MAX_USER_FACING_RESULTS))


class RecommendationService:
    """
    모든 하위 컴포넌트(Planner, Retriever, Reasoner 등)를 연결하여 
    단일 엔드포인트에서 추천 프로세스를 수행하도록 돕는 파사드(Facade) 클래스입니다.
    """
    def __init__(
        self,
        *,
        planner: Planner,
        retriever: QdrantHybridRetriever,
        filter_compiler: QdrantFilterCompiler,
        card_builder: CandidateCardBuilder,
        evidence_selector: EvidenceSelector,
        reason_generator: ReasonGenerator,
        feedback_store: FeedbackStore,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
        self.evidence_selector = evidence_selector
        self.reason_generator = reason_generator
        self.feedback_store = feedback_store

    async def search_candidates(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        [/recommend 내부용] RRF 기반 후보자 검색 파이프라인.
        """
        return await self._run_search_pipeline(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
            retrieve=self.retriever.search,
            mode_label="grouped_hybrid_rrf",
        )

    async def search_weighted_candidates(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        [/search/candidates 전용] grouped RRF로 통일된 검색 파이프라인(/recommend과 동일 경로).
        query_points_groups(group_by=researcher_id) + 앱단 RRF 누적. (구 가중 0.6/0.4 융합 폐기.)
        """
        return await self._run_search_pipeline(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
            retrieve=self.retriever.search_weighted,
            mode_label="grouped_hybrid_rrf",
        )

    async def _run_search_pipeline(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None,
        include_orgs: list[str] | None,
        exclude_orgs: list[str] | None,
        top_k: int | None,
        retrieve: Any,
        mode_label: str,
    ) -> dict[str, Any]:
        logger.info(
            "검색 파이프라인 시작: query_chars=%d top_k=%s include_orgs=%d exclude_orgs=%d override_filter_keys=%s",
            len(query),
            top_k,
            len(include_orgs or []),
            len(exclude_orgs or []),
            _sorted_filter_keys(filters_override),
        )
        logger.info("플래너 단계 시작: query=%r", query)
        with Timer() as plan_timer:
            plan = await self.planner.plan(
                query=query,
                filters_override=filters_override,
                include_orgs=include_orgs,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )

        planner_trace = self._extract_component_trace(self.planner)
        result_limit = _clamp_result_limit(top_k, default=plan.top_k)
        retrieval_keywords = QueryTextBuilder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        logger.info(
            "플래너 단계 완료: elapsed_ms=%.2f mode=%s intent=%r retrieval_core=%s core_keywords=%s retrieval_keywords=%s role_terms=%s action_terms=%s semantic_query=%r bundle_ids=%s include_orgs=%s exclude_orgs=%s hard_filters=%s",
            plan_timer.elapsed_ms,
            (planner_trace or {}).get("mode", "unknown"),
            plan.intent_summary,
            plan.retrieval_core,
            plan.core_keywords,
            retrieval_keywords,
            plan.role_terms,
            plan.action_terms,
            plan.semantic_query,
            plan.bundle_ids,
            plan.include_orgs,
            plan.exclude_orgs,
            _filter_summary(plan.hard_filters),
        )

        query_filter = self.filter_compiler.compile(
            plan.hard_filters,
            plan.exclude_orgs,
            include_orgs=plan.include_orgs,
        )
        logger.info(
            "검색 필터 컴파일 완료: has_filter=%s include_orgs=%s exclude_orgs=%s hard_filters=%s",
            query_filter is not None,
            plan.include_orgs,
            plan.exclude_orgs,
            _filter_summary(plan.hard_filters),
        )

        if not retrieval_keywords:
            logger.warning(
                "검색 단계 스킵: query=%r reason=%s",
                query,
                EMPTY_RETRIEVAL_KEYWORDS_REASON,
            )
            return {
                "planner": plan,
                "planner_trace": planner_trace,
                "query_filter": query_filter,
                "retrieved_count": 0,
                "candidates": [],
                "query_payload": {
                    "skipped": True,
                    "reason": EMPTY_RETRIEVAL_KEYWORDS_REASON,
                },
                "branch_queries": {},
                "retrieval_keywords": retrieval_keywords,
                "retrieval_score_traces": [],
                "filtered_out_candidates": [],
                "raw_query": query,
                "retrieval_skipped_reason": EMPTY_RETRIEVAL_KEYWORDS_REASON,
                "final_sort_policy": FINAL_SORT_POLICY,
                "top_k_used": result_limit,
                "timers": {
                    "plan_ms": plan_timer.elapsed_ms,
                    "search_ms": 0.0,
                },
            }

        logger.info(
            "검색 단계 시작: mode=%s retrieval_keywords=%s semantic_query=%r query_filter=%s",
            mode_label,
            retrieval_keywords,
            plan.semantic_query,
            query_filter is not None,
        )
        with Timer() as search_timer:
            retrieval = await retrieve(
                query=query,
                plan=plan,
                query_filter=query_filter,
            )

        retrieval_payload = retrieval.query_payload or {}
        logger.info(
            "검색 단계 완료: elapsed_ms=%.2f hits=%d cache_hit=%s mode=%s retrieval_keywords=%s group_count=%s aggregated=%s org_filtered=%s final=%s",
            search_timer.elapsed_ms,
            len(retrieval.hits),
            retrieval.cache_hit,
            retrieval_payload.get("retrieval_mode"),
            retrieval.retrieval_keywords,
            retrieval_payload.get("group_count"),
            retrieval_payload.get("aggregated_candidate_count"),
            retrieval_payload.get("org_filtered_count"),
            retrieval_payload.get("final_hit_count"),
        )

        display_hits = retrieval.hits[:result_limit]
        cards = self.card_builder.build_small_cards(display_hits, plan)
        logger.info(
            "후보 카드 생성 완료: display_hits=%d cards=%d requested_top_k=%s top_k_used=%d total_retrieved=%d",
            len(display_hits),
            len(cards),
            top_k,
            result_limit,
            len(retrieval.hits),
        )
        return {
            "planner": plan,
            "planner_trace": planner_trace,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "hits_with_support": display_hits,
            # v2.0 집계(RRF 누적 + chunk_cap)에는 branch-cross support omission rule이 없다.
            # support_rule_*_min=0(비활성)인 deprecated legacy trace 필드 → 항상 미적용(False).
            "support_rule_applied": False,
            "cache_hit": retrieval.cache_hit,
            "filtered_out_candidates": retrieval.filtered_out_candidates,
            "query_payload": retrieval.query_payload,
            "branch_queries": {"stable": retrieval.queries.stable, "expanded": retrieval.queries.expanded},
            "retrieval_keywords": retrieval.retrieval_keywords,
            "retrieval_score_traces": retrieval.retrieval_score_traces,
            "expanded_shadow_hits": self._serialize_shadow_hits(retrieval.expanded_shadow_hits),
            "raw_query": query,
            "retrieval_skipped_reason": None,
            "final_sort_policy": FINAL_SORT_POLICY,
            "top_k_used": result_limit,
            "timers": {
                "plan_ms": plan_timer.elapsed_ms,
                "search_ms": search_timer.elapsed_ms,
            },
        }

    async def recommend(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        사용자의 질의를 바탕으로 전체 추천 프로세스(검색 + 심사)를 수행합니다.
        """
        logger.info(
            "추천 파이프라인 시작: query_chars=%d top_k=%s include_orgs=%d exclude_orgs=%d override_filter_keys=%s query=%r",
            len(query),
            top_k,
            len(include_orgs or []),
            len(exclude_orgs or []),
            _sorted_filter_keys(filters_override),
            query,
        )
        total_timer = Timer()
        total_timer.start()

        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )

        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        top_k_used = int(
            search_result.get("top_k_used")
            or _clamp_result_limit(top_k, default=plan.top_k)
        )
        recommendation_strict_exclusions = self._build_recommendation_strict_exclusions(
            candidate_cards
        )
        filtered_out_candidates = [
            *(search_result.get("filtered_out_candidates") or []),
            *recommendation_strict_exclusions,
        ]
        eligible_candidate_cards = [
            card for card in candidate_cards if not card.missing_concepts
        ]
        shortlist = eligible_candidate_cards[:top_k_used]

        logger.info(
            "추천 후보 확정: retrieved_count=%d candidate_cards=%d strict_eligible=%d strict_excluded=%d top_k_used=%d selected=%d shortlist_ids=%s",
            search_result["retrieved_count"],
            len(candidate_cards),
            len(eligible_candidate_cards),
            len(recommendation_strict_exclusions),
            top_k_used,
            len(shortlist),
            [candidate.expert_id for candidate in shortlist],
        )
        logger.info(
            "추천 후보 상세: %s",
            [
                {
                    "expert_id": c.expert_id,
                    "name": c.name,
                    "org": c.organization,
                    "rank_score": c.rank_score,
                    "doc_types": c.doc_types_present,
                    "counts": c.counts,
                }
                for c in shortlist
            ],
        )

        if search_result["retrieved_count"] == 0 or not shortlist:
            logger.info(
                "추천 파이프라인 종료: reason=no_shortlist query=%r retrieved=%d shortlist=%d",
                query,
                search_result["retrieved_count"],
                len(shortlist),
            )
            return self._build_recommendation_response(
                plan=plan,
                candidate_cards=candidate_cards,
                query_payload=search_result["query_payload"],
                branch_queries=search_result["branch_queries"],
                raw_query=search_result["raw_query"],
                retrieval_keywords=search_result.get("retrieval_keywords") or [],
                retrieval_score_traces=search_result.get("retrieval_score_traces") or [],
                filtered_out_candidates=filtered_out_candidates,
                retrieval_skipped_reason=search_result.get("retrieval_skipped_reason"),
                retrieved_count=search_result["retrieved_count"],
                recommendations=[],
                data_gaps=(
                    [search_result["retrieval_skipped_reason"]]
                    if search_result.get("retrieval_skipped_reason")
                    else []
                ),
                not_selected_reasons=[NO_MATCHING_CANDIDATE_REASON],
                planner_trace=search_result.get("planner_trace"),
                reason_generation_trace=None,
                final_sort_policy=search_result["final_sort_policy"],
                top_k_used=top_k_used,
                timers=search_result.get("timers"),
                retrieval_cache_hit=search_result.get("cache_hit", False),
            )

        # researcher_id hydration: '참고 프로필' 보강(질의 매칭 evidence와 분리, 점수/랭킹 무영향).
        # retriever/card_builder가 보강 인터페이스를 갖춘 경우에만(테스트 더블은 skip).
        hydration_settings = getattr(self.retriever, "settings", None)
        hydrate = getattr(self.retriever, "hydrate_profile_evidence", None)
        attach_profiles = getattr(self.card_builder, "attach_profile_evidence", None)
        if (
            hydration_settings is not None
            and getattr(hydration_settings, "profile_hydration_enabled", False)
            and callable(hydrate)
            and callable(attach_profiles)
        ):
            try:
                profile_by_researcher = await hydrate(
                    [card.expert_id for card in shortlist],
                    per_doc_type_cap=getattr(hydration_settings, "profile_evidence_per_doc_type_cap", 3),
                    fetch_limit=getattr(hydration_settings, "profile_hydration_fetch_limit", 4000),
                )
                attach_profiles(shortlist, profile_by_researcher)
                logger.info(
                    "프로필 보강 완료: candidates=%d hydrated=%d",
                    len(shortlist),
                    sum(1 for card in shortlist if card.profile_evidence),
                )
            except Exception as exc:  # noqa: BLE001 — 보강 실패는 격리(추천 흐름 계속).
                logger.warning("프로필 보강 실패(무시): %s", exc)

        logger.info(
            "증거 선별 시작: candidates=%d candidate_ids=%s",
            len(shortlist),
            [candidate.expert_id for candidate in shortlist],
        )
        with Timer() as evidence_selection_timer:
            relevant_evidence_by_expert_id = self.evidence_selector.select(
                candidates=shortlist,
                plan=plan,
            )
        evidence_item_count = _count_relevant_evidence_items(
            relevant_evidence_by_expert_id
        )
        empty_evidence_count = sum(
            1
            for bundle in relevant_evidence_by_expert_id.values()
            if not bundle.all_items()
        )
        logger.info(
            "증거 선별 완료: elapsed_ms=%.2f candidates=%d evidence_items=%d empty_candidates=%d",
            evidence_selection_timer.elapsed_ms,
            len(relevant_evidence_by_expert_id),
            evidence_item_count,
            empty_evidence_count,
        )
        retrieval_score_traces_by_expert_id = {
            trace["expert_id"]: trace
            for trace in (search_result.get("retrieval_score_traces") or [])
            if trace.get("expert_id")
        }

        reason_batch_count = len(
            self._batch_candidates(
                shortlist,
                batch_size=REASON_GENERATION_BATCH_SIZE,
            )
        )
        logger.info(
            "추천 사유 생성 시작: candidates=%d batches=%d batch_size=%d",
            len(shortlist),
            reason_batch_count,
            REASON_GENERATION_BATCH_SIZE,
        )
        with Timer() as reason_timer:
            reason_output, reason_batch_traces = await self._generate_reasons_in_batches(
                query=query,
                plan=plan,
                candidates=shortlist,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            )
        logger.info(
            "추천 사유 생성 완료: elapsed_ms=%.2f batches=%d output_items=%d data_gaps=%d",
            reason_timer.elapsed_ms,
            len(reason_batch_traces),
            len(reason_output.items),
            len(reason_output.data_gaps),
        )

        reason_generation_trace = self._build_reason_generation_trace(
            candidates=shortlist,
            reason_output=reason_output,
            batch_traces=reason_batch_traces,
        )
        evidence_selection_trace = self._extract_component_trace(self.evidence_selector)
        if evidence_selection_trace is not None:
            reason_generation_trace = dict(reason_generation_trace)
            reason_generation_trace["evidence_selection"] = evidence_selection_trace
        (
            recommendations,
            selected_evidence_trace,
            server_fallback_reasons,
        ) = self._build_recommendations(
            shortlist,
            reason_output,
            plan=plan,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
        )
        reason_generation_trace = dict(reason_generation_trace)
        reason_generation_trace["selected_evidence"] = selected_evidence_trace
        reason_generation_trace["server_fallback_reasons"] = server_fallback_reasons
        timers = dict(search_result.get("timers", {}))
        timers["evidence_selection_ms"] = evidence_selection_timer.elapsed_ms
        timers["reason_generation_ms"] = reason_timer.elapsed_ms

        total_timer.stop()
        timers["total_ms"] = total_timer.elapsed_ms

        logger.info(
            "추천 파이프라인 완료: query=%r retrieved=%d recommended=%d data_gaps=%d total_ms=%.2f",
            query,
            search_result["retrieved_count"],
            len(recommendations),
            len(reason_output.data_gaps),
            total_timer.elapsed_ms,
        )

        return self._build_recommendation_response(
            plan=plan,
            candidate_cards=candidate_cards,
            query_payload=search_result["query_payload"],
            branch_queries=search_result["branch_queries"],
            raw_query=search_result["raw_query"],
            retrieval_keywords=search_result.get("retrieval_keywords") or [],
            retrieval_score_traces=search_result.get("retrieval_score_traces") or [],
            filtered_out_candidates=filtered_out_candidates,
            retrieval_skipped_reason=search_result.get("retrieval_skipped_reason"),
            retrieved_count=search_result["retrieved_count"],
            recommendations=recommendations,
            data_gaps=list(reason_output.data_gaps),
            not_selected_reasons=[],
            planner_trace=search_result.get("planner_trace"),
            reason_generation_trace=reason_generation_trace,
            final_sort_policy=search_result["final_sort_policy"],
            top_k_used=top_k_used,
            timers=timers,
            retrieval_cache_hit=search_result.get("cache_hit", False),
        )

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
        """
        추천 결과에 대한 사용자의 피드백을 저장합니다.
        """
        logger.info(
            "Feedback saved: query=%r selected=%d rejected=%d",
            query,
            len(selected_expert_ids),
            len(rejected_expert_ids),
        )
        return self.feedback_store.save_feedback(
            query=query,
            selected_expert_ids=selected_expert_ids,
            rejected_expert_ids=rejected_expert_ids,
            notes=notes,
            metadata=metadata,
        )

    @staticmethod
    def _extract_component_trace(component: Any) -> dict[str, Any] | None:
        trace = getattr(component, "last_trace", None)
        if isinstance(trace, dict):
            return trace
        return None

    @staticmethod
    def _build_recommendation_strict_exclusions(
        cards: list[CandidateCard],
    ) -> list[dict[str, Any]]:
        return [
            {
                "expert_id": card.expert_id,
                "name": card.name,
                "reason": "relevance_concepts_missing",
                "matched_concepts": list(card.matched_concepts),
                "missing_concepts": list(card.missing_concepts),
            }
            for card in cards
            if card.missing_concepts
        ]

    @staticmethod
    def _batch_candidates(
        candidates: list[CandidateCard], *, batch_size: int
    ) -> list[list[CandidateCard]]:
        if batch_size <= 0:
            return [candidates]
        return [
            candidates[index : index + batch_size]
            for index in range(0, len(candidates), batch_size)
        ]

    async def _generate_reasons_in_batches(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]],
    ) -> tuple[ReasonGenerationOutput, list[dict[str, Any]]]:
        batch_outputs: list[ReasonGenerationOutput] = []
        batch_traces: list[dict[str, Any]] = []
        candidate_batches = self._batch_candidates(
            candidates, batch_size=REASON_GENERATION_BATCH_SIZE
        )

        for batch_index, candidate_batch in enumerate(candidate_batches, start=1):
            batch_candidate_ids = [candidate.expert_id for candidate in candidate_batch]
            logger.info(
                "Reason generation batch started: batch=%d/%d size=%d candidate_ids=%s",
                batch_index,
                len(candidate_batches),
                len(candidate_batch),
                batch_candidate_ids,
            )
            batch_output = await self.reason_generator.generate(
                query=query,
                plan=plan,
                candidates=candidate_batch,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            )
            batch_outputs.append(batch_output)

            raw_trace = dict(self._extract_component_trace(self.reason_generator) or {})
            batch_trace = {
                "batch_index": batch_index,
                "batch_size": len(candidate_batch),
                "candidate_ids": batch_candidate_ids,
                "returned_ids": list(raw_trace.get("returned_ids", [])),
                "missing_candidate_ids": list(
                    raw_trace.get("missing_candidate_ids", [])
                ),
                "empty_reason_candidate_ids": list(
                    raw_trace.get("empty_reason_candidate_ids", [])
                ),
                "empty_selected_evidence_candidate_ids": list(
                    raw_trace.get("empty_selected_evidence_candidate_ids", [])
                ),
                "invalid_selected_evidence_candidate_ids": list(
                    raw_trace.get("invalid_selected_evidence_candidate_ids", [])
                ),
                "invalid_selected_evidence_ids_by_candidate": dict(
                    raw_trace.get("invalid_selected_evidence_ids_by_candidate", {})
                ),
                "mode": raw_trace.get("mode", "unknown"),
                "seed": raw_trace.get("seed"),
                "retry_count": raw_trace.get("retry_count", 0),
                "returned_ratio": raw_trace.get(
                    "returned_ratio",
                    round(
                        len(list(raw_trace.get("returned_ids", []))) / len(candidate_batch),
                        3,
                    )
                    if candidate_batch
                    else 0.0,
                ),
                "prompt_budget_mode": raw_trace.get("prompt_budget_mode"),
                "trim_applied": raw_trace.get("trim_applied", False),
                "payload_token_estimate": raw_trace.get("payload_token_estimate"),
                "attempts": list(raw_trace.get("attempts", [])),
                "raw_output_count": raw_trace.get(
                    "raw_output_count", len(batch_output.items)
                ),
                "output_count": raw_trace.get("output_count", len(batch_output.items)),
            }
            batch_traces.append(batch_trace)
            logger.info(
                "Reason generation batch completed: batch=%d/%d size=%d returned=%d missing=%d empty_reasons=%d",
                batch_index,
                len(candidate_batches),
                len(candidate_batch),
                len(batch_trace["returned_ids"]),
                len(batch_trace["missing_candidate_ids"]),
                len(batch_trace["empty_reason_candidate_ids"]),
            )

        merged_data_gaps: list[str] = []
        merged_items: list[Any] = []
        for batch_output in batch_outputs:
            merged_items.extend(batch_output.items)
            merged_data_gaps = _merge_unique_strings(
                [*merged_data_gaps, *batch_output.data_gaps]
            )

        return (
            ReasonGenerationOutput(items=merged_items, data_gaps=merged_data_gaps),
            batch_traces,
        )

    @staticmethod
    def _build_reason_generation_trace(
        *,
        candidates: list[CandidateCard],
        reason_output: ReasonGenerationOutput,
        batch_traces: list[dict[str, Any]],
    ) -> dict[str, Any]:
        modes = {
            trace.get("mode")
            for trace in batch_traces
            if isinstance(trace.get("mode"), str) and trace.get("mode")
        }
        return {
            "mode": next(iter(modes)) if len(modes) == 1 else "mixed",
            "candidate_count": len(candidates),
            "output_count": len(reason_output.items),
            "batch_count": len(batch_traces),
            "batch_size": REASON_GENERATION_BATCH_SIZE,
            "reason_generation_failed": any(
                float(trace.get("returned_ratio", 0.0)) == 0.0 for trace in batch_traces
            ),
            "batches": batch_traces,
        }

    @staticmethod
    def _build_profile_evidence(card: CandidateCard) -> EvidenceItem | None:
        if not (card.organization or card.degree):
            return None
        return EvidenceItem(
            type="profile",
            title=card.name,
            detail=" / ".join(
                [
                    card.organization or "unknown organization",
                    card.degree or "unknown degree",
                ]
            ),
        )

    @staticmethod
    def _build_evidence_item(item: RelevantEvidenceItem) -> EvidenceItem:
        return EvidenceItem(
            type=item.type,
            title=item.title,
            date=item.date,
            detail=item.detail,
            snippet=item.snippet,
            chunk_id=item.item_id,
        )

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _concept_label_map(plan: PlannerOutput) -> dict[str, str]:
        labels: dict[str, str] = {}
        for spec in plan.concept_specs:
            labels[spec.id] = spec.label or spec.id
        for concept in [*plan.required_concepts, *plan.optional_concepts]:
            labels.setdefault(concept, concept)
        return labels

    @staticmethod
    def _format_concept_labels(concepts: list[str], labels: dict[str, str]) -> str:
        return ", ".join(labels.get(concept, concept) for concept in concepts)

    @classmethod
    def _build_match_details(
        cls, *, card: CandidateCard, evidence: list[EvidenceItem]
    ) -> dict[str, Any]:
        direct_evidence = [item for item in evidence if item.type != "profile"]
        coverage_type = card.coverage_type
        if not coverage_type and card.missing_concepts:
            coverage_type = "partial"
        return {
            "matched_concepts": list(card.matched_concepts),
            "missing_concepts": list(card.missing_concepts),
            "coverage_type": coverage_type,
            "matched_doc_types": _merge_unique_strings(
                [item.type for item in direct_evidence]
            ),
            "direct_evidence_count": len(direct_evidence),
        }

    @classmethod
    def _build_match_badges(
        cls, *, plan: PlannerOutput, match_details: dict[str, Any]
    ) -> list[str]:
        labels = cls._concept_label_map(plan)
        badges = [
            f"{labels.get(concept, concept)} 충족"
            for concept in match_details.get("matched_concepts", [])
        ]
        coverage_type = match_details.get("coverage_type")
        if coverage_type == "joint":
            badges.append("복합 조건 동시 근거")
        elif coverage_type == "separate":
            badges.append("조건별 근거 충족")
        if int(match_details.get("direct_evidence_count") or 0) > 0:
            badges.append("직접 수행 근거 있음")
        return _merge_unique_strings(badges)

    @classmethod
    def _build_match_summary(
        cls, *, plan: PlannerOutput, match_details: dict[str, Any]
    ) -> str:
        labels = cls._concept_label_map(plan)
        matched = list(match_details.get("matched_concepts") or [])
        missing = list(match_details.get("missing_concepts") or [])
        direct_count = int(match_details.get("direct_evidence_count") or 0)
        if missing:
            missing_text = cls._format_concept_labels(missing, labels)
            if matched:
                matched_text = cls._format_concept_labels(matched, labels)
                return (
                    f"{matched_text} 근거는 확인되었지만 "
                    f"{missing_text} 근거는 아직 부족합니다."
                )
            return f"{missing_text} 근거는 아직 부족합니다."
        if matched:
            matched_text = cls._format_concept_labels(matched, labels)
            return f"{matched_text} 근거가 확인되었습니다."
        if direct_count:
            return f"질의와 매칭된 직접 근거 {direct_count}건이 확인되었습니다."
        return "직접적인 질의 일치 근거는 제한적입니다."

    @classmethod
    def _build_score_explanation(
        cls,
        *,
        card: CandidateCard,
        retrieval_trace: dict[str, Any] | None,
    ) -> dict[str, Any]:
        trace = retrieval_trace or {}
        title_by_chunk_id = {
            item.get("chunk_id"): item.get("title")
            for item in card.top_chunks
            if item.get("chunk_id")
        }
        top_chunks: list[dict[str, Any]] = []
        trace_matches = trace.get("matches") if isinstance(trace.get("matches"), list) else []
        for item in trace_matches[:10]:
            if not isinstance(item, dict):
                continue
            chunk_id = item.get("chunk_id")
            score = item.get("score", item.get("fused_score", item.get("contribution", 0.0)))
            top_chunks.append(
                {
                    "doc_type": item.get("doc_type", ""),
                    "title": item.get("title") or title_by_chunk_id.get(chunk_id),
                    "concepts": list(item.get("concepts") or []),
                    "display_only_concepts": list(item.get("display_only_concepts") or []),
                    "sources": list(item.get("sources") or []),
                    "score": round(cls._safe_float(score), 6),
                }
            )
        if not top_chunks:
            top_chunks = [
                {
                    "doc_type": item.get("doc_type", ""),
                    "title": item.get("title"),
                    "concepts": list(item.get("concepts") or []),
                    "display_only_concepts": list(item.get("display_only_concepts") or []),
                    "sources": list(item.get("sources") or []),
                    "score": round(cls._safe_float(item.get("score")), 6),
                }
                for item in card.top_chunks[:10]
                if isinstance(item, dict)
            ]
        return {
            "final_score": round(
                cls._safe_float(trace.get("final_score", card.raw_score)), 6
            ),
            "rank_score": round(cls._safe_float(card.rank_score), 6),
            "score_breakdown": dict(
                trace.get("score_breakdown") or card.score_breakdown
            ),
            "top_chunks": top_chunks,
        }

    @staticmethod
    def _build_evidence_summary(
        *,
        card: CandidateCard,
        evidence: list[EvidenceItem],
        relevant_bundle: RelevantEvidenceBundle,
    ) -> dict[str, Any]:
        return {
            "total_profile_counts": dict(card.counts),
            "matched_evidence_count": len(relevant_bundle.all_items()),
            "shown_evidence_count": len(evidence),
            "profile_evidence_count": len(card.profile_evidence),
        }

    @staticmethod
    def _build_profile_evidence_items(
        *, card: CandidateCard, exclude_chunk_ids: set[str]
    ) -> list[EvidenceItem]:
        """card.profile_evidence(researcher hydration)를 응답 EvidenceItem(kind=profile)으로 변환.

        이미 표시되는 질의-매칭 evidence(chunk_id)는 제외해 중복을 막는다. 점수/랭킹 무관.
        """
        valid_types = {"paper", "patent", "project", "assessor_activity", "specialty"}
        items: list[EvidenceItem] = []
        seen: set[str] = set()
        for ev in card.profile_evidence:
            if ev.chunk_id in exclude_chunk_ids or ev.chunk_id in seen:
                continue
            title = " ".join((ev.title or "").split())
            if not title:
                continue
            seen.add(ev.chunk_id)
            items.append(
                EvidenceItem(
                    type=ev.doc_type if ev.doc_type in valid_types else "profile",
                    title=title,
                    date=ev.date,
                    detail=None,
                    snippet=(" ".join((ev.snippet or "").split()) or None),
                    chunk_id=ev.chunk_id,
                    evidence_kind="profile",
                )
            )
        return items

    @staticmethod
    def _build_strict_filter_trace(
        query_payload: dict[str, Any] | None,
        filtered_out_candidates: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload = query_payload or {}
        search_query_plan = payload.get("search_query_plan") or {}
        required_concepts = list(
            payload.get("relevance_gate_active_concepts")
            or search_query_plan.get("required_concepts")
            or []
        )
        enabled_value = payload.get("relevance_gate_enabled")
        enabled = bool(enabled_value) if enabled_value is not None else bool(required_concepts)
        strict_exclusions: list[dict[str, Any]] = []
        for item in filtered_out_candidates or []:
            if not isinstance(item, dict):
                continue
            if item.get("reason") != "relevance_concepts_missing":
                continue
            strict_exclusions.append(
                {
                    "expert_id": item.get("expert_id"),
                    "name": item.get("name"),
                    "reason": item.get("reason"),
                    "matched_concepts": list(item.get("matched_concepts") or []),
                    "missing_concepts": list(item.get("missing_concepts") or []),
                }
            )
        return {
            "enabled": enabled,
            "required_concepts": required_concepts,
            "excluded_candidate_count": len(strict_exclusions),
            "excluded_reasons": strict_exclusions,
        }

    @staticmethod
    def _sort_relevant_items(
        bundle: RelevantEvidenceBundle,
    ) -> list[RelevantEvidenceItem]:
        return sorted(
            bundle.all_items(),
            key=lambda item: (-item.match_score, item.type, item.title),
        )

    @staticmethod
    def _build_server_fallback_reason(
        *,
        evidence: list[EvidenceItem],
        fallback: str,
        match_summary: str = "",
    ) -> str:
        # concept-aware match_summary(예: '인공지능, 반도체 근거가 확인되었습니다')를 앞세우고
        # 실제 evidence 제목을 구체적으로 인용한 뒤, 직접 근거 부족을 정직하게 명시한다.
        summary = " ".join((match_summary or "").split())
        if not evidence:
            return summary or "직접적인 질의 일치 근거를 확인하지 못했습니다."

        if all(item.type == "profile" for item in evidence):
            lead = summary or "질의와 직접 매칭된 근거는 제한적입니다."
            return f"{lead} 프로필 기반으로 검토된 후보입니다."

        type_labels = {
            "project": "과제",
            "paper": "논문",
            "patent": "특허",
            "assessor_activity": "심사위원 활동",
            "specialty": "전문분야",
            "profile": "프로필",
        }
        referenced_items: list[str] = []
        seen_titles: set[tuple[str, str]] = set()
        for item in evidence:
            if item.type == "profile":
                continue
            key = (item.type, item.title)
            if key in seen_titles:
                continue
            seen_titles.add(key)
            referenced_items.append(f"'{item.title}' {type_labels.get(item.type, item.type)}")
            if len(referenced_items) == 2:
                break

        if not referenced_items:
            lead = summary or "질의와 직접 매칭된 근거는 제한적입니다."
            return f"{lead} 프로필 기반으로 검토된 후보입니다."

        referenced = "와 ".join(referenced_items)
        lead = summary or "질의와 관련된 전문성 근거가 확인되었습니다."
        return (
            f"{lead} 구체적으로 {referenced} 등에서 관련 내용이 확인되었으며, "
            "직접 근거 수가 제한적이라 추가 검토가 권장됩니다."
        )

    @classmethod
    def _build_candidate_evidence(
        cls,
        *,
        card: CandidateCard,
        generated: Any | None,
        relevant_bundle: RelevantEvidenceBundle,
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        # EvidenceSelector가 doc_type/family cap(FAMILY_EVIDENCE_CAP: achievement=10/assessment=6/
        # expertise=6/identity=1)으로 grounding evidence를 선별해 둔 상태다. LLM의 selected_evidence_ids와
        # 무관하게 relevant_bundle의 모든 항목을 최종 증거로 확정해 추천 사유-증거 100% 싱크를 보장한다.
        resolved_items = relevant_bundle.all_items()
        provided_evidence_ids = [item.item_id for item in resolved_items]
        
        # LLM이 선택한 ID (트래킹용)
        requested_ids = list(getattr(generated, "selected_evidence_ids", []) or [])

        if not resolved_items:
            profile_item = cls._build_profile_evidence(card)
            if profile_item is not None:
                return [profile_item], {
                    "expert_id": card.expert_id,
                    "provided_evidence_ids": [],
                    "selected_evidence_ids": requested_ids,
                    "resolved_evidence_ids": ["profile"],
                    "fallback": "profile",
                }
            return [], {
                "expert_id": card.expert_id,
                "provided_evidence_ids": [],
                "selected_evidence_ids": requested_ids,
                "resolved_evidence_ids": [],
                "fallback": "empty",
            }

        return (
            [cls._build_evidence_item(item) for item in resolved_items],
            {
                "expert_id": card.expert_id,
                "provided_evidence_ids": provided_evidence_ids,
                "selected_evidence_ids": requested_ids,
                "resolved_evidence_ids": provided_evidence_ids,
                "fallback": "none",
            },
        )

    def _build_recommendations(
        self,
        cards: list[CandidateCard],
        reason_output: ReasonGenerationOutput,
        *,
        plan: PlannerOutput,
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]],
    ) -> tuple[list[RecommendationDecision], list[dict[str, Any]], list[dict[str, Any]]]:
        generated_by_expert_id = {
            item.expert_id: item for item in reason_output.items
        }
        recommendations: list[RecommendationDecision] = []
        selected_evidence_trace: list[dict[str, Any]] = []
        server_fallback_reasons: list[dict[str, Any]] = []

        for rank, card in enumerate(cards, start=1):
            generated = generated_by_expert_id.get(card.expert_id)
            relevant_bundle = relevant_evidence_by_expert_id.get(
                card.expert_id,
                RelevantEvidenceBundle(expert_id=card.expert_id),
            )
            fit = "보통"
            recommendation_reason = ""
            risks = list(card.risks)
            if generated is not None:
                fit = generated.fit if generated.fit in {"높음", "중간", "보통"} else "보통"
                recommendation_reason = generated.recommendation_reason
                risks = list(generated.risks) or list(card.risks)
            evidence, evidence_trace = self._build_candidate_evidence(
                card=card,
                generated=generated,
                relevant_bundle=relevant_bundle,
            )
            match_details = self._build_match_details(card=card, evidence=evidence)
            match_badges = self._build_match_badges(
                plan=plan,
                match_details=match_details,
            )
            match_summary = self._build_match_summary(
                plan=plan,
                match_details=match_details,
            )
            score_explanation = self._build_score_explanation(
                card=card,
                retrieval_trace=retrieval_score_traces_by_expert_id.get(
                    card.expert_id
                ),
            )
            evidence_summary = self._build_evidence_summary(
                card=card,
                evidence=evidence,
                relevant_bundle=relevant_bundle,
            )
            profile_evidence = self._build_profile_evidence_items(
                card=card,
                exclude_chunk_ids={item.chunk_id for item in evidence if item.chunk_id},
            )
            selected_evidence_trace.append(evidence_trace)
            if not recommendation_reason:
                logger.warning(
                    "Recommendation reason is empty after reason generation: expert_id=%s fit=%s resolved_evidence_ids=%s fallback=%s",
                    card.expert_id,
                    fit,
                    evidence_trace.get("resolved_evidence_ids", []),
                    evidence_trace.get("fallback"),
                )
                fallback_source = (
                    evidence_trace.get("fallback")
                    if evidence_trace.get("fallback") not in {None, "none"}
                    else "selected_evidence"
                )
                recommendation_reason = self._build_server_fallback_reason(
                    evidence=evidence,
                    fallback=fallback_source,
                    match_summary=match_summary,
                )
                server_fallback_reasons.append(
                    {
                        "expert_id": card.expert_id,
                        "source": fallback_source,
                        "resolved_evidence_ids": list(
                            evidence_trace.get("resolved_evidence_ids", [])
                        ),
                    }
                )
                logger.warning(
                    "Recommendation reason fallback generated: expert_id=%s source=%s resolved_evidence_ids=%s",
                    card.expert_id,
                    fallback_source,
                    evidence_trace.get("resolved_evidence_ids", []),
                )

            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    organization=card.organization,
                    fit=fit,
                    recommendation_reason=recommendation_reason,
                    match_badges=match_badges,
                    match_summary=match_summary,
                    match_details=match_details,
                    score_explanation=score_explanation,
                    evidence_summary=evidence_summary,
                    evidence=evidence,
                    profile_evidence=profile_evidence,
                    risks=risks,
                    rank_score=card.rank_score,
                )
            )

        return recommendations, selected_evidence_trace, server_fallback_reasons

    @staticmethod
    def _serialize_shadow_hits(hits: list[ResearcherCandidate]) -> list[dict[str, str]]:
        return [
            {
                "expert_id": hit.researcher_id,
                "name": hit.researcher_name or "",
            }
            for hit in hits
        ]

    @staticmethod
    def _build_recommendation_response(
        *,
        plan: PlannerOutput,
        candidate_cards: list[CandidateCard],
        query_payload: dict[str, Any],
        branch_queries: dict[str, Any],
        raw_query: str,
        retrieval_keywords: list[str],
        retrieval_score_traces: list[dict[str, Any]],
        filtered_out_candidates: list[dict[str, Any]],
        retrieval_skipped_reason: str | None,
        retrieved_count: int,
        recommendations: list[RecommendationDecision],
        data_gaps: list[str],
        not_selected_reasons: list[str],
        planner_trace: dict[str, Any] | None,
        reason_generation_trace: dict[str, Any] | None,
        final_sort_policy: str,
        top_k_used: int,
        timers: dict[str, Any] | None,
        retrieval_cache_hit: bool = False,
        expanded_shadow_hits: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        merged_data_gaps = _merge_unique_strings(data_gaps)
        return {
            "intent_summary": plan.intent_summary,
            "applied_filters": plan.hard_filters,
            "searched_branches": list(DOC_TYPES),
            "retrieved_count": retrieved_count,
            "recommendations": recommendations,
            "data_gaps": merged_data_gaps,
            "not_selected_reasons": not_selected_reasons,
            "trace": {
                "planner": plan.model_dump(mode="json"),
                "planner_trace": planner_trace or {},
                "reason_generation_trace": reason_generation_trace or {},
                "raw_query": raw_query,
                "cache": {
                    "canonical_plan": (planner_trace or {}).get("cache", {}).get("canonical_plan", "miss"),
                    "retrieval": "hit" if retrieval_cache_hit else "miss",
                },
                "planner_keywords": (
                    (planner_trace or {}).get("planner_keywords") or []
                ),
                "retrieval_keywords": retrieval_keywords,
                "bundle_ids": plan.bundle_ids,
                "expanded_shadow_hits": expanded_shadow_hits or [],
                "filtered_out_count": len(filtered_out_candidates),
                "filtered_out_candidates": filtered_out_candidates,
                "strict_filter": RecommendationService._build_strict_filter_trace(
                    query_payload,
                    filtered_out_candidates,
                ),
                "planner_retry_count": (
                    (planner_trace or {}).get("planner_retry_count", 0)
                ),
                "retrieval_skipped_reason": retrieval_skipped_reason,
                "branch_queries": branch_queries,
                "include_orgs": plan.include_orgs,
                "exclude_orgs": plan.exclude_orgs,
                "candidate_ids": [card.expert_id for card in candidate_cards],
                "retrieval_score_traces": retrieval_score_traces,
                "recommendation_ids": [item.expert_id for item in recommendations],
                "final_sort_policy": final_sort_policy,
                "top_k_used": top_k_used,
                "query_payload": RecommendationService._serialize_query_payload(
                    query_payload
                ),
                "timers": timers or {},
            },
        }

    @staticmethod
    def _serialize_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def _mask_vectors(data: Any) -> Any:
            if hasattr(data, "model_dump"):
                try:
                    data = data.model_dump()
                except Exception:
                    pass
            elif hasattr(data, "dict") and callable(data.dict):
                try:
                    data = data.dict()
                except Exception:
                    pass

            if isinstance(data, dict):
                return {key: _mask_vectors(value) for key, value in data.items()}
            if isinstance(data, list):
                if len(data) > 100 and all(
                    isinstance(item, (float, int)) for item in data[:10]
                ):
                    return f"<Dense Vector: {len(data)} dimensions>"
                return [_mask_vectors(item) for item in data]
            if isinstance(data, (int, float, bool)) or data is None:
                return data
            return str(data)

        serialized = dict(payload)
        serialized["prefetch"] = [
            _mask_vectors(item) for item in payload.get("prefetch", [])
        ]
        query_filter = payload.get("query_filter")
        serialized["query_filter"] = str(query_filter) if query_filter else None
        query_value = payload.get("query")
        serialized["query"] = str(query_value) if query_value is not None else None
        return serialized
