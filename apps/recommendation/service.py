from __future__ import annotations

import logging
import re
import json
from typing import Any, AsyncGenerator

from apps.core.feedback_store import FeedbackStore
from apps.core.timer import Timer
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import (
    CandidateCard,
    EvidenceItem,
    PlannerOutput,
    RecommendationDecision,
    SearchHit,
)
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.evidence_selector import (
    EvidenceSelector,
    RelevantEvidenceBundle,
    RelevantEvidenceItem,
    normalize_phrase_keywords,
)
from apps.recommendation.planner import Planner
from apps.recommendation.reasoner import ReasonGenerationOutput, ReasonGenerator
from apps.search.filters import QdrantFilterCompiler
from apps.search.query_builder import CompiledBranchQueries, QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES

logger = logging.getLogger(__name__)

NO_MATCHING_CANDIDATE_REASON = "No matching candidates were found."
NO_GATE_PASSED_CANDIDATE_REASON = "No candidates satisfied deterministic evidence gates."
EMPTY_RETRIEVAL_KEYWORDS_REASON = (
    "Retrieval skipped because planner core_keywords were empty after retry."
)
FINAL_SORT_POLICY = "rrf_score_desc_name_asc"
REASON_GENERATION_BATCH_SIZE = 5
INTERNAL_EVIDENCE_ID_PATTERN = re.compile(r"\b(?:paper|project|patent):\d+\b")
STRONG_REASON_MARKERS = (
    "추천",
    "적합",
    "전문",
    "권장",
    "우수",
    "높음",
    "recommend",
    "recommended",
    "strong fit",
    "well suited",
)


class RecommendationService:
    """
    전문가 추천의 전체 비즈니스 흐름을 관장하는 핵심 서비스 클래스입니다.
    검색 계획(Planner) -> 검색(Retriever) -> 필터링(Gate) -> 증거 선별(EvidenceSelector) -> 추천 사유 생성(Reasoner)
    의 파이프라인(Pipeline)을 조율합니다.
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
        limit_candidates: bool = True,
    ) -> dict[str, Any]:
        """
        1단계 추천 로직: 사용자 쿼리를 바탕으로 검색 계획(Plan)을 세우고, 검색 엔진(Retriever)을 호출하여 후보자 목록을 가져옵니다.
        여기까지는 LLM 추론(추천 사유 생성)이 포함되지 않은 순수 검색(Search) 결과입니다.
        """
        with Timer() as plan_timer:
            plan = await self.planner.plan(
                query=query,
                filters_override=filters_override,
                include_orgs=include_orgs,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )

        planner_trace = self._extract_component_trace(self.planner)
        retrieval_keywords = QueryTextBuilder.normalize_keywords(plan.core_keywords)
        logger.info(
            "Planner completed: elapsed_ms=%.2f intent=%r filters=%r retrieval_core=%s must_aspects=%s removed_meta_terms=%s",
            plan_timer.elapsed_ms,
            plan.intent_summary,
            plan.hard_filters,
            plan.retrieval_core,
            getattr(plan, "must_aspects", []),
            (planner_trace or {}).get("removed_meta_terms", []),
        )

        query_filter = self.filter_compiler.compile(
            plan.hard_filters,
            plan.exclude_orgs,
            include_orgs=plan.include_orgs,
        )

        if not retrieval_keywords:
            logger.warning(
                "Retriever skipped: query=%r reason=%s removed_meta_terms=%s",
                query,
                EMPTY_RETRIEVAL_KEYWORDS_REASON,
                (planner_trace or {}).get("removed_meta_terms", []),
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
                "expanded_shadow_hits": [],
                "raw_query": query,
                "retrieval_skipped_reason": EMPTY_RETRIEVAL_KEYWORDS_REASON,
                "final_sort_policy": FINAL_SORT_POLICY,
                "timers": {
                    "plan_ms": plan_timer.elapsed_ms,
                    "search_ms": 0.0,
                },
            }

        with Timer() as search_timer:
            retrieval = await self.retriever.search(
                query=query,
                plan=plan,
                query_filter=query_filter,
            )

        logger.info(
            "Retriever completed: elapsed_ms=%.2f hits=%d",
            search_timer.elapsed_ms,
            len(retrieval.hits),
        )

        display_hits = (
            retrieval.hits[:top_k]
            if limit_candidates and top_k is not None
            else retrieval.hits
        )
        cards = self.card_builder.build_small_cards(display_hits, plan)
        return {
            "planner": plan,
            "planner_trace": planner_trace,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "hits_with_support": display_hits,
            "support_rule_applied": True,
            "cache_hit": retrieval.cache_hit,
            "filtered_out_candidates": retrieval.filtered_out_candidates,
            "query_payload": retrieval.query_payload,
            "branch_queries": retrieval.branch_queries,
            "retrieval_keywords": retrieval.retrieval_keywords,
            "retrieval_score_traces": retrieval.retrieval_score_traces,
            "expanded_shadow_hits": self._serialize_shadow_hits(retrieval.expanded_shadow_hits),
            "raw_query": query,
            "retrieval_skipped_reason": None,
            "final_sort_policy": FINAL_SORT_POLICY,
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
        2단계 추천 로직: `search_candidates`의 결과에 대해 엄격한 증거 필터링(Shortlist Gate)을 적용하고, 
        살아남은 소수 정예 후보자들에 대해 LLM 기반의 추천 사유(Reasoner)를 생성하여 최종 추천 결과를 반환합니다.
        """
        logger.info("Recommendation started: query=%r", query)
        total_timer = Timer()
        total_timer.start()

        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
            limit_candidates=False,
        )

        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        top_k_used = top_k if top_k is not None else max(plan.top_k, 1)

        if search_result["retrieved_count"] == 0 or not candidate_cards:
            logger.info(
                "Recommendation finished with no candidates: query=%r retrieved=%d",
                query,
                search_result["retrieved_count"],
            )
            total_timer.stop()
            timers = dict(search_result.get("timers", {}))
            timers["total_ms"] = total_timer.elapsed_ms
            return self._build_recommendation_response(
                plan=plan,
                candidate_cards=candidate_cards,
                query_payload=search_result["query_payload"],
                branch_queries=search_result["branch_queries"],
                raw_query=search_result["raw_query"],
                retrieval_keywords=search_result.get("retrieval_keywords") or [],
                retrieval_score_traces=search_result.get("retrieval_score_traces") or [],
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
                timers=timers,
                expanded_shadow_hits=search_result.get("expanded_shadow_hits"),
            )

        with Timer() as evidence_selection_timer:
            relevant_evidence_by_expert_id = self.evidence_selector.select(
                candidates=candidate_cards,
                plan=plan,
            )
        evidence_selection_trace = self._extract_component_trace(self.evidence_selector) or {}
        shortlist, shortlist_gate_trace = self._apply_shortlist_gates(
            cards=candidate_cards,
            plan=plan,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            strict_evidence_gating=self.retriever.settings.strict_evidence_gating,
        )
        shortlist = shortlist[:top_k_used]
        logger.info(
            "Shortlist gate completed: kept=%d low_coverage=%d generic_only=%d dropped=%d shortlist=%s",
            len(shortlist_gate_trace["kept_candidate_ids"]),
            len(shortlist_gate_trace["low_coverage_candidate_ids"]),
            len(shortlist_gate_trace["generic_only_candidate_ids"]),
            len(shortlist_gate_trace["dropped_candidate_ids"]),
            [card.expert_id for card in shortlist],
        )

        if not shortlist:
            total_timer.stop()
            timers = dict(search_result.get("timers", {}))
            timers["evidence_selection_ms"] = evidence_selection_timer.elapsed_ms
            timers["total_ms"] = total_timer.elapsed_ms
            reason_generation_trace = {
                "mode": "gated_empty",
                "candidate_count": len(candidate_cards),
                "output_count": 0,
                "batch_count": 0,
                "batch_size": REASON_GENERATION_BATCH_SIZE,
                "reason_generation_failed": False,
                "batches": [],
                "evidence_selection": evidence_selection_trace,
                "shortlist_gate": shortlist_gate_trace,
                "selected_evidence": [],
                "server_fallback_reasons": [],
                "reason_sync_validator": {
                    "checked_candidate_ids": [],
                    "fallback_count": 0,
                    "fallback_ratio": 0.0,
                    "violations": [],
                },
            }
            logger.warning(
                "Recommendation ended after gates removed every candidate: query=%r",
                query,
            )
            return self._build_recommendation_response(
                plan=plan,
                candidate_cards=candidate_cards,
                query_payload=search_result["query_payload"],
                branch_queries=search_result["branch_queries"],
                raw_query=search_result["raw_query"],
                retrieval_keywords=search_result.get("retrieval_keywords") or [],
                retrieval_score_traces=search_result.get("retrieval_score_traces") or [],
                retrieval_skipped_reason=search_result.get("retrieval_skipped_reason"),
                retrieved_count=search_result["retrieved_count"],
                recommendations=[],
                data_gaps=[],
                not_selected_reasons=[NO_GATE_PASSED_CANDIDATE_REASON],
                planner_trace=search_result.get("planner_trace"),
                reason_generation_trace=reason_generation_trace,
                final_sort_policy=search_result["final_sort_policy"],
                top_k_used=top_k_used,
                timers=timers,
                expanded_shadow_hits=search_result.get("expanded_shadow_hits"),
            )

        retrieval_score_traces_by_expert_id = {
            trace["expert_id"]: trace
            for trace in (search_result.get("retrieval_score_traces") or [])
            if trace.get("expert_id")
        }

        with Timer() as reason_timer:
            reason_output, reason_batch_traces = await self._generate_reasons_in_batches(
                query=query,
                plan=plan,
                candidates=shortlist,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            )

        reason_generation_trace = self._build_reason_generation_trace(
            candidates=shortlist,
            reason_output=reason_output,
            batch_traces=reason_batch_traces,
        )
        reason_generation_trace = dict(reason_generation_trace)
        reason_generation_trace["evidence_selection"] = evidence_selection_trace
        reason_generation_trace["shortlist_gate"] = shortlist_gate_trace
        (
            recommendations,
            selected_evidence_trace,
            server_fallback_reasons,
            reason_sync_validator_trace,
        ) = self._build_recommendations(
            shortlist,
            reason_output,
            plan=plan,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
        )
        reason_generation_trace["selected_evidence"] = selected_evidence_trace
        reason_generation_trace["server_fallback_reasons"] = server_fallback_reasons
        reason_generation_trace["reason_sync_validator"] = reason_sync_validator_trace

        timers = dict(search_result.get("timers", {}))
        timers["evidence_selection_ms"] = evidence_selection_timer.elapsed_ms
        timers["reason_generation_ms"] = reason_timer.elapsed_ms
        total_timer.stop()
        timers["total_ms"] = total_timer.elapsed_ms

        logger.info(
            "Recommendation finished: query=%r recommended=%d total_ms=%.2f validator_fallbacks=%d",
            query,
            len(recommendations),
            total_timer.elapsed_ms,
            reason_sync_validator_trace["fallback_count"],
        )

        return self._build_recommendation_response(
            plan=plan,
            candidate_cards=candidate_cards,
            query_payload=search_result["query_payload"],
            branch_queries=search_result["branch_queries"],
            raw_query=search_result["raw_query"],
            retrieval_keywords=search_result.get("retrieval_keywords") or [],
            retrieval_score_traces=search_result.get("retrieval_score_traces") or [],
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
            expanded_shadow_hits=search_result.get("expanded_shadow_hits"),
        )

    async def recommend_stream(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        include_orgs: list[str] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> AsyncGenerator[str, None]:
        logger.info("Recommendation stream started: query=%r", query)
        total_timer = Timer()
        total_timer.start()

        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            include_orgs=include_orgs,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
            limit_candidates=False,
        )

        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        top_k_used = top_k if top_k is not None else max(plan.top_k, 1)

        # Yield search completed event
        yield f"event: search_completed\ndata: {json.dumps({'retrieved_count': search_result['retrieved_count']})}\n\n"

        if search_result["retrieved_count"] == 0 or not candidate_cards:
            yield f"event: stream_completed\ndata: {json.dumps({'reason': NO_MATCHING_CANDIDATE_REASON})}\n\n"
            return

        with Timer() as evidence_selection_timer:
            relevant_evidence_by_expert_id = self.evidence_selector.select(
                candidates=candidate_cards,
                plan=plan,
            )
        evidence_selection_trace = self._extract_component_trace(self.evidence_selector) or {}
        shortlist, shortlist_gate_trace = self._apply_shortlist_gates(
            cards=candidate_cards,
            plan=plan,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            strict_evidence_gating=self.retriever.settings.strict_evidence_gating,
        )
        shortlist = shortlist[:top_k_used]

        yield f"event: shortlist_completed\ndata: {json.dumps({'shortlist_count': len(shortlist)})}\n\n"

        if not shortlist:
            yield f"event: stream_completed\ndata: {json.dumps({'reason': NO_GATE_PASSED_CANDIDATE_REASON})}\n\n"
            return

        retrieval_score_traces_by_expert_id = {
            trace["expert_id"]: trace
            for trace in (search_result.get("retrieval_score_traces") or [])
            if trace.get("expert_id")
        }

        # Stream reasons
        async for batch_output, batch_trace in self._stream_reasons_in_batches(
            query=query,
            plan=plan,
            candidates=shortlist,
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
        ):
            # Find the original cards for this batch
            batch_candidate_ids = set(batch_trace["candidate_ids"])
            batch_cards = [card for card in shortlist if card.expert_id in batch_candidate_ids]
            
            (
                batch_recommendations,
                _,
                _,
                _
            ) = self._build_recommendations(
                batch_cards,
                batch_output,
                plan=plan,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
            )
            
            # Serialize decisions
            decisions = [rec.model_dump(mode="json") for rec in batch_recommendations]
            yield f"event: reason_batch_completed\ndata: {json.dumps({'recommendations': decisions})}\n\n"

        total_timer.stop()
        yield f"event: stream_completed\ndata: {json.dumps({'status': 'success', 'total_ms': total_timer.elapsed_ms})}\n\n"

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
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
    def _chunk_candidates(
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
        """
        최종 후보자 목록이 너무 많을 경우 LLM 컨텍스트 한계나 응답 누락 현상을 방지하기 위해 
        일정 크기(REASON_GENERATION_BATCH_SIZE)로 나누어 배치(Batch) 처리로 추천 사유를 생성합니다.
        """
        batch_outputs: list[ReasonGenerationOutput] = []
        batch_traces: list[dict[str, Any]] = []
        candidate_batches = self._chunk_candidates(
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
                "selected_evidence_count": raw_trace.get("selected_evidence_count", 0),
                "retry_triggered": raw_trace.get("retry_triggered", False),
                "retry_trigger": raw_trace.get("retry_trigger"),
                "retry_reason": raw_trace.get("retry_reason"),
                "attempts": list(raw_trace.get("attempts", [])),
                "raw_output_count": raw_trace.get(
                    "raw_output_count", len(batch_output.items)
                ),
                "output_count": raw_trace.get("output_count", len(batch_output.items)),
            }
            batch_traces.append(batch_trace)
            logger.info(
                "Reason generation batch completed: batch=%d/%d size=%d returned=%d missing=%d empty_reasons=%d retry_triggered=%s retry_reason=%s",
                batch_index,
                len(candidate_batches),
                len(candidate_batch),
                len(batch_trace["returned_ids"]),
                len(batch_trace["missing_candidate_ids"]),
                len(batch_trace["empty_reason_candidate_ids"]),
                batch_trace["retry_triggered"],
                batch_trace["retry_reason"],
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

    async def _stream_reasons_in_batches(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        candidates: list[CandidateCard],
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
        retrieval_score_traces_by_expert_id: dict[str, dict[str, Any]],
    ) -> AsyncGenerator[tuple[ReasonGenerationOutput, dict[str, Any]], None]:
        candidate_batches = self._chunk_candidates(
            candidates, batch_size=REASON_GENERATION_BATCH_SIZE
        )
        for batch_index, candidate_batch in enumerate(candidate_batches, start=1):
            batch_candidate_ids = [candidate.expert_id for candidate in candidate_batch]
            logger.info(
                "Reason generation stream batch started: batch=%d/%d size=%d",
                batch_index, len(candidate_batches), len(candidate_batch)
            )
            batch_output = await self.reason_generator.generate(
                query=query,
                plan=plan,
                candidates=candidate_batch,
                relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
                retrieval_score_traces_by_expert_id=retrieval_score_traces_by_expert_id,
            )
            raw_trace = dict(self._extract_component_trace(self.reason_generator) or {})
            batch_trace = {
                "batch_index": batch_index,
                "batch_size": len(candidate_batch),
                "candidate_ids": batch_candidate_ids,
                "returned_ids": list(raw_trace.get("returned_ids", [])),
                "missing_candidate_ids": list(raw_trace.get("missing_candidate_ids", [])),
                "empty_reason_candidate_ids": list(raw_trace.get("empty_reason_candidate_ids", [])),
                "mode": raw_trace.get("mode", "unknown"),
                "seed": raw_trace.get("seed"),
                "retry_count": raw_trace.get("retry_count", 0),
                "returned_ratio": raw_trace.get("returned_ratio", 0.0),
                "output_count": raw_trace.get("output_count", len(batch_output.items)),
            }
            yield batch_output, batch_trace

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
    def _normalize_text(value: str | None) -> str:
        return " ".join((value or "").lower().split())

    @classmethod
    def _required_aspect_coverage(cls, plan: PlannerOutput) -> int:
        # evidence_aspects가 있으면 그것을 기준으로 threshold 계산.
        # evidence_aspects는 bilingual이고 must_aspects보다 세분화된 용어이므로,
        # 더 많은 수의 aspects 중 최소 2개만 매칭해도 충분히 관련성 있는 후보로 판단.
        target_aspects = normalize_phrase_keywords(
            plan.evidence_aspects
            or getattr(plan, "must_aspects", None)
            or plan.retrieval_core
            or plan.core_keywords
        )
        if not target_aspects:
            return 0
        return min(2, len(target_aspects))

    @classmethod
    def _apply_shortlist_gates(
        cls,
        *,
        cards: list[CandidateCard],
        plan: PlannerOutput,
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
        strict_evidence_gating: bool = True,
    ) -> tuple[list[CandidateCard], dict[str, Any]]:
        """
        검색된 후보자들 중 LLM 추천 사유 생성으로 넘길 가치가 있는 우수 후보(Shortlist)만 남기는 관문(Gate) 역할을 합니다.
        - 쿼리와 직접 매칭된 증거가 최소 1개 이상 있는지 여부
        - 요구된 속성(aspect)들을 충분히 커버하는지 여부
        를 평가하여 미달하는 후보를 탈락(drop)시키거나 후순위로 강등(demote)시킵니다.
        """
        # gate diagnostic 표시용 — evidence_selector가 실제로 사용한 것과 동일한 소스 사용
        target_aspects = normalize_phrase_keywords(
            plan.evidence_aspects
            or getattr(plan, "must_aspects", None)
            or plan.retrieval_core
            or plan.core_keywords
        )
        coverage_threshold = cls._required_aspect_coverage(plan)
        keep_cards: list[CandidateCard] = []
        low_coverage_cards: list[CandidateCard] = []
        generic_only_cards: list[CandidateCard] = []
        dropped_cards: list[CandidateCard] = []
        candidate_diagnostics: list[dict[str, Any]] = []

        for card in cards:
            bundle = relevant_evidence_by_expert_id.get(
                card.expert_id,
                RelevantEvidenceBundle(expert_id=card.expert_id),
            )
            gate_status = "keep"
            if bundle.direct_match_count < 1:
                if strict_evidence_gating:
                    gate_status = "drop_no_direct_evidence"
                    dropped_cards.append(card)
                else:
                    gate_status = "demote_no_direct_evidence"
                    low_coverage_cards.append(card)
            elif coverage_threshold and bundle.aspect_coverage < coverage_threshold:
                gate_status = "demote_low_aspect_coverage"
                low_coverage_cards.append(card)
            elif bundle.generic_only:
                gate_status = "demote_generic_only"
                generic_only_cards.append(card)
            else:
                keep_cards.append(card)

            candidate_diagnostics.append(
                {
                    "expert_id": card.expert_id,
                    "gate_status": gate_status,
                    "direct_match_count": bundle.direct_match_count,
                    "aspect_coverage": bundle.aspect_coverage,
                    "generic_only": bundle.generic_only,
                    "matched_aspects": list(bundle.matched_aspects),
                    "matched_generic_terms": list(bundle.matched_generic_terms),
                    "future_selected_evidence_ids": list(
                        bundle.future_selected_evidence_ids
                    ),
                }
            )

        aspect_source = (
            "evidence_aspects" if plan.evidence_aspects
            else "must_aspects" if plan.must_aspects
            else "retrieval_core"
        )
        return [*keep_cards, *low_coverage_cards, *generic_only_cards], {
            "coverage_threshold": coverage_threshold,
            "coverage_threshold_basis": "phrase",
            "aspect_source": aspect_source,
            "target_aspects": target_aspects,
            "kept_candidate_ids": [card.expert_id for card in keep_cards],
            "low_coverage_candidate_ids": [card.expert_id for card in low_coverage_cards],
            "generic_only_candidate_ids": [card.expert_id for card in generic_only_cards],
            "dropped_candidate_ids": [card.expert_id for card in dropped_cards],
            "candidate_diagnostics": candidate_diagnostics,
        }

    @staticmethod
    def _build_profile_evidence(card: CandidateCard) -> EvidenceItem | None:
        if not (card.organization or card.degree or card.major):
            return None
        return EvidenceItem(
            type="profile",
            title=card.name,
            detail=" / ".join(
                [
                    card.organization or "unknown organization",
                    card.degree or "unknown degree",
                    card.major or "unknown major",
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
        )

    @staticmethod
    def _build_server_fallback_reason(
        *,
        evidence: list[EvidenceItem],
        fallback: str,
    ) -> str:
        if not evidence:
            return "직접적인 질의 일치 근거를 확인하지 못했습니다."

        if all(item.type == "profile" for item in evidence):
            return "직접적인 질의 일치 근거는 제한적이지만 프로필 기반 후보로 검토되었습니다."

        type_labels = {
            "project": "과제",
            "paper": "논문",
            "patent": "특허",
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
            referenced_items.append(
                f"'{item.title}' {type_labels.get(item.type, item.type)}"
            )
            if len(referenced_items) == 2:
                break

        if not referenced_items:
            return "직접적인 질의 일치 근거는 제한적이지만 프로필 기반 후보로 검토되었습니다."
        if len(referenced_items) == 1:
            return f"{referenced_items[0]}을 통해 질의와 관련된 직접 근거를 확인했습니다."
        return (
            f"{referenced_items[0]}과 {referenced_items[1]}을 통해 "
            "질의와 관련된 직접 근거를 확인했습니다."
        )

    @classmethod
    def _build_candidate_evidence(
        cls,
        *,
        card: CandidateCard,
        relevant_bundle: RelevantEvidenceBundle,
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        resolved_items = relevant_bundle.all_items()
        provided_evidence_ids = [item.item_id for item in resolved_items]

        if not resolved_items:
            profile_item = cls._build_profile_evidence(card)
            if profile_item is not None:
                return [profile_item], {
                    "expert_id": card.expert_id,
                    "provided_evidence_ids": [],
                    "selected_evidence_ids": [],
                    "resolved_evidence_ids": ["profile"],
                    "direct_match_count": relevant_bundle.direct_match_count,
                    "aspect_coverage": relevant_bundle.aspect_coverage,
                    "matched_aspects": list(relevant_bundle.matched_aspects),
                    "future_selected_evidence_ids": list(
                        relevant_bundle.future_selected_evidence_ids
                    ),
                    "fallback": "profile",
                }
            return [], {
                "expert_id": card.expert_id,
                "provided_evidence_ids": [],
                "selected_evidence_ids": [],
                "resolved_evidence_ids": [],
                "direct_match_count": relevant_bundle.direct_match_count,
                "aspect_coverage": relevant_bundle.aspect_coverage,
                    "matched_aspects": list(relevant_bundle.matched_aspects),
                    "future_selected_evidence_ids": [],
                    "fallback": "empty",
                }

        return (
            [cls._build_evidence_item(item) for item in resolved_items],
            {
                "expert_id": card.expert_id,
                "provided_evidence_ids": provided_evidence_ids,
                "selected_evidence_ids": [],
                "resolved_evidence_ids": provided_evidence_ids,
                "direct_match_count": relevant_bundle.direct_match_count,
                "aspect_coverage": relevant_bundle.aspect_coverage,
                "matched_aspects": list(relevant_bundle.matched_aspects),
                "future_selected_evidence_ids": list(
                    relevant_bundle.future_selected_evidence_ids
                ),
                "fallback": "none",
            },
        )

    @classmethod
    def _validate_reason_sync(
        cls,
        *,
        plan: PlannerOutput,
        card: CandidateCard,
        recommendation_reason: str,
        evidence: list[EvidenceItem],
        relevant_bundle: RelevantEvidenceBundle,
        candidate_names: list[str],
    ) -> list[str]:
        normalized_reason = cls._normalize_text(recommendation_reason)
        if not normalized_reason:
            return []

        violations: list[str] = []
        for candidate_name in candidate_names:
            normalized_name = cls._normalize_text(candidate_name)
            if (
                candidate_name != card.name
                and len(normalized_name) >= 2
                and normalized_name in normalized_reason
            ):
                violations.append("other_candidate_name")
                break

        if INTERNAL_EVIDENCE_ID_PATTERN.search(recommendation_reason):
            violations.append("internal_evidence_id")

        target_aspects = normalize_phrase_keywords(
            getattr(plan, "must_aspects", None) or plan.retrieval_core or plan.core_keywords
        )
        evidence_scope_texts = [
            cls._normalize_text(
                " ".join(
                    value
                    for value in [item.title, item.detail, item.snippet]
                    if value
                )
            )
            for item in relevant_bundle.all_items()
        ]
        if not evidence_scope_texts:
            evidence_scope_texts = [
                cls._normalize_text(" ".join(value for value in [item.title, item.detail] if value))
                for item in evidence
                if item.type != "profile"
            ]
        if target_aspects and evidence_scope_texts:
            if not any(
                cls._normalize_text(aspect) in scope_text
                for aspect in target_aspects
                for scope_text in evidence_scope_texts
            ):
                violations.append("aspect_scope_miss")

        non_profile_evidence = [item for item in evidence if item.type != "profile"]
        if not non_profile_evidence and any(
            marker in normalized_reason for marker in STRONG_REASON_MARKERS
        ):
            violations.append("strong_claim_without_direct_evidence")

        return violations

    def _build_recommendations(
        self,
        cards: list[CandidateCard],
        reason_output: ReasonGenerationOutput,
        *,
        plan: PlannerOutput,
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
    ) -> tuple[
        list[RecommendationDecision],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        generated_by_expert_id = {
            item.expert_id: item for item in reason_output.items
        }
        recommendations: list[RecommendationDecision] = []
        selected_evidence_trace: list[dict[str, Any]] = []
        server_fallback_reasons: list[dict[str, Any]] = []
        validator_violations: list[dict[str, Any]] = []
        candidate_names = [card.name for card in cards]

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
                fit = (
                    generated.fit
                    if generated.fit in {"높음", "중간", "보통"}
                    else "보통"
                )
                recommendation_reason = generated.recommendation_reason
                risks = list(generated.risks) or list(card.risks)

            evidence, evidence_trace = self._build_candidate_evidence(
                card=card,
                relevant_bundle=relevant_bundle,
            )
            selected_evidence_trace.append(evidence_trace)

            if not recommendation_reason:
                fallback_source = (
                    evidence_trace.get("fallback")
                    if evidence_trace.get("fallback") not in {None, "none"}
                    else "selected_evidence"
                )
                recommendation_reason = self._build_server_fallback_reason(
                    evidence=evidence,
                    fallback=fallback_source,
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
            else:
                violations = self._validate_reason_sync(
                    plan=plan,
                    card=card,
                    recommendation_reason=recommendation_reason,
                    evidence=evidence,
                    relevant_bundle=relevant_bundle,
                    candidate_names=candidate_names,
                )
                if violations:
                    recommendation_reason = self._build_server_fallback_reason(
                        evidence=evidence,
                        fallback="reason_sync_validator",
                    )
                    violation_entry = {
                        "expert_id": card.expert_id,
                        "violations": violations,
                        "resolved_evidence_ids": list(
                            evidence_trace.get("resolved_evidence_ids", [])
                        ),
                    }
                    validator_violations.append(violation_entry)
                    server_fallback_reasons.append(
                        {
                            "expert_id": card.expert_id,
                            "source": "reason_sync_validator",
                            "resolved_evidence_ids": list(
                                evidence_trace.get("resolved_evidence_ids", [])
                            ),
                            "violations": violations,
                        }
                    )
                    logger.warning(
                        "Recommendation reason validator fallback generated: expert_id=%s violations=%s resolved_evidence_ids=%s",
                        card.expert_id,
                        violations,
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
                    evidence=evidence,
                    risks=risks,
                    rank_score=card.rank_score,
                )
            )

        validator_trace = {
            "checked_candidate_ids": [card.expert_id for card in cards],
            "fallback_count": len(validator_violations),
            "fallback_ratio": round(len(validator_violations) / len(cards), 3)
            if cards
            else 0.0,
            "violations": validator_violations,
        }
        return (
            recommendations,
            selected_evidence_trace,
            server_fallback_reasons,
            validator_trace,
        )

    @staticmethod
    def _serialize_shadow_hits(hits: list[SearchHit]) -> list[dict[str, str]]:
        return [
            {
                "expert_id": hit.expert_id,
                "name": hit.payload.basic_info.researcher_name or "",
            }
            for hit in hits
        ]

    @staticmethod
    def _build_recommendation_response(
        *,
        plan: PlannerOutput,
        candidate_cards: list[CandidateCard],
        query_payload: dict[str, Any],
        branch_queries: dict[str, CompiledBranchQueries],
        raw_query: str,
        retrieval_keywords: list[str],
        retrieval_score_traces: list[dict[str, Any]],
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
        expanded_shadow_hits: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        merged_data_gaps = _merge_unique_strings(data_gaps)
        return {
            "intent_summary": plan.intent_summary,
            "applied_filters": plan.hard_filters,
            "searched_branches": list(BRANCHES),
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
                    "retrieval": "hit" if timers and timers.get("search_ms", 0) < 5 else "miss",
                },
                "planner_keywords": (
                    (planner_trace or {}).get("planner_keywords") or []
                ),
                "retrieval_keywords": retrieval_keywords,
                "bundle_ids": plan.bundle_ids,
                "expanded_shadow_hits": expanded_shadow_hits or [],
                "filtered_out_candidates": (planner_trace or {}).get("filtered_out_candidates") or [],
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
