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
from apps.search.filters import QdrantFilterCompiler
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES

logger = logging.getLogger(__name__)

NO_MATCHING_CANDIDATE_REASON = "No matching candidates were found."
EMPTY_RETRIEVAL_KEYWORDS_REASON = (
    "Retrieval skipped because planner core_keywords were empty after retry."
)
FINAL_SORT_POLICY = "rrf_score_desc_name_asc"
REASON_GENERATION_BATCH_SIZE = 5


class RecommendationService:
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
            "Planner completed: elapsed_ms=%.2f intent=%r filters=%r keywords=%d",
            plan_timer.elapsed_ms,
            plan.intent_summary,
            plan.hard_filters,
            len(retrieval_keywords),
        )

        query_filter = self.filter_compiler.compile(
            plan.hard_filters,
            plan.exclude_orgs,
            include_orgs=plan.include_orgs,
        )

        if not retrieval_keywords:
            logger.warning(
                "Retriever skipped: query=%r reason=%s",
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

        display_hits = retrieval.hits[:top_k] if top_k is not None else retrieval.hits
        cards = self.card_builder.build_small_cards(display_hits, plan)
        return {
            "planner": plan,
            "planner_trace": planner_trace,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "query_payload": retrieval.query_payload,
            "branch_queries": retrieval.branch_queries,
            "retrieval_keywords": retrieval.retrieval_keywords,
            "retrieval_score_traces": retrieval.retrieval_score_traces,
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
        logger.info("Recommendation started: query=%r", query)
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
        top_k_used = top_k if top_k is not None else max(plan.top_k, 1)
        shortlist = candidate_cards[:top_k_used]

        logger.info(
            "Top-k selected for reason generation: candidates=%d top_k=%d selected=%d",
            len(candidate_cards),
            top_k_used,
            len(shortlist),
        )

        if search_result["retrieved_count"] == 0 or not shortlist:
            logger.info(
                "Recommendation finished with no shortlist: query=%r retrieved=%d shortlist=%d",
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
            )

        with Timer() as evidence_selection_timer:
            relevant_evidence_by_expert_id = self.evidence_selector.select(
                candidates=shortlist,
                plan=plan,
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
            relevant_evidence_by_expert_id=relevant_evidence_by_expert_id,
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
            "Recommendation finished: query=%r recommended=%d total_ms=%.2f",
            query,
            len(recommendations),
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
            referenced_items.append(f"'{item.title}' {type_labels.get(item.type, item.type)}")
            if len(referenced_items) == 2:
                break

        if not referenced_items:
            return "직접적인 질의 일치 근거는 제한적이지만 프로필 기반 후보로 검토되었습니다."

        if len(referenced_items) == 1:
            return (
                f"{referenced_items[0]}이 확인되어 질의와 관련된 전문성 근거로 참고할 수 있습니다."
            )
        return (
            f"{referenced_items[0]}와 {referenced_items[1]}이 확인되어 "
            "질의와 관련된 전문성 근거로 참고할 수 있습니다."
        )

    @classmethod
    def _build_candidate_evidence(
        cls,
        *,
        card: CandidateCard,
        generated: Any | None,
        relevant_bundle: RelevantEvidenceBundle,
    ) -> tuple[list[EvidenceItem], dict[str, Any]]:
        evidence_lookup = relevant_bundle.by_item_id()
        provided_evidence_ids = [
            item.item_id for item in cls._sort_relevant_items(relevant_bundle)
        ]
        requested_ids = list(getattr(generated, "selected_evidence_ids", []) or [])[:4]
        invalid_selected_evidence_ids = [
            item_id
            for item_id in requested_ids
            if not VALID_EVIDENCE_ID_PATTERN.fullmatch(item_id)
        ]
        unresolved_selected_evidence_ids = [
            item_id
            for item_id in requested_ids
            if item_id not in invalid_selected_evidence_ids and item_id not in evidence_lookup
        ]

        resolved_items: list[RelevantEvidenceItem] = []
        seen_ids: set[str] = set()
        for item_id in requested_ids:
            item = evidence_lookup.get(item_id)
            if item is None or item_id in seen_ids:
                continue
            resolved_items.append(item)
            seen_ids.add(item_id)

        fallback = "none"
        if not resolved_items:
            if requested_ids:
                logger.warning(
                    "Selected evidence ids could not be resolved; using fallback: expert_id=%s requested_ids=%s invalid_ids=%s unresolved_ids=%s available_ids=%s",
                    card.expert_id,
                    requested_ids,
                    invalid_selected_evidence_ids,
                    unresolved_selected_evidence_ids,
                    provided_evidence_ids,
                )
            ranked_items = cls._sort_relevant_items(relevant_bundle)
            if ranked_items:
                resolved_items = [ranked_items[0]]
                fallback = "top_relevant"
            else:
                profile_item = cls._build_profile_evidence(card)
                if profile_item is not None:
                    return [profile_item], {
                        "expert_id": card.expert_id,
                        "provided_evidence_ids": provided_evidence_ids,
                        "selected_evidence_ids": requested_ids,
                        "llm_selected_evidence_ids": requested_ids,
                        "resolver_available_evidence_ids": provided_evidence_ids,
                        "invalid_selected_evidence_ids": invalid_selected_evidence_ids,
                        "unresolved_selected_evidence_ids": unresolved_selected_evidence_ids,
                        "resolved_evidence_ids": ["profile"],
                        "relevant_evidence_count": len(provided_evidence_ids),
                        "relevant_bundle_empty": not provided_evidence_ids,
                        "fallback": "profile",
                    }
                return [], {
                    "expert_id": card.expert_id,
                    "provided_evidence_ids": provided_evidence_ids,
                    "selected_evidence_ids": requested_ids,
                    "llm_selected_evidence_ids": requested_ids,
                    "resolver_available_evidence_ids": provided_evidence_ids,
                    "invalid_selected_evidence_ids": invalid_selected_evidence_ids,
                    "unresolved_selected_evidence_ids": unresolved_selected_evidence_ids,
                    "resolved_evidence_ids": [],
                    "relevant_evidence_count": len(provided_evidence_ids),
                    "relevant_bundle_empty": not provided_evidence_ids,
                    "fallback": "empty",
                }

        return (
            [cls._build_evidence_item(item) for item in resolved_items],
            {
                "expert_id": card.expert_id,
                "provided_evidence_ids": provided_evidence_ids,
                "selected_evidence_ids": requested_ids,
                "llm_selected_evidence_ids": requested_ids,
                "resolver_available_evidence_ids": provided_evidence_ids,
                "invalid_selected_evidence_ids": invalid_selected_evidence_ids,
                "unresolved_selected_evidence_ids": unresolved_selected_evidence_ids,
                "resolved_evidence_ids": [item.item_id for item in resolved_items],
                "relevant_evidence_count": len(provided_evidence_ids),
                "relevant_bundle_empty": not provided_evidence_ids,
                "fallback": fallback,
            },
        )

    def _build_recommendations(
        self,
        cards: list[CandidateCard],
        reason_output: ReasonGenerationOutput,
        *,
        relevant_evidence_by_expert_id: dict[str, RelevantEvidenceBundle],
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
                    evidence=evidence,
                    risks=risks,
                    rank_score=card.rank_score,
                )
            )

        return recommendations, selected_evidence_trace, server_fallback_reasons

    @staticmethod
    def _build_recommendation_response(
        *,
        plan: PlannerOutput,
        candidate_cards: list[CandidateCard],
        query_payload: dict[str, Any],
        branch_queries: dict[str, str],
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
                "planner_keywords": (
                    (planner_trace or {}).get("planner_keywords") or []
                ),
                "retrieval_keywords": retrieval_keywords,
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
