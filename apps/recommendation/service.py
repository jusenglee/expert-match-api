from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.core.timer import Timer
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import CandidateCard, EvidenceItem, PlannerOutput, RecommendationDecision
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.planner import Planner
from apps.recommendation.reasoner import ReasonGenerationOutput, ReasonGenerator
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


class RecommendationService:
    def __init__(
        self,
        *,
        planner: Planner,
        retriever: QdrantHybridRetriever,
        filter_compiler: QdrantFilterCompiler,
        card_builder: CandidateCardBuilder,
        reason_generator: ReasonGenerator,
        feedback_store: FeedbackStore,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
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

        with Timer() as reason_timer:
            reason_output = await self.reason_generator.generate(
                query=query,
                plan=plan,
                candidates=shortlist,
            )

        reason_generation_trace = self._extract_component_trace(self.reason_generator)
        recommendations = self._build_recommendations(shortlist, reason_output)
        timers = dict(search_result.get("timers", {}))
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
    def _build_candidate_evidence(card: CandidateCard) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        if card.top_papers:
            paper = card.top_papers[0]
            evidence.append(
                EvidenceItem(
                    type="paper",
                    title=paper.publication_title,
                    date=paper.publication_year_month,
                    detail=paper.journal_name,
                )
            )
        if card.top_patents:
            patent = card.top_patents[0]
            evidence.append(
                EvidenceItem(
                    type="patent",
                    title=patent.intellectual_property_title,
                    date=patent.registration_date or patent.application_date,
                    detail=patent.application_registration_type
                    or patent.application_country,
                )
            )
        if card.top_projects:
            project = card.top_projects[0]
            evidence.append(
                EvidenceItem(
                    type="project",
                    title=project.display_title,
                    date=project.project_end_date or project.project_start_date,
                    detail=project.managing_agency or project.performing_organization,
                )
            )
        if card.organization or card.degree or card.major:
            evidence.append(
                EvidenceItem(
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
            )
        return evidence

    def _build_recommendations(
        self,
        cards: list[CandidateCard],
        reason_output: ReasonGenerationOutput,
    ) -> list[RecommendationDecision]:
        generated_by_expert_id = {
            item.expert_id: item for item in reason_output.items
        }
        recommendations: list[RecommendationDecision] = []

        for rank, card in enumerate(cards, start=1):
            generated = generated_by_expert_id.get(card.expert_id)
            fit = "보통"
            recommendation_reason = ""
            risks = list(card.risks)
            if generated is not None:
                fit = generated.fit if generated.fit in {"높음", "중간", "보통"} else "보통"
                recommendation_reason = generated.recommendation_reason
                risks = list(generated.risks) or list(card.risks)

            recommendations.append(
                RecommendationDecision(
                    rank=rank,
                    expert_id=card.expert_id,
                    name=card.name,
                    organization=card.organization,
                    fit=fit,
                    recommendation_reason=recommendation_reason,
                    evidence=self._build_candidate_evidence(card),
                    risks=risks,
                    rank_score=card.rank_score,
                )
            )

        return recommendations

    @staticmethod
    def _build_recommendation_response(
        *,
        plan: PlannerOutput,
        candidate_cards: list[CandidateCard],
        query_payload: dict[str, Any],
        branch_queries: dict[str, str],
        raw_query: str,
        retrieval_keywords: list[str],
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
