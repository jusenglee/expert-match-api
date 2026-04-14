from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.core.timer import Timer
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import CandidateCard, PlannerOutput, RecommendationDecision
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import Judge
from apps.recommendation.planner import Planner
from apps.search.filters import QdrantFilterCompiler
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES

logger = logging.getLogger(__name__)

NO_MATCHING_CANDIDATE_REASON = "No matching candidates were found."
INSUFFICIENT_EVIDENCE_REASON = (
    "Recommendations were dropped because supporting evidence was insufficient."
)
EMPTY_RETRIEVAL_KEYWORDS_REASON = (
    "Retrieval skipped because planner core_keywords were empty after retry."
)


class RecommendationService:
    def __init__(
        self,
        *,
        planner: Planner,
        retriever: QdrantHybridRetriever,
        filter_compiler: QdrantFilterCompiler,
        card_builder: CandidateCardBuilder,
        judge: Judge,
        feedback_store: FeedbackStore,
        shortlist_limit: int,
        use_map_reduce_judging: bool = True,
        final_recommendation_max: int = 20,
        final_recommendation_min: int = 1,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
        self.judge = judge
        self.feedback_store = feedback_store
        self.shortlist_limit = shortlist_limit
        self.use_map_reduce_judging = use_map_reduce_judging
        self.final_recommendation_max = final_recommendation_max
        self.final_recommendation_min = final_recommendation_min

        judge_settings = getattr(self.judge, "settings", None)
        if judge_settings is not None and hasattr(
            judge_settings, "use_map_reduce_judging"
        ):
            judge_settings.use_map_reduce_judging = use_map_reduce_judging

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

        cards = self.card_builder.build_small_cards(retrieval.hits, plan)
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
        shortlist = self.card_builder.shortlist(candidate_cards, self.shortlist_limit)

        logger.info(
            "Shortlist built: candidates=%d shortlist=%d limit=%d",
            len(candidate_cards),
            len(shortlist),
            self.shortlist_limit,
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
                judge_trace=None,
                timers=search_result.get("timers"),
            )

        with Timer() as judge_timer:
            judge_output = await self.judge.judge(
                query=query,
                plan=plan,
                shortlist=shortlist,
            )

        judge_trace = self._extract_component_trace(self.judge)
        evidence_less_count = sum(
            1 for item in judge_output.recommended if not item.evidence
        )
        recommendations = [
            item for item in judge_output.recommended if item.evidence
        ]

        if len(recommendations) > self.final_recommendation_max:
            logger.warning(
                "Recommendation count truncated: current=%d max=%d",
                len(recommendations),
                self.final_recommendation_max,
            )
            recommendations = recommendations[: self.final_recommendation_max]

        if 0 < len(recommendations) < self.final_recommendation_min:
            logger.warning(
                "Recommendation count below minimum: current=%d min=%d",
                len(recommendations),
                self.final_recommendation_min,
            )

        not_selected_reasons = _merge_unique_strings(judge_output.not_selected_reasons)
        if not judge_output.recommended:
            not_selected_reasons = _merge_unique_strings(
                not_selected_reasons,
                [NO_MATCHING_CANDIDATE_REASON],
            )
        elif evidence_less_count:
            not_selected_reasons = _merge_unique_strings(
                not_selected_reasons,
                [INSUFFICIENT_EVIDENCE_REASON],
            )

        timers = dict(search_result.get("timers", {}))
        timers["judge_ms"] = judge_timer.elapsed_ms

        total_timer.stop()
        timers["total_ms"] = total_timer.elapsed_ms

        logger.info(
            "Recommendation finished: query=%r recommended=%d dropped_without_evidence=%d total_ms=%.2f",
            query,
            len(recommendations),
            evidence_less_count,
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
            data_gaps=judge_output.data_gaps,
            not_selected_reasons=not_selected_reasons,
            planner_trace=search_result.get("planner_trace"),
            judge_trace=judge_trace,
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
        judge_trace: dict[str, Any] | None,
        timers: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "intent_summary": plan.intent_summary,
            "applied_filters": plan.hard_filters,
            "searched_branches": list(BRANCHES),
            "retrieved_count": retrieved_count,
            "recommendations": recommendations,
            "data_gaps": data_gaps,
            "not_selected_reasons": not_selected_reasons,
            "trace": {
                "planner": plan.model_dump(mode="json"),
                "planner_trace": planner_trace or {},
                "judge_trace": judge_trace or {},
                "final_reduce_candidate_count": (
                    (judge_trace or {}).get("final_reduce_candidate_count")
                ),
                "final_reduce_token_estimate": (
                    (judge_trace or {}).get("final_reduce_token_estimate")
                ),
                "final_reduce_gate_reason": (
                    (judge_trace or {}).get("final_reduce_gate_reason")
                ),
                "raw_query": raw_query,
                "retrieval_keywords": retrieval_keywords,
                "planner_retry_count": (
                    (planner_trace or {}).get("planner_retry_count", 0)
                ),
                "retrieval_skipped_reason": retrieval_skipped_reason,
                "branch_queries": branch_queries,
                "include_orgs": plan.include_orgs,
                "exclude_orgs": plan.exclude_orgs,
                "candidate_ids": [card.expert_id for card in candidate_cards],
                "recommendation_evidence_summary": [
                    {
                        "expert_id": item.expert_id,
                        "evidence_titles": [
                            evidence.title for evidence in item.evidence
                        ],
                    }
                    for item in recommendations
                ],
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
        query = payload.get("query")
        serialized["query"] = str(query) if query is not None else None
        return serialized
