from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.domain.models import CandidateCard, JudgeOutput, PlannerOutput
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import Judge
from apps.recommendation.planner import Planner
from apps.search.filters import QdrantFilterCompiler
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
        self.judge = judge
        self.feedback_store = feedback_store
        self.shortlist_limit = shortlist_limit

    async def search_candidates(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        plan = await self.planner.plan(
            query=query,
            filters_override=filters_override,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )
        query_filter = self.filter_compiler.compile(plan.hard_filters, plan.exclude_orgs)
        retrieval = await self.retriever.search(query=query, plan=plan, query_filter=query_filter)
        cards = self.card_builder.build_small_cards(retrieval.hits, plan.hard_filters)
        return {
            "planner": plan,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "query_payload": retrieval.query_payload,
            "branch_queries": retrieval.branch_queries,
        }

    async def recommend(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )
        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        shortlist = self.card_builder.shortlist(candidate_cards, self.shortlist_limit)
        judge_output: JudgeOutput = await self.judge.judge(query=query, plan=plan, shortlist=shortlist)

        # 운영 경로에서는 evidence 없는 추천을 허용하지 않는다.
        # 추천이 나갔는데 근거가 없다면 품질 문제를 경고가 아니라 실패로 본다.
        if not judge_output.recommended:
            raise RuntimeError("추천 결과가 비어 있습니다.")
        if not all(item.evidence for item in judge_output.recommended):
            raise RuntimeError("evidence가 비어 있는 추천 결과가 포함되어 있습니다.")

        return {
            "intent_summary": plan.intent_summary,
            "applied_filters": plan.hard_filters,
            "searched_branches": list(BRANCHES),
            "retrieved_count": search_result["retrieved_count"],
            "recommendations": judge_output.recommended,
            "data_gaps": judge_output.data_gaps,
            "not_selected_reasons": judge_output.not_selected_reasons,
            "trace": {
                "planner": plan.model_dump(mode="json"),
                "branch_queries": search_result["branch_queries"],
                "exclude_orgs": plan.exclude_orgs,
                "candidate_ids": [card.expert_id for card in candidate_cards],
                "recommendation_evidence_summary": [
                    {
                        "expert_id": item.expert_id,
                        "evidence_titles": [evidence.title for evidence in item.evidence],
                    }
                    for item in judge_output.recommended
                ],
                "query_payload": self._serialize_query_payload(search_result["query_payload"]),
            },
        }

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
        return self.feedback_store.save_feedback(
            query=query,
            selected_expert_ids=selected_expert_ids,
            rejected_expert_ids=rejected_expert_ids,
            notes=notes,
            metadata=metadata,
        )

    @staticmethod
    def _serialize_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(payload)
        serialized["prefetch"] = [str(item) for item in payload.get("prefetch", [])]
        query_filter = payload.get("query_filter")
        serialized["query_filter"] = str(query_filter) if query_filter else None
        serialized["query"] = str(payload.get("query"))
        return serialized
