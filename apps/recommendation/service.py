from __future__ import annotations

import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
from apps.core.utils import merge_unique_strings as _merge_unique_strings
from apps.domain.models import CandidateCard, JudgeOutput, PlannerOutput, RecommendationDecision
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import Judge
from apps.recommendation.planner import Planner
from apps.core.timer import Timer
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
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        with Timer() as t_plan:
            plan = await self.planner.plan(
                query=query,
                filters_override=filters_override,
                exclude_orgs=exclude_orgs,
                top_k=top_k,
            )
        logger.info("질의 분석 완료: 소요시간=%.2fms 의도=%r 필터=%r", t_plan.elapsed_ms, plan.intent_summary, plan.hard_filters)
        logger.info("플래너 상세 답변(PlannerOutput): %s", plan.model_dump_json(indent=2))
        
        query_filter = self.filter_compiler.compile(
            plan.hard_filters,
            plan.exclude_orgs,
            include_orgs=plan.include_orgs,
        )
        with Timer() as t_search:
            retrieval = await self.retriever.search(query=query, plan=plan, query_filter=query_filter)
        logger.info("검색 완료: 소요시간=%.2fms 검색된 히트 수=%d", t_search.elapsed_ms, len(retrieval.hits))
        
        cards = self.card_builder.build_small_cards(retrieval.hits, plan)
        return {
            "planner": plan,
            "query_filter": query_filter,
            "retrieved_count": len(retrieval.hits),
            "candidates": cards,
            "query_payload": retrieval.query_payload,
            "branch_queries": retrieval.branch_queries,
            "timers": {
                "plan_ms": t_plan.elapsed_ms,
                "search_ms": t_search.elapsed_ms,
            }
        }

    async def recommend(
        self,
        *,
        query: str,
        filters_override: dict[str, Any] | None = None,
        exclude_orgs: list[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        logger.info("전체 추천 프로세스 시작: 질의=%r", query)
        t_total = Timer()
        t_total.start()
        search_result = await self.search_candidates(
            query=query,
            filters_override=filters_override,
            exclude_orgs=exclude_orgs,
            top_k=top_k,
        )

        plan: PlannerOutput = search_result["planner"]
        candidate_cards: list[CandidateCard] = search_result["candidates"]
        shortlist = self.card_builder.shortlist(candidate_cards, self.shortlist_limit)
        logger.info("후보자 압축 완료: 초기 후보=%d명 -> 최종 후보(Shortlist)=%d명 (제한=%d)", 
                    len(candidate_cards), len(shortlist), self.shortlist_limit)

        if search_result["retrieved_count"] == 0 or not shortlist:
            logger.info(
                "추천 결과 없음: 질의=%r 검색된 수=%d 압축된 수=%d 사유=적합한 후보 없음",
                query,
                search_result["retrieved_count"],
                len(shortlist),
            )
            return self._build_recommendation_response(
                plan=plan,
                candidate_cards=candidate_cards,
                query_payload=search_result["query_payload"],
                branch_queries=search_result["branch_queries"],
                retrieved_count=search_result["retrieved_count"],
                recommendations=[],
                data_gaps=[],
                not_selected_reasons=["조건을 만족하는 추천 후보를 찾지 못했습니다."],
            )

        with Timer() as t_judge:
            logger.info(
                "최종 판정(Judging) 단계 시작: 후보 수=%d, judge_managed_map_reduce=%s",
                len(shortlist),
                self.use_map_reduce_judging,
            )
            judge_output = await self.judge.judge(
                query=query,
                plan=plan,
                shortlist=shortlist,
            )
        
        evidence_less_count = sum(1 for item in judge_output.recommended if not item.evidence)
        recommendations = [item for item in judge_output.recommended if item.evidence]

        # final_recommendation_max 상한 적용
        if len(recommendations) > self.final_recommendation_max:
            logger.warning(
                "추천 결과가 최대 허용치를 초과하여 절단: %d → %d",
                len(recommendations),
                self.final_recommendation_max,
            )
            recommendations = recommendations[: self.final_recommendation_max]

        # final_recommendation_min 하한 경고 (결과 자체는 유지)
        if 0 < len(recommendations) < self.final_recommendation_min:
            logger.warning(
                "추천 결과가 최소 기준 미달: 현재=%d 기준=%d (결과 그대로 반환)",
                len(recommendations),
                self.final_recommendation_min,
            )

        not_selected_reasons = _merge_unique_strings(judge_output.not_selected_reasons)

        if not judge_output.recommended:
            not_selected_reasons = _merge_unique_strings(
                not_selected_reasons,
                ["조건을 만족하는 추천 후보를 찾지 못했습니다."],
            )
        elif evidence_less_count:
            not_selected_reasons = _merge_unique_strings(
                not_selected_reasons,
                ["근거가 충분한 추천 결과를 생성하지 못했습니다."],
            )

        logger.info("추천 프로세스 완료: 질의=%r 최종 추천=%d명 (근거 부족 제외=%d명) 소요시간(Judge)=%.2fms", 
                    query, len(recommendations), evidence_less_count, t_judge.elapsed_ms)

        result = self._build_recommendation_response(
            plan=plan,
            candidate_cards=candidate_cards,
            query_payload=search_result["query_payload"],
            branch_queries=search_result["branch_queries"],
            retrieved_count=search_result["retrieved_count"],
            recommendations=recommendations,
            data_gaps=judge_output.data_gaps,
            not_selected_reasons=not_selected_reasons,
        )
        
        timers = search_result.get("timers", {})
        t_plan_sec = float(timers.get("plan_ms", 0.0)) / 1000.0
        t_search_sec = float(timers.get("search_ms", 0.0)) / 1000.0
        t_judge_sec = t_judge.elapsed_ms / 1000.0
        
        t_total.stop()
        t_total_sec = t_total.elapsed_ms / 1000.0
        
        logger.info(
            "전체 추천 프로세스 종료: 총 소요시간=%.2f초 [세부 구간: 의도분석=%.2f초 | 하이브리드 검색=%.2f초 | LLM 심사=%.2f초]", 
            t_total_sec, t_plan_sec, t_search_sec, t_judge_sec
        )
        return result

    def save_feedback(
        self,
        *,
        query: str,
        selected_expert_ids: list[str],
        rejected_expert_ids: list[str],
        notes: str | None,
        metadata: dict[str, Any],
    ) -> int:
        logger.info("사용자 피드백 저장: 질의=%r 선택된 전문가=%d명 거절된 전문가=%d명", 
                    query, len(selected_expert_ids), len(rejected_expert_ids))
        return self.feedback_store.save_feedback(
            query=query,
            selected_expert_ids=selected_expert_ids,
            rejected_expert_ids=rejected_expert_ids,
            notes=notes,
            metadata=metadata,
        )

    @staticmethod
    def _build_recommendation_response(
        *,
        plan: PlannerOutput,
        candidate_cards: list[CandidateCard],
        query_payload: dict[str, Any],
        branch_queries: dict[str, str],
        retrieved_count: int,
        recommendations: list[RecommendationDecision],
        data_gaps: list[str],
        not_selected_reasons: list[str],
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
                "branch_queries": branch_queries,
                "exclude_orgs": plan.exclude_orgs,
                "candidate_ids": [card.expert_id for card in candidate_cards],
                "recommendation_evidence_summary": [
                    {
                        "expert_id": item.expert_id,
                        "evidence_titles": [evidence.title for evidence in item.evidence],
                    }
                    for item in recommendations
                ],
                "query_payload": RecommendationService._serialize_query_payload(query_payload),
            },
        }

    @staticmethod
    def _serialize_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
        def _mask_vectors(data: Any) -> Any:
            if hasattr(data, "model_dump"):
                try: data = data.model_dump()
                except Exception: pass
            elif hasattr(data, "dict") and callable(data.dict):
                try: data = data.dict()
                except Exception: pass

            if isinstance(data, dict):
                return {k: _mask_vectors(v) for k, v in data.items()}
            elif isinstance(data, list):
                if len(data) > 100 and all(isinstance(x, (float, int)) for x in data[:10]):
                    return f"<Dense Vector: {len(data)} dimensions>"
                return [_mask_vectors(x) for x in data]
            # Qdrant 내부 객체(Enum 등)의 직렬화를 위해 문자열로 변환 (dict, list가 아닌 기본형들)
            return str(data) if not isinstance(data, (int, float, bool, type(None))) else data

        serialized = dict(payload)
        serialized["prefetch"] = [_mask_vectors(item) for item in payload.get("prefetch", [])]
        query_filter = payload.get("query_filter")
        serialized["query_filter"] = str(query_filter) if query_filter else None
        serialized["query"] = str(payload.get("query"))
        return serialized

    # _merge_unique_strings 는 apps.core.utils 로 이전됨 (import 위 참조)
