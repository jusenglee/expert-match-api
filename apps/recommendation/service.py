from __future__ import annotations

import asyncio
import logging
from typing import Any

from apps.core.feedback_store import FeedbackStore
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
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.filter_compiler = filter_compiler
        self.card_builder = card_builder
        self.judge = judge
        self.feedback_store = feedback_store
        self.shortlist_limit = shortlist_limit
        self.use_map_reduce_judging = use_map_reduce_judging

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
        
        query_filter = self.filter_compiler.compile(plan.hard_filters, plan.exclude_orgs)
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
        with Timer() as t_total:
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

        # LLM Context Limit 대응: Map-Reduce 병렬 심사
        chunk_size = 10
        with Timer() as t_judge:
            if not self.use_map_reduce_judging or len(shortlist) <= chunk_size:
                logger.info("최종 판정(Judging) 단계 시작: 단일 심사 (후보 수=%d, Map-Reduce=%s)", len(shortlist), self.use_map_reduce_judging)
                judge_output: JudgeOutput = await self.judge.judge(query=query, plan=plan, shortlist=shortlist)
            else:
                logger.info("최종 판정(Judging) 단계 시작: Map-Reduce 병렬 심사 (총 후보 수=%d, 청크 크기=%d명)", len(shortlist), chunk_size)
                chunks = [shortlist[i:i + chunk_size] for i in range(0, len(shortlist), chunk_size)]
                
                # 1. Map Phase
                map_tasks = [self.judge.judge(query=query, plan=plan, shortlist=chunk) for chunk in chunks]
                map_outputs: list[JudgeOutput] = await asyncio.gather(*map_tasks, return_exceptions=False)
                
                # Map 결과 풀링
                winner_ids = set()
                all_data_gaps = []
                all_not_selected_reasons = []
                for out in map_outputs:
                    for rec in out.recommended:
                        winner_ids.add(rec.expert_id)
                    all_data_gaps.extend(out.data_gaps)
                    all_not_selected_reasons.extend(out.not_selected_reasons)
                
                reduce_shortlist = [card for card in shortlist if card.expert_id in winner_ids]
                logger.info("Map 단계 완료: %d개 청크에서 총 %d명의 예비 후보 선정됨. Reduce 실행", len(chunks), len(reduce_shortlist))
                
                # 2. Reduce Phase
                if not reduce_shortlist:
                    judge_output = JudgeOutput(
                        recommended=[],
                        not_selected_reasons=self._merge_unique_strings(all_not_selected_reasons),
                        data_gaps=self._merge_unique_strings(all_data_gaps)
                    )
                else:
                    final_output = await self.judge.judge(query=query, plan=plan, shortlist=reduce_shortlist)
                    judge_output = JudgeOutput(
                        recommended=final_output.recommended,
                        not_selected_reasons=self._merge_unique_strings(all_not_selected_reasons, final_output.not_selected_reasons),
                        data_gaps=self._merge_unique_strings(all_data_gaps, final_output.data_gaps)
                    )
        
        evidence_less_count = sum(1 for item in judge_output.recommended if not item.evidence)
        recommendations = [item for item in judge_output.recommended if item.evidence]
        not_selected_reasons = self._merge_unique_strings(judge_output.not_selected_reasons)

        if not judge_output.recommended:
            not_selected_reasons = self._merge_unique_strings(
                not_selected_reasons,
                ["조건을 만족하는 추천 후보를 찾지 못했습니다."],
            )
        elif evidence_less_count:
            not_selected_reasons = self._merge_unique_strings(
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
        
        t_plan_sec = search_result["timers"]["plan_ms"] / 1000.0
        t_search_sec = search_result["timers"]["search_ms"] / 1000.0
        t_judge_sec = t_judge.elapsed_ms / 1000.0
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

    @staticmethod
    def _merge_unique_strings(*groups: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            for item in group:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(normalized)
        return merged
