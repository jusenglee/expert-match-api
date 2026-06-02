"""사용자 질의 + 의도(PlannerOutput)를 검색 텍스트 쿼리로 조립하는 모듈.

flat 단일 벡터 모델(v2.1): doc_type별 named vector가 없으므로 브랜치별 쿼리 fan-out도 없다.
단일 stable/expanded 쿼리 쌍은 trace 및 retriever 경로 호환을 위해 유지하되, v1 계열 lexicon
확장은 제거한다. 현재 expanded는 stable과 동일하다.
임베딩 텍스트 정제 원칙 유지: role/action 등 요청 어투는 넣지 않는다(planner가 분리).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from apps.domain.models import PlannerOutput

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompiledQueries:
    """정제된 쿼리 쌍(stable/expanded)."""

    stable: str
    expanded: str

    def startswith(self, prefix: str) -> bool:
        return self.stable.startswith(prefix)

    def __contains__(self, value: str) -> bool:
        return value in self.stable

    def __str__(self) -> str:
        return self.stable


class QueryTextBuilder:
    def build_queries(self, query: str, plan: PlannerOutput) -> CompiledQueries:
        """하이브리드 단계용 stable/expanded 쿼리. semantic_query 우선, 없으면 키워드."""
        stable_base = plan.semantic_query.strip() or self.build_keyword_query_text(query, plan)
        return self._build_queries_from_base(stable_base, plan)

    def build_keyword_queries(self, query: str, plan: PlannerOutput) -> CompiledQueries:
        """1차 sparse 키워드 풀링용 쿼리. 핵심 명사 위주."""
        base_text = self.build_keyword_query_text(query, plan)
        return self._build_queries_from_base(base_text, plan)

    def build_keyword_query_text(self, query: str, plan: PlannerOutput) -> str:
        keywords = plan.retrieval_core or plan.core_keywords
        if keywords:
            return " ".join(keywords)
        return query.strip()

    def _build_queries_from_base(self, stable_base: str, plan: PlannerOutput) -> CompiledQueries:
        _ = plan
        stable_text = stable_base.strip()
        return CompiledQueries(stable=stable_text, expanded=stable_text)

    def build_query_text(self, plan: PlannerOutput) -> str:
        """전체 검색 맥락 대표 텍스트(UI/Trace용)."""
        keywords = plan.retrieval_core or plan.core_keywords
        parts = []
        if keywords:
            parts.append(" ".join(keywords))
        if plan.semantic_query:
            parts.append(plan.semantic_query)
        if parts:
            return " ".join(parts)
        return plan.intent_summary

    @staticmethod
    def normalize_keywords(keywords: list[str]) -> list[str]:
        """키워드 리스트를 정규화하고 중복을 제거합니다."""
        normalized: list[str] = []
        for kw in keywords:
            if not kw:
                continue
            for part in kw.split():
                clean = part.strip()
                if clean and clean not in normalized:
                    normalized.append(clean)
        return normalized
