"""사용자 질의 + 의도(PlannerOutput)를 검색 텍스트 쿼리로 조립하는 모듈.

flat 단일 벡터 모델(v2.1): doc_type별 named vector가 없으므로 브랜치별 쿼리 fan-out은 없다.
다만 dense와 sparse는 서로 다른 실패 모드를 갖기 때문에 검색 채널별 쿼리를 분리한다.

concept(개념) 감지·검색문·alias는 더 이상 이 모듈에 하드코딩하지 않는다 — 질의별 동적 Concept Evidence
Plan(relevance.ConceptPlan)을 입력으로 받아 concept 관련 필드를 파생시킨다(planner 산출 우선, registry 안전망).
이 모듈은 채널 텍스트(raw/dense/sparse_focus) 조립만 담당한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from apps.domain.models import PlannerOutput

if TYPE_CHECKING:
    from apps.search.relevance import ConceptPlan

logger = logging.getLogger(__name__)


GENERIC_SEARCH_TERMS: tuple[str, ...] = (
    "전문",
    "전문성",
    "분야",
    "연구자",
    "가진",
    "보유",
    "또는",
    "상세",
    "관련",
    "다양한",
    "경력",
    "경험",
    "연구",
    "개발",
    "산업",
    "expert",
    "expertise",
    "researcher",
    "related",
)

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


@dataclass(slots=True)
class SearchQueryPlan:
    """검색 채널별 쿼리 정의.

    dense는 planner semantic_query를 우선 사용하고, sparse는 SPLADE 과확장을 줄이기 위해
    짧은 자연문/명사구와 concept별 보조 문장으로 분리한다.
    """

    raw_query: str
    dense_query: str
    sparse_joint_query: str
    sparse_concept_queries: dict[str, str]
    required_concepts: list[str]
    optional_concepts: list[str]
    drop_terms_for_sparse: list[str]

    def sparse_queries(self) -> dict[str, str]:
        queries = {"sparse_joint": self.sparse_joint_query}
        queries.update(
            {
                f"sparse_{name}": query
                for name, query in self.sparse_concept_queries.items()
                if query.strip()
            }
        )
        return {name: query for name, query in queries.items() if query.strip()}


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

    def build_search_query_plan(
        self,
        query: str,
        plan: PlannerOutput,
        concept_plan: "ConceptPlan | None" = None,
    ) -> SearchQueryPlan:
        """채널 쿼리 plan 조립. concept 필드는 concept_plan(동적)에서 파생, 없으면 비운다.

        sparse_focus(=sparse_joint_query)는 concept label 중심 짧은 명사구를 우선 쓰고(과확장 억제),
        concept이 없으면 핵심 키워드로 폴백한다. concept 감지/검색문은 이 모듈에 하드코딩하지 않는다.
        """
        raw_query = " ".join(query.strip().split())
        dense_query = (
            " ".join((plan.semantic_query or "").split())
            or raw_query
            or " ".join(self.build_query_text(plan).split())
        )

        if concept_plan is not None:
            required_concepts = list(concept_plan.required)
            optional_concepts = list(concept_plan.optional)
            sparse_concept_queries = dict(concept_plan.concept_queries)
            focus = concept_plan.focus_query
        else:
            required_concepts, optional_concepts, sparse_concept_queries, focus = [], [], {}, ""

        sparse_joint_query = focus or self._keyword_focus_query(query, plan)
        return SearchQueryPlan(
            raw_query=raw_query,
            dense_query=dense_query,
            sparse_joint_query=sparse_joint_query,
            sparse_concept_queries=sparse_concept_queries,
            required_concepts=required_concepts,
            optional_concepts=optional_concepts,
            drop_terms_for_sparse=list(GENERIC_SEARCH_TERMS),
        )

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

    def _keyword_focus_query(self, query: str, plan: PlannerOutput) -> str:
        """concept이 없을 때 sparse_focus 폴백: 핵심 키워드(일반어 제거) 짧은 명사구."""
        parts: list[str] = []
        for keyword in plan.retrieval_core or plan.core_keywords:
            cleaned = self._clean_sparse_keyword(keyword)
            if cleaned:
                parts.append(cleaned)
        if not parts:
            for token in query.split():
                cleaned = self._clean_sparse_keyword(token)
                if cleaned:
                    parts.append(cleaned)
        return " ".join(self._dedupe(parts)).strip() or query.strip()

    @classmethod
    def _clean_sparse_keyword(cls, keyword: str) -> str:
        text = " ".join(str(keyword).strip().split())
        if not text:
            return ""
        lowered = text.casefold()
        if lowered in {term.casefold() for term in GENERIC_SEARCH_TERMS}:
            return ""
        parts = [part for part in text.split() if part.casefold() not in {term.casefold() for term in GENERIC_SEARCH_TERMS}]
        if len(parts) == 1 and parts[0] != text:
            return parts[0]
        if parts:
            return " ".join(parts)
        return text if len(text) > 1 else ""

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        output: list[str] = []
        for value in values:
            key = value.casefold()
            if key in seen:
                continue
            seen.add(key)
            output.append(value)
        return output

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
