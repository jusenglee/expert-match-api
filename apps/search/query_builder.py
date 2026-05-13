"""
사용자의 자연어 질의와 LLM이 파악한 의도(PlannerOutput)를 바탕으로,
실제 검색 엔진(Qdrant)에 전달될 최종 텍스트 쿼리를 생성하고 조립하는 모듈입니다.

[Architecture Overview]
사용자의 질의는 단순히 하나의 문자열로 검색되지 않고, 검색 정확도를 높이기 위해 다음과 같이 분할/확장됩니다.
1. Stable Query: 원본 질의나 핵심 키워드로만 구성되어, 정확한(Exact) 매칭을 노리는 쿼리입니다.
2. Expanded Query: 핵심 키워드 외에 관련된 동의어(Synonyms)나 사전 정의된 확장 어휘(Bundle)를 덧붙여, 
                   유의어 누락을 방지(Recall 향상)하는 넓은 범위의 쿼리입니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from apps.domain.models import PlannerOutput
from apps.search.expansion_lexicon import get_expanded_keywords

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompiledBranchQueries:
    """브랜치별 정제된 쿼리 세트입니다."""
    stable: str
    expanded: str

    def startswith(self, prefix: str) -> bool:
        return self.stable.startswith(prefix)

    def __contains__(self, value: str) -> bool:
        return value in self.stable

    def __str__(self) -> str:
        return self.stable


class QueryTextBuilder:
    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, CompiledBranchQueries]:
        """
        플랜 정보를 바탕으로 각 브랜치(basic, art, pat, pjt)의 Stable 및 Expanded 쿼리를 생성합니다.
        """
        stable_base = plan.semantic_query.strip() or self.build_keyword_query_text(query, plan)
        return self._build_queries_from_base(stable_base, plan)

    def build_keyword_branch_queries(
        self, query: str, plan: PlannerOutput
    ) -> dict[str, CompiledBranchQueries]:
        """
        [1차 키워드 검색용 Sparse 전용 쿼리 생성]
        하이브리드 모드보다 앞서 초고속 풀링(Pooling)을 수행할 때 사용됩니다.
        복잡한 문장보다는 철저히 핵심 명사(Keyword) 위주로 구성되어야 검색 성능이 높습니다.
        """
        base_text = self.build_keyword_query_text(query, plan)
        return self._build_queries_from_base(base_text, plan)

    def build_keyword_query_text(self, query: str, plan: PlannerOutput) -> str:
        keywords = plan.retrieval_core or plan.core_keywords
        if keywords:
            return " ".join(keywords)
        return query.strip()

    def _build_queries_from_base(
        self,
        stable_base: str,
        plan: PlannerOutput,
    ) -> dict[str, CompiledBranchQueries]:
        results = {}
        branches = ["basic", "art", "pat", "pjt"]

        for branch in branches:
            # 1. Stable Path 구축 (힌트 단어 없이 순수 쿼리 사용)
            stable_text = stable_base.strip()

            # 2. Expanded Path 구축 (Shadow Mode용, 힌트 없이 번들 확장어만 추가)
            expanded_keywords = get_expanded_keywords(plan.bundle_ids, branch)
            expanded_text = stable_text
            if expanded_keywords:
                expanded_text = f"{stable_text} {' '.join(expanded_keywords)}"

            results[branch] = CompiledBranchQueries(
                stable=stable_text,
                expanded=expanded_text
            )

        return results

    def build_query_text(self, plan: PlannerOutput) -> str:
        """전체 검색 맥락을 대표하는 텍스트를 생성합니다. (UI/Trace용)"""
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
