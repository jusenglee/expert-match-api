from __future__ import annotations

import logging
from dataclasses import dataclass

from apps.domain.models import PlannerOutput
from apps.search.expansion_lexicon import get_expanded_keywords

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompiledBranchQueries:
    """Compiled per-branch query texts for stable and expanded paths."""

    stable: str
    expanded: str
    stable_dense: str
    stable_sparse: str
    expanded_dense: str
    expanded_sparse: str
    dense_base_source: str
    sparse_base_source: str

    def expanded_differs(self) -> bool:
        return (
            " ".join(self.stable_dense.split()) != " ".join(self.expanded_dense.split())
            or " ".join(self.stable_sparse.split()) != " ".join(self.expanded_sparse.split())
        )


class QueryTextBuilder:
    # Branch hints are now handled as instruct prefixes in the retriever.
    # They are kept here only for reference / future use.
    BRANCH_HINTS: dict[str, str] = {}

    @staticmethod
    def _planner_keywords(plan: PlannerOutput) -> list[str]:
        return list(plan.retrieval_core or plan.core_keywords)

    @classmethod
    def _stable_sparse_base(cls, *, query: str, plan: PlannerOutput) -> tuple[str, str]:
        """Sparse(BM25) 쿼리 베이스: 키워드 조인이 TF-IDF 계열에 최적."""
        planner_keywords = cls._planner_keywords(plan)
        if planner_keywords:
            return " ".join(planner_keywords), "retrieval_core"
        normalized_query = " ".join(query.split()) or plan.intent_summary
        return normalized_query, "raw_query"

    @classmethod
    def _stable_dense_base(
        cls,
        *,
        query: str,
        plan: PlannerOutput,
        stable_sparse_base: str,
    ) -> tuple[str, str]:
        """Dense(e5-instruct) 쿼리 베이스: 자연어 문장이 임베딩 품질에 최적.

        우선순위: semantic_query(LLM 생성 자연어 문장) > retrieval_core 조인 > raw_query
        """
        semantic_query = " ".join((plan.semantic_query or "").split())
        if semantic_query:
            return semantic_query, "semantic_query"
        if stable_sparse_base:
            return stable_sparse_base, "retrieval_core"
        normalized_query = " ".join(query.split()) or plan.intent_summary
        return normalized_query, "raw_query"

    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, CompiledBranchQueries]:
        stable_sparse_base, sparse_base_source = self._stable_sparse_base(query=query, plan=plan)
        stable_dense_base, dense_base_source = self._stable_dense_base(
            query=query,
            plan=plan,
            stable_sparse_base=stable_sparse_base,
        )

        results: dict[str, CompiledBranchQueries] = {}
        for branch in ("basic", "art", "pat", "pjt"):
            # Branch-specific context는 retriever의 instruct prefix가 담당.
            # query_builder는 순수 텍스트만 조립한다.
            stable_sparse_text = stable_sparse_base
            stable_dense_text = stable_dense_base

            expanded_keywords = get_expanded_keywords(plan.bundle_ids, branch)
            expanded_sparse_base = stable_sparse_base
            expanded_dense_base = stable_dense_base
            if expanded_keywords:
                expansion_suffix = " ".join(expanded_keywords)
                expanded_sparse_base = f"{stable_sparse_base} {expansion_suffix}".strip()
                # semantic_query가 dense base인 경우 확장 키워드를 붙이지 않는다.
                # 자연어 문장에 무관한 기술 키워드를 suffix로 붙이면 임베딩 방향이 흐려짐.
                if dense_base_source != "semantic_query":
                    expanded_dense_base = expanded_sparse_base

            expanded_sparse_text = expanded_sparse_base
            expanded_dense_text = expanded_dense_base

            results[branch] = CompiledBranchQueries(
                stable=stable_sparse_text,
                expanded=expanded_sparse_text,
                stable_dense=stable_dense_text,
                stable_sparse=stable_sparse_text,
                expanded_dense=expanded_dense_text,
                expanded_sparse=expanded_sparse_text,
                dense_base_source=dense_base_source,
                sparse_base_source=sparse_base_source,
            )

        return results

    def build_query_text(self, plan: PlannerOutput) -> str:
        planner_keywords = self._planner_keywords(plan)
        if planner_keywords:
            return " ".join(planner_keywords)
        return plan.intent_summary

    @staticmethod
    def normalize_keywords(keywords: list[str]) -> list[str]:
        normalized: list[str] = []
        for keyword in keywords:
            if not keyword:
                continue
            for part in keyword.split():
                clean = part.strip()
                if clean and clean not in normalized:
                    normalized.append(clean)
        return normalized

    def _compose_query_text(self, base_text: str, branch: str) -> str:
        """Deprecated: 더 이상 branch hint를 텍스트에 삽입하지 않는다.

        branch 컨텍스트는 retriever의 instruct prefix가 담당.
        하위호환을 위해 메서드는 유지하되, base_text를 그대로 반환한다.
        """
        return base_text.strip()
