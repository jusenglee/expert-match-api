from __future__ import annotations

import logging
from dataclasses import dataclass

from apps.domain.models import PlannerOutput
from apps.search.expansion_lexicon import get_expanded_keywords

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompiledBranchQueries:
    """
    특정 브랜치(논문, 특허, 과제 등)에 대해 최종 조립된 쿼리 문자열들을 담는 데이터 구조입니다.
    안정적 쿼리(stable)와 확장 키워드가 포함된 쿼리(expanded), 그리고 각각의 Dense/Sparse 엔진용 텍스트를 포괄합니다.
    """

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
    """
    LLM 플래너가 추출한 검색 조건(PlannerOutput)과 원본 사용자 쿼리를 조합하여, 
    Qdrant 검색 엔진(Retriever)에 던질 최적의 쿼리 텍스트(Dense/Sparse 용)를 동적으로 생성하는 빌더 클래스입니다.
    """
    # Branch hints are now handled as instruct prefixes in the retriever.
    # They are kept here only for reference / future use.
    BRANCH_HINTS: dict[str, str] = {}

    @staticmethod
    def _planner_keywords(plan: PlannerOutput) -> list[str]:
        return list(plan.retrieval_core or plan.core_keywords)

    @classmethod
    def _stable_sparse_base(cls, *, query: str, plan: PlannerOutput) -> tuple[str, str]:
        """
        Sparse(BM25) 검색에 최적화된 기본 쿼리 텍스트를 생성합니다.
        키워드 단어들의 단순 나열이 TF-IDF 계열 검색(BM25)에 유리하므로, 플래너가 추출한 핵심 키워드(retrieval_core)를 최우선으로 조합합니다.
        """
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
        """
        Dense(의미 기반) 검색에 최적화된 기본 쿼리 텍스트를 생성합니다.
        E5-Instruct와 같은 Dense 임베딩 모델은 자연어 문장 형태일 때 가장 품질이 좋으므로,
        플래너가 생성한 가상 문서(HyDE)나 자연어 쿼리(semantic_query)를 최우선으로 사용합니다.
        우선순위: bounded_hyde_document(가상 문서) > semantic_query(자연어 문장) > retrieval_core 키워드 나열 > raw_query(원본 쿼리)
        """
        bounded_hyde_document = " ".join((getattr(plan, "bounded_hyde_document", "") or "").split())
        if bounded_hyde_document:
            return bounded_hyde_document, "bounded_hyde_document"

        semantic_query = " ".join((plan.semantic_query or "").split())
        if semantic_query:
            return semantic_query, "semantic_query"
            
        if stable_sparse_base:
            return stable_sparse_base, "retrieval_core"
            
        normalized_query = " ".join(query.split()) or plan.intent_summary
        return normalized_query, "raw_query"

    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, CompiledBranchQueries]:
        """
        기본 쿼리, 논문(art), 특허(pat), 과제(pjt) 브랜치별로 수행할 실제 Qdrant 검색 쿼리 세트를 조립합니다.
        필요한 경우 해당 도메인에 맞는 확장 키워드(expanded_keywords)를 Sparse 쿼리에 덧붙입니다.
        """
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
