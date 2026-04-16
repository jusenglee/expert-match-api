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


class QueryTextBuilder:
    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, CompiledBranchQueries]:
        """
        플랜 정보를 바탕으로 각 브랜치(basic, art, pat, pjt)의 Stable 및 Expanded 쿼리를 생성합니다.
        """
        # Stable: Planner가 추출한 핵심 기술어 (모든 브랜치 공통 기준축)
        stable_base = " ".join(plan.retrieval_core) if plan.retrieval_core else query
        
        results = {}
        branches = ["basic", "art", "pat", "pjt"]
        
        for branch in branches:
            # 1. Stable Path 구축
            stable_text = self._compose_stable(stable_base, branch)
            
            # 2. Expanded Path 구축 (Shadow Mode용)
            expanded_keywords = get_expanded_keywords(plan.bundle_ids, branch)
            expanded_base = stable_base
            if expanded_keywords:
                expanded_base = stable_base + " " + " ".join(expanded_keywords)
            
            expanded_text = self._compose_expanded(expanded_base, branch)
            
            results[branch] = CompiledBranchQueries(
                stable=stable_text,
                expanded=expanded_text
            )
            
        return results

    def build_query_text(self, plan: PlannerOutput) -> str:
        """전체 검색 맥락을 대표하는 텍스트를 생성합니다. (UI/Trace용)"""
        if plan.retrieval_core:
            return " ".join(plan.retrieval_core)
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

    def _compose_stable(self, base_text: str, branch: str) -> str:
        hints = {
            "basic": "전공 학위 소속유형 직위 기술분류 전문가 프로필",
            "art": "논문명 키워드 초록 학술지 최근 연구실적",
            "pat": "특허 발명명 출원 등록 사업화 지식재산",
            "pjt": "과제명 연구목표 연구내용 전문기관 연구수행 경험",
        }
        return f"{base_text.strip()}\n{hints.get(branch, '')}"

    def _compose_expanded(self, base_text: str, branch: str) -> str:
        # 현재는 stable과 같은 hint를 사용하되, base_text에 번들 확장어가 포함됨
        return self._compose_stable(base_text, branch)
