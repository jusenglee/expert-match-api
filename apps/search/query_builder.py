from __future__ import annotations

from apps.domain.models import PlannerOutput


class QueryTextBuilder:
    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, str]:
        hints = plan.branch_query_hints
        return {
            "basic": self._compose(query, "전공 학위 소속유형 직위 기술분류 전문가 프로필", hints.get("basic")),
            "art": self._compose(query, "논문명 키워드 초록 학술지 최근 연구실적", hints.get("art")),
            "pat": self._compose(query, "특허 발명명 출원 등록 사업화 지식재산", hints.get("pat")),
            "pjt": self._compose(query, "과제명 연구목표 연구내용 전문기관 연구수행 경험", hints.get("pjt")),
        }

    @staticmethod
    def _compose(query: str, default_hint: str, extra_hint: str | None) -> str:
        parts = [query.strip(), default_hint]
        if extra_hint:
            parts.append(extra_hint.strip())
        return "\n".join(part for part in parts if part)

