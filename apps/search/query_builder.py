from __future__ import annotations

from apps.domain.models import PlannerOutput
from apps.search.schema_registry import BRANCHES


class QueryTextBuilder:
    @staticmethod
    def normalize_keywords(values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        for value in values:
            normalized = " ".join(str(value).split())
            if normalized and normalized not in normalized_values:
                normalized_values.append(normalized)
        return normalized_values

    def build_query_text(self, plan: PlannerOutput) -> str:
        return "\n".join(self.normalize_keywords(plan.core_keywords))

    def build_branch_queries(self, query: str, plan: PlannerOutput) -> dict[str, str]:
        _ = query
        query_text = self.build_query_text(plan)
        return {branch: query_text for branch in BRANCHES}
