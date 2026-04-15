from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder


def test_query_builder_uses_core_keywords_only_for_branch_queries():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="Recommend fire reviewers",
        core_keywords=["fire suppression", "drone"],
        task_terms=["reviewer recommendation"],
    )

    query_text = builder.build_query_text(plan)
    branch_queries = builder.build_branch_queries(
        query="Recommend fire reviewers with drone expertise",
        plan=plan,
    )

    assert query_text == "fire suppression\ndrone"
    assert branch_queries == {
        "basic": "fire suppression\ndrone",
        "art": "fire suppression\ndrone",
        "pat": "fire suppression\ndrone",
        "pjt": "fire suppression\ndrone",
    }


def test_query_builder_normalizes_and_deduplicates_core_keywords():
    assert QueryTextBuilder.normalize_keywords(
        ["  fire suppression  ", "", "drone", "fire suppression"]
    ) == ["fire suppression", "drone"]
