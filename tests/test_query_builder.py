from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder


def test_query_builder_uses_semantic_query_for_dense_only():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="Recommend experts for drone-assisted fire suppression",
        core_keywords=["fire suppression", "drone"],
        retrieval_core=["fire suppression", "drone"],
        semantic_query="drone-assisted fire suppression expert",
    )

    query_text = builder.build_query_text(plan)
    branch_queries = builder.build_branch_queries(
        query="Recommend experts for drone-assisted fire suppression",
        plan=plan,
    )

    assert query_text == "fire suppression drone"
    assert branch_queries["basic"].stable.startswith("fire suppression drone")
    assert branch_queries["basic"].stable_sparse.startswith("fire suppression drone")
    assert branch_queries["basic"].stable_dense.startswith(
        "drone-assisted fire suppression expert"
    )
    assert branch_queries["basic"].dense_base_source == "semantic_query"
    assert branch_queries["basic"].sparse_base_source == "retrieval_core"


def test_query_builder_prefers_retrieval_core_over_legacy_core_keywords():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="Recommend AI semiconductor experts",
        core_keywords=["expert recommendation", "AI semiconductor"],
        retrieval_core=["AI semiconductor", "chip reliability"],
        must_aspects=["AI semiconductor"],
    )

    query_text = builder.build_query_text(plan)
    branch_queries = builder.build_branch_queries(
        query="Recommend AI semiconductor experts",
        plan=plan,
    )

    assert query_text == "AI semiconductor chip reliability"
    assert branch_queries["basic"].stable.startswith("AI semiconductor chip reliability")
    assert branch_queries["basic"].stable_dense.startswith("AI semiconductor chip reliability")
    assert "expert recommendation" not in branch_queries["basic"].stable


def test_query_builder_normalizes_and_deduplicates_core_keywords():
    keywords = QueryTextBuilder.normalize_keywords(
        ["  fire suppression  ", "", "drone", "fire", "drone"]
    )

    assert keywords == ["fire", "suppression", "drone"]
