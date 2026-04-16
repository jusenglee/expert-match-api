from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder


def test_query_builder_uses_core_keywords_only_for_branch_queries():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="화재 진압 전문가 추천",
        core_keywords=["화재진압", "드론"],
        task_terms=["평가위원 추천"],
    )

    query_text = builder.build_query_text(plan)
    # build_branch_queries는 기본적으로 dense(semantic_query)를 사용함
    branch_queries = builder.build_branch_queries(
        query="드론을 활용한 화재 진압 전문가를 추천해줘",
        plan=plan,
    )

    # Kiwi가 "화재진압"을 "화재", "진압"으로 분리함
    assert "화재" in query_text
    assert "진압" in query_text
    assert "드론" in query_text
    
    assert branch_queries == {
        "basic": "드론을 활용한 화재 진압 전문가를 추천해줘",
        "art": "드론을 활용한 화재 진압 전문가를 추천해줘",
        "pat": "드론을 활용한 화재 진압 전문가를 추천해줘",
        "pjt": "드론을 활용한 화재 진압 전문가를 추천해줘",
    }


def test_query_builder_normalizes_and_deduplicates_core_keywords():
    # Kiwi를 통한 명사 추출 및 중복 제거 검증
    keywords = QueryTextBuilder.normalize_keywords(
        ["  화재진압  ", "", "드론", "화재"]
    )
    assert "화재" in keywords
    assert "진압" in keywords
    assert "드론" in keywords
    # "화재"가 중복되어 들어가지 않았는지 확인
    assert keywords.count("화재") == 1
