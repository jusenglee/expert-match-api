import pytest

from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder

# WO-0: 514403e "검색 로직 변경 - 설계 변경"으로 superseded된 구설계(branch query) 검증 stale 테스트.
# WO-C에서 재작성/제거 예정.
_WOC_STALE = pytest.mark.xfail(
    reason="WO-C 이연: 514403e 설계 변경으로 superseded된 구설계 검증(stale). WO-C에서 재작성/제거.",
    strict=False,
)


@_WOC_STALE
def test_query_builder_uses_core_keywords_only_for_branch_queries():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="화재 진압 전문가 추천",
        core_keywords=["화재진압", "드론"],
        task_terms=["평가위원 추천"],
        semantic_query="화재 진압 현장 드론 활용 연구 전문가"
    )

    query_text = builder.build_query_text(plan)
    # build_branch_queries는 기본적으로 dense(semantic_query)를 사용함
    branch_queries = builder.build_branch_queries(
        query="드론을 활용한 화재 진압 전문가를 추천해줘",
        plan=plan,
    )

    assert "화재진압" in query_text
    assert "드론" in query_text
    assert "화재 진압" in query_text
    
    assert branch_queries["basic"].startswith("화재 진압 현장 드론 활용 연구 전문가")
    assert "전공 학위" in branch_queries["basic"]


def test_query_builder_normalizes_and_deduplicates_core_keywords():
    # 중복 제거 및 공백 처리 검증 (Kiwi 사용 안 함)
    keywords = QueryTextBuilder.normalize_keywords(
        ["  화재진압  ", "", "드론", "화재"]
    )
    assert "화재진압" in keywords
    assert "드론" in keywords
    assert "화재" in keywords
    # 중복 제거 확인
    assert len(keywords) == 3


def test_query_builder_does_not_expand_bundle_ids():
    builder = QueryTextBuilder()
    plan = PlannerOutput(
        intent_summary="드론 화재 진압 전문가",
        retrieval_core=["드론", "화재 진압"],
        core_keywords=["드론", "화재 진압"],
        semantic_query="드론 화재 진압 기술 전문가",
        bundle_ids=["legacy_uav"],
    )

    queries = builder.build_queries(
        query="드론 화재 진압 전문가 추천",
        plan=plan,
    )
    keyword_queries = builder.build_keyword_queries(
        query="드론 화재 진압 전문가 추천",
        plan=plan,
    )

    assert queries.stable == "드론 화재 진압 기술 전문가"
    assert queries.expanded == queries.stable
    assert keyword_queries.stable == "드론 화재 진압"
    assert keyword_queries.expanded == keyword_queries.stable
