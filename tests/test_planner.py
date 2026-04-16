import asyncio

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.recommendation.planner import OpenAICompatPlanner


class FakePlannerModel:
    def __init__(self, content):
        self.contents = list(content) if isinstance(content, list) else [content]
        self.call_count = 0
        self.last_kwargs = None

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        index = min(self.call_count, len(self.contents) - 1)
        self.call_count += 1
        return AIMessage(content=self.contents[index])


def test_openai_compat_planner_extracts_pure_keywords_and_request_terms():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "난접근성 화재 진압과 드론 접목 전문가 탐색",
          "retrieval_core": ["난접근성 화재 진압", "드론", "드론"],
          "semantic_query": "난접근성 화재 현장 드론 전문가",
          "role_terms": ["전문가"],
          "action_terms": ["추천"],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query="난접근성 화재 진압에서 드론 전문가 추천"))

    assert result.intent_summary == "난접근성 화재 진압과 드론 접목 전문가 탐색"
    assert result.retrieval_core == ["난접근성 화재 진압", "드론"]
    assert result.core_keywords == ["난접근성 화재 진압", "드론"]
    assert result.role_terms == ["전문가"]
    assert result.action_terms == ["추천"]
    assert planner.last_trace["planner_keywords"] == ["난접근성 화재 진압", "드론"]
    assert planner.last_trace["retrieval_keywords"] == ["난접근성 화재 진압", "드론"]
    assert planner.last_trace["planner_retry_count"] == 0
    assert planner.model.call_count == 1


def test_openai_compat_planner_retries_when_keywords_are_empty():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "first",
              "retrieval_core": [],
              "role_terms": ["평가위원"],
              "action_terms": ["추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "second",
              "retrieval_core": ["NTIS", "국가과학기술지식정보서비스", "운영"],
              "role_terms": ["평가위원"],
              "action_terms": ["추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="NTIS 운영 사업 제안 평가위원 추천"))

    assert result.intent_summary == "second"
    assert result.retrieval_core == ["NTIS", "국가과학기술지식정보서비스", "운영"]
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.last_trace["attempts"][0]["status"] == "empty_keywords"
    assert planner.last_trace["attempts"][1]["status"] == "ok"
    assert planner.model.call_count == 2


def test_openai_compat_planner_falls_back_after_retry_exhaustion():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "attempt one",
              "retrieval_core": [],
              "role_terms": ["추천"],
              "action_terms": [],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "attempt two",
              "retrieval_core": [],
              "role_terms": ["추천"],
              "action_terms": [],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="추천해줘"))

    # 신규 Fallback 로직에서는 [Fallback] 접두사가 붙고, 역할어를 제거한 뒤 남은 단어를 키워드로 추출함
    assert "[Fallback]" in result.intent_summary
    assert result.role_terms == ["추천"]
    assert planner.last_trace["mode"] == "fallback_broad_search"
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.model.call_count == 2


def test_openai_compat_planner_selects_expansion_bundles():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "드론 화재 진압 전문가",
          "retrieval_core": ["드론", "화재 진압"],
          "bundle_ids": ["uav", "fire_response", "invalid_id"],
          "semantic_query": "드론 화재 진압 기술 전문가",
          "role_terms": [],
          "action_terms": [],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query="드론 화재 진압 전문가 추천"))

    # 유효한 번들 ID만 남아야 함
    assert "uav" in result.bundle_ids
    assert "fire_response" in result.bundle_ids
    assert "invalid_id" not in result.bundle_ids
    assert len(result.bundle_ids) == 2
