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
          "core_keywords": ["난접근성 화재 진압", "드론", "드론"],
          "task_terms": ["전문가 추천"],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query="난접근성 화재 진압에서 드론 전문가 추천"))

    assert result.intent_summary == "난접근성 화재 진압과 드론 접목 전문가 탐색"
    assert result.core_keywords == ["난접근성 화재 진압", "드론"]
    assert result.task_terms == ["전문가 추천"]
    assert planner.last_trace["planner_keywords"] == ["난접근성 화재 진압", "드론"]
    assert planner.last_trace["retrieval_keywords"] == ["난접근성 화재 진압", "드론"]
    assert planner.last_trace["planner_retry_count"] == 0
    assert planner.model.call_count == 1
    assert planner.model.last_kwargs["temperature"] == 0.0
    assert planner.model.last_kwargs["top_p"] == 0.2


def test_openai_compat_planner_retries_when_keywords_are_empty():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "first",
              "core_keywords": [],
              "task_terms": ["평가위원 추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "second",
              "core_keywords": ["NTIS", "국가과학기술지식정보서비스", "운영"],
              "task_terms": ["제안 평가위원 추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="NTIS 운영 사업 제안 평가위원 추천"))

    assert result.intent_summary == "second"
    assert result.core_keywords == ["NTIS", "국가과학기술지식정보서비스", "운영"]
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
              "core_keywords": [],
              "task_terms": ["추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "attempt two",
              "core_keywords": [],
              "task_terms": ["추천"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="추천해줘"))

    assert result.intent_summary == "추천해줘"
    assert result.core_keywords == []
    assert result.task_terms == []
    assert planner.last_trace["mode"] == "deterministic_fallback"
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.model.call_count == 2
