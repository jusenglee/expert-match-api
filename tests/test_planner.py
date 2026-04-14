import asyncio

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.recommendation.planner import OpenAICompatPlanner


class FakePlannerModel:
    def __init__(self, content):
        if isinstance(content, list):
            self.contents = list(content)
        else:
            self.contents = [content]
        self.call_count = 0
        self.last_kwargs = None

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        index = min(self.call_count, len(self.contents) - 1)
        self.call_count += 1
        return AIMessage(content=self.contents[index])


def test_openai_compat_planner_parses_code_fenced_json_and_keeps_core_keywords():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    fake_model = FakePlannerModel(
        """```json
        {
          "intent_summary": "AI semiconductor search",
          "core_keywords": ["AI semiconductor", "chip design"],
          "hard_filters": {},
          "exclude_orgs": [],
          "soft_preferences": [],
          "top_k": 5
        }
        ```"""
    )
    planner.model = fake_model

    result = asyncio.run(planner.plan(query="AI semiconductor experts"))

    assert result.intent_summary == "AI semiconductor search"
    assert result.core_keywords == ["AI semiconductor", "chip design"]
    assert result.branch_query_hints == {}
    assert planner.last_trace["planner_retry_count"] == 0
    assert planner.last_trace["retrieval_keywords"] == ["AI semiconductor", "chip design"]
    assert fake_model.last_kwargs["temperature"] == 0.0
    assert fake_model.last_kwargs["top_p"] == 0.2
    assert fake_model.last_kwargs["reasoning_effort"] == "low"
    assert fake_model.last_kwargs["include_reasoning"] is False
    assert fake_model.last_kwargs["disable_thinking"] is True


def test_openai_compat_planner_retries_when_core_keywords_are_empty():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "First attempt",
              "core_keywords": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Second attempt",
              "core_keywords": ["난접근성 화재 진압", "드론"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(
        planner.plan(query="난접근성 화재 진압에서 드론을 접목하려고해")
    )

    assert result.intent_summary == "Second attempt"
    assert result.core_keywords == ["난접근성 화재 진압", "드론"]
    assert planner.model.call_count == 2
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.last_trace["attempts"][0]["status"] == "empty_keywords"
    assert planner.last_trace["attempts"][1]["status"] == "ok"


def test_openai_compat_planner_falls_back_after_retry_exhaustion():
    query = "화재 평가위원 추천"
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "Attempt one",
              "core_keywords": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Attempt two",
              "core_keywords": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query=query))

    assert result.intent_summary == query
    assert result.core_keywords == []
    assert result.branch_query_hints == {}
    assert planner.last_trace["mode"] == "deterministic_fallback"
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.model.call_count == 2
