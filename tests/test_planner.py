import asyncio

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.recommendation.planner import OpenAICompatPlanner


class FakePlannerModel:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        return AIMessage(content=self.content)


def test_openai_compat_planner_parses_code_fenced_json():
    planner = OpenAICompatPlanner(Settings(app_env="test", strict_runtime_validation=False))
    fake_model = FakePlannerModel(
        """```json
        {
          "intent_summary": "AI semiconductor experts",
          "hard_filters": {},
          "exclude_orgs": [],
          "soft_preferences": [],
          "branch_query_hints": {
            "basic": "profile",
            "art": "papers",
            "pat": "patents",
            "pjt": "projects"
          },
          "top_k": 5
        }
        ```"""
    )
    planner.model = fake_model

    result = asyncio.run(planner.plan(query="AI semiconductor experts"))

    assert result.intent_summary == "AI semiconductor experts"
    assert result.branch_query_hints["art"] == "papers"
    assert fake_model.last_kwargs["temperature"] == 0.0
    assert fake_model.last_kwargs["top_p"] == 0.2
    assert fake_model.last_kwargs["reasoning_effort"] == "low"
    assert fake_model.last_kwargs["include_reasoning"] is False
    assert fake_model.last_kwargs["disable_thinking"] is True


def test_openai_compat_planner_falls_back_on_invalid_branch_query_hints_shape():
    query = "반도체 논문 전문가 추천"
    planner = OpenAICompatPlanner(Settings(app_env="test", strict_runtime_validation=False))
    planner.model = FakePlannerModel(
        """{
          "hard_filters": {},
          "exclude_orgs": [],
          "soft_preferences": [],
          "branch_query_hints": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query=query))

    assert result.intent_summary == query
    assert isinstance(result.branch_query_hints, dict)
    assert result.branch_query_hints["basic"]
