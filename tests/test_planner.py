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


def test_openai_compat_planner_runs_verifier_and_keeps_clean_keywords():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """```json
            {
              "intent_summary": "AI semiconductor search",
              "core_keywords": ["AI semiconductor", "chip design"],
              "task_terms": ["expert recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }
            ```""",
            """{
              "intent_summary": "AI semiconductor search",
              "core_keywords": ["AI semiconductor", "chip design"],
              "task_terms": ["expert recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="AI semiconductor experts"))

    assert result.intent_summary == "AI semiconductor search"
    assert result.core_keywords == ["AI semiconductor", "chip design"]
    assert result.task_terms == ["expert recommendation"]
    assert result.branch_query_hints == {}
    assert planner.last_trace["planner_retry_count"] == 0
    assert planner.last_trace["planner_raw_keywords"] == [
        "AI semiconductor",
        "chip design",
    ]
    assert planner.last_trace["verifier_keywords"] == [
        "AI semiconductor",
        "chip design",
    ]
    assert planner.last_trace["retrieval_keywords"] == [
        "AI semiconductor",
        "chip design",
    ]
    assert planner.last_trace["verifier_applied"] is True
    assert planner.model.call_count == 2
    assert planner.model.last_kwargs["temperature"] == 0.0
    assert planner.model.last_kwargs["top_p"] == 0.2
    assert planner.model.last_kwargs["reasoning_effort"] == "low"
    assert planner.model.last_kwargs["include_reasoning"] is False
    assert planner.model.last_kwargs["disable_thinking"] is True


def test_openai_compat_planner_verifier_removes_meta_terms_from_retrieval_keywords():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "NTIS proposal reviewer recommendation",
              "core_keywords": ["NTIS", "proposal", "reviewer"],
              "task_terms": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "NTIS proposal reviewer recommendation",
              "core_keywords": ["NTIS", "proposal"],
              "task_terms": ["reviewer recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="NTIS proposal reviewer recommendation"))

    assert result.core_keywords == ["NTIS", "proposal"]
    assert result.task_terms == ["reviewer recommendation"]
    assert planner.last_trace["planner_raw_keywords"] == [
        "NTIS",
        "proposal",
        "reviewer",
    ]
    assert planner.last_trace["verifier_keywords"] == ["NTIS", "proposal"]


def test_openai_compat_planner_retries_when_verifier_returns_empty_keywords():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "First attempt",
              "core_keywords": ["reviewer", "recommendation"],
              "task_terms": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "First attempt",
              "core_keywords": [],
              "task_terms": ["reviewer recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Second attempt",
              "core_keywords": ["fire suppression", "drone"],
              "task_terms": ["reviewer recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Second attempt",
              "core_keywords": ["fire suppression", "drone"],
              "task_terms": ["reviewer recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(
        planner.plan(query="Recommend fire-suppression reviewers with drone expertise")
    )

    assert result.intent_summary == "Second attempt"
    assert result.core_keywords == ["fire suppression", "drone"]
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.last_trace["attempts"][0]["status"] == "empty_keywords"
    assert planner.last_trace["attempts"][1]["status"] == "ok"
    assert planner.model.call_count == 4


def test_openai_compat_planner_falls_back_after_planner_verifier_retry_exhaustion():
    query = "Recommend fire reviewers"
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "Attempt one",
              "core_keywords": ["reviewer", "recommendation"],
              "task_terms": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Attempt one",
              "core_keywords": [],
              "task_terms": ["reviewer recommendation"],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Attempt two",
              "core_keywords": ["reviewer", "recommendation"],
              "task_terms": [],
              "hard_filters": {},
              "exclude_orgs": [],
              "soft_preferences": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "Attempt two",
              "core_keywords": [],
              "task_terms": ["reviewer recommendation"],
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
    assert result.task_terms == []
    assert result.branch_query_hints == {}
    assert planner.last_trace["mode"] == "deterministic_fallback"
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.last_trace["planner_raw_keywords"] == [
        "reviewer",
        "recommendation",
    ]
    assert planner.last_trace["verifier_applied"] is True
    assert planner.model.call_count == 4
