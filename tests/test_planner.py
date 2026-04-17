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


def test_openai_compat_planner_strips_meta_terms_and_derives_must_aspects():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "Recommend AI semiconductor experts",
          "retrieval_core": ["AI semiconductor", "expert recommendation"],
          "must_aspects": ["expert recommendation"],
          "generic_terms": ["experience"],
          "semantic_query": "AI semiconductor expert recommendation",
          "role_terms": ["expert"],
          "action_terms": ["recommend"],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(
        planner.plan(query="Recommend AI semiconductor expert recommendation")
    )

    assert result.retrieval_core == ["AI semiconductor"]
    assert result.must_aspects == ["AI semiconductor"]
    assert result.core_keywords == ["AI semiconductor"]
    assert "expert" in result.role_terms
    assert "recommend" in result.action_terms
    assert "experience" in result.generic_terms
    assert "expert" in planner.last_trace["removed_meta_terms"]
    assert planner.last_trace["pruned_must_aspects"] == ["AI semiconductor"]


def test_openai_compat_planner_retries_when_meta_stripping_leaves_no_keywords():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "first",
              "retrieval_core": ["reviewer recommendation"],
              "role_terms": ["reviewer"],
              "action_terms": ["recommend"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "second",
              "retrieval_core": ["NTIS platform", "evaluation analytics"],
              "role_terms": ["reviewer"],
              "action_terms": ["recommend"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(
        planner.plan(query="Recommend NTIS platform reviewer recommendation")
    )

    assert result.retrieval_core == ["NTIS platform", "evaluation analytics"]
    assert result.must_aspects == ["NTIS platform", "evaluation analytics"]
    assert planner.last_trace["planner_retry_count"] == 1
    assert planner.last_trace["attempts"][0]["status"] == "empty_keywords"
    assert planner.model.call_count == 2


def test_openai_compat_planner_fallback_keeps_broad_search_without_meta_terms():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        [
            """{
              "intent_summary": "attempt one",
              "retrieval_core": [],
              "role_terms": ["expert"],
              "action_terms": ["recommend"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
            """{
              "intent_summary": "attempt two",
              "retrieval_core": [],
              "role_terms": ["expert"],
              "action_terms": ["recommend"],
              "hard_filters": {},
              "include_orgs": [],
              "exclude_orgs": [],
              "top_k": 5
            }""",
        ]
    )

    result = asyncio.run(planner.plan(query="Recommend experts"))

    assert "[Fallback]" in result.intent_summary
    assert result.retrieval_core == []
    assert result.must_aspects == []
    assert planner.last_trace["mode"] == "fallback_broad_search"
    assert planner.last_trace["fallback_terms"] == []
    assert planner.model.call_count == 2


def test_openai_compat_planner_selects_only_valid_expansion_bundles():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "drone fire response",
          "retrieval_core": ["drone", "fire response"],
          "bundle_ids": ["uav", "fire_response", "invalid_id"],
          "semantic_query": "drone fire response expert",
          "role_terms": [],
          "action_terms": [],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query="Recommend drone fire response experts"))

    assert result.bundle_ids == ["uav", "fire_response"]


def test_openai_compat_planner_moves_contextual_evaluation_to_intent_flags():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "AI 기반 의료영상 과제 평가 전문가 추천",
          "retrieval_core": ["AI 기반 의료영상", "과제 평가", "전문가 추천"],
          "must_aspects": ["과제 평가", "AI 기반", "의료영상 분석"],
          "generic_terms": ["평가"],
          "semantic_query": "AI 기반 의료영상 과제 평가 전문가 추천",
          "role_terms": ["전문가"],
          "action_terms": ["추천"],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(
        planner.plan(query="AI 기반 의료영상 과제를 평가할 수 있는 전문가를 추천해 주세요")
    )

    assert result.retrieval_core == ["AI 기반 의료영상"]
    assert result.must_aspects == ["의료영상 분석"]
    assert result.intent_flags["review_context"] is True
    assert result.intent_flags["review_targets"] == ["과제 평가"]
    assert planner.last_trace["retained_contextual_terms"] == ["과제 평가"]
    assert planner.last_trace["pruned_must_aspects"] == ["의료영상 분석"]


def test_openai_compat_planner_prunes_generic_phrases_from_must_aspects():
    planner = OpenAICompatPlanner(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    planner.model = FakePlannerModel(
        """{
          "intent_summary": "AI 기반 의료영상 분석 기술 개발",
          "retrieval_core": ["의료영상 분석", "AI 기반", "기술"],
          "must_aspects": ["의료영상 분석", "AI 기반", "기술 개발"],
          "generic_terms": [],
          "semantic_query": "AI 기반 의료영상 분석 기술 개발",
          "role_terms": [],
          "action_terms": [],
          "hard_filters": {},
          "include_orgs": [],
          "exclude_orgs": [],
          "top_k": 5
        }"""
    )

    result = asyncio.run(planner.plan(query="AI 기반 의료영상 분석 기술 개발"))

    assert result.retrieval_core == ["의료영상 분석", "AI 기반", "기술"]
    assert result.must_aspects == ["의료영상 분석"]
    assert planner.last_trace["raw_must_aspects"] == ["의료영상 분석", "AI 기반", "기술 개발"]
    assert planner.last_trace["normalized_must_aspects"] == [
        "의료영상 분석",
        "AI 기반",
        "기술 개발",
    ]
