import asyncio
import json
import logging

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.domain.models import CandidateCard, PlannerOutput, PublicationEvidence
from apps.recommendation.evidence_selector import RelevantEvidenceBundle, RelevantEvidenceItem
from apps.recommendation.reasoner import FIT_HIGH, OpenAICompatReasonGenerator


class FakeReasonModel:
    def __init__(self, content="", *, tool_calls=None, responses=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.responses = list(responses or [])
        self.last_kwargs = None
        self.last_messages = None
        self.call_count = 0
        self.calls: list[dict[str, object]] = []

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        self.last_messages = messages
        self.call_count += 1
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        if self.responses:
            return self.responses.pop(0)
        additional_kwargs = {"tool_calls": self.tool_calls} if self.tool_calls else {}
        return AIMessage(content=self.content, additional_kwargs=additional_kwargs)


def _candidate(expert_id: str, name: str, score: float) -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=name,
        organization="Test Institute",
        counts={"article_cnt": 1},
        top_papers=[
            PublicationEvidence(
                publication_title="Latest but unrelated robotics study",
                publication_year_month="2026-01",
                abstract="Soft robot manipulation",
            )
        ],
        rank_score=score,
        shortlist_score=score,
    )


def _bundle(*items: RelevantEvidenceItem) -> RelevantEvidenceBundle:
    papers = [item for item in items if item.type == "paper"]
    projects = [item for item in items if item.type == "project"]
    patents = [item for item in items if item.type == "patent"]
    return RelevantEvidenceBundle(
        expert_id="1",
        papers=papers,
        projects=projects,
        patents=patents,
        direct_match_count=len(items),
        aspect_coverage=1 if items else 0,
        matched_aspects=["medical imaging"] if items else [],
    )


def test_reason_generator_normalizes_output_to_input_order():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    generator.model = FakeReasonModel(
        """{
          "items": [
            {
              "expert_id": "2",
              "fit": "중간",
              "recommendation_reason": "Reason for second",
              "risks": []
            },
            {
              "expert_id": "1",
              "fit": "높음",
              "recommendation_reason": "Reason for first",
              "risks": []
            }
          ],
          "data_gaps": []
        }"""
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend reviewers",
                retrieval_core=["semiconductor"],
                must_aspects=["semiconductor"],
                core_keywords=["semiconductor"],
            ),
            candidates=[
                _candidate("1", "Alpha", 98.0),
                _candidate("2", "Bravo", 95.0),
            ],
        )
    )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert output.items[0].recommendation_reason == "Reason for first"
    assert output.items[1].recommendation_reason == "Reason for second"
    assert output.items[0].selected_evidence_ids == []
    assert generator.last_trace["mode"] == "json_fallback"
    assert generator.last_trace["returned_ids"] == ["2", "1"]
    assert generator.last_trace["missing_candidate_ids"] == []
    assert generator.last_trace["retry_count"] == 0


def test_reason_generator_serializes_selected_evidence_and_do_not_mention_only():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    fake_model = FakeReasonModel(
        """{
          "items": [
            {
              "expert_id": "1",
              "fit": "높음",
              "recommendation_reason": "Relevant imaging evidence is present.",
              "risks": []
            }
          ],
          "data_gaps": []
        }"""
    )
    generator.model = fake_model

    asyncio.run(
        generator.generate(
            query="Recommend medical imaging reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend medical imaging reviewers",
                retrieval_core=["medical imaging"],
                must_aspects=["medical imaging"],
                core_keywords=["medical imaging"],
            ),
            candidates=[_candidate("1", "Alpha", 98.0), _candidate("2", "Bravo", 95.0)],
            relevant_evidence_by_expert_id={
                "1": _bundle(
                    RelevantEvidenceItem(
                        item_id="paper:0",
                        type="paper",
                        title="Medical imaging segmentation",
                        date="2024-01",
                        snippet="Segmentation for emergency CT scans",
                        matched_keywords=["medical imaging"],
                        aspect_matches=["medical imaging"],
                        direct_match=True,
                        match_score=11.5,
                    )
                )
            },
        )
    )

    first_call = fake_model.calls[0]
    payload = json.loads(first_call["messages"][1].content)
    candidate_payload = payload["candidates"][0]

    assert payload["must_aspects"] == ["medical imaging"]
    assert "selected_evidence" in candidate_payload
    assert candidate_payload["selected_evidence"][0]["evidence_id"] == "paper:0"
    assert candidate_payload["selected_evidence"][0]["title"] == "Medical imaging segmentation"
    assert "do_not_mention" in candidate_payload
    assert "Bravo" in candidate_payload["do_not_mention"]
    assert "paper:0" in candidate_payload["do_not_mention"]
    assert "all_projects" not in candidate_payload
    assert "relevant_papers" not in candidate_payload
    tool_properties = first_call["kwargs"]["tools"][0]["function"]["parameters"][
        "properties"
    ]["items"]["items"]["properties"]
    assert "selected_evidence_ids" not in tool_properties


def test_reason_generator_logs_missing_and_empty_reasons(caplog):
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    generator.model = FakeReasonModel(
        """{
          "items": [
            {
              "expert_id": "1",
              "fit": "보통",
              "recommendation_reason": "",
              "risks": []
            }
          ],
          "data_gaps": []
        }"""
    )

    with caplog.at_level(logging.INFO, logger="apps.recommendation.reasoner"):
        output = asyncio.run(
            generator.generate(
                query="Recommend reviewers",
                plan=PlannerOutput(
                    intent_summary="Recommend reviewers",
                    retrieval_core=["semiconductor"],
                    must_aspects=["semiconductor"],
                    core_keywords=["semiconductor"],
                ),
                candidates=[
                    _candidate("1", "Alpha", 98.0),
                    _candidate("2", "Bravo", 95.0),
                ],
            )
        )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert "omitted candidates from output" in caplog.text
    assert "empty recommendation reasons" in caplog.text
    assert "scheduling compact retry" in caplog.text
    assert generator.model.call_count == 2
    assert generator.last_trace["returned_ids"] == ["1"]
    assert generator.last_trace["missing_candidate_ids"] == ["2"]
    assert generator.last_trace["empty_reason_candidate_ids"] == ["1", "2"]
    assert generator.last_trace["retry_count"] == 1
    assert generator.last_trace["prompt_budget_mode"] == "retry_compact"
    assert generator.last_trace["retry_triggered"] is True
    assert generator.last_trace["retry_trigger"] == "missing_candidate_ids"


def test_reason_generator_prefers_tool_call_arguments_when_present():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    tool_payload = {
        "items": [
            {
                "expert_id": "1",
                "fit": FIT_HIGH,
                "recommendation_reason": "Structured tool output is available.",
                "risks": [],
            }
        ],
        "data_gaps": [],
    }
    generator.model = FakeReasonModel(
        content='{"items":[],"data_gaps":[]}',
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "submit_recommendation_batch",
                    "arguments": json.dumps(tool_payload, ensure_ascii=False),
                },
            }
        ],
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend reviewers",
                retrieval_core=["semiconductor"],
                must_aspects=["semiconductor"],
                core_keywords=["semiconductor"],
            ),
            candidates=[_candidate("1", "Alpha", 98.0)],
        )
    )

    assert output.items[0].recommendation_reason == "Structured tool output is available."
    assert output.items[0].selected_evidence_ids == []
    assert generator.last_trace["mode"] == "tool_call"


def test_reason_generator_retries_with_compact_payload_when_first_attempt_returns_no_candidate_ids():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    generator.model = FakeReasonModel(
        responses=[
            AIMessage(content='{"items":[],"data_gaps":[]}'),
            AIMessage(
                content="""{
                  "items": [
                    {
                      "expert_id": "1",
                      "fit": "보통",
                      "recommendation_reason": "Retry payload succeeded.",
                      "risks": []
                    }
                  ],
                  "data_gaps": []
                }"""
            ),
        ]
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend reviewers",
                retrieval_core=["medical imaging"],
                must_aspects=["medical imaging"],
                core_keywords=["medical imaging"],
            ),
            candidates=[_candidate("1", "Alpha", 98.0)],
            relevant_evidence_by_expert_id={
                "1": _bundle(
                    RelevantEvidenceItem(
                        item_id="paper:0",
                        type="paper",
                        title="Medical imaging study A",
                        date="2024-01",
                        matched_keywords=["medical imaging"],
                        aspect_matches=["medical imaging"],
                        direct_match=True,
                    ),
                    RelevantEvidenceItem(
                        item_id="paper:1",
                        type="paper",
                        title="Medical imaging study B",
                        date="2024-02",
                        matched_keywords=["medical imaging"],
                        aspect_matches=["medical imaging"],
                        direct_match=True,
                    ),
                    RelevantEvidenceItem(
                        item_id="project:0",
                        type="project",
                        title="Medical imaging project",
                        date="2024-03",
                        matched_keywords=["medical imaging"],
                        aspect_matches=["medical imaging"],
                        direct_match=True,
                    ),
                    RelevantEvidenceItem(
                        item_id="patent:0",
                        type="patent",
                        title="Medical imaging patent",
                        date="2024-04",
                        matched_keywords=["medical imaging"],
                        aspect_matches=["medical imaging"],
                        direct_match=True,
                    ),
                )
            },
        )
    )

    assert output.items[0].recommendation_reason == "Retry payload succeeded."
    assert generator.model.call_count == 2
    first_payload = json.loads(generator.model.calls[0]["messages"][1].content)
    second_payload = json.loads(generator.model.calls[1]["messages"][1].content)
    assert len(first_payload["candidates"][0]["selected_evidence"]) == 4
    assert len(second_payload["candidates"][0]["selected_evidence"]) == 2
    assert generator.last_trace["retry_count"] == 1
    assert generator.last_trace["prompt_budget_mode"] == "retry_compact"


def test_reason_generator_retries_with_compact_payload_when_first_attempt_returns_partial_candidates():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    generator.model = FakeReasonModel(
        responses=[
            AIMessage(
                content="""{
                  "items": [
                    {
                      "expert_id": "1",
                      "fit": "보통",
                      "recommendation_reason": "First attempt returned only one candidate.",
                      "risks": []
                    }
                  ],
                  "data_gaps": []
                }"""
            ),
            AIMessage(
                content="""{
                  "items": [
                    {
                      "expert_id": "1",
                      "fit": "보통",
                      "recommendation_reason": "Retry kept candidate one.",
                      "risks": []
                    },
                    {
                      "expert_id": "2",
                      "fit": "높음",
                      "recommendation_reason": "Retry recovered candidate two.",
                      "risks": []
                    }
                  ],
                  "data_gaps": []
                }"""
            ),
        ]
    )

    output = asyncio.run(
        generator.generate(
            query="Recommend reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend reviewers",
                retrieval_core=["medical imaging"],
                must_aspects=["medical imaging"],
                core_keywords=["medical imaging"],
            ),
            candidates=[
                _candidate("1", "Alpha", 98.0),
                _candidate("2", "Bravo", 95.0),
            ],
        )
    )

    assert [item.expert_id for item in output.items] == ["1", "2"]
    assert output.items[1].recommendation_reason == "Retry recovered candidate two."
    assert generator.model.call_count == 2
    assert generator.last_trace["retry_count"] == 1
    assert generator.last_trace["retry_triggered"] is True
    assert generator.last_trace["retry_trigger"] == "missing_candidate_ids"
    assert generator.last_trace["prompt_budget_mode"] == "retry_compact"
