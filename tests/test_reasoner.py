import asyncio
import json
import logging

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.domain.models import (
    CandidateCard,
    EvaluationActivity,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)
from apps.recommendation.evidence_selector import RelevantEvidenceBundle, RelevantEvidenceItem
from apps.recommendation.reasoner import OpenAICompatReasonGenerator


class FakeReasonModel:
    def __init__(self, content):
        self.content = content
        self.last_kwargs = None
        self.last_messages = None

    async def ainvoke_non_stream(self, messages, **kwargs):
        self.last_kwargs = dict(kwargs)
        self.last_messages = messages
        return AIMessage(content=self.content)


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
              "selected_evidence_ids": [],
              "risks": []
            },
            {
              "expert_id": "1",
              "fit": "높음",
              "recommendation_reason": "Reason for first",
              "selected_evidence_ids": [],
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
    assert generator.last_trace["mode"] == "openai_compat"
    assert generator.last_trace["returned_ids"] == ["2", "1"]
    assert generator.last_trace["missing_candidate_ids"] == []


def test_reason_generator_serializes_relevant_evidence_only():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    fake_model = FakeReasonModel(
        """{
          "items": [
            {
              "expert_id": "1",
              "fit": "?믪쓬",
              "recommendation_reason": "Relevant imaging project is present.",
              "selected_evidence_ids": ["paper:0"],
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
                core_keywords=["medical imaging"],
            ),
            candidates=[_candidate("1", "Alpha", 98.0)],
            relevant_evidence_by_expert_id={
                "1": RelevantEvidenceBundle(
                    expert_id="1",
                    papers=[
                        RelevantEvidenceItem(
                            item_id="paper:0",
                            type="paper",
                            title="Medical imaging segmentation",
                            date="2024-01",
                            snippet="Segmentation for emergency CT scans",
                            matched_keywords=["medical imaging"],
                            match_score=11.5,
                        )
                    ],
                )
            },
        )
    )

    payload = json.loads(fake_model.last_messages[1].content)
    candidate_payload = payload["candidates"][0]

    assert "relevant_papers" in candidate_payload
    assert "top_papers" not in candidate_payload
    assert candidate_payload["relevant_papers"][0]["evidence_id"] == "paper:0"
    assert candidate_payload["relevant_papers"][0]["title"] == "Medical imaging segmentation"
    assert candidate_payload["relevant_papers"][0]["matched_keywords"] == ["medical imaging"]
    assert "match_score" not in candidate_payload["relevant_papers"][0]


def test_reason_generator_serializes_raw_candidate_context_and_retrieval_grounding():
    generator = OpenAICompatReasonGenerator(
        Settings(app_env="test", strict_runtime_validation=False)
    )
    fake_model = FakeReasonModel(
        """{
          "items": [
            {
              "expert_id": "1",
              "fit": "蹂댄넻",
              "recommendation_reason": "Profile and project context are available.",
              "selected_evidence_ids": [],
              "risks": []
            }
          ],
          "data_gaps": []
        }"""
    )
    generator.model = fake_model
    candidate = _candidate("1", "Alpha", 98.0)
    candidate.technical_classifications = ["smart factory", "semiconductor"]
    candidate.evaluation_activity_cnt = 2
    candidate.evaluation_activities = [
        EvaluationActivity(
            appoint_org_nm="NTIS",
            committee_nm="Smart factory review committee",
            appoint_period="2025",
            appoint_dt="2025-05-01",
        )
    ]
    candidate.top_projects = [
        ResearchProjectEvidence(
            project_title_korean="스마트팩토리 공정 자동화",
            project_end_date="2025-12-31",
            research_content_summary="반도체 공정 자동화 프로젝트",
        )
    ]

    asyncio.run(
        generator.generate(
            query="Recommend smart factory reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend smart factory reviewers",
                core_keywords=["smart factory"],
            ),
            candidates=[candidate],
            retrieval_score_traces_by_expert_id={
                "1": {
                    "expert_id": "1",
                    "point_id": "1_basic",
                    "primary_branch": "basic",
                    "final_score": 0.5,
                    "branch_matches": [{"branch": "basic", "rank": 1, "score": 0.5}],
                }
            },
        )
    )

    payload = json.loads(fake_model.last_messages[1].content)
    candidate_payload = payload["candidates"][0]

    assert candidate_payload["retrieval_grounding"]["primary_branch"] == "basic"
    assert candidate_payload["technical_classifications"] == [
        "smart factory",
        "semiconductor",
    ]
    assert candidate_payload["evaluation_activity_cnt"] == 2
    assert candidate_payload["evaluation_activities"][0]["committee_nm"] == (
        "Smart factory review committee"
    )
    assert candidate_payload["all_projects"][0]["title"] == "스마트팩토리 공정 자동화"


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
              "selected_evidence_ids": [],
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
    assert "no selected evidence ids" in caplog.text
    assert generator.last_trace["returned_ids"] == ["1"]
    assert generator.last_trace["missing_candidate_ids"] == ["2"]
    assert generator.last_trace["empty_reason_candidate_ids"] == ["1", "2"]
