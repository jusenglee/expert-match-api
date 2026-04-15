import asyncio
import json

from langchain_core.messages import AIMessage

from apps.core.config import Settings
from apps.domain.models import CandidateCard, PlannerOutput, PublicationEvidence
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
    assert candidate_payload["relevant_papers"][0]["title"] == "Medical imaging segmentation"
    assert candidate_payload["relevant_papers"][0]["matched_keywords"] == ["medical imaging"]
    assert "match_score" not in candidate_payload["relevant_papers"][0]
