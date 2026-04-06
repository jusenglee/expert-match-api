import asyncio
from types import SimpleNamespace

from apps.core.config import Settings
from apps.domain.models import (
    CandidateCard,
    IntellectualPropertyEvidence,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)
from apps.recommendation.judge import HeuristicJudge, OpenAICompatJudge


def _plan() -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend AI semiconductor reviewers",
        branch_query_hints={
            "basic": "profile",
            "art": "papers",
            "pat": "patents",
            "pjt": "projects",
        },
        top_k=5,
    )


def _full_card() -> CandidateCard:
    return CandidateCard(
        expert_id="1",
        name="Alice Kim",
        organization="Test Lab",
        degree="PhD",
        major="AI",
        branch_coverage={"basic": True, "art": True, "pat": True, "pjt": True},
        counts={"article_cnt": 2, "scie_cnt": 1, "patent_cnt": 1, "project_cnt": 1},
        top_papers=[
            PublicationEvidence(
                publication_title="Efficient AI Chips",
                publication_year_month="2024-09-01",
                journal_name="IEEE Access",
            )
        ],
        top_patents=[
            IntellectualPropertyEvidence(
                intellectual_property_title="Neural Accelerator",
                application_registration_type="registered",
                registration_date="2024-03-10",
            )
        ],
        top_projects=[
            ResearchProjectEvidence(
                project_title_korean="AI Semiconductor Program",
                project_start_date="2023-01-01",
                project_end_date="2025-12-31",
                managing_agency="NIPA",
            )
        ],
        shortlist_score=25.0,
    )


def test_heuristic_judge_uses_current_domain_field_names_for_evidence():
    judge = HeuristicJudge()

    result = asyncio.run(judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()]))

    assert len(result.recommended) == 1
    evidence = result.recommended[0].evidence
    assert [item.type for item in evidence] == ["paper", "patent", "project", "profile"]
    assert evidence[0].title == "Efficient AI Chips"
    assert evidence[0].date == "2024-09-01"
    assert evidence[0].detail == "IEEE Access"
    assert evidence[1].title == "Neural Accelerator"
    assert evidence[1].date == "2024-03-10"
    assert evidence[1].detail == "registered"
    assert evidence[2].title == "AI Semiconductor Program"
    assert evidence[2].date == "2025-12-31"
    assert evidence[2].detail == "NIPA"


def test_heuristic_judge_handles_empty_nested_evidence_lists_without_attribute_errors():
    judge = HeuristicJudge()
    card = CandidateCard(
        expert_id="2",
        name="Bob Lee",
        branch_coverage={"basic": False, "art": True, "pat": True, "pjt": True},
        counts={"article_cnt": 0, "scie_cnt": 0, "patent_cnt": 0, "project_cnt": 0},
        top_papers=[],
        top_patents=[],
        top_projects=[],
        shortlist_score=5.0,
    )

    result = asyncio.run(judge.judge(query="fallback judge safety", plan=_plan(), shortlist=[card]))

    assert len(result.recommended) == 1
    assert result.recommended[0].evidence == []


def test_openai_compat_judge_normalizes_recoverable_output_before_validation():
    judge = OpenAICompatJudge(Settings(app_env="test", strict_runtime_validation=False))
    judge.model = SimpleNamespace(ainvoke_non_stream=_fake_recoverable_judge_response)

    result = asyncio.run(judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()]))

    assert len(result.recommended) == 1
    recommendation = result.recommended[0]
    assert recommendation.expert_id == "1"
    assert recommendation.fit == "높음"
    assert recommendation.reasons == ["Publication evidence is strong."]
    assert recommendation.risks == ["Patent evidence is thinner."]
    assert result.not_selected_reasons == ["Other shortlisted candidates had weaker alignment."]
    assert result.data_gaps == ["Patent coverage is limited."]


def test_openai_compat_judge_falls_back_to_heuristic_judge_when_required_fields_are_missing():
    judge = OpenAICompatJudge(Settings(app_env="test", strict_runtime_validation=False))
    judge.model = SimpleNamespace(ainvoke_non_stream=_fake_missing_fit_judge_response)

    result = asyncio.run(judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()]))

    assert len(result.recommended) == 1
    assert result.recommended[0].expert_id == "1"
    assert result.recommended[0].evidence[0].title == "Efficient AI Chips"


async def _fake_recoverable_judge_response(messages, **kwargs):
    return SimpleNamespace(
        content="""```json
{
  "recommended": [
    {
      "rank": "1",
      "name": "Alice Kim",
      "fit": "높음",
      "reasons": "Publication evidence is strong.",
      "evidence": [
        {
          "type": "paper",
          "title": "Efficient AI Chips",
          "date": "2024-09-01",
          "detail": "IEEE Access"
        }
      ],
      "risks": "Patent evidence is thinner."
    }
  ],
  "not_selected_reasons": "Other shortlisted candidates had weaker alignment.",
  "data_gaps": "Patent coverage is limited."
}
```"""
    )


async def _fake_missing_fit_judge_response(messages, **kwargs):
    return SimpleNamespace(
        content="""{
  "recommended": [
    {
      "rank": 1,
      "name": "Alice Kim",
      "reasons": ["Strong evidence"],
      "evidence": [
        {
          "type": "paper",
          "title": "Efficient AI Chips"
        }
      ],
      "risks": []
    }
  ],
  "not_selected_reasons": [],
  "data_gaps": []
}"""
    )
