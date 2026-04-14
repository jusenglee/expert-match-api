import asyncio
import json
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


def _plan(top_k: int = 5) -> PlannerOutput:
    return PlannerOutput(
        intent_summary="Recommend AI semiconductor reviewers",
        branch_query_hints={
            "basic": "profile",
            "art": "papers",
            "pat": "patents",
            "pjt": "projects",
        },
        top_k=top_k,
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
        rank_score=25.0,
    )


def _simple_card(expert_id: str, *, organization: str = "Test Lab") -> CandidateCard:
    return CandidateCard(
        expert_id=expert_id,
        name=f"Researcher {expert_id}",
        organization=organization,
        degree="PhD",
        major="AI",
        branch_coverage={"basic": True, "art": True, "pat": False, "pjt": False},
        counts={"article_cnt": 1, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 0},
        top_papers=[
            PublicationEvidence(
                publication_title=f"Paper {expert_id}",
                publication_year_month="2024-09-01",
                journal_name="IEEE Access",
            )
        ],
        shortlist_score=100.0,
        rank_score=100.0,
    )


def _build_shortlist(count: int, *, prefix: str = "expert") -> list[CandidateCard]:
    return [_simple_card(f"{prefix}-{index:02d}") for index in range(count)]


class RecordingBatchModel:
    def __init__(
        self,
        *,
        sleep_s: float = 0.0,
        fail_prefix: str | None = None,
        shared_not_selected_reason: str | None = None,
        shared_data_gap: str | None = None,
    ) -> None:
        self.sleep_s = sleep_s
        self.fail_prefix = fail_prefix
        self.shared_not_selected_reason = shared_not_selected_reason
        self.shared_data_gap = shared_data_gap
        self.calls: list[list[str]] = []
        self.call_kwargs: list[dict[str, object]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def ainvoke_non_stream(self, messages, **kwargs):
        payload = json.loads(messages[1].content)
        shortlist = payload["shortlist"]
        expert_ids = [card["expert_id"] for card in shortlist]
        self.calls.append(expert_ids)
        self.call_kwargs.append(dict(kwargs))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.sleep_s:
                await asyncio.sleep(self.sleep_s)
            if self.fail_prefix and any(
                expert_id.startswith(self.fail_prefix) for expert_id in expert_ids
            ):
                raise RuntimeError("synthetic batch failure")

            top_k = int(payload["plan"].get("top_k", 5))
            recommended = []
            for rank, card in enumerate(shortlist[: min(top_k, len(shortlist))], start=1):
                top_papers = card.get("top_papers") or [{}]
                first_paper = top_papers[0] if top_papers else {}
                recommended.append(
                    {
                        "rank": rank,
                        "expert_id": card["expert_id"],
                        "name": card["name"],
                        "organization": card.get("organization"),
                        "fit": "높음",
                        "reasons": ["Strong evidence"],
                        "evidence": [
                            {
                                "type": "paper",
                                "title": first_paper.get(
                                    "publication_title",
                                    f"Paper {card['expert_id']}",
                                ),
                            }
                        ],
                        "risks": [],
                        "rank_score": card.get("rank_score", 0.0),
                    }
                )

            response = {
                "recommended": recommended,
                "not_selected_reasons": (
                    [self.shared_not_selected_reason]
                    if self.shared_not_selected_reason
                    else []
                ),
                "data_gaps": [self.shared_data_gap] if self.shared_data_gap else [],
            }
            return SimpleNamespace(content=json.dumps(response, ensure_ascii=False))
        finally:
            self.in_flight -= 1


def test_heuristic_judge_uses_current_domain_field_names_for_evidence():
    judge = HeuristicJudge()

    result = asyncio.run(
        judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()])
    )

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

    result = asyncio.run(
        judge.judge(query="fallback judge safety", plan=_plan(), shortlist=[card])
    )

    assert len(result.recommended) == 1
    assert result.recommended[0].evidence == []


def test_openai_compat_judge_normalizes_recoverable_output_before_validation():
    judge = OpenAICompatJudge(Settings(app_env="test", strict_runtime_validation=False))
    judge.model = SimpleNamespace(ainvoke_non_stream=_fake_recoverable_judge_response)

    result = asyncio.run(
        judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()])
    )

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

    result = asyncio.run(
        judge.judge(query="AI semiconductor", plan=_plan(), shortlist=[_full_card()])
    )

    assert len(result.recommended) == 1
    assert result.recommended[0].expert_id == "1"
    assert result.recommended[0].evidence[0].title == "Efficient AI Chips"


def test_openai_compat_judge_uses_single_call_when_shortlist_fits_batch():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_judge_batch_size=10,
        llm_judge_max_concurrency=10,
    )
    judge = OpenAICompatJudge(settings)
    model = RecordingBatchModel()
    judge.model = model

    result = asyncio.run(
        judge.judge(
            query="AI semiconductor",
            plan=_plan(top_k=5),
            shortlist=_build_shortlist(5),
        )
    )

    assert len(model.calls) == 1
    assert len(model.calls[0]) == 5
    assert model.call_kwargs[0]["temperature"] == 0.0
    assert model.call_kwargs[0]["top_p"] == 0.2
    assert model.call_kwargs[0]["reasoning_effort"] == "low"
    assert model.call_kwargs[0]["include_reasoning"] is False
    assert model.call_kwargs[0]["disable_thinking"] is True
    assert "max_tokens_hint" not in model.call_kwargs[0]
    assert len(result.recommended) == 5


def test_openai_compat_judge_disables_batching_when_flag_is_false():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        use_map_reduce_judging=False,
        llm_judge_batch_size=10,
        llm_judge_max_concurrency=10,
    )
    judge = OpenAICompatJudge(settings)
    model = RecordingBatchModel()
    judge.model = model

    result = asyncio.run(
        judge.judge(
            query="AI semiconductor",
            plan=_plan(top_k=5),
            shortlist=_build_shortlist(40),
        )
    )

    assert len(model.calls) == 1
    assert len(model.calls[0]) == 40
    assert len(result.recommended) == 5


def test_openai_compat_judge_batches_large_shortlist_into_tournament_rounds():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_judge_batch_size=10,
        llm_judge_max_concurrency=10,
    )
    judge = OpenAICompatJudge(settings)
    model = RecordingBatchModel(
        shared_not_selected_reason="shared-reason",
        shared_data_gap="shared-gap",
    )
    judge.model = model

    result = asyncio.run(
        judge.judge(
            query="AI semiconductor",
            plan=_plan(top_k=5),
            shortlist=_build_shortlist(40),
        )
    )

    assert len(model.calls) == 7
    assert all(len(call) == 10 for call in model.calls)
    assert all(kwargs["temperature"] == 0.0 for kwargs in model.call_kwargs)
    assert all(kwargs["top_p"] == 0.2 for kwargs in model.call_kwargs)
    assert all(kwargs["reasoning_effort"] == "low" for kwargs in model.call_kwargs)
    assert all(kwargs["include_reasoning"] is False for kwargs in model.call_kwargs)
    assert all(kwargs["disable_thinking"] is True for kwargs in model.call_kwargs)
    assert all(
        kwargs.get("max_tokens_hint") == 3000 for kwargs in model.call_kwargs[:-1]
    )
    assert "max_tokens_hint" not in model.call_kwargs[-1]
    assert len(result.recommended) == 5
    assert result.not_selected_reasons == ["shared-reason"]
    assert result.data_gaps == ["shared-gap"]


def test_openai_compat_judge_respects_semaphore_limit_during_batched_calls():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_judge_batch_size=10,
        llm_judge_max_concurrency=2,
    )
    judge = OpenAICompatJudge(settings)
    model = RecordingBatchModel(sleep_s=0.01)
    judge.model = model

    result = asyncio.run(
        judge.judge(
            query="AI semiconductor",
            plan=_plan(top_k=5),
            shortlist=_build_shortlist(40),
        )
    )

    assert len(result.recommended) == 5
    assert model.max_in_flight <= 2


def test_openai_compat_judge_falls_back_per_batch_without_failing_entire_round():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_judge_batch_size=10,
        llm_judge_max_concurrency=10,
    )
    judge = OpenAICompatJudge(settings)
    model = RecordingBatchModel(fail_prefix="fail")
    judge.model = model
    shortlist = _build_shortlist(10, prefix="fail") + _build_shortlist(10, prefix="ok")

    result = asyncio.run(
        judge.judge(
            query="AI semiconductor",
            plan=_plan(top_k=5),
            shortlist=shortlist,
        )
    )

    assert len(model.calls) == 3
    assert len(result.recommended) == 5
    assert any(item.expert_id.startswith("fail") for item in result.recommended)


async def _fake_judge_with_pjt_type(messages, **kwargs):
    """LLM이 evidence.type에 'pjt' 축약어를 반환하는 경우를 재현합니다.
    Pydantic validator가 자동 정규화(pjt → project)하므로 ValidationError 없이
    통과되어야 합니다 (postmortem-judge-errors 회귀 방지 케이스).
    """
    return SimpleNamespace(
        content="""{
  "recommended": [
    {
      "rank": 1,
      "expert_id": "1",
      "name": "Alice Kim",
      "organization": "Test Lab",
      "fit": "높음",
      "reasons": ["연구과제 실적 우수"],
      "evidence": [
        {
          "type": "pjt",
          "title": "AI Semiconductor Program",
          "date": "2025-12-31",
          "detail": "NIPA"
        }
      ],
      "risks": [],
      "rank_score": 25.0
    }
  ],
  "not_selected_reasons": [],
  "data_gaps": []
}"""
    )


def test_openai_compat_judge_normalizes_pjt_type_in_evidence():
    """
    회귀 방지: LLM이 evidence.type에 브랜치 축약어 'pjt'를 반환해도
    Pydantic validator가 'project'로 자동 정규화하여 ValidationError 없이 통과해야 한다.
    이전에는 이 오류로 Reduce 라운드 전체가 Fallback으로 강등되었음.
    """
    judge = OpenAICompatJudge(Settings(app_env="test", strict_runtime_validation=False))
    judge.model = SimpleNamespace(ainvoke_non_stream=_fake_judge_with_pjt_type)

    result = asyncio.run(
        judge.judge(query="AI 반도체 전문가", plan=_plan(), shortlist=[_full_card()])
    )

    assert len(result.recommended) == 1
    evidence = result.recommended[0].evidence
    assert len(evidence) == 1
    assert evidence[0].type == "project", (
        "'pjt' 타입이 'project'로 정규화되지 않았습니다. "
        "EvidenceItem._normalize_evidence_type validator를 확인하세요."
    )
    assert evidence[0].title == "AI Semiconductor Program"


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
