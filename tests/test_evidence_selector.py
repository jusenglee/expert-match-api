from apps.domain.models import (
    CandidateCard,
    IntellectualPropertyEvidence,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)
from apps.recommendation.evidence_selector import KeywordEvidenceSelector


def _plan(*keywords: str) -> PlannerOutput:
    return PlannerOutput(
        intent_summary="test",
        core_keywords=list(keywords),
    )


def _candidate_card() -> CandidateCard:
    return CandidateCard(
        expert_id="1",
        name="Alpha",
        top_papers=[],
        top_projects=[],
        top_patents=[],
    )


def test_selector_matches_whitespace_variants_in_korean():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean="화재진압용 드론 시스템",
            project_end_date="2025-12-31",
            research_objective_summary="재난 대응 자동화",
        )
    ]

    bundles = selector.select(candidates=[card], plan=_plan("화재 진압"))

    assert bundles["1"].projects[0].title == "화재진압용 드론 시스템"
    assert bundles["1"].projects[0].matched_keywords == ["화재 진압"]


def test_selector_is_case_insensitive_for_english_keywords():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title="Medical Imaging Assessment for CT",
            publication_year_month="2024-01",
            abstract="Emergency imaging workflow",
        )
    ]

    bundles = selector.select(candidates=[card], plan=_plan("medical imaging"))

    assert bundles["1"].papers[0].title == "Medical Imaging Assessment for CT"
    assert bundles["1"].papers[0].matched_keywords == ["medical imaging"]
    assert bundles["1"].papers[0].item_id == "paper:0"


def test_selector_prefers_keyword_matched_evidence_over_newer_irrelevant_items():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title="Soft robot manipulation",
            publication_year_month="2026-01",
            abstract="Anthropomorphic hand control",
        ),
        PublicationEvidence(
            publication_title="Medical imaging segmentation for emergency CT",
            publication_year_month="2022-05",
            abstract="Imaging triage support",
        ),
    ]

    bundles = selector.select(candidates=[card], plan=_plan("medical imaging"))

    assert [item.title for item in bundles["1"].papers] == [
        "Medical imaging segmentation for emergency CT"
    ]


def test_selector_enforces_branch_limits():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title=f"Medical imaging paper {index}",
            publication_year_month=f"2024-{index:02d}",
            abstract="Imaging",
        )
        for index in range(1, 6)
    ]
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean=f"의료영상 과제 {index}",
            project_end_date=f"2024-{index:02d}-01",
            research_objective_summary="의료영상 분석",
        )
        for index in range(1, 6)
    ]
    card.top_patents = [
        IntellectualPropertyEvidence(
            intellectual_property_title=f"medical imaging patent {index}",
            application_date=f"2024-{index:02d}-01",
        )
        for index in range(1, 6)
    ]

    bundles = selector.select(candidates=[card], plan=_plan("medical imaging", "의료영상"))

    assert len(bundles["1"].papers) == 4
    assert len(bundles["1"].projects) == 4
    assert len(bundles["1"].patents) == 4


def test_selector_returns_empty_relevant_evidence_when_keywords_are_empty():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title="Medical imaging assessment",
            publication_year_month="2024-01",
        )
    ]

    bundles = selector.select(candidates=[card], plan=_plan())

    assert bundles["1"].papers == []
    assert selector.last_trace["empty_candidate_ids"] == ["1"]
