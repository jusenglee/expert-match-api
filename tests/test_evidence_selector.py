from apps.domain.models import (
    CandidateCard,
    IntellectualPropertyEvidence,
    PlannerOutput,
    PublicationEvidence,
    ResearchProjectEvidence,
)
from apps.recommendation.evidence_selector import KeywordEvidenceSelector


def _plan(*aspects: str, generic_terms: list[str] | None = None) -> PlannerOutput:
    return PlannerOutput(
        intent_summary="test",
        core_keywords=list(aspects),
        retrieval_core=list(aspects),
        must_aspects=list(aspects),
        generic_terms=list(generic_terms or []),
    )


def _candidate_card() -> CandidateCard:
    return CandidateCard(
        expert_id="1",
        name="Alpha",
        top_papers=[],
        top_projects=[],
        top_patents=[],
    )


def test_selector_matches_whitespace_variants_in_project_titles():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean="AI  semiconductor process platform",
            project_end_date="2025-12-31",
            research_objective_summary="Process optimization",
        )
    ]

    bundles = selector.select(candidates=[card], plan=_plan("AI semiconductor"))

    assert bundles["1"].projects[0].title == "AI  semiconductor process platform"
    assert bundles["1"].projects[0].matched_keywords == ["ai semiconductor"]
    assert bundles["1"].direct_match_count == 1
    assert bundles["1"].aspect_coverage == 1


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


def test_selector_excludes_irrelevant_newer_evidence():
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
    assert bundles["1"].direct_match_count == 1


def test_selector_deduplicates_same_title_and_year():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean="AI semiconductor process platform",
            project_end_date="2025-12-31",
            research_objective_summary="Process optimization",
        ),
        ResearchProjectEvidence(
            project_title_korean="AI semiconductor process platform",
            project_end_date="2025-06-01",
            research_objective_summary="Duplicate title same year",
        ),
    ]

    bundles = selector.select(candidates=[card], plan=_plan("AI semiconductor"))

    assert len(bundles["1"].projects) == 1
    assert bundles["1"].dedup_dropped_count == 1


def test_selector_caps_selected_evidence_and_tracks_aspect_coverage():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title=f"Medical imaging paper {index}",
            publication_year_month=f"2024-{index:02d}",
            abstract="Emergency CT workflow",
        )
        for index in range(1, 6)
    ]
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean=f"Emergency CT project {index}",
            project_end_date=f"2024-{index:02d}-01",
            research_objective_summary="Medical imaging support",
        )
        for index in range(1, 6)
    ]
    card.top_patents = [
        IntellectualPropertyEvidence(
            intellectual_property_title=f"medical imaging emergency ct patent {index}",
            application_date=f"2024-{index:02d}-01",
        )
        for index in range(1, 6)
    ]

    bundles = selector.select(
        candidates=[card],
        plan=_plan("medical imaging", "emergency ct"),
    )

    assert len(bundles["1"].all_items()) <= 4
    assert bundles["1"].aspect_coverage == 2
    assert set(bundles["1"].matched_aspects) == {"medical imaging", "emergency ct"}
    assert selector.last_trace["candidate_evidence_counts"][0]["total"] <= 4


def test_selector_marks_generic_only_candidates_without_direct_evidence():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_papers = [
        PublicationEvidence(
            publication_title="Research experience overview",
            publication_year_month="2024-01",
            abstract="Long experience in multiple domains",
        )
    ]

    bundles = selector.select(
        candidates=[card],
        plan=_plan("medical imaging", generic_terms=["experience"]),
    )

    assert bundles["1"].papers == []
    assert bundles["1"].generic_only is True
    assert bundles["1"].direct_match_count == 0
    assert selector.last_trace["empty_candidate_ids"] == ["1"]


def test_selector_keeps_future_projects_and_traces_future_selected_evidence():
    selector = KeywordEvidenceSelector(reference_year=2026)
    card = _candidate_card()
    card.top_projects = [
        ResearchProjectEvidence(
            project_title_korean="Medical imaging platform",
            project_end_date="2030-12-31",
            research_objective_summary="Medical imaging workflow",
        )
    ]

    bundles = selector.select(candidates=[card], plan=_plan("medical imaging"))

    assert bundles["1"].projects[0].is_future_item is True
    assert bundles["1"].future_selected_evidence_ids == ["project:0"]
    assert (
        selector.last_trace["candidate_evidence_counts"][0]["future_selected_evidence_ids"]
        == ["project:0"]
    )
