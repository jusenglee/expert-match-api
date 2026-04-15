import pytest
from pydantic import ValidationError

from apps.domain.models import EvidenceItem, ExpertPayload, PublicationEvidence


def test_expert_payload_normalizes_blank_string_fields():
    payload = {
        "basic_info": {"researcher_id": "1", "researcher_name": "Tester"},
        "researcher_profile": {
            "publication_count": "",
            "scie_publication_count": " ",
            "intellectual_property_count": "2",
            "research_project_count": None,
        },
        "publications": "",
        "intellectual_properties": "",
        "research_projects": [{"project_title_korean": "Project", "reference_year": ""}],
        "technical_classifications": "",
        "evaluation_activity_cnt": "",
        "external_activity_cnt": " ",
        "evaluation_activities": "",
    }

    parsed = ExpertPayload.model_validate(payload)

    assert parsed.publications == []
    assert parsed.intellectual_properties == []
    assert parsed.technical_classifications == []
    assert parsed.evaluation_activities == []
    assert parsed.researcher_profile.publication_count == 0
    assert parsed.researcher_profile.scie_publication_count == 0
    assert parsed.researcher_profile.intellectual_property_count == 2
    assert parsed.researcher_profile.research_project_count == 0
    assert parsed.evaluation_activity_cnt == 0
    assert parsed.external_activity_cnt == 0
    assert parsed.research_projects[0].reference_year is None


def test_publication_evidence_normalizes_keyword_strings_to_lists():
    publication = PublicationEvidence.model_validate(
        {
            "publication_title": "Test paper",
            "korean_keywords": "AI",
            "english_keywords": "",
        }
    )

    assert publication.korean_keywords == ["AI"]
    assert publication.english_keywords == []


@pytest.mark.parametrize("raw_type", ["project", "paper", "patent", "profile"])
def test_evidence_item_accepts_canonical_type(raw_type):
    item = EvidenceItem(type=raw_type, title="Test Title")
    assert item.type == raw_type


def test_evidence_item_rejects_unknown_type():
    with pytest.raises(ValidationError):
        EvidenceItem(type="unknown_type", title="Test")


@pytest.mark.parametrize("legacy_type", ["pjt", "art", "pat", "PJT", "ART", "PAT"])
def test_evidence_item_rejects_legacy_branch_alias_type(legacy_type):
    with pytest.raises(ValidationError):
        EvidenceItem(type=legacy_type, title="Test")
