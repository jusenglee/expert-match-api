import pytest
from pydantic import ValidationError

from apps.domain.models import EvidenceItem, ExpertPayload, PublicationEvidence


def test_expert_payload_normalizes_legacy_blank_string_fields():
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


# ── EvidenceItem 스키마 동기화 회귀 방지 ────────────────────────────────────
# 발생 배경: LLM이 플래너 브랜치 축약어(pjt, art, pat)를 evidence.type에 그대로
# 출력하여 Pydantic Validation 실패가 연속 발생한 사건 (postmortem-judge-errors 참조)

@pytest.mark.parametrize("raw_type,expected", [
    ("pjt", "project"),   # 과제 축약어
    ("art", "paper"),     # 논문 축약어
    ("pat", "patent"),    # 특허 축약어
    ("PJT", "project"),   # 대문자 변형
    ("ART", "paper"),
    ("PAT", "patent"),
    ("project", "project"),  # 정규값은 그대로 통과
    ("paper", "paper"),
    ("patent", "patent"),
    ("profile", "profile"),
])
def test_evidence_item_normalizes_branch_alias_to_canonical_type(raw_type, expected):
    """브랜치 축약어 → 정규 type 값으로 자동 정규화되어야 한다."""
    item = EvidenceItem(type=raw_type, title="Test Title")
    assert item.type == expected


def test_evidence_item_rejects_unknown_type():
    """허용 값 목록에 없는 type은 ValidationError를 발생시켜야 한다."""
    with pytest.raises(ValidationError):
        EvidenceItem(type="unknown_type", title="Test")
