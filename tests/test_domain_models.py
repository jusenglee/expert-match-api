import pytest
from pydantic import ValidationError

from apps.domain.chunk_view import (
    clean_value,
    derive_date,
    derive_title,
    normalize_doc_date,
    parse_year,
    snippet,
)
from apps.domain.models import (
    COUNT_FIELDS,
    CandidateCard,
    ChunkEvidence,
    ChunkHit,
    ChunkPayload,
    EvidenceItem,
    ResearcherCandidate,
)


# ---------------------------------------------------------------------------
# ChunkPayload — flat contract: count coercion, doc_date NONE→None, extra ignored
# ---------------------------------------------------------------------------


def _minimal_payload(**overrides):
    base = {
        "researcher_id": "100",
        "doc_type": "paper",
        "chunk_id": "paper_100000045256_c000",
    }
    base.update(overrides)
    return base


def test_count_fields_constant_matches_payload_fields():
    assert COUNT_FIELDS == (
        "publication_count",
        "scie_publication_count",
        "intellectual_property_count",
        "research_project_count",
        "researcher_assessor_activity_count",
    )


def test_chunk_payload_minimal_defaults():
    payload = ChunkPayload.model_validate(_minimal_payload())

    assert payload.researcher_id == "100"
    assert payload.researcher_name == ""
    assert payload.doc_type == "paper"
    assert payload.doc_id == ""
    assert payload.chunk_id == "paper_100000045256_c000"
    assert payload.chunk_text == ""
    assert payload.doc_date is None
    assert payload.affiliated_organization is None
    assert payload.highest_degree is None
    assert payload.doc_attrs == {}
    # all counts default to 0
    assert payload.counts() == {field: 0 for field in COUNT_FIELDS}


def test_chunk_payload_coerces_count_strings_blanks_and_none_to_int():
    payload = ChunkPayload.model_validate(
        _minimal_payload(
            publication_count="",
            scie_publication_count=" ",
            intellectual_property_count="2",
            research_project_count=None,
            researcher_assessor_activity_count="NONE",
        )
    )

    assert payload.publication_count == 0
    assert payload.scie_publication_count == 0
    assert payload.intellectual_property_count == 2
    assert payload.research_project_count == 0
    # "NONE" is not numeric → coerces to 0
    assert payload.researcher_assessor_activity_count == 0


def test_chunk_payload_coerces_float_and_float_string_counts():
    payload = ChunkPayload.model_validate(
        _minimal_payload(
            publication_count=3.0,
            scie_publication_count="4.0",
        )
    )

    assert payload.publication_count == 3
    assert payload.scie_publication_count == 4


def test_chunk_payload_counts_returns_all_five_fields():
    payload = ChunkPayload.model_validate(
        _minimal_payload(
            publication_count=1,
            scie_publication_count=2,
            intellectual_property_count=3,
            research_project_count=4,
            researcher_assessor_activity_count=5,
        )
    )

    assert payload.counts() == {
        "publication_count": 1,
        "scie_publication_count": 2,
        "intellectual_property_count": 3,
        "research_project_count": 4,
        "researcher_assessor_activity_count": 5,
    }


@pytest.mark.parametrize("raw", ["NONE", "none", "", "  ", "없음", "-", "null", "n/a"])
def test_chunk_payload_doc_date_sentinels_become_none(raw):
    payload = ChunkPayload.model_validate(_minimal_payload(doc_date=raw))
    assert payload.doc_date is None


def test_chunk_payload_keeps_real_doc_date_and_optional_strings():
    payload = ChunkPayload.model_validate(
        _minimal_payload(
            doc_date="2021-05-01",
            affiliated_organization="  KAIST  ",
            highest_degree="박사",
        )
    )

    assert payload.doc_date == "2021-05-01"
    # clean_value strips surrounding whitespace
    assert payload.affiliated_organization == "KAIST"
    assert payload.highest_degree == "박사"


def test_chunk_payload_optional_str_sentinels_become_none():
    payload = ChunkPayload.model_validate(
        _minimal_payload(affiliated_organization="NONE", highest_degree="")
    )
    assert payload.affiliated_organization is None
    assert payload.highest_degree is None


def test_chunk_payload_ignores_unknown_extra_keys():
    payload = ChunkPayload.model_validate(
        _minimal_payload(some_unknown_key="x", legacy_nested={"a": 1})
    )
    # extra="ignore" → the field simply does not exist
    assert not hasattr(payload, "some_unknown_key")
    dumped = payload.model_dump()
    assert "some_unknown_key" not in dumped
    assert "legacy_nested" not in dumped


def test_chunk_payload_doc_attrs_passthrough():
    attrs = {"main_language_title": "Deep Nets", "indexing_database": "SCIE"}
    payload = ChunkPayload.model_validate(_minimal_payload(doc_attrs=attrs))
    assert payload.doc_attrs == attrs


def test_chunk_payload_requires_core_identifiers():
    with pytest.raises(ValidationError):
        ChunkPayload.model_validate({"doc_type": "paper", "chunk_id": "paper_1_c000"})
    with pytest.raises(ValidationError):
        ChunkPayload.model_validate({"researcher_id": "1", "chunk_id": "paper_1_c000"})
    with pytest.raises(ValidationError):
        ChunkPayload.model_validate({"researcher_id": "1", "doc_type": "paper"})


# ---------------------------------------------------------------------------
# ChunkHit — score + flat payload, proxy properties
# ---------------------------------------------------------------------------


def test_chunk_hit_proxy_properties():
    payload = ChunkPayload.model_validate(
        _minimal_payload(researcher_id="42", doc_type="patent", chunk_id="patent_9_c001")
    )
    hit = ChunkHit(score=0.83, payload=payload)

    assert hit.score == 0.83
    assert hit.chunk_id == "patent_9_c001"
    assert hit.doc_type == "patent"
    assert hit.researcher_id == "42"


def test_chunk_hit_default_score():
    payload = ChunkPayload.model_validate(_minimal_payload())
    hit = ChunkHit(payload=payload)
    assert hit.score == 0.0


# ---------------------------------------------------------------------------
# ResearcherCandidate — doc_types_present (ordered, dedup) + chunks_of
# ---------------------------------------------------------------------------


def _hit(doc_type, chunk_id, score=0.0, researcher_id="7"):
    payload = ChunkPayload.model_validate(
        _minimal_payload(researcher_id=researcher_id, doc_type=doc_type, chunk_id=chunk_id)
    )
    return ChunkHit(score=score, payload=payload)


def test_researcher_candidate_doc_types_present_is_ordered_and_deduped():
    candidate = ResearcherCandidate(
        researcher_id="7",
        researcher_name="Kim",
        counts={"publication_count": 3},
        chunks=[
            _hit("paper", "paper_1_c000"),
            _hit("patent", "patent_2_c000"),
            _hit("paper", "paper_1_c001"),
            _hit("project", "project_3_c000"),
        ],
    )

    assert candidate.doc_types_present == ["paper", "patent", "project"]


def test_researcher_candidate_chunks_of_filters_by_doc_type():
    candidate = ResearcherCandidate(
        researcher_id="7",
        chunks=[
            _hit("paper", "paper_1_c000"),
            _hit("paper", "paper_1_c001"),
            _hit("patent", "patent_2_c000"),
        ],
    )

    paper_chunks = candidate.chunks_of("paper")
    assert [h.chunk_id for h in paper_chunks] == ["paper_1_c000", "paper_1_c001"]
    assert candidate.chunks_of("patent")[0].chunk_id == "patent_2_c000"
    assert candidate.chunks_of("specialty") == []


def test_researcher_candidate_defaults():
    candidate = ResearcherCandidate(researcher_id="7")
    assert candidate.researcher_name == ""
    assert candidate.affiliated_organization is None
    assert candidate.highest_degree is None
    assert candidate.counts == {}
    assert candidate.group_score == 0.0
    assert candidate.rank_score == 0.0
    assert candidate.chunks == []
    assert candidate.doc_types_present == []


# ---------------------------------------------------------------------------
# ChunkEvidence — unified display/grounding unit
# ---------------------------------------------------------------------------


def test_chunk_evidence_defaults_and_fields():
    evidence = ChunkEvidence(chunk_id="paper_1_c000", doc_type="paper")
    assert evidence.title is None
    assert evidence.date is None
    assert evidence.snippet == ""
    assert evidence.doc_attrs == {}
    assert evidence.score == 0.0

    full = ChunkEvidence(
        chunk_id="patent_2_c000",
        doc_type="patent",
        title="Some Patent",
        date="2020-01-01",
        snippet="abstract...",
        doc_attrs={"intellectual_property_title": "Some Patent"},
        score=0.5,
    )
    assert full.title == "Some Patent"
    assert full.date == "2020-01-01"
    assert full.score == 0.5


# ---------------------------------------------------------------------------
# EvidenceItem — exactly the 5 doc_types + synthetic 'profile'
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_type",
    ["paper", "patent", "project", "assessor_activity", "specialty", "profile"],
)
def test_evidence_item_accepts_canonical_type(raw_type):
    item = EvidenceItem(type=raw_type, title="Test Title")
    assert item.type == raw_type


def test_evidence_item_optional_fields():
    item = EvidenceItem(type="paper", title="T")
    assert item.date is None
    assert item.detail is None
    assert item.chunk_id is None

    item2 = EvidenceItem(
        type="paper",
        title="T",
        date="2021",
        detail="detail text",
        chunk_id="paper_1_c000",
    )
    assert item2.chunk_id == "paper_1_c000"


def test_evidence_item_rejects_unknown_type():
    with pytest.raises(ValidationError):
        EvidenceItem(type="unknown_type", title="Test")


@pytest.mark.parametrize("legacy_type", ["pjt", "art", "pat", "PJT", "ART", "PAT"])
def test_evidence_item_rejects_legacy_branch_alias_type(legacy_type):
    with pytest.raises(ValidationError):
        EvidenceItem(type=legacy_type, title="Test")


# ---------------------------------------------------------------------------
# CandidateCard — evidence_by_type, doc_types_present, evidence_of, all_evidence
# ---------------------------------------------------------------------------


def _build_card():
    return CandidateCard(
        expert_id="7",
        name="Kim",
        organization="KAIST",
        degree="박사",
        counts={"article_cnt": 3, "patent_cnt": 1},
        evidence_by_type={
            "paper": [
                ChunkEvidence(chunk_id="paper_1_c000", doc_type="paper", title="P1"),
                ChunkEvidence(chunk_id="paper_1_c001", doc_type="paper", title="P2"),
            ],
            "patent": [
                ChunkEvidence(chunk_id="patent_2_c000", doc_type="patent", title="IP1"),
            ],
            "project": [],
        },
        matched_filter_summary=["org=KAIST"],
        shortlist_score=1.5,
        rank_score=2.0,
    )


def test_candidate_card_doc_types_present_excludes_empty_buckets():
    card = _build_card()
    # 'project' bucket is empty → excluded
    assert set(card.doc_types_present) == {"paper", "patent"}
    assert "project" not in card.doc_types_present


def test_candidate_card_evidence_of():
    card = _build_card()
    papers = card.evidence_of("paper")
    assert [e.chunk_id for e in papers] == ["paper_1_c000", "paper_1_c001"]
    assert card.evidence_of("project") == []
    # absent doc_type returns empty list, not KeyError
    assert card.evidence_of("specialty") == []


def test_candidate_card_all_evidence_flattens_all_buckets():
    card = _build_card()
    chunk_ids = sorted(e.chunk_id for e in card.all_evidence())
    assert chunk_ids == ["paper_1_c000", "paper_1_c001", "patent_2_c000"]


def test_candidate_card_defaults():
    card = CandidateCard(expert_id="1", name="Lee")
    assert card.organization is None
    assert card.degree is None
    assert card.counts == {}
    assert card.evidence_by_type == {}
    assert card.matched_filter_summary == []
    assert card.risks == []
    assert card.data_gaps == []
    assert card.shortlist_score == 0.0
    assert card.rank_score == 0.0
    assert card.doc_types_present == []
    assert card.all_evidence() == []


# ---------------------------------------------------------------------------
# chunk_view helpers (pure derivation)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ("NONE", None),
        ("none", None),
        ("", None),
        ("  ", None),
        ("없음", None),
        ("  KAIST  ", "KAIST"),
        (5, 5),
    ],
)
def test_clean_value(raw, expected):
    assert clean_value(raw) == expected


def test_normalize_doc_date():
    assert normalize_doc_date("2021-05-01") == "2021-05-01"
    assert normalize_doc_date("NONE") is None
    assert normalize_doc_date(None) is None
    # non-string clean values are not dates
    assert normalize_doc_date(2021) is None


def test_parse_year():
    assert parse_year("2021-05-01") == 2021
    assert parse_year("1999") == 1999
    assert parse_year("NONE") is None
    assert parse_year("abcd-01") is None
    assert parse_year(None) is None


def test_derive_title_from_doc_attrs_priority():
    # paper prefers main_language_title over journal_name
    title = derive_title(
        "paper",
        {"main_language_title": "Main", "journal_name": "Nature"},
    )
    assert title == "Main"


def test_derive_title_falls_back_to_first_line():
    title = derive_title("paper", {}, fallback="First line\nSecond line")
    assert title == "First line"


def test_derive_title_none_when_no_attrs_no_fallback():
    assert derive_title("specialty", {}) is None


def test_derive_title_joins_list_values():
    title = derive_title("specialty", {"specific_specialty_name": ["AI", "ML"]})
    assert title == "AI, ML"


def test_derive_date_prefers_root_doc_date():
    assert derive_date("paper", "2021-05", {"publication_year_month": "1900-01"}) == "2021-05"


def test_derive_date_falls_back_to_doc_attrs():
    assert derive_date("paper", "NONE", {"publication_year_month": "2019-09"}) == "2019-09"
    assert derive_date("patent", None, {"application_date": "2018-02-02"}) == "2018-02-02"


def test_derive_date_parses_project_period_start():
    assert derive_date("project", None, {"project_period": "2017-01-01 ~ 2019-12-31"}) == "2017-01-01"


def test_derive_date_none_when_unavailable():
    assert derive_date("specialty", None, {}) is None


def test_snippet_trims_and_collapses_whitespace():
    assert snippet("  hello   world\n\tagain  ") == "hello world again"
    assert snippet(None) == ""
    assert snippet("a" * 500, limit=10) == "a" * 10
