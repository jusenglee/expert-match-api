"""WO-0 산출물 검증: apps.search.doc_types 의 chunk_id 코덱 + doc_type/family 상수.

검증 항목:
  ① 11 doc_type build 결과 == 샘플 payload(전체 샘플 payload (도메인별).txt)의 실제 chunk_id
  ② parse 라운드트립 (build(*parse(x)) == x)
  ③ unknown PREFIX / 형식 위반 → ValueError
  ④ DOC_TYPE_TO_FAMILY 커버리지·중복 없음 + FAMILY_EVIDENCE_CAP
  ⑤ DOC_TYPE_TO_PREFIX 값 전역 유일
"""
from __future__ import annotations

import pytest

from apps.search.doc_types import (
    DOC_TYPE_TO_FAMILY,
    DOC_TYPE_TO_PREFIX,
    DOC_TYPES,
    FAMILY_EVIDENCE_CAP,
    Family,
    build_chunk_id,
    build_doc_id,
    parse_chunk_id,
)

# 샘플 payload의 실제 chunk_id (researcher_id=M1006328, seq=0001, chunk_index=0)
SAMPLE_CHUNK_IDS: dict[str, str] = {
    "publication": "PUB_M1006328_0001_c0",
    "intellectual_property": "IP_M1006328_0001_c0",
    "research_project": "PJT_M1006328_0001_c0",
    "researcher_assessor": "RAS_M1006328_0001_c0",
    "expert_assessor": "EAS_M1006328_0001_c0",
    "researcher_tech": "RTC_M1006328_0001_c0",
    "expert_tech": "ETC_M1006328_0001_c0",
    "researcher_core": "RCO_M1006328_0001_c0",
    "researcher_major": "RMJ_M1006328_0001_c0",
    "expert_specific": "ESP_M1006328_0001_c0",
    "profile": "PRF_M1006328_0001_c0",
}


def test_sample_chunk_ids_cover_all_eleven_doc_types():
    assert set(SAMPLE_CHUNK_IDS) == set(DOC_TYPES)
    assert len(DOC_TYPES) == 11


@pytest.mark.parametrize("doc_type,expected", sorted(SAMPLE_CHUNK_IDS.items()))
def test_build_chunk_id_matches_sample(doc_type: str, expected: str):
    assert build_chunk_id(doc_type, "M1006328", "0001", 0) == expected
    # seq는 int로 넣어도 4자리 zero-pad 정규화
    assert build_chunk_id(doc_type, "M1006328", 1, 0) == expected


@pytest.mark.parametrize("doc_type,chunk_id", sorted(SAMPLE_CHUNK_IDS.items()))
def test_parse_roundtrip(doc_type: str, chunk_id: str):
    parts = parse_chunk_id(chunk_id)
    assert parts.doc_type == doc_type
    assert parts.researcher_id == "M1006328"
    assert parts.seq == "0001"
    assert parts.chunk_index == 0
    rebuilt = build_chunk_id(
        parts.doc_type, parts.researcher_id, parts.seq, parts.chunk_index
    )
    assert rebuilt == chunk_id


def test_build_doc_id_has_no_chunk_suffix():
    assert build_doc_id("publication", "M1006328", "0001") == "PUB_M1006328_0001"
    assert build_doc_id("intellectual_property", "M1006328", 1) == "IP_M1006328_0001"


def test_parse_handles_researcher_id_with_underscores():
    chunk_id = build_chunk_id("publication", "ORG_123_X", "0007", 2)
    parts = parse_chunk_id(chunk_id)
    assert parts.researcher_id == "ORG_123_X"
    assert parts.seq == "0007"
    assert parts.chunk_index == 2
    assert build_chunk_id(parts.doc_type, parts.researcher_id, parts.seq, parts.chunk_index) == chunk_id


def test_build_rejects_unknown_doc_type():
    with pytest.raises(ValueError):
        build_chunk_id("not_a_doc_type", "M1", "0001", 0)


@pytest.mark.parametrize(
    "bad",
    [
        "ZZZ_M1006328_0001_c0",  # unknown prefix
        "PUB_M1006328_0001",  # missing _c suffix
        "PUB_M1006328_c0",  # missing seq
        "not-a-chunk-id",
        "",
    ],
)
def test_parse_rejects_malformed(bad: str):
    with pytest.raises(ValueError):
        parse_chunk_id(bad)


def test_family_mapping_covers_all_doc_types_without_gaps():
    assert set(DOC_TYPE_TO_FAMILY) == set(DOC_TYPES)
    assert set(DOC_TYPE_TO_FAMILY.values()) <= set(Family)
    # family 분포 (DATA_MODEL §4)
    assert DOC_TYPE_TO_FAMILY["profile"] == Family.IDENTITY
    assert DOC_TYPE_TO_FAMILY["publication"] == Family.ACHIEVEMENT
    assert DOC_TYPE_TO_FAMILY["researcher_assessor"] == Family.ASSESSMENT
    assert DOC_TYPE_TO_FAMILY["expert_assessor"] == Family.ASSESSMENT
    assert DOC_TYPE_TO_FAMILY["expert_specific"] == Family.EXPERTISE


def test_family_evidence_cap_values():
    assert FAMILY_EVIDENCE_CAP == {
        Family.ACHIEVEMENT: 10,
        Family.ASSESSMENT: 6,
        Family.EXPERTISE: 6,
        Family.IDENTITY: 1,
    }


def test_prefix_values_are_globally_unique():
    prefixes = list(DOC_TYPE_TO_PREFIX.values())
    assert len(prefixes) == len(set(prefixes))
    assert len(prefixes) == 11
