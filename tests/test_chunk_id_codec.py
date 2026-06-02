"""flat chunk_id 코덱 + doc_type/family 상수 검증 (v2.1 flat contract).

검증 항목:
  ① doc_type 5종 단일 출처 (paper/patent/project/assessor_activity/specialty)
  ② build_doc_id / build_chunk_id 결정적 생성 + 샘플 chunk_id 일치
  ③ parse_chunk_id 라운드트립 (assessor_activity 내부 underscore 포함)
  ④ unknown doc_type / 형식 위반 → ValueError
  ⑤ doc_type_of_chunk_id (chunk_id 및 doc_id 모두 지원, 위반 시 None)
  ⑥ DOC_TYPE_TO_FAMILY 커버리지·중복 없음 + FAMILY_EVIDENCE_CAP

근거: apps/search/doc_types.py (authoritative flat codec).
"""
from __future__ import annotations

import pytest

from apps.search.doc_types import (
    CHUNK_INDEX_PAD,
    DOC_TYPE_TO_FAMILY,
    DOC_TYPES,
    FAMILY_EVIDENCE_CAP,
    ChunkIdParts,
    DocType,
    Family,
    build_chunk_id,
    build_doc_id,
    doc_type_of_chunk_id,
    parse_chunk_id,
)

# 샘플 flat payload의 실제 chunk_id (chunk_index=0 → c000)
SAMPLE_CHUNK_IDS: dict[str, str] = {
    "paper": "paper_100000045256_c000",
    "patent": "patent_100000045256_c000",
    "project": "project_100000045256_c000",
    "assessor_activity": "assessor_activity_100000045256_c000",
    "specialty": "specialty_100000045256_c000",
}


# ---------------------------------------------------------------------------
# ① doc_type 5종 단일 출처
# ---------------------------------------------------------------------------
def test_doc_types_are_exactly_five_flat_types():
    assert len(DOC_TYPES) == 5
    assert set(DOC_TYPES) == {
        "paper",
        "patent",
        "project",
        "assessor_activity",
        "specialty",
    }
    # DOC_TYPES는 DocType enum 값 순서를 그대로 따른다.
    assert DOC_TYPES == tuple(dt.value for dt in DocType)


def test_sample_chunk_ids_cover_all_five_doc_types():
    assert set(SAMPLE_CHUNK_IDS) == set(DOC_TYPES)


# ---------------------------------------------------------------------------
# ② build_doc_id / build_chunk_id
# ---------------------------------------------------------------------------
def test_build_doc_id_has_no_chunk_suffix():
    assert build_doc_id("paper", "100000045256") == "paper_100000045256"
    # 숫자 doc_num을 int로 넣어도 동일.
    assert build_doc_id("patent", 100000045256) == "patent_100000045256"


@pytest.mark.parametrize("doc_type,expected", sorted(SAMPLE_CHUNK_IDS.items()))
def test_build_chunk_id_matches_sample(doc_type: str, expected: str):
    # chunk_index는 CHUNK_INDEX_PAD(=3) 자리 zero-pad.
    assert build_chunk_id(doc_type, "100000045256", 0) == expected
    # doc_num을 int로 넣어도 동일 결과(순수/결정적).
    assert build_chunk_id(doc_type, 100000045256, 0) == expected


def test_build_chunk_id_zero_pads_index():
    assert CHUNK_INDEX_PAD == 3
    assert build_chunk_id("paper", "42", 7) == "paper_42_c007"
    assert build_chunk_id("paper", "42", 123) == "paper_42_c123"
    # pad 자릿수를 넘어가는 index도 자르지 않고 그대로 확장.
    assert build_chunk_id("paper", "42", 4567) == "paper_42_c4567"


def test_build_chunk_id_is_deterministic():
    assert build_chunk_id("project", "999", 1) == build_chunk_id("project", "999", 1)


# ---------------------------------------------------------------------------
# ③ parse_chunk_id 라운드트립
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_type,chunk_id", sorted(SAMPLE_CHUNK_IDS.items()))
def test_parse_roundtrip(doc_type: str, chunk_id: str):
    parts = parse_chunk_id(chunk_id)
    assert isinstance(parts, ChunkIdParts)
    assert parts.doc_type == doc_type
    assert parts.doc_num == "100000045256"
    assert parts.chunk_index == 0
    assert parts.doc_id == f"{doc_type}_100000045256"
    rebuilt = build_chunk_id(parts.doc_type, parts.doc_num, parts.chunk_index)
    assert rebuilt == chunk_id


def test_parse_handles_assessor_activity_internal_underscore():
    """doc_type 자체에 '_'가 있어도(assessor_activity) 모호성 없이 파싱."""
    chunk_id = build_chunk_id("assessor_activity", "100000045256", 3)
    assert chunk_id == "assessor_activity_100000045256_c003"
    parts = parse_chunk_id(chunk_id)
    assert parts.doc_type == "assessor_activity"
    assert parts.doc_num == "100000045256"
    assert parts.chunk_index == 3
    assert parts.doc_id == "assessor_activity_100000045256"
    assert (
        build_chunk_id(parts.doc_type, parts.doc_num, parts.chunk_index) == chunk_id
    )


def test_parse_preserves_multi_digit_index():
    parts = parse_chunk_id("paper_5_c012")
    assert parts.doc_type == "paper"
    assert parts.doc_num == "5"
    assert parts.chunk_index == 12


# ---------------------------------------------------------------------------
# ④ unknown doc_type / 형식 위반 → ValueError
# ---------------------------------------------------------------------------
def test_build_doc_id_rejects_unknown_doc_type():
    with pytest.raises(ValueError):
        build_doc_id("not_a_doc_type", "1")


def test_build_chunk_id_rejects_unknown_doc_type():
    with pytest.raises(ValueError):
        build_chunk_id("not_a_doc_type", "1", 0)


def test_build_doc_id_accepts_non_numeric_body():
    # 실데이터: specialty/assessor_activity는 연구자ID형 doc_id 본문을 쓴다(숫자 제한 없음).
    assert build_doc_id("specialty", "M1006328") == "specialty_M1006328"
    assert build_doc_id("assessor_activity", "M77") == "assessor_activity_M77"
    assert build_doc_id("paper", "100000045256") == "paper_100000045256"
    with pytest.raises(ValueError):
        build_doc_id("paper", "   ")  # 빈 본문은 거부


@pytest.mark.parametrize(
    "bad",
    [
        "unknown_100000045256_c000",  # unknown doc_type
        "paper_100000045256",  # missing _c suffix
        "paper_c000",  # missing doc_num
        "paper_100_cXY",  # non-numeric chunk index
        "PAPER_100_c000",  # uppercase doc_type not allowed
        "not-a-chunk-id",
        "",
    ],
)
def test_parse_rejects_malformed(bad: str):
    with pytest.raises(ValueError):
        parse_chunk_id(bad)


# ---------------------------------------------------------------------------
# ⑤ doc_type_of_chunk_id
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc_type,chunk_id", sorted(SAMPLE_CHUNK_IDS.items()))
def test_doc_type_of_chunk_id_from_chunk_id(doc_type: str, chunk_id: str):
    assert doc_type_of_chunk_id(chunk_id) == doc_type


@pytest.mark.parametrize("doc_type", sorted(DOC_TYPES))
def test_doc_type_of_chunk_id_from_doc_id(doc_type: str):
    doc_id = build_doc_id(doc_type, "100000045256")
    assert doc_type_of_chunk_id(doc_id) == doc_type


@pytest.mark.parametrize(
    "bad",
    [
        "unknown_1_c000",
        "not-a-chunk-id",
        "",
    ],
)
def test_doc_type_of_chunk_id_returns_none_for_malformed(bad: str):
    assert doc_type_of_chunk_id(bad) is None


# ---------------------------------------------------------------------------
# ⑥ family 매핑 + evidence cap
# ---------------------------------------------------------------------------
def test_family_mapping_covers_all_doc_types_without_gaps():
    assert set(DOC_TYPE_TO_FAMILY) == set(DOC_TYPES)
    assert set(DOC_TYPE_TO_FAMILY.values()) <= set(Family)
    # family 분포 (flat contract):
    #   achievement = {paper, patent, project}
    #   assessment  = {assessor_activity}
    #   expertise   = {specialty}
    assert DOC_TYPE_TO_FAMILY["paper"] == Family.ACHIEVEMENT
    assert DOC_TYPE_TO_FAMILY["patent"] == Family.ACHIEVEMENT
    assert DOC_TYPE_TO_FAMILY["project"] == Family.ACHIEVEMENT
    assert DOC_TYPE_TO_FAMILY["assessor_activity"] == Family.ASSESSMENT
    assert DOC_TYPE_TO_FAMILY["specialty"] == Family.EXPERTISE


def test_identity_family_has_no_doc_type():
    """identity는 합성 family로, 어떤 doc_type에도 매핑되지 않는다."""
    assert Family.IDENTITY not in set(DOC_TYPE_TO_FAMILY.values())
    # 그러나 4 family enum에는 존재한다.
    assert Family.IDENTITY in set(Family)


def test_family_evidence_cap_values():
    assert FAMILY_EVIDENCE_CAP == {
        Family.ACHIEVEMENT: 10,
        Family.ASSESSMENT: 6,
        Family.EXPERTISE: 6,
        Family.IDENTITY: 1,
    }
    # 4 family 전부를 덮는다.
    assert set(FAMILY_EVIDENCE_CAP) == set(Family)
