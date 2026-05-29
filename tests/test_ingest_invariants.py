"""WO-A: §6 적재 불변식 + GOLDEN_TESTS 시나리오 1·7 + meta 집계 검증."""
from __future__ import annotations

from apps.domain.models import ChunkPayload, ResearcherMeta
from apps.ingest.converters import build_chunk
from apps.ingest.embedder import ChunkEmbedder
from apps.ingest.meta_aggregator import (
    build_researcher_meta,
    count_records_by_doc_type,
    meta_discrepancies,
)
from apps.ingest.validate_chunks import validate_chunks
from apps.search.doc_types import DocType

from test_ingest_converters import SAMPLE_META, SAMPLES, build_all_sample_chunks

_PUB_ATTRS = SAMPLES[0]["attrs"]
_PJT_ATTRS = SAMPLES[2]["attrs"]


def _codes(report):
    return {v.code for v in report.violations}


def _chunk(**overrides) -> ChunkPayload:
    base = dict(
        researcher_id="M1006328",
        researcher_name="홍길동",
        doc_type="publication",
        doc_id="PUB_M1006328_0001",
        chunk_id="PUB_M1006328_0001_c0",
        chunk_text="논문명: x",
        chunk_text_len=len("논문명: x"),
        researcher_meta=SAMPLE_META,
        event_date="2024-05-01",
        event_year=2024,
        tags=[],
        domain_attrs={},
    )
    base.update(overrides)
    base["chunk_text_len"] = len(base["chunk_text"])
    return ChunkPayload(**base)


# --- 정상 케이스 -----------------------------------------------------------
def test_all_sample_chunks_pass_invariants():
    report = validate_chunks(build_all_sample_chunks())
    assert report.ok, report.render()
    assert report.checked == 11


def test_golden_scenario_7_chunk_per_point_same_meta():
    """publication 3 + research_project 2 → 5 Point, 모두 동일 researcher_id/researcher_meta."""
    chunks = [
        build_chunk(
            DocType.PUBLICATION,
            researcher_id="M1006328",
            researcher_name="홍길동",
            doc_seq=f"{i:04d}",
            domain_attrs=_PUB_ATTRS,
            researcher_meta=SAMPLE_META,
        )
        for i in range(1, 4)
    ] + [
        build_chunk(
            DocType.RESEARCH_PROJECT,
            researcher_id="M1006328",
            researcher_name="홍길동",
            doc_seq=f"{i:04d}",
            domain_attrs=_PJT_ATTRS,
            researcher_meta=SAMPLE_META,
        )
        for i in range(1, 3)
    ]

    assert len(chunks) == 5
    assert len({c.chunk_id for c in chunks}) == 5  # 전역 유일
    assert {c.researcher_id for c in chunks} == {"M1006328"}
    assert all(c.researcher_meta == SAMPLE_META for c in chunks)

    report = validate_chunks(chunks)
    assert report.ok, report.render()


# --- 고의 위반 감지 --------------------------------------------------------
def test_detects_duplicate_chunk_id():
    report = validate_chunks([_chunk(), _chunk()])  # 동일 chunk_id 2개
    assert "dup_chunk_id" in _codes(report)


def test_detects_meta_mismatch_within_researcher():
    other_meta = ResearcherMeta(publication_count=99)
    chunks = [
        _chunk(chunk_id="PUB_M1006328_0001_c0"),
        _chunk(chunk_id="PUB_M1006328_0002_c0", researcher_meta=other_meta),
    ]
    assert "meta_mismatch" in _codes(validate_chunks(chunks))


def test_detects_name_mismatch_within_researcher():
    chunks = [
        _chunk(chunk_id="PUB_M1006328_0001_c0"),
        _chunk(chunk_id="PUB_M1006328_0002_c0", researcher_name="다른이름"),
    ]
    assert "meta_name_mismatch" in _codes(validate_chunks(chunks))


def test_detects_event_year_mismatch():
    report = validate_chunks([_chunk(event_date="2024-05-01", event_year=2023)])
    assert "event_year_mismatch" in _codes(report)


def test_detects_partial_null_event():
    report = validate_chunks([_chunk(event_date="2024-05-01", event_year=None)])
    assert "event_partial_null" in _codes(report)


def test_dateless_both_null_is_valid():
    report = validate_chunks([_chunk(event_date=None, event_year=None, doc_type="profile",
                                     chunk_id="PRF_M1006328_0001_c0", doc_id="PRF_M1006328_0001")])
    assert report.ok, report.render()


def test_detects_uppercase_tag():
    report = validate_chunks([_chunk(tags=["RAG"])])
    assert "tag_not_normalized" in _codes(report)


def test_detects_bad_doc_type():
    report = validate_chunks([_chunk(doc_type="not_a_doc_type")])
    assert "bad_doc_type" in _codes(report)


def test_detects_role_action_leak():
    report = validate_chunks([_chunk(chunk_text="이 사람을 평가위원으로 추천합니다")])
    assert "role_action_leak" in _codes(report)


def test_detects_v1x_leak_phrase():
    report = validate_chunks([_chunk(chunk_text="심사평가위원 활동: 5건")])
    assert "role_action_leak" in _codes(report)


# --- 임베딩 입력 동일성 (§6-6) ---------------------------------------------
class _FakeDense:
    model_name = "fake-dense"
    vector_size = 4

    def embed(self, text: str) -> list[float]:
        return [float(len(text))] * self.vector_size


class _FakeSparse:
    model_name = "fake-sparse"

    def embed(self, text: str) -> dict[int, float]:
        return {1: float(len(text))}


def test_embedding_hash_match_and_mismatch():
    chunks = build_all_sample_chunks()
    embedder = ChunkEmbedder(dense_encoder=_FakeDense(), sparse_encoder=_FakeSparse())
    embeddings = embedder.embed_chunks(chunks)
    hashes = {e.chunk_id: e.input_sha256 for e in embeddings}

    # dense·sparse가 동일 chunk_text에서 생성됨 → 통과
    ok_report = validate_chunks(chunks, embedding_hashes=hashes)
    assert ok_report.ok, ok_report.render()

    # 한 chunk의 임베딩 입력 해시를 변조 → 위반
    tampered = dict(hashes)
    tampered[chunks[0].chunk_id] = "deadbeef"
    bad_report = validate_chunks(chunks, embedding_hashes=tampered)
    assert "embedding_input_mismatch" in _codes(bad_report)


def test_embedding_uses_same_chunk_text_for_dense_and_sparse():
    chunk = build_all_sample_chunks()[0]
    embedder = ChunkEmbedder(dense_encoder=_FakeDense(), sparse_encoder=_FakeSparse())
    emb = embedder.embed_chunk(chunk)
    # 두 인코더 모두 len(chunk_text) 기반 값 → 동일 입력 확인
    assert emb.dense_e5i[0] == float(len(chunk.chunk_text))
    assert emb.sparse_splade[1] == float(len(chunk.chunk_text))


# --- researcher_meta 집계 --------------------------------------------------
def test_meta_aggregator_prefers_profile_counts():
    profile = {
        "highest_degree": "박사",
        "publication_count": 15,
        "scie_publication_count": 5,
        "intellectual_property_count": 3,
        "research_project_count": 5,
        "researcher_assessor_count": 5,
        "expert_assessor_count": 2,
    }
    meta = build_researcher_meta(
        profile_attrs=profile,
        affiliated_organization="주식회사 미소테크",
        record_counts={"publication_count": 3},  # profile이 우선
    )
    assert meta == SAMPLE_META


def test_meta_aggregator_falls_back_to_record_counts():
    counts = count_records_by_doc_type(
        [DocType.PUBLICATION, DocType.PUBLICATION, DocType.RESEARCH_PROJECT,
         DocType.RESEARCHER_ASSESSOR, DocType.EXPERT_ASSESSOR, DocType.EXPERT_ASSESSOR]
    )
    meta = build_researcher_meta(
        profile_attrs=None, affiliated_organization="X", record_counts=counts
    )
    assert meta.publication_count == 2
    assert meta.research_project_count == 1
    assert meta.researcher_assessor_count == 1
    assert meta.expert_assessor_count == 2
    assert meta.affiliated_organization == "X"


def test_meta_discrepancies_flags_profile_vs_records():
    diff = meta_discrepancies({"publication_count": 15}, {"publication_count": 3})
    assert diff == {"publication_count": {"profile": 15, "records": 3}}
    assert meta_discrepancies({"publication_count": 5}, {"publication_count": 5}) == {}
