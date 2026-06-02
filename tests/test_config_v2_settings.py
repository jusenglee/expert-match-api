"""flat 계약(v2.1): apps.core.config.Settings default 검증."""
from __future__ import annotations

from apps.core.config import Settings


def test_v2_settings_defaults():
    s = Settings()
    # 집계/리랭커 HARD 기본값
    assert s.doc_type_priors is None          # 미설정 = equal (ADR 0005 §2)
    assert s.candidate_reranker == "off"      # 후보 리랭커 기본 OFF (ADR 0005 §3)
    assert s.retrieval_doc_types is None      # 미설정 = 전체 5종 (ADR 0003)
    assert s.doc_type_chunk_cap == 3
    assert s.evidence_family_cap == {
        "achievement": 10,
        "assessment": 6,
        "expertise": 6,
        "identity": 1,
    }
    # evidence 리랭커 / cross-encoder 설정
    assert s.evidence_reranker_backend == "lexical"
    assert s.cross_encoder_model_name is None
    assert s.ce_relevance_floor == 0.30
    assert s.ce_pregate_per_type == 20
    assert s.ce_max_pairs_per_request == 256
    assert s.ce_top_n_per_type == 5


def test_v2_settings_env_override(monkeypatch):
    monkeypatch.setenv("NTIS_CANDIDATE_RERANKER", "band")
    monkeypatch.setenv("NTIS_DOC_TYPE_CHUNK_CAP", "5")
    s = Settings()
    assert s.candidate_reranker == "band"
    assert s.doc_type_chunk_cap == 5


def test_flat_collection_and_pipeline_defaults():
    """flat 단일 벡터 컬렉션 default 및 검색 파이프라인 한계."""
    s = Settings()
    # flat chunk 모델 단일 벡터 컬렉션 (레거시 researcher_recommend_proto 폐기)
    assert s.qdrant_collection_name == "researcher_recommend_v1"
    assert s.embedding_vector_size == 1024
    assert s.prefetch_limit == 256
    assert s.group_size == 10
