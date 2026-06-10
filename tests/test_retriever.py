"""QdrantHybridRetriever 테스트 (flat chunk 모델, v2.1 — 멀티뷰 flat 검색 + 관련도 재점수).

검증 계약:
- search()는 view별 flat query_points를 1콜씩(dense_full + sparse_raw/focus + concept:<id>).
  dense는 planner semantic_query를 우선 쓰고, sparse_raw는 원문을 보존한다.
- 결과는 payload.chunk_id 기준 병합(point_id 아님). chunk 점수 = Σ view_weight × normalized rank.
- chunk는 검색 후 concept 태깅(view hit ∪ alias). 후보 점수 = capped evidence score(relevance).
- required_concepts 전부 충족(joint/separate)은 main tier, 부분충족(partial)은 fallback tier.
- query_points_groups는 search_grouped_diagnostic()로 분리(진단/AB 전용).
네트워크/실모델 금지 — FakeFlatClient + Recording 인코더 스텁만.
"""

import asyncio
from types import SimpleNamespace

from apps.core.cache import RetrievalResultCache
from apps.core.config import Settings
from apps.domain.models import ConceptSpec, PlannerOutput, ResearcherCandidate
from apps.search.query_builder import CompiledQueries, QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import SparseRuntimeConfig

# concept view 텍스트 = planner concept_specs의 query_terms join (registry 제거 후 명시 주입).
AI_VIEW_TEXT = "인공지능 AI 머신러닝 딥러닝 신경망"
SEMI_VIEW_TEXT = "반도체 시스템반도체 반도체소자"
JOINT_FOCUS_TEXT = "인공지능 반도체"  # sparse_focus = ConceptPlan.focus_query(concept label 중심)

# query_terms join이 AI_VIEW_TEXT / SEMI_VIEW_TEXT와 일치해야 concept view 텍스트가 동일하다.
AI_SPEC = ConceptSpec(
    id="ai", label="인공지능", role="required",
    query_terms=["인공지능", "AI", "머신러닝", "딥러닝", "신경망"],
    evidence_terms=["인공지능", "AI", "머신러닝", "딥러닝", "신경망"],
    weak_terms=["지능형", "스마트", "자동화"],
)
SEMI_SPEC = ConceptSpec(
    id="semiconductor", label="반도체", role="required",
    query_terms=["반도체", "시스템반도체", "반도체소자"],
    evidence_terms=["반도체", "시스템반도체", "반도체소자", "집적회로", "웨이퍼"],
    weak_terms=["지능형", "시스템", "센서", "회로", "소자", "공정"],
)
AI_SEMI_SPECS = [AI_SPEC, SEMI_SPEC]


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeFlatClient:
    """query_points/query_points_groups를 흉내.

    points_by_view: {view_key: [(payload, score), ...]} — view_key = "dense" 또는 ("sparse", text).
    default_points: view_key 미지정 시 모든 view가 반환할 기본 목록.
    """

    def __init__(self, *, default_points=None, points_by_view=None, groups_spec=None) -> None:
        self.default_points = default_points or []
        self.points_by_view = points_by_view or {}
        self.groups_spec = groups_spec or []
        self.calls: list[dict] = []
        self.group_calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        spec = self.points_by_view.get(self._view_key(kwargs), self.default_points)
        points = [
            SimpleNamespace(id=payload.get("chunk_id"), payload=payload, score=score)
            for payload, score in spec
        ]
        return SimpleNamespace(points=points)

    def query_points_groups(self, **kwargs):
        self.group_calls.append(kwargs)
        groups = []
        for rid, hits in self.groups_spec:
            pts = [
                SimpleNamespace(id=payload.get("chunk_id"), payload=payload, score=score)
                for payload, score in hits
            ]
            groups.append(SimpleNamespace(id=rid, hits=pts))
        return SimpleNamespace(groups=groups)

    @staticmethod
    def _view_key(kwargs):
        if kwargs.get("using") == DENSE_VECTOR_NAME:
            return "dense"
        return ("sparse", getattr(kwargs.get("query"), "text", None))


class RecordingDenseEncoder:
    def __init__(self) -> None:
        self.model_name = "hash"
        self.vector_size = 8
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.1] * self.vector_size


def _settings(**overrides) -> Settings:
    base = dict(
        app_env="test",
        strict_runtime_validation=False,
        cache_enabled=False,  # L3 캐시 비활성(테스트 격리)
        embedding_vector_size=8,
        prefetch_limit=64,
        group_size=10,
        retrieval_limit=40,
    )
    base.update(overrides)
    return Settings(**base)


def _chunk_payload(
    researcher_id: str,
    name: str,
    *,
    doc_type: str = "paper",
    chunk_index: int = 0,
    text: str | None = None,
    organization: str | None = None,
    doc_attrs: dict | None = None,
) -> dict:
    digits = "".join(ch for ch in researcher_id if ch.isdigit()) or "0"
    doc_num = f"1000000000{int(digits):02d}"
    return {
        "researcher_id": researcher_id,
        "researcher_name": name,
        "doc_type": doc_type,
        "doc_id": f"{doc_type}_{doc_num}",
        "chunk_id": f"{doc_type}_{doc_num}_c{chunk_index:03d}",
        "chunk_text": text if text is not None else f"{name} {doc_type} chunk",
        "doc_date": None,
        "affiliated_organization": organization,
        "highest_degree": "박사",
        "publication_count": 3,
        "scie_publication_count": 1,
        "intellectual_property_count": 0,
        "research_project_count": 2,
        "researcher_assessor_activity_count": 0,
        "doc_attrs": doc_attrs or {},
    }


def _retriever(client, **settings_overrides) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(
        client=client,
        settings=_settings(**settings_overrides),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )


def _run(retriever, *, query="검색 질의", core=None, **plan_kwargs):
    plan = PlannerOutput(
        intent_summary=query,
        retrieval_core=core or ["키워드"],
        core_keywords=core or ["키워드"],
        **plan_kwargs,
    )
    return asyncio.run(retriever.search(query=query, plan=plan, query_filter=None))


# ---------------------------------------------------------------------------
# _sort_hits
# ---------------------------------------------------------------------------
def test_sort_hits_breaks_ties_by_name_then_researcher_id():
    bravo = ResearcherCandidate(researcher_id="2", researcher_name="Bravo", group_score=0.5)
    alpha = ResearcherCandidate(researcher_id="1", researcher_name="Alpha", group_score=0.5)
    higher = ResearcherCandidate(researcher_id="3", researcher_name="Zulu", group_score=0.9)
    ordered = QdrantHybridRetriever._sort_hits([bravo, alpha, higher])
    assert [c.researcher_id for c in ordered] == ["3", "1", "2"]


# ---------------------------------------------------------------------------
# view shape — one flat query_points per view, semantic dense + raw sparse preserved
# ---------------------------------------------------------------------------
def test_search_runs_one_flat_query_per_view_dense_first():
    raw_query = "인공지능 분야 전문성과 반도체 연구개발 또는 반도체 산업 경험을 가진 연구자"
    dense_encoder = RecordingDenseEncoder()
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Alpha"), 0.9)])
    retriever = QdrantHybridRetriever(
        client=client, settings=_settings(),
        dense_encoder=dense_encoder, query_builder=QueryTextBuilder(),
    )
    result = asyncio.run(
        retriever.search(
            query=raw_query,
            plan=PlannerOutput(
                intent_summary=raw_query,
                retrieval_core=["인공지능", "반도체", "반도체 연구개발", "반도체 산업 경험"],
                core_keywords=["인공지능", "반도체", "반도체 연구개발", "반도체 산업 경험"],
                semantic_query="인공지능 반도체 경험 연구자",
                concept_specs=AI_SEMI_SPECS,
            ),
            query_filter=None,
        )
    )

    # dense는 planner semantic_query를 우선 임베드한다.
    assert dense_encoder.inputs == ["인공지능 반도체 경험 연구자"]
    # dense_full + sparse_raw + sparse_focus + concept:ai + concept:semiconductor = 5 view.
    assert len(client.calls) == 5
    assert client.calls[0]["using"] == DENSE_VECTOR_NAME
    assert all(call["using"] == SPARSE_VECTOR_NAME for call in client.calls[1:])
    sparse_texts = [call["query"].text for call in client.calls[1:]]
    assert sparse_texts == [
        raw_query,
        JOINT_FOCUS_TEXT,
        AI_VIEW_TEXT,
        SEMI_VIEW_TEXT,
    ]
    assert result.query_payload["retrieval_mode"] == "multiview_flat_relevance"
    assert result.query_payload["search_query_plan"]["dense_query"] == "인공지능 반도체 경험 연구자"
    assert result.query_payload["relevance_gate_active_concepts"] == ["ai", "semiconductor"]
    assert isinstance(result.queries, CompiledQueries)


# ---------------------------------------------------------------------------
# chunk_id merge across views
# ---------------------------------------------------------------------------
def test_search_merges_chunks_by_chunk_id_across_views():
    points = [
        (_chunk_payload("M1", "Alpha", chunk_index=0, text="인공지능 반도체 설계"), 0.9),
        (_chunk_payload("M1", "Alpha", chunk_index=1, text="반도체 공정 인공지능 가속기"), 0.8),
    ]
    client = FakeFlatClient(default_points=points)
    result = _run(_retriever(client), query="인공지능 반도체", core=["인공지능", "반도체"], concept_specs=AI_SEMI_SPECS)

    assert len(result.hits) == 1
    candidate = result.hits[0]
    assert candidate.researcher_id == "M1"
    # 같은 chunk가 여러 view에서 회수돼도 chunk_id로 1건 병합(중복 아님).
    assert len(candidate.chunks) == 2
    # dense + sparse view 모두에서 잡혔으므로 sources 다중.
    assert "dense_full" in candidate.chunks[0].sources
    assert any(src.startswith("sparse") or src.startswith("concept:") for src in candidate.chunks[0].sources)
    assert candidate.coverage_type == "joint"  # 두 concept view가 동일 chunk를 잡음
    assert candidate.group_score > 0.0
    assert result.query_payload["merged_chunk_count"] == 2


# ---------------------------------------------------------------------------
# concept tagging + required gate (main vs fallback tier)
# ---------------------------------------------------------------------------
def test_search_partial_coverage_routed_to_fallback_tier():
    # concept:ai view에서만 잡힌 중립 텍스트 chunk → 'ai'만 태깅 → 부분충족(partial).
    neutral = _chunk_payload("M1", "Alpha", text="인공지능 데이터 분석 연구")
    client = FakeFlatClient(points_by_view={("sparse", AI_VIEW_TEXT): [(neutral, 0.5)]})
    result = _run(_retriever(client), query="인공지능 반도체", core=["인공지능", "반도체"], concept_specs=AI_SEMI_SPECS)

    assert len(result.hits) == 1
    candidate = result.hits[0]
    assert candidate.matched_concepts == ["ai"]
    assert candidate.coverage_type == "partial"
    assert result.query_payload["main_count"] == 0
    assert result.query_payload["fallback_count"] == 1


def test_search_main_tier_ranks_above_fallback_tier():
    joint_chunk = _chunk_payload("1", "Full", chunk_index=0, text="인공지능 반도체 설계 연구")
    partial_chunk = _chunk_payload("2", "Half", chunk_index=0, text="인공지능 데이터 분석 연구")
    client = FakeFlatClient(
        points_by_view={
            "dense": [(joint_chunk, 0.9)],
            ("sparse", AI_VIEW_TEXT): [(joint_chunk, 0.9), (partial_chunk, 0.5)],
            ("sparse", SEMI_VIEW_TEXT): [(joint_chunk, 0.9)],
        }
    )
    result = _run(_retriever(client), query="인공지능 반도체", core=["인공지능", "반도체"], concept_specs=AI_SEMI_SPECS)

    ids = [hit.researcher_id for hit in result.hits]
    assert ids == ["1", "2"]  # required 충족 후보가 부분충족보다 항상 위.
    assert result.hits[0].coverage_type in {"joint", "separate"}
    assert result.hits[1].coverage_type == "partial"


def test_search_gate_drops_partial_when_fallback_disabled():
    neutral = _chunk_payload("M1451331", "나관식", text="인공지능 데이터 분석 연구")
    client = FakeFlatClient(points_by_view={("sparse", AI_VIEW_TEXT): [(neutral, 0.5)]})
    result = _run(
        _retriever(client, relevance_fallback_tier=False),
        query="인공지능 반도체",
        core=["인공지능", "반도체"],
        concept_specs=AI_SEMI_SPECS,
    )

    assert result.hits == []
    assert result.filtered_out_candidates == [
        {
            "expert_id": "M1451331",
            "name": "나관식",
            "reason": "relevance_concepts_missing",
            "matched_concepts": ["ai"],
            "missing_concepts": ["semiconductor"],
        }
    ]


def test_search_generic_query_without_concepts_ranks_by_evidence():
    # 전부 일반어(GENERIC_SEARCH_TERMS) → query_exact 합성도 비어 source=none, generic 폴백.
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Generic", text="모호성 가설 재검토"), 0.8)])
    result = _run(_retriever(client), query="연구 산업 개발", core=["연구", "산업", "개발"])

    assert [hit.researcher_id for hit in result.hits] == ["M1"]
    assert result.hits[0].group_score > 0.0
    assert result.hits[0].coverage_type == ""  # concept 미감지 → generic 폴백
    assert result.query_payload["relevance_gate_active_concepts"] == []
    assert result.query_payload["concept_plan"]["source"] == "none"


def test_search_query_exact_when_planner_and_registry_miss():
    # registry 밖 도메인 + 비일반어 retrieval_core → query_exact concept(optional) 합성으로 정렬 견고화.
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Meta", text="메타물질 음굴절 광학 소자"), 0.8)])
    result = _run(_retriever(client), query="메타물질 음굴절 전문가", core=["메타물질", "음굴절"])

    assert result.query_payload["concept_plan"]["source"] == "query_exact"
    assert result.query_payload["relevance_gate_active_concepts"] == []  # optional이라 gate 미작동
    assert result.hits[0].researcher_id == "M1"
    assert set(result.hits[0].matched_concepts) == {"메타물질", "음굴절"}


# ---------------------------------------------------------------------------
# org exclusion / invalid / empty
# ---------------------------------------------------------------------------
def test_search_excludes_candidates_by_org():
    points = [
        (_chunk_payload("1", "Keep", text="인공지능 반도체", organization="서울대학교"), 0.9),
        (_chunk_payload("2", "Drop", text="인공지능 반도체", organization="한국전자통신연구원"), 0.9),
    ]
    client = FakeFlatClient(default_points=points)
    result = asyncio.run(
        _retriever(client).search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x", retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"], exclude_orgs=["한국전자통신연구원"],
            ),
            query_filter=None,
        )
    )
    assert {hit.researcher_id for hit in result.hits} == {"1"}
    assert any(f["expert_id"] == "2" and f["reason"] == "excluded_org" for f in result.filtered_out_candidates)


def test_search_filters_include_orgs_by_root_affiliation_only():
    points = [
        (_chunk_payload("1", "Keep", text="인공지능 반도체", organization="주식회사 대원테크"), 0.9),
        (_chunk_payload("2", "Drop", text="인공지능 반도체", organization="한국전자통신연구원"), 0.9),
    ]
    client = FakeFlatClient(default_points=points)
    result = asyncio.run(
        _retriever(client).search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x",
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
                include_orgs=["대원테크"],
            ),
            query_filter=None,
        )
    )

    assert [hit.researcher_id for hit in result.hits] == ["1"]
    assert any(
        f["expert_id"] == "2" and f["reason"] == "include_org_mismatch"
        for f in result.filtered_out_candidates
    )


def test_search_org_filter_ignores_project_doc_attrs_organizations():
    points = [
        (
            _chunk_payload(
                "1",
                "Keep",
                doc_type="project",
                text="인공지능 반도체 과제",
                organization="서울대학교",
                doc_attrs={"performing_organization": "한국전자통신연구원"},
            ),
            0.9,
        )
    ]
    client = FakeFlatClient(default_points=points)
    result = asyncio.run(
        _retriever(client).search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x",
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
                exclude_orgs=["한국전자통신연구원"],
            ),
            query_filter=None,
        )
    )

    assert [hit.researcher_id for hit in result.hits] == ["1"]
    assert result.filtered_out_candidates == []


def test_l3_cache_does_not_bypass_org_filter(tmp_path):
    """회귀: L3 캐시 적중이 include/exclude org 필터를 우회하면 안 된다.

    버그(수정 전): 캐시 키가 org를 빼고 (compiled_json|filter_json|snapshot)만으로 구성돼,
    무필터 질의가 캐시한 결과를 exclude_orgs 요청이 그대로 재사용 → 제외 대상이 결과에 남았다.
    """
    points = [
        (_chunk_payload("1", "Keep", text="인공지능 반도체", organization="서울대학교"), 0.9),
        (_chunk_payload("2", "Drop", text="인공지능 반도체", organization="한국전자통신연구원"), 0.9),
    ]
    client = FakeFlatClient(default_points=points)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(cache_enabled=True),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
        l3_cache=RetrievalResultCache(tmp_path),
    )

    base_plan = dict(
        intent_summary="x",
        retrieval_core=["인공지능", "반도체"],
        core_keywords=["인공지능", "반도체"],
    )

    # 1) 무필터 질의 → 두 후보 모두 회수되고 L3에 캐시된다.
    first = asyncio.run(
        retriever.search(
            query="인공지능 반도체",
            plan=PlannerOutput(**base_plan),
            query_filter=None,
        )
    )
    assert {hit.researcher_id for hit in first.hits} == {"1", "2"}

    # 2) 같은 질의 + exclude_orgs → 캐시 적중이어도 제외 대상이 남으면 안 된다.
    excluded = asyncio.run(
        retriever.search(
            query="인공지능 반도체",
            plan=PlannerOutput(exclude_orgs=["한국전자통신연구원"], **base_plan),
            query_filter=None,
        )
    )
    assert {hit.researcher_id for hit in excluded.hits} == {"1"}

    # 3) 같은 질의 + include_orgs → 소속 일치 후보만 남아야 한다.
    included = asyncio.run(
        retriever.search(
            query="인공지능 반도체",
            plan=PlannerOutput(include_orgs=["서울대학교"], **base_plan),
            query_filter=None,
        )
    )
    assert {hit.researcher_id for hit in included.hits} == {"1"}


def test_search_skips_invalid_points():
    points = [
        ({"researcher_id": "bad", "researcher_name": "Broken"}, 0.9),  # chunk_id/doc_type 없음
        (_chunk_payload("good", "Valid", text="인공지능 반도체"), 0.8),
    ]
    client = FakeFlatClient(default_points=points)
    result = _run(_retriever(client), query="인공지능 반도체", core=["인공지능", "반도체"], concept_specs=AI_SEMI_SPECS)

    assert [hit.researcher_id for hit in result.hits] == ["good"]
    assert len(result.hits[0].chunks) == 1


def test_search_returns_empty_when_no_points():
    client = FakeFlatClient(default_points=[])
    result = _run(_retriever(client))
    assert result.hits == []
    assert result.query_payload["merged_chunk_count"] == 0
    assert result.query_payload["final_hit_count"] == 0


def test_search_uses_active_sparse_runtime_model():
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Alpha"), 0.9)])
    retriever = QdrantHybridRetriever(
        client=client, settings=_settings(),
        dense_encoder=RecordingDenseEncoder(), query_builder=QueryTextBuilder(),
        sparse_runtime=SparseRuntimeConfig(
            backend="fastembed_builtin", active_model_name="Qdrant/bm25",
            requires_idf_modifier=True, used_fallback=True,
        ),
    )
    _run(retriever)
    sparse_calls = [c for c in retriever.client.calls if c["using"] == SPARSE_VECTOR_NAME]
    assert sparse_calls
    assert all(c["query"].model == "Qdrant/bm25" for c in sparse_calls)


# ---------------------------------------------------------------------------
# search_weighted alias + grouped diagnostic
# ---------------------------------------------------------------------------
def test_search_weighted_is_multiview_alias():
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Alpha", text="인공지능 반도체"), 0.9)])
    retriever = _retriever(client)
    result = asyncio.run(retriever.search_weighted(
        query="인공지능 반도체",
        plan=PlannerOutput(intent_summary="x", retrieval_core=["인공지능", "반도체"], core_keywords=["인공지능", "반도체"]),
        query_filter=None,
    ))
    assert [hit.researcher_id for hit in result.hits] == ["M1"]
    assert result.query_payload["retrieval_mode"] == "multiview_flat_relevance"
    assert client.calls  # flat 검색 사용
    assert client.group_calls == []  # grouped 미사용


# ---------------------------------------------------------------------------
# 사용자 선택형 검색 모드 (search_mode)
# ---------------------------------------------------------------------------
def test_hybrid_mode_uses_dense_plus_single_sparse_view():
    raw = "인공지능 반도체 연구자"
    dense_encoder = RecordingDenseEncoder()
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Alpha", text="인공지능 반도체"), 0.9)])
    retriever = QdrantHybridRetriever(
        client=client, settings=_settings(),
        dense_encoder=dense_encoder, query_builder=QueryTextBuilder(),
    )
    result = asyncio.run(
        retriever.search(
            query=raw,
            plan=PlannerOutput(
                intent_summary=raw,
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
                concept_specs=AI_SEMI_SPECS,
            ),
            query_filter=None,
            search_mode="hybrid",
        )
    )

    # hybrid = dense_full + sparse_raw 2뷰만(focus/concept 뷰 미사용).
    assert len(client.calls) == 2
    assert client.calls[0]["using"] == DENSE_VECTOR_NAME
    assert client.calls[1]["using"] == SPARSE_VECTOR_NAME
    assert client.calls[1]["query"].text == raw  # sparse_raw = 원문 보존
    assert result.query_payload["retrieval_mode"] == "hybrid_dense_sparse_rrf"
    assert result.query_payload["search_mode"] == "hybrid"
    assert result.query_payload["concept_plan"]["view_sources"] == ["dense_full", "sparse_raw"]
    # hybrid 가중은 균등(1.0/1.0) — multiview의 sparse_raw=0.25 보조채널과 구분.
    assert result.query_payload["weights"]["view"] == {"dense_full": 1.0, "sparse_raw": 1.0}
    assert [hit.researcher_id for hit in result.hits] == ["M1"]


def test_keyword_similarity_mode_sparse_shortlist_then_dense_rerank():
    from qdrant_client import models

    client = FakeFlatClient(
        default_points=[(_chunk_payload("M1", "Alpha", text="인공지능 반도체"), 0.87)]
    )
    retriever = _retriever(client)
    result = asyncio.run(
        retriever.search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x",
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
            ),
            query_filter=None,
            search_mode="keyword_similarity",
        )
    )

    # 2단계 cascade: 1차 SPLADE(키워드) → 2차 dense(유사도) = 정확히 2콜.
    assert len(client.calls) == 2
    assert client.calls[0]["using"] == SPARSE_VECTOR_NAME  # 1차 키워드 검색
    assert client.calls[0]["with_payload"] is False  # 1차는 point id만 필요
    assert client.calls[1]["using"] == DENSE_VECTOR_NAME  # 2차 dense 유사도 재정렬
    # 2차 dense는 1차 후보 point id 집합으로만 한정(HasIdCondition).
    stage2_filter = client.calls[1]["query_filter"]
    has_id_conds = [c for c in (stage2_filter.must or []) if isinstance(c, models.HasIdCondition)]
    assert has_id_conds and set(has_id_conds[0].has_id) == {"paper_100000000001_c000"}

    assert [hit.researcher_id for hit in result.hits] == ["M1"]
    assert result.query_payload["retrieval_mode"] == "keyword_then_dense_similarity"
    assert result.query_payload["search_mode"] == "keyword_similarity"
    assert result.query_payload["view_counts"] == {"sparse_keyword": 1, "dense_rerank": 1}
    # chunk 점수 = dense 유사도(raw) — 순위 RRF 융합이 아님.
    chunk = result.hits[0].chunks[0]
    assert chunk.score == 0.87
    assert chunk.sources == ["dense_similarity"]


def test_keyword_similarity_mode_empty_when_no_keyword_candidates():
    client = FakeFlatClient(default_points=[])  # 1차 SPLADE가 후보 0건
    retriever = _retriever(client)
    result = asyncio.run(
        retriever.search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x",
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
            ),
            query_filter=None,
            search_mode="keyword_similarity",
        )
    )
    # 1차 후보가 없으면 2차 dense는 호출하지 않는다(콜 1건).
    assert len(client.calls) == 1
    assert result.hits == []
    assert result.query_payload["view_counts"] == {"sparse_keyword": 0, "dense_rerank": 0}


def test_unknown_search_mode_falls_back_to_multiview():
    client = FakeFlatClient(default_points=[(_chunk_payload("M1", "Alpha", text="인공지능 반도체"), 0.9)])
    result = asyncio.run(
        _retriever(client).search(
            query="인공지능 반도체",
            plan=PlannerOutput(
                intent_summary="x",
                retrieval_core=["인공지능", "반도체"],
                core_keywords=["인공지능", "반도체"],
            ),
            query_filter=None,
            search_mode="does_not_exist",
        )
    )
    assert result.query_payload["retrieval_mode"] == "multiview_flat_relevance"
    assert result.query_payload["search_mode"] == "multiview"


def test_search_grouped_diagnostic_uses_query_points_groups():
    client = FakeFlatClient(
        groups_spec=[("M1", [(_chunk_payload("M1", "Alpha", text="인공지능 반도체 설계"), 0.9)])]
    )
    retriever = _retriever(client)
    result = asyncio.run(retriever.search_grouped_diagnostic(
        query="인공지능 반도체",
        plan=PlannerOutput(intent_summary="x", retrieval_core=["인공지능", "반도체"], core_keywords=["인공지능", "반도체"]),
        query_filter=None,
    ))
    assert [hit.researcher_id for hit in result.hits] == ["M1"]
    assert result.query_payload["retrieval_mode"] == "grouped_hybrid_rrf"
    assert len(client.group_calls) == 1
    assert client.group_calls[0]["group_by"] == "researcher_id"
    assert result.query_payload["group_count"] == 1
