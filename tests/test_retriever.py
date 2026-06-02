"""QdrantHybridRetriever 테스트 (flat chunk 모델, v2.1 — query_points_groups 기반).

검증 계약:
- 단일 grouped 하이브리드 검색(query_points_groups, group_by=researcher_id) 1콜.
  prefetch=[dense(vector_e5i), sparse(vector_splade)], query=FusionQuery.RRF, group_size/limit 설정값.
- 그룹 → ResearcherCandidate. group_score = Σ chunk_score × doc_type_prior (doc_type별 chunk cap).
- evidence chunk는 cap과 무관하게 모두 보관. exclude_orgs 후처리 배제.
- search_weighted()는 grouped RRF로 통일(search()와 동일 경로).
네트워크/실모델 금지 — FakeGroupsClient + Recording 인코더 스텁만.
"""

import asyncio
from types import SimpleNamespace

from qdrant_client import models

from apps.core.config import Settings
from apps.domain.models import PlannerOutput, ResearcherCandidate
from apps.search.query_builder import CompiledQueries, QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import SparseRuntimeConfig


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeGroupsClient:
    """query_points_groups를 흉내. groups_spec: [(researcher_id, [(payload, score), ...]), ...]."""

    def __init__(self, groups_spec: list[tuple[str, list[tuple[dict, float]]]]) -> None:
        self._spec = groups_spec
        self.calls: list[dict] = []

    def query_points_groups(self, **kwargs):
        self.calls.append(kwargs)
        groups = []
        for rid, hits in self._spec:
            points = [
                SimpleNamespace(id=payload.get("chunk_id"), payload=payload, score=score)
                for payload, score in hits
            ]
            groups.append(SimpleNamespace(id=rid, hits=points))
        return SimpleNamespace(groups=groups)


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


def _run(retriever, *, query="검색 질의", core=None, **plan_kwargs):
    plan = PlannerOutput(
        intent_summary=query,
        retrieval_core=core or ["키워드"],
        core_keywords=core or ["키워드"],
        **plan_kwargs,
    )
    return asyncio.run(retriever.search(query=query, plan=plan, query_filter=None))


def _retriever(client, **settings_overrides) -> QdrantHybridRetriever:
    return QdrantHybridRetriever(
        client=client,
        settings=_settings(**settings_overrides),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )


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
# search() — grouped hybrid RRF
# ---------------------------------------------------------------------------
def test_search_groups_into_researcher_candidates():
    client = FakeGroupsClient(
        [
            ("M2", [(_chunk_payload("M2", "Bravo"), 0.91)]),
            ("M1", [(_chunk_payload("M1", "Alpha"), 0.90)]),
        ]
    )
    result = _run(_retriever(client), core=["AI semiconductor", "chip design"])

    assert all(isinstance(hit, ResearcherCandidate) for hit in result.hits)
    assert {hit.researcher_id for hit in result.hits} == {"M1", "M2"}
    assert all(len(hit.chunks) >= 1 for hit in result.hits)
    assert isinstance(result.queries, CompiledQueries)
    assert result.retrieval_keywords == ["AI", "semiconductor", "chip", "design"]

    # query_points_groups 단일 콜.
    assert len(client.calls) == 1
    assert result.query_payload["retrieval_mode"] == "grouped_hybrid_rrf"
    assert result.query_payload["group_count"] == 2
    assert result.query_payload["aggregated_candidate_count"] == 2
    assert result.query_payload["final_hit_count"] == 2

    # score traces: 후보별 1건 + doc_types/family_contributions.
    assert len(result.retrieval_score_traces) == 2
    trace = result.retrieval_score_traces[0]
    assert trace["expert_id"] in {"M1", "M2"}
    assert set(trace["doc_types"]) == {"paper"}
    assert "achievement" in trace["family_contributions"]


def test_search_uses_grouped_hybrid_query_shape():
    client = FakeGroupsClient([("M1", [(_chunk_payload("M1", "Alpha"), 0.9)])])
    settings_obj = _settings()
    retriever = QdrantHybridRetriever(
        client=client, settings=settings_obj,
        dense_encoder=RecordingDenseEncoder(), query_builder=QueryTextBuilder(),
    )
    asyncio.run(retriever.search(
        query="single vector check",
        plan=PlannerOutput(intent_summary="x", retrieval_core=["alpha"], core_keywords=["alpha"]),
        query_filter=None,
    ))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["collection_name"] == settings_obj.qdrant_collection_name
    assert call["group_by"] == "researcher_id"
    assert call["group_size"] == settings_obj.group_size
    assert call["limit"] == settings_obj.retrieval_limit
    prefetch = call["prefetch"]
    assert len(prefetch) == 2
    assert prefetch[0].using == DENSE_VECTOR_NAME
    assert prefetch[1].using == SPARSE_VECTOR_NAME
    assert prefetch[0].limit == settings_obj.prefetch_limit
    assert isinstance(call["query"], models.FusionQuery)
    assert call["query"].fusion == models.Fusion.RRF


def test_search_app_side_rrf_accumulation_with_doc_type_prior():
    # 그룹에 paper(prior 1.0) + specialty(prior 0.5) → group_score = 0.8*1.0 + 0.4*0.5 = 1.0
    group_hits = [
        (_chunk_payload("M1", "Alpha", doc_type="paper", chunk_index=0), 0.8),
        (_chunk_payload("M1", "Alpha", doc_type="specialty", chunk_index=0), 0.4),
    ]
    client = FakeGroupsClient([("M1", group_hits)])
    retriever = _retriever(client, doc_type_priors={"specialty": 0.5})
    result = _run(retriever)

    assert len(result.hits) == 1
    assert result.hits[0].group_score == 0.8 * 1.0 + 0.4 * 0.5


def test_search_doc_type_chunk_cap_limits_score_contribution():
    # 같은 (researcher, doc_type)에서 chunk가 cap보다 많아도 점수 기여는 cap개까지만.
    hits = [(_chunk_payload("M1", "Prolific", doc_type="paper", chunk_index=i), 0.9) for i in range(5)]
    client = FakeGroupsClient([("M1", hits)])
    result = _run(_retriever(client, doc_type_chunk_cap=2))

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert len(hit.chunks) == 5  # evidence chunk는 모두 보관
    # 점수 기여는 cap(2)개 × 0.9 = 1.8
    assert hit.group_score == 0.9 * 2


def test_search_skips_invalid_points():
    hits = [
        ({"researcher_id": "bad", "researcher_name": "Broken"}, 0.9),  # chunk_id/doc_type 없음
        (_chunk_payload("good", "Valid"), 0.8),
    ]
    client = FakeGroupsClient([("good", hits)])
    result = _run(_retriever(client))

    assert len(result.hits) == 1
    assert result.hits[0].researcher_id == "good"
    assert len(result.hits[0].chunks) == 1  # 깨진 point는 skip


def test_search_excludes_candidates_by_org():
    client = FakeGroupsClient(
        [
            ("1", [(_chunk_payload("1", "Keep", organization="서울대학교"), 0.9)]),
            ("2", [(_chunk_payload("2", "Drop", organization="한국전자통신연구원"), 0.9)]),
        ]
    )
    retriever = _retriever(client)
    result = asyncio.run(retriever.search(
        query="exclude org",
        plan=PlannerOutput(intent_summary="x", core_keywords=["x"], exclude_orgs=["한국전자통신연구원"]),
        query_filter=None,
    ))
    ids = {hit.researcher_id for hit in result.hits}
    assert ids == {"1"}
    assert any(f["expert_id"] == "2" for f in result.filtered_out_candidates)


def test_search_returns_empty_when_no_groups():
    client = FakeGroupsClient([])
    result = _run(_retriever(client))
    assert result.hits == []
    assert result.query_payload["group_count"] == 0


def test_search_uses_active_sparse_runtime_model():
    client = FakeGroupsClient([("M1", [(_chunk_payload("M1", "Alpha"), 0.9)])])
    retriever = QdrantHybridRetriever(
        client=client, settings=_settings(),
        dense_encoder=RecordingDenseEncoder(), query_builder=QueryTextBuilder(),
        sparse_runtime=SparseRuntimeConfig(
            backend="fastembed_builtin", active_model_name="Qdrant/bm25",
            requires_idf_modifier=True, used_fallback=True,
        ),
    )
    _run(retriever)
    # sparse prefetch(Document)의 model이 활성 런타임 모델.
    sparse_prefetch = client.calls[0]["prefetch"][1]
    assert sparse_prefetch.query.model == "Qdrant/bm25"


def test_search_weighted_is_grouped_alias():
    client = FakeGroupsClient([("M1", [(_chunk_payload("M1", "Alpha"), 0.9)])])
    retriever = _retriever(client)
    result = asyncio.run(retriever.search_weighted(
        query="weighted unified",
        plan=PlannerOutput(intent_summary="x", retrieval_core=["alpha"], core_keywords=["alpha"]),
        query_filter=None,
    ))
    assert [hit.researcher_id for hit in result.hits] == ["M1"]
    assert result.query_payload["retrieval_mode"] == "grouped_hybrid_rrf"
    assert len(client.calls) == 1  # grouped 단일 콜(가중 fan-out 폐기)
