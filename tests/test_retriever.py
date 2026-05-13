import asyncio
import logging
from types import SimpleNamespace

from qdrant_client import models

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry
from apps.search.sparse_runtime import SparseRuntimeConfig


class FakeQdrantClient:
    def __init__(self, payloads: list[dict], scores: list[float] | None = None) -> None:
        self.payloads = payloads
        self.scores = scores or [0.88 - (index * 0.01) for index in range(len(payloads))]
        self.calls: list[dict] = []
        self.main_kwargs = None
        self.branch_kwargs: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("with_payload"):
            self.main_kwargs = kwargs
        else:
            self.branch_kwargs.append(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=str(index + 1),
                    payload=payload,
                    score=self.scores[index],
                )
                for index, payload in enumerate(self.payloads)
            ]
        )


class StageAwareFakeQdrantClient:
    def __init__(
        self,
        *,
        keyword_payloads: list[dict],
        hybrid_payloads: list[dict],
    ) -> None:
        self.keyword_payloads = keyword_payloads
        self.hybrid_payloads = hybrid_payloads
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        payloads = self.hybrid_payloads if "prefetch" in kwargs else self.keyword_payloads
        return SimpleNamespace(
            points=[
                SimpleNamespace(id=str(index + 1), payload=payload, score=0.9)
                for index, payload in enumerate(payloads)
            ]
        )


class RecordingDenseEncoder:
    def __init__(self) -> None:
        self.model_name = "hash"
        self.vector_size = 8
        self.inputs: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.inputs.append(text)
        return [0.1] * self.vector_size


def _settings() -> Settings:
    return Settings(
        app_env="test",
        strict_runtime_validation=False,
        embedding_vector_size=8,
        branch_prefetch_limit=80,
        branch_output_limit=50,
        retrieval_limit=40,
    )


def _payload(researcher_id: str, name: str) -> dict:
    return {
        "basic_info": {"researcher_id": researcher_id, "researcher_name": name},
        "researcher_profile": {},
        "publications": [{"publication_title": f"{name} paper"}],
        "intellectual_properties": [],
        "research_projects": [],
    }


def _keyword_calls(client) -> list[dict]:
    return [call for call in client.calls if "prefetch" not in call]


def _hybrid_calls(client) -> list[dict]:
    return [call for call in client.calls if "prefetch" in call]


def test_retriever_uses_single_clean_query_across_all_branches_and_name_tiebreak():
    payloads = [
        {
            "basic_info": {"researcher_id": "2", "researcher_name": "Bravo"},
            "researcher_profile": {},
            "publications": [{"publication_title": "B paper"}],
            "intellectual_properties": [],
            "research_projects": [],
        },
        {
            "basic_info": {"researcher_id": "1", "researcher_name": "Alpha"},
            "researcher_profile": {},
            "publications": [{"publication_title": "A paper"}],
            "intellectual_properties": [],
            "research_projects": [],
        },
    ]
    client = FakeQdrantClient(payloads=payloads, scores=[0.91, 0.91])
    encoder = RecordingDenseEncoder()
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=encoder,
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="Recommend AI semiconductor reviewers",
            plan=PlannerOutput(
                intent_summary="Recommend AI semiconductor reviewers",
                retrieval_core=["AI semiconductor", "chip design"],
                core_keywords=["AI semiconductor", "chip design"],
            ),
            query_filter=None,
        ),
    )

    assert result.retrieval_keywords == ["AI", "semiconductor", "chip", "design"]
    assert "AI semiconductor" in result.branch_queries["basic"].stable
    assert "chip design" in result.branch_queries["basic"].stable
    assert "AI semiconductor" in result.branch_queries["art"].stable
    assert "chip design" in result.branch_queries["art"].stable
    assert "AI semiconductor" in result.branch_queries["pat"].stable
    assert "chip design" in result.branch_queries["pat"].stable
    assert "AI semiconductor" in result.branch_queries["pjt"].stable
    assert "chip design" in result.branch_queries["pjt"].stable
    assert {hit.expert_id for hit in result.hits} == {"1", "2"}
    assert len(encoder.inputs) == 6
    assert all("AI semiconductor" in text for text in encoder.inputs)
    assert all("chip design" in text for text in encoder.inputs)
    keyword_calls = _keyword_calls(client)
    hybrid_calls = _hybrid_calls(client)
    assert len(keyword_calls) == 6
    assert len(hybrid_calls) == 6
    keyword_sparse_texts = [
        call["query"].text
        for call in keyword_calls
    ]
    sparse_texts = [
        call["prefetch"][1].query.text
        for call in hybrid_calls
    ]
    assert keyword_sparse_texts == encoder.inputs
    assert sparse_texts == encoder.inputs
    assert len(client.calls) == 12
    assert result.query_payload["retrieval_mode"] == "keyword_pool_then_hybrid"
    assert result.query_payload["retrieval_keywords"] == [
        "AI",
        "semiconductor",
        "chip",
        "design",
    ]
    assert "AI semiconductor" in result.query_payload["keyword_stage_queries"]["basic"]["stable"]
    assert "chip design" in result.query_payload["hybrid_stage_queries"]["art"]["stable"]
    assert result.query_payload["keyword_stage_candidate_count"] == 2
    assert len(result.retrieval_score_traces) == 2
    assert result.retrieval_score_traces[0]["expert_id"] in {"1", "2"}
    assert {item["branch"] for item in result.retrieval_score_traces[0]["branch_matches"]} == {
        "basic",
        "art",
        "pat",
        "pjt",
    }


def test_retriever_limits_hybrid_stage_to_keyword_candidate_pool_and_existing_filter(caplog):
    caplog.set_level(logging.INFO)
    keyword_payloads = [_payload("1", "Keyword Match")]
    hybrid_payloads = [
        _payload("1", "Keyword Match"),
        _payload("2", "Outside Pool"),
    ]
    client = StageAwareFakeQdrantClient(
        keyword_payloads=keyword_payloads,
        hybrid_payloads=hybrid_payloads,
    )
    base_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="researcher_profile.highest_degree",
                match=models.MatchAny(any=["PhD"]),
            )
        ]
    )
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="keyword constrained hybrid",
            plan=PlannerOutput(
                intent_summary="keyword constrained hybrid",
                retrieval_core=["keyword"],
                core_keywords=["keyword"],
            ),
            query_filter=base_filter,
        ),
    )

    assert [hit.expert_id for hit in result.hits] == ["1"]
    assert len(_keyword_calls(client)) == 6
    assert len(_hybrid_calls(client)) == 6
    hybrid_filter = _hybrid_calls(client)[0]["query_filter"]
    must_conditions = hybrid_filter.must or []
    assert any(
        getattr(condition, "key", None) == "researcher_profile.highest_degree"
        for condition in must_conditions
    )
    candidate_conditions = [
        condition
        for condition in must_conditions
        if getattr(condition, "key", None) == "basic_info.researcher_id"
    ]
    assert len(candidate_conditions) == 1
    assert candidate_conditions[0].match.any == ["1"]
    assert result.query_payload["hybrid_stage_candidate_filter_count"] == 1
    assert result.query_payload["hybrid_stage_raw_branch_counts"] == {
        "basic:stable": 2,
        "art:stable": 2,
        "art:expanded": 2,
        "pat:stable": 2,
        "pjt:stable": 2,
        "pjt:expanded": 2,
    }
    assert result.query_payload["aggregated_candidate_count"] == 1
    assert result.query_payload["support_pass_count"] == 1
    assert result.query_payload["support_filtered_count"] == 0
    assert "1차 키워드 검색 시작" in caplog.text
    assert "2차 하이브리드 검색 시작" in caplog.text
    assert "검색 집계 완료" in caplog.text


def test_retriever_returns_empty_when_keyword_stage_has_no_candidates():
    client = StageAwareFakeQdrantClient(
        keyword_payloads=[],
        hybrid_payloads=[_payload("1", "Hybrid Only")],
    )
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="no keyword candidates",
            plan=PlannerOutput(
                intent_summary="no keyword candidates",
                retrieval_core=["missing"],
                core_keywords=["missing"],
            ),
            query_filter=None,
        ),
    )

    assert result.hits == []
    assert len(_keyword_calls(client)) == 6
    assert _hybrid_calls(client) == []
    assert result.query_payload["retrieval_mode"] == "keyword_pool_then_hybrid"
    assert result.query_payload["keyword_stage_candidate_count"] == 0
    assert result.query_payload["hybrid_stage_skipped_reason"] == "keyword_stage_empty"
    assert result.query_payload["aggregated_candidate_count"] == 0


def test_retriever_skips_invalid_points_and_keeps_valid_hits():
    payloads = [
        {
            "basic_info": {"researcher_id": "bad", "researcher_name": "Broken"},
            "researcher_profile": {},
            "publications": "broken payload",
            "intellectual_properties": [],
            "research_projects": [],
        },
        {
            "basic_info": {"researcher_id": "good", "researcher_name": "Valid"},
            "researcher_profile": {},
            "publications": [{"publication_title": "Valid paper"}],
            "intellectual_properties": [],
            "research_projects": [],
        },
    ]
    client = FakeQdrantClient(payloads=payloads)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="skip broken payloads",
            plan=PlannerOutput(
                intent_summary="skip broken payloads",
                core_keywords=["valid"],
            ),
            query_filter=None,
        ),
    )

    assert len(result.hits) == 1
    assert result.hits[0].expert_id == "good"
    assert len(result.retrieval_score_traces) == 1
    assert result.retrieval_score_traces[0]["expert_id"] == "good"


def test_retriever_uses_active_sparse_runtime_model_for_builtin_queries():
    payloads = [
        {
            "basic_info": {"researcher_id": "1", "researcher_name": "Alpha"},
            "researcher_profile": {},
            "publications": [{"publication_title": "A paper"}],
            "intellectual_properties": [],
            "research_projects": [],
        },
    ]
    client = FakeQdrantClient(payloads=payloads)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
        sparse_runtime=SparseRuntimeConfig(
            backend="fastembed_builtin",
            active_model_name="Qdrant/bm25",
            requires_idf_modifier=True,
            used_fallback=True,
        ),
    )

    asyncio.run(
        retriever.search(
            query="bm25 fallback query",
            plan=PlannerOutput(
                intent_summary="bm25 fallback query",
                core_keywords=["bm25", "fallback"],
            ),
            query_filter=None,
        )
    )

    sparse_models = [
        call["prefetch"][1].query.model if "prefetch" in call else call["query"].model
        for call in client.calls
    ]
    assert sparse_models == ["Qdrant/bm25"] * 12
