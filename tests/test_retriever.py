import asyncio
from types import SimpleNamespace

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import (
    RETRIEVAL_QUERY_SCHEMA_VERSION,
    QdrantHybridRetriever,
)
from apps.search.schema_registry import SearchSchemaRegistry
from apps.search.sparse_runtime import SparseRuntimeConfig


class FakeQdrantClient:
    def __init__(self, payloads: list[dict], scores: list[float] | None = None) -> None:
        self.payloads = payloads
        self.scores = scores or [0.88 - (index * 0.01) for index in range(len(payloads))]
        self.calls: list[dict] = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
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


def test_retriever_uses_semantic_query_for_dense_and_keywords_for_sparse():
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
                semantic_query="AI semiconductor expert for chip design",
            ),
            query_filter=None,
        ),
    )

    assert result.retrieval_keywords == ["AI", "semiconductor", "chip", "design"]
    assert result.branch_queries["basic"].stable.startswith("AI semiconductor chip design")
    assert result.branch_queries["basic"].stable_dense.startswith(
        "AI semiconductor expert for chip design"
    )
    assert result.branch_queries["basic"].stable_sparse.startswith(
        "AI semiconductor chip design"
    )
    assert result.query_payload["query_schema_version"] == RETRIEVAL_QUERY_SCHEMA_VERSION
    assert result.query_payload["branch_query_sources"]["basic"]["dense_base_source"] == (
        "semantic_query"
    )
    assert len(encoder.inputs) == 4
    assert all("expert for chip design" in text for text in encoder.inputs)

    sparse_texts = [call["prefetch"][1].query.text for call in client.calls]
    assert sparse_texts != encoder.inputs
    assert all("AI semiconductor chip design" in text for text in sparse_texts)
    assert len(client.calls) == 4
    assert len(result.retrieval_score_traces) == 2
    assert result.query_payload["rrf_mode"] == "equal_weight"
    assert result.query_payload["expanded_path_policy"] == "distinct_expanded_all_branches"


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

    sparse_models = [call["prefetch"][1].query.model for call in client.calls]
    assert sparse_models == ["Qdrant/bm25"] * 4
