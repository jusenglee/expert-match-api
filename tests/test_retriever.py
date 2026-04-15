import asyncio
from types import SimpleNamespace

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry


class FakeQdrantClient:
    def __init__(self, payloads: list[dict], scores: list[float] | None = None) -> None:
        self.payloads = payloads
        self.scores = scores or [0.88 - (index * 0.01) for index in range(len(payloads))]
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
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
                core_keywords=["AI semiconductor", "chip design"],
            ),
            query_filter=None,
        ),
    )

    assert result.retrieval_keywords == ["AI semiconductor", "chip design"]
    assert result.branch_queries == {
        "basic": "AI semiconductor\nchip design",
        "art": "AI semiconductor\nchip design",
        "pat": "AI semiconductor\nchip design",
        "pjt": "AI semiconductor\nchip design",
    }
    assert [hit.expert_id for hit in result.hits] == ["1", "2"]
    assert encoder.inputs == [
        "AI semiconductor\nchip design",
        "AI semiconductor\nchip design",
        "AI semiconductor\nchip design",
        "AI semiconductor\nchip design",
    ]
    assert len(client.last_kwargs["prefetch"]) == 4
    sparse_texts = [
        branch_prefetch.prefetch[1].query.text
        for branch_prefetch in client.last_kwargs["prefetch"]
    ]
    assert sparse_texts == encoder.inputs


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
