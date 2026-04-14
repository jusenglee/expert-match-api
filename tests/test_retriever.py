import asyncio
from types import SimpleNamespace

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry


class FakeQdrantClient:
    def __init__(self, payload: dict | None = None, payloads: list[dict] | None = None) -> None:
        self.payloads = payloads or ([payload] if payload is not None else [])
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=str(index + 1),
                    payload=payload,
                    score=0.88 - (index * 0.01),
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


def test_retriever_uses_same_core_keyword_query_for_all_branches():
    payload = {
        "basic_info": {
            "researcher_id": "1",
            "researcher_name": "Tester",
            "affiliated_organization": "Test Institute",
            "affiliated_organization_exact": "Test Institute",
        },
        "researcher_profile": {
            "highest_degree": "PhD",
            "major_field": "AI",
            "publication_count": 1,
            "scie_publication_count": 1,
            "intellectual_property_count": 1,
            "research_project_count": 1,
        },
        "publications": [{"publication_title": "Test paper"}],
        "intellectual_properties": [{"intellectual_property_title": "Test patent"}],
        "research_projects": [{"project_title_korean": "Test project"}],
    }
    client = FakeQdrantClient(payload=payload)
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
                branch_query_hints={
                    "basic": "ignored",
                    "art": "ignored",
                    "pat": "ignored",
                    "pjt": "ignored",
                },
            ),
            query_filter=None,
        ),
    )

    expected_query = "AI semiconductor\nchip design"

    assert len(result.hits) == 1
    assert result.retrieval_keywords == ["AI semiconductor", "chip design"]
    assert result.branch_queries == {
        "basic": expected_query,
        "art": expected_query,
        "pat": expected_query,
        "pjt": expected_query,
    }
    assert encoder.inputs == [expected_query] * 4
    assert client.last_kwargs is not None
    assert len(client.last_kwargs["prefetch"]) == 4
    sparse_texts = [
        branch_prefetch.prefetch[1].query.text
        for branch_prefetch in client.last_kwargs["prefetch"]
    ]
    assert sparse_texts == [expected_query] * 4


def test_retriever_normalizes_legacy_blank_string_payload_fields():
    payload = {
        "basic_info": {
            "researcher_id": "2",
            "researcher_name": "Legacy Payload",
        },
        "researcher_profile": {
            "publication_count": "",
            "scie_publication_count": " ",
            "intellectual_property_count": "",
            "research_project_count": "1",
        },
        "publications": [
            {
                "publication_title": "Legacy paper",
                "korean_keywords": "",
                "english_keywords": "semiconductor",
            }
        ],
        "intellectual_properties": "",
        "research_projects": [{"project_title_korean": "Legacy project", "reference_year": ""}],
        "technical_classifications": "",
        "evaluation_activity_cnt": "",
        "external_activity_cnt": "",
        "evaluation_activities": "",
    }
    client = FakeQdrantClient(payload=payload)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=_settings(),
        registry=SearchSchemaRegistry.default(),
        dense_encoder=RecordingDenseEncoder(),
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="legacy payload normalization",
            plan=PlannerOutput(
                intent_summary="legacy payload normalization",
                core_keywords=["legacy", "normalization"],
            ),
            query_filter=None,
        ),
    )

    assert len(result.hits) == 1
    hit = result.hits[0]
    assert hit.payload.publications[0].korean_keywords == []
    assert hit.payload.publications[0].english_keywords == ["semiconductor"]
    assert hit.payload.intellectual_properties == []
    assert hit.payload.researcher_profile.publication_count == 0
    assert hit.payload.research_projects[0].reference_year is None
    assert hit.branch_coverage["pat"] is False


def test_retriever_skips_invalid_points_and_keeps_valid_hits():
    invalid_payload = {
        "basic_info": {"researcher_id": "bad", "researcher_name": "Broken"},
        "researcher_profile": {},
        "publications": "broken payload",
        "intellectual_properties": [],
        "research_projects": [],
    }
    valid_payload = {
        "basic_info": {"researcher_id": "good", "researcher_name": "Valid"},
        "researcher_profile": {},
        "publications": [{"publication_title": "Valid paper"}],
        "intellectual_properties": [],
        "research_projects": [],
    }
    client = FakeQdrantClient(payloads=[invalid_payload, valid_payload])
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
