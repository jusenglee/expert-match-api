import asyncio
from types import SimpleNamespace

from apps.core.config import Settings
from apps.domain.models import PlannerOutput
from apps.search.encoders import HashingDenseEncoder
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry


class FakeQdrantClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_kwargs = None

    def query_points(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(points=[SimpleNamespace(payload=self.payload, score=0.88)])


def test_retriever_always_builds_four_branch_prefetches():
    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        embedding_vector_size=8,
        branch_prefetch_limit=80,
        branch_output_limit=50,
        retrieval_limit=40,
    )
    payload = {
        "basic_info": {
            "researcher_id": "1",
            "researcher_name": "홍길동",
            "affiliated_organization": "테스트기관",
            "affiliated_organization_exact": "테스트기관",
        },
        "researcher_profile": {
            "highest_degree": "박사",
            "major_field": "AI",
            "publication_count": 1,
            "scie_publication_count": 1,
            "intellectual_property_count": 1,
            "research_project_count": 1,
        },
        "publications": [{"publication_title": "테스트 논문"}],
        "intellectual_properties": [{"intellectual_property_title": "테스트 특허"}],
        "research_projects": [{"project_title_korean": "테스트 과제"}],
    }
    client = FakeQdrantClient(payload=payload)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=settings,
        registry=SearchSchemaRegistry.default(),
        dense_encoder=HashingDenseEncoder(model_name="hash", vector_size=8),
        query_builder=QueryTextBuilder(),
    )

    result = asyncio.run(
        retriever.search(
            query="AI 반도체 평가위원 추천",
            plan=PlannerOutput(
                intent_summary="AI 반도체 평가위원 추천",
                branch_query_hints={
                    "basic": "기본",
                    "art": "논문",
                    "pat": "특허",
                    "pjt": "과제",
                },
            ),
            query_filter=None,
        ),
    )

    assert len(result.hits) == 1
    assert client.last_kwargs is not None
    assert len(client.last_kwargs["prefetch"]) == 4
    assert client.last_kwargs["limit"] == 40
    assert result.branch_queries["basic"]
