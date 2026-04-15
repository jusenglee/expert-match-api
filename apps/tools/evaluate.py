"""
Simple evaluation helper for the current retrieval-first recommendation flow.
"""

from __future__ import annotations

import asyncio
import logging

from qdrant_client import QdrantClient

from apps.api.main import build_dense_encoder
from apps.core.config import Settings
from apps.core.feedback_store import FeedbackStore
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.evidence_selector import KeywordEvidenceSelector
from apps.recommendation.planner import OpenAICompatPlanner
from apps.recommendation.reasoner import OpenAICompatReasonGenerator
from apps.recommendation.service import RecommendationService
from apps.search.filters import QdrantFilterCompiler
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import SearchSchemaRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")


GOLDEN_DATA = [
    "반도체 설계와 AI 칩 연구 경험이 있는 전문가를 추천해줘",
    "소방 드론과 화재 대응 과제 경험이 있는 전문가를 추천해줘",
]


async def evaluate() -> None:
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    registry = SearchSchemaRegistry.default()
    dense_encoder = build_dense_encoder(settings)

    service = RecommendationService(
        planner=OpenAICompatPlanner(settings),
        retriever=QdrantHybridRetriever(
            client=client,
            settings=settings,
            registry=registry,
            dense_encoder=dense_encoder,
            query_builder=QueryTextBuilder(),
        ),
        filter_compiler=QdrantFilterCompiler(),
        card_builder=CandidateCardBuilder(),
        evidence_selector=KeywordEvidenceSelector(),
        reason_generator=OpenAICompatReasonGenerator(settings),
        feedback_store=FeedbackStore(settings.feedback_db_path, settings.feedback_table),
    )

    for index, query in enumerate(GOLDEN_DATA, start=1):
        print(f"[{index}] {query}")
        result = await service.recommend(query=query)
        print(f"  retrieved_count={result['retrieved_count']}")
        print(f"  recommendations={len(result['recommendations'])}")
        print(f"  retrieval_keywords={result['trace']['retrieval_keywords']}")
        if result["recommendations"]:
            top_item = result["recommendations"][0]
            print(f"  top_1={top_item.name} ({top_item.rank_score})")
        print()


if __name__ == "__main__":
    asyncio.run(evaluate())
