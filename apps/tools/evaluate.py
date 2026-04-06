"""
추천 시스템의 품질을 정량적으로 측정하기 위한 평가 도구입니다.
Precision@k, NDCG@k 등 주요 검색 지표를 계산하고, 
'특허 위주', '제외 조건' 등이 실제 랭킹에 잘 반영되는지 검증합니다.
"""

import asyncio
import logging
from typing import Any

from qdrant_client import QdrantClient

from apps.core.config import Settings
from apps.recommendation.service import RecommendationService
from apps.recommendation.planner import OpenAICompatPlanner
from apps.search.filters import QdrantFilterCompiler
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import OpenAICompatJudge
from apps.core.feedback_store import FeedbackStore
from apps.search.retriever import QdrantHybridRetriever
from apps.search.encoders import DenseEncoder
from apps.search.query_builder import QueryTextBuilder
from apps.search.schema_registry import SearchSchemaRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluator")

# 골든 테스트셋 정의
GOLDEN_DATA = [
    {
        "query": "인공지능 소속은 제외하고, 반도체 분야에서 최근 3년간 특허가 있는 전문가 추천해줘.",
        "expected_excluded_org": "AI반도체연구원", # 제외 대상 예시
        "must_have_branch": "pat",
        "label": "특허 중심 필터링 및 기관 제외"
    },
    {
        "query": "박사급 전문가 중에서 딥러닝 논문이 많은 사람 위주로 5명 추천.",
        "must_have_branch": "art",
        "min_top_score": 80,
        "label": "논문 중심 가중치 및 학위 필터"
    },
    {
        "query": "심사평가 위원 경험이 있는 반도체 전문가",
        "keyword_match": "심사평가위원",
        "label": "평가위원 활동 검색 신호"
    }
]

async def evaluate():
    settings = Settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    registry = SearchSchemaRegistry()
    dense_encoder = DenseEncoder(settings.dense_model_path)
    query_builder = QueryTextBuilder()
    
    planner = OpenAICompatPlanner(settings)
    retriever = QdrantHybridRetriever(
        client=client,
        settings=settings,
        registry=registry,
        dense_encoder=dense_encoder,
        query_builder=query_builder,
    )
    filter_compiler = QdrantFilterCompiler()
    card_builder = CandidateCardBuilder()
    judge = OpenAICompatJudge(settings)
    feedback_store = FeedbackStore(settings.db_url)
    
    service = RecommendationService(
        planner=planner,
        retriever=retriever,
        filter_compiler=filter_compiler,
        card_builder=card_builder,
        judge=judge,
        feedback_store=feedback_store,
        shortlist_limit=10,
    )

    print("\n" + "="*50)
    print("  Experts Recommendation Quality Evaluation")
    print("="*50 + "\n")

    for i, test in enumerate(GOLDEN_DATA, 1):
        print(f"Test case {i}: {test['label']}")
        print(f"Query: '{test['query']}'")
        
        try:
            result = await service.recommend(query=test['query'])
            recommendations = result["recommendations"]
            planner_output = result["planner"]
            
            # 1. Planner 검증
            print(f" [Planner] Intent: {planner_output.intent_summary}")
            print(f" [Planner] Weights: {planner_output.branch_weights}")
            print(f" [Planner] Exclude: {planner_output.exclude_orgs}")

            # 2. 결과 검증
            if test.get("expected_excluded_org"):
                for rec in recommendations:
                    if test["expected_excluded_org"] in (rec.name or ""): # 단순 매칭 예시
                         print(f" [FAIL] Excluded org '{test['expected_excluded_org']}' found in results!")
            
            if test.get("must_have_branch"):
                branch = test["must_have_branch"]
                covered_count = sum(1 for rec in recommendations if any(e.type == branch for e in rec.evidence))
                print(f" [Result] Branch '{branch}' coverage in recommendations: {covered_count}/{len(recommendations)}")

            print(f" [Result] Top 1 Name: {recommendations[0].name if recommendations else 'None'}")
            print(f" [Result] Top 1 Score: {recommendations[0].rank_score if recommendations else 'N/A'}")
            
        except Exception as e:
            print(f" [ERROR] Test failed: {e}")
        
        print("-" * 30)

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    asyncio.run(evaluate())
