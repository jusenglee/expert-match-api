"""
Qdrant를 사용하여 하이브리드 검색을 수행하는 리트리버 모델입니다.
Dense(의미론적) 검색과 Sparse(키워드) 검색을 결합하고,
여러 데이터 브랜치(Basic, Art, Pat, Pjt)의 결과를 RRF(Reciprocal Rank Fusion)로 통합합니다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import ExpertPayload, PlannerOutput, SearchHit
from apps.search.encoders import DenseEncoder
from apps.search.query_builder import QueryTextBuilder
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    """검색 결과를 담는 데이터 클래스입니다."""

    hits: list[SearchHit]  # 검색된 전문가 히트 리스트
    query_payload: dict[str, Any]  # Qdrant에 전달된 최종 쿼리 페이로드 (디버깅용)
    branch_queries: dict[str, str]  # 브랜치별로 최적화된 질의문 (디버깅용)


class QdrantHybridRetriever:
    """
    Qdrant의 다중 벡터 지원 기능을 활용하여 복합적인 전문가 검색을 수행합니다.
    """

    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        dense_encoder: DenseEncoder,
        query_builder: QueryTextBuilder,
    ) -> None:
        """
        리트리버 초기화 및 필요한 의존성 주입.
        """
        self.client = client
        self.settings = settings
        self.registry = registry
        self.dense_encoder = dense_encoder
        self.query_builder = query_builder

    async def search(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        """
        하이브리드 RRF 검색을 실행합니다.

        파이프라인:
        1. 브랜치별(논문/특허/과제/기본) 질의문 생성
        2. 각 브랜치 내부에서 Dense + Sparse 하이브리드 검색용 Prefetch 조립
        3. 모든 브랜치 결과를 최상위에서 RRF로 다시 결합하여 최종 순위 산정
        """
        # 1. 브랜치별 최적화된 질의 텍스트 구성
        branch_queries = self.query_builder.build_branch_queries(query, plan)
        logger.debug("브랜치별 생성 쿼리: %s", branch_queries)
        prefetches: list[models.Prefetch] = []

        # 2. 각 데이터 브랜치별로 Prefetch 쿼리 조립
        for branch in BRANCHES:
            # 의미론적 검색을 위한 Dense 임베딩 생성
            dense_query = self.dense_encoder.embed(branch_queries[branch])
            # 키워드 검색을 위한 Sparse 쿼리 구성 (BM25 로컬 인코딩)
            sparse_query = models.Document(
                text=branch_queries[branch], model=self.settings.bm25_model_name
            )

            # 브랜치 레벨 RRF: Dense와 Sparse 결과를 합침
            prefetches.append(
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=dense_query,
                            using=self.registry.dense_vector_by_branch[branch],
                            limit=self.settings.branch_prefetch_limit,
                        ),
                        models.Prefetch(
                            query=sparse_query,
                            using=self.registry.sparse_vector_by_branch[branch],
                            limit=self.settings.branch_prefetch_limit,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.settings.branch_output_limit,
                )
            )

        # 3. 최종 통합 쿼리 구성 (모든 브랜치 통합 RRF)
        query_payload = {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": prefetches,
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "query_filter": query_filter,
            "limit": self.settings.retrieval_limit,
            "with_payload": True,
            "with_vectors": False,
        }

        # Qdrant 호출 (동기 서버 메서드를 비동기 스레드에서 실행)
        logger.info(
            "Qdrant 하이브리드 검색 실행: 컬렉션=%s 필터 적용 여부=%s",
            self.settings.qdrant_collection_name,
            "예" if query_filter else "아니오",
        )
        points = await asyncio.to_thread(self.client.query_points, **query_payload)

        hits: list[SearchHit] = []
        skipped_invalid_payloads = 0

        # 4. 검색 결과 파싱 및 도메인 모델 변환
        point_list = getattr(points, "points", points)
        for point in point_list:
            point_id = getattr(point, "id", None)
            if point_id is None and isinstance(point, dict):
                point_id = point.get("id")

            payload_data = (
                point.payload if hasattr(point, "payload") else point.get("payload", {})
            )

            try:
                # Pydantic 모델을 사용한 데이터 무결성 검증
                payload = ExpertPayload.model_validate(payload_data)
            except ValidationError as exc:
                skipped_invalid_payloads += 1
                logger.warning(
                    "유효하지 않은 Qdrant 페이로드 스킵: point_id=%s 오류=%s",
                    point_id,
                    exc.errors(include_url=False),
                )
                continue

            # 브랜치 커버리지 분석: 실제 데이터 유무 확인
            branch_coverage = {
                "basic": True,
                "art": bool(payload.publications),
                "pat": bool(payload.intellectual_properties),
                "pjt": bool(payload.research_projects),
            }

            hits.append(
                SearchHit(
                    expert_id=payload.basic_info.researcher_id,
                    score=float(getattr(point, "score", 0.0)),
                    payload=payload,
                    branch_coverage=branch_coverage,
                )
            )

        if skipped_invalid_payloads:
            logger.info(
                "검색 완료: 총 %d명의 히트 중 %d명의 데이터가 유효하지 않아 스킵되었습니다.",
                len(point_list),
                skipped_invalid_payloads,
            )
        else:
            logger.info("검색 완료: 총 %d명의 전문가 히트 반환", len(hits))

        return RetrievalResult(
            hits=hits, query_payload=query_payload, branch_queries=branch_queries
        )
