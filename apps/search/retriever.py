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
    hits: list[SearchHit]
    query_payload: dict[str, Any]
    branch_queries: dict[str, str]


class QdrantHybridRetriever:
    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        dense_encoder: DenseEncoder,
        query_builder: QueryTextBuilder,
    ) -> None:
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
        # planner는 branch를 끄지 않고, branch별 질의문만 다르게 만든다.
        # 따라서 retrieval은 basic/art/pat/pjt를 항상 모두 조립한다.
        branch_queries = self.query_builder.build_branch_queries(query, plan)
        prefetches: list[models.Prefetch] = []
        for branch in BRANCHES:
            dense_query = self.dense_encoder.embed(branch_queries[branch])
            sparse_query = models.Document(text=branch_queries[branch], model="qdrant/bm25")
            # 각 branch 내부에서는 dense와 sparse를 먼저 RRF로 결합하고,
            # 최상위에서는 다시 branch 결과를 RRF로 합친다.
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

        query_payload = {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": prefetches,
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "query_filter": query_filter,
            "limit": self.settings.retrieval_limit,
            "with_payload": True,
            "with_vectors": False,
        }
        points = await asyncio.to_thread(self.client.query_points, **query_payload)
        hits: list[SearchHit] = []
        skipped_invalid_payloads = 0
        for point in getattr(points, "points", points):
            point_id = getattr(point, "id", None)
            if point_id is None and isinstance(point, dict):
                point_id = point.get("id")
            payload_data = point.payload if hasattr(point, "payload") else point.get("payload", {})
            try:
                payload = ExpertPayload.model_validate(payload_data)
            except ValidationError as exc:
                skipped_invalid_payloads += 1
                logger.warning(
                    "Skipping invalid Qdrant payload during retrieval: point_id=%s errors=%s",
                    point_id,
                    exc.errors(include_url=False),
                )
                continue
            # MVP에서는 branch별 세부 점수 분해 대신,
            # 실제 payload 안에 해당 근거가 존재하는지를 coverage로 요약한다.
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
            logger.info("Skipped %d invalid Qdrant payload(s) during retrieval.", skipped_invalid_payloads)
        return RetrievalResult(hits=hits, query_payload=query_payload, branch_queries=branch_queries)
