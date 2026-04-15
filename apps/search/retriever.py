"""
Hybrid Qdrant retriever for NTIS expert search.
"""

from __future__ import annotations

import asyncio
import logging
import pprint
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.core.timer import Timer
from apps.domain.models import ExpertPayload, PlannerOutput, SearchHit
from apps.search.encoders import DenseEncoder
from apps.search.query_builder import QueryTextBuilder
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.text_utils import normalize_org_name

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    query_payload: dict[str, Any]
    branch_queries: dict[str, str]
    retrieval_keywords: list[str]


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

    @staticmethod
    def _sanitize_payload_for_log(data: Any) -> Any:
        if hasattr(data, "model_dump"):
            try:
                data = data.model_dump()
            except Exception:
                pass
        elif hasattr(data, "dict") and callable(data.dict):
            try:
                data = data.dict()
            except Exception:
                pass

        if isinstance(data, dict):
            return {
                key: QdrantHybridRetriever._sanitize_payload_for_log(value)
                for key, value in data.items()
            }
        if isinstance(data, list):
            if len(data) > 100 and all(
                isinstance(value, (float, int)) for value in data[:10]
            ):
                return f"<Dense Vector: {len(data)} dimensions>"
            return [
                QdrantHybridRetriever._sanitize_payload_for_log(value)
                for value in data
            ]
        return data

    @staticmethod
    def _sort_hits(hits: list[SearchHit]) -> list[SearchHit]:
        return sorted(
            hits,
            key=lambda hit: (
                -hit.score,
                " ".join((hit.payload.basic_info.researcher_name or "").split()),
                hit.expert_id,
            ),
        )

    async def search(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        retrieval_keywords = self.query_builder.normalize_keywords(plan.core_keywords)
        branch_queries = self.query_builder.build_branch_queries(query, plan)
        logger.debug(
            "Generated branch queries:\n%s",
            pprint.pformat(branch_queries, indent=2, width=100),
        )
        prefetches: list[models.Prefetch] = []

        for branch in BRANCHES:
            dense_query_text = branch_queries[branch]
            if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
                instruct_prefix = (
                    "Instruct: Find experts whose profile, papers, patents, or projects "
                    "match the given query.\nQuery: "
                )
                dense_query_text = f"{instruct_prefix}{dense_query_text}"
            dense_query = self.dense_encoder.embed(dense_query_text)
            sparse_query = models.Document(
                text=branch_queries[branch],
                model=self.settings.bm25_model_name,
            )

            prefetches.append(
                models.Prefetch(
                    prefetch=[
                        models.Prefetch(
                            query=dense_query,
                            using=self.registry.dense_vector_by_branch[branch],
                            limit=self.settings.branch_prefetch_limit,
                            filter=query_filter,
                        ),
                        models.Prefetch(
                            query=sparse_query,
                            using=self.registry.sparse_vector_by_branch[branch],
                            limit=self.settings.branch_prefetch_limit,
                            filter=query_filter,
                        ),
                    ],
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=self.settings.branch_output_limit,
                    filter=query_filter,
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

        logger.info(
            "Qdrant hybrid search: collection=%s filter_applied=%s",
            self.settings.qdrant_collection_name,
            query_filter is not None,
        )
        logger.info(
            "Qdrant query payload:\n%s",
            pprint.pformat(
                self._sanitize_payload_for_log(query_payload), indent=2, width=120
            ),
        )
        with Timer() as timer:
            points = await asyncio.to_thread(self.client.query_points, **query_payload)
        logger.info("Qdrant search finished: elapsed_ms=%.2f", timer.elapsed_ms)

        hits: list[SearchHit] = []
        skipped_invalid_payloads = 0
        point_list = getattr(points, "points", points)
        for point in point_list:
            point_id = getattr(point, "id", None)
            if point_id is None and isinstance(point, dict):
                point_id = point.get("id")

            payload_data = (
                point.payload if hasattr(point, "payload") else point.get("payload", {})
            )

            try:
                payload = ExpertPayload.model_validate(payload_data)
            except ValidationError as exc:
                skipped_invalid_payloads += 1
                logger.warning(
                    "Skipping invalid payload: point_id=%s errors=%s",
                    point_id,
                    exc.errors(include_url=False),
                )
                continue

            data_presence_flags = {
                "basic": True,
                "art": bool(payload.publications),
                "pat": bool(payload.intellectual_properties),
                "pjt": bool(payload.research_projects),
            }

            if plan.exclude_orgs:
                organization = payload.basic_info.affiliated_organization or ""
                normalized_org = normalize_org_name(organization) or ""
                should_exclude = False
                for excluded_org in plan.exclude_orgs:
                    normalized_excluded_org = normalize_org_name(excluded_org) or ""
                    if normalized_excluded_org and normalized_org:
                        if (
                            normalized_excluded_org in normalized_org
                            or normalized_org in normalized_excluded_org
                        ):
                            should_exclude = True
                            break
                if should_exclude:
                    continue

            hits.append(
                SearchHit(
                    expert_id=payload.basic_info.researcher_id,
                    score=float(getattr(point, "score", 0.0)),
                    payload=payload,
                    data_presence_flags=data_presence_flags,
                )
            )

        hits = self._sort_hits(hits)

        if skipped_invalid_payloads:
            logger.info(
                "Search completed with payload skips: total=%d skipped=%d returned=%d",
                len(point_list),
                skipped_invalid_payloads,
                len(hits),
            )
        else:
            logger.info("Search completed: hits=%d", len(hits))

        return RetrievalResult(
            hits=hits,
            query_payload=query_payload,
            branch_queries=branch_queries,
            retrieval_keywords=retrieval_keywords,
        )
