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
    retrieval_score_traces: list[dict[str, Any]]


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

    @staticmethod
    def _sort_hit_records(
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            records,
            key=lambda record: (
                -record["hit"].score,
                " ".join(
                    (record["hit"].payload.basic_info.researcher_name or "").split()
                ),
                record["hit"].expert_id,
            ),
        )

    @staticmethod
    def _point_key(point_id: Any) -> str:
        return str(point_id)

    @staticmethod
    def _infer_point_branch(point_id: str) -> str | None:
        for branch in BRANCHES:
            if point_id.endswith(f"_{branch}") or f"_{branch}_" in point_id:
                return branch
        return None

    def _build_branch_query_payload(
        self,
        *,
        branch: str,
        dense_query: list[float],
        sparse_query: models.Document,
        query_filter: models.Filter | None,
    ) -> dict[str, Any]:
        return {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": [
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
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "query_filter": query_filter,
            "limit": self.settings.branch_output_limit,
            "with_payload": False,
            "with_vectors": False,
        }

    @staticmethod
    def _build_retrieval_score_trace(
        *,
        expert_id: str,
        point_id: str,
        final_score: float,
        branch_matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sorted_branch_matches = sorted(
            branch_matches,
            key=lambda item: (item["rank"], -item["score"], item["branch"]),
        )
        primary_branch = (
            sorted_branch_matches[0]["branch"] if sorted_branch_matches else None
        )
        return {
            "expert_id": expert_id,
            "point_id": point_id,
            "point_branch_hint": QdrantHybridRetriever._infer_point_branch(point_id),
            "final_score": round(float(final_score), 6),
            "primary_branch": primary_branch,
            "branch_matches": [
                {
                    "branch": item["branch"],
                    "rank": item["rank"],
                    "score": round(float(item["score"]), 6),
                }
                for item in sorted_branch_matches
            ],
        }

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
        branch_query_payloads: dict[str, dict[str, Any]] = {}
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
            branch_query_payloads[branch] = self._build_branch_query_payload(
                branch=branch,
                dense_query=dense_query,
                sparse_query=sparse_query,
                query_filter=query_filter,
            )

            prefetches.append(
                models.Prefetch(
                    prefetch=branch_query_payloads[branch]["prefetch"],
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
            query_results = await asyncio.gather(
                asyncio.to_thread(self.client.query_points, **query_payload),
                *[
                    asyncio.to_thread(self.client.query_points, **branch_payload)
                    for branch_payload in branch_query_payloads.values()
                ],
                return_exceptions=True,
            )
        if isinstance(query_results[0], Exception):
            raise query_results[0]
        points = query_results[0]
        logger.info("Qdrant search finished: elapsed_ms=%.2f", timer.elapsed_ms)

        branch_trace_map: dict[str, list[dict[str, Any]]] = {}
        branch_query_results = dict(zip(BRANCHES, query_results[1:]))
        for branch, branch_result in branch_query_results.items():
            if isinstance(branch_result, Exception):
                logger.warning(
                    "Skipping retrieval branch trace for %s: %s",
                    branch,
                    branch_result,
                )
                continue
            branch_points = getattr(branch_result, "points", branch_result)
            for rank, point in enumerate(branch_points, start=1):
                point_id = getattr(point, "id", None)
                if point_id is None and isinstance(point, dict):
                    point_id = point.get("id")
                if point_id is None:
                    continue
                branch_trace_map.setdefault(self._point_key(point_id), []).append(
                    {
                        "branch": branch,
                        "rank": rank,
                        "score": float(getattr(point, "score", 0.0)),
                    }
                )

        hit_records: list[dict[str, Any]] = []
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
                "basic": False,
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

            hit = SearchHit(
                expert_id=payload.basic_info.researcher_id,
                score=float(getattr(point, "score", 0.0)),
                payload=payload,
                data_presence_flags=data_presence_flags,
            )
            point_key = self._point_key(point_id)
            hit_records.append(
                {
                    "hit": hit,
                    "point_id": point_key,
                    "branch_matches": branch_trace_map.get(point_key, []),
                }
            )

        hit_records = self._sort_hit_records(hit_records)
        hits = [record["hit"] for record in hit_records]
        trace_gap_records = [
            {
                "expert_id": record["hit"].expert_id,
                "point_id": record["point_id"],
            }
            for record in hit_records
            if not record["branch_matches"]
        ]
        retrieval_score_traces = [
            self._build_retrieval_score_trace(
                expert_id=record["hit"].expert_id,
                point_id=record["point_id"],
                final_score=record["hit"].score,
                branch_matches=record["branch_matches"],
            )
            for record in hit_records
        ]
        if trace_gap_records:
            logger.warning(
                "Retrieval score traces missing branch matches: count=%d samples=%s",
                len(trace_gap_records),
                trace_gap_records[:5],
            )

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
            retrieval_score_traces=retrieval_score_traces,
        )
