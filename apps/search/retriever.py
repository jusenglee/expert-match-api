"""
Hybrid Qdrant retriever for NTIS expert search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from qdrant_client import QdrantClient, models

from apps.core.cache import RetrievalResultCache
from apps.core.config import Settings
from apps.core.timer import Timer
from apps.domain.models import ExpertPayload, PlannerOutput, SearchHit
from apps.search.encoders import DenseEncoder, SparseEncoder
from apps.search.query_builder import CompiledBranchQueries, QueryTextBuilder
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.sparse_runtime import SparseRuntimeConfig
from apps.search.text_utils import normalize_org_name

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalResult:
    hits: list[SearchHit]
    query_payload: dict[str, Any]
    branch_queries: dict[str, CompiledBranchQueries]
    retrieval_keywords: list[str]
    retrieval_score_traces: list[dict[str, Any]]
    expanded_shadow_hits: list[SearchHit] = field(default_factory=list)
    filtered_out_candidates: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False


# 가중치 설정 (아키텍처 가이드라인 준수)
BRANCH_WEIGHTS = {
    "basic": 0.6,
    "art": 1.0,
    "pat": 0.8,
    "pjt": 1.0,
}

PATH_WEIGHTS = {
    "stable": 1.0,
    "expanded": 0.25,
}

RETRIEVAL_MODE = "keyword_pool_then_hybrid"


class QdrantHybridRetriever:
    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        dense_encoder: DenseEncoder,
        query_builder: QueryTextBuilder,
        sparse_encoder: SparseEncoder | None = None,
        sparse_runtime: SparseRuntimeConfig | None = None,
        l3_cache: RetrievalResultCache | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry
        self.dense_encoder = dense_encoder
        self.query_builder = query_builder
        self.sparse_encoder = sparse_encoder
        self.sparse_runtime = sparse_runtime
        self.l3_cache = l3_cache

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
    def _point_key(point_id: Any) -> str:
        return str(point_id)

    @staticmethod
    def _infer_point_branch(point_id: str) -> str | None:
        for branch in BRANCHES:
            if point_id.endswith(f"_{branch}") or f"_{branch}_" in point_id:
                return branch
        return None

    @staticmethod
    def _as_condition_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    @staticmethod
    def _point_payload_data(point: Any) -> dict[str, Any]:
        return getattr(point, "payload", {}) or (
            point.get("payload", {}) if isinstance(point, dict) else {}
        )

    @staticmethod
    def _point_id(point: Any) -> Any:
        return getattr(point, "id", None) or (
            point.get("id") if isinstance(point, dict) else None
        )

    @staticmethod
    def _point_count(points: Any) -> int:
        try:
            return len(points)
        except TypeError:
            return 0

    @staticmethod
    def _build_path_tasks(
        branch_compiled_queries: dict[str, CompiledBranchQueries],
    ) -> list[tuple[str, str, str]]:
        tasks_meta: list[tuple[str, str, str]] = []
        for branch in BRANCHES:
            tasks_meta.append((branch, "stable", branch_compiled_queries[branch].stable))
            if branch in ["art", "pjt"]:
                tasks_meta.append((branch, "expanded", branch_compiled_queries[branch].expanded))
        return tasks_meta

    def _build_branch_query_payload(
        self,
        *,
        branch: str,
        dense_query: list[float],
        sparse_query: models.Document | models.SparseVector,
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
            "with_payload": True,
            "with_vectors": False,
        }

    def _build_keyword_query_payload(
        self,
        *,
        branch: str,
        sparse_query: models.Document | models.SparseVector,
        query_filter: models.Filter | None,
    ) -> dict[str, Any]:
        return {
            "collection_name": self.settings.qdrant_collection_name,
            "query": sparse_query,
            "using": self.registry.sparse_vector_by_branch[branch],
            "query_filter": query_filter,
            "limit": self.settings.branch_prefetch_limit,
            "with_payload": True,
            "with_vectors": False,
        }

    def _build_sparse_query(self, query_text: str) -> models.Document | models.SparseVector:
        if self.sparse_encoder:
            sparse_map = self.sparse_encoder.embed(query_text)
            return models.SparseVector(
                indices=list(sparse_map.keys()),
                values=list(sparse_map.values()),
            )

        sparse_model_name = (
            self.sparse_runtime.active_model_name
            if self.sparse_runtime is not None
            else self.settings.sparse_model_name
        )
        return models.Document(
            text=query_text,
            model=sparse_model_name,
        )

    @staticmethod
    def _with_candidate_pool_filter(
        query_filter: models.Filter | None,
        researcher_ids: list[str],
    ) -> models.Filter:
        candidate_condition = models.FieldCondition(
            key="basic_info.researcher_id",
            match=models.MatchAny(any=researcher_ids),
        )
        if query_filter is None:
            return models.Filter(must=[candidate_condition])

        return models.Filter(
            should=query_filter.should,
            min_should=query_filter.min_should,
            must=[
                *QdrantHybridRetriever._as_condition_list(query_filter.must),
                candidate_condition,
            ],
            must_not=query_filter.must_not,
        )

    @staticmethod
    def _collect_keyword_candidate_pool(
        keyword_results: list[Any],
    ) -> tuple[list[str], dict[str, int]]:
        candidate_ids: list[str] = []
        seen_ids: set[str] = set()
        branch_counts: dict[str, int] = {}

        for result in keyword_results:
            if isinstance(result, Exception):
                logger.error("Keyword query task failed: %s", result)
                continue

            branch, path, points = result
            valid_count = 0
            for point in points:
                try:
                    payload = ExpertPayload.model_validate(
                        QdrantHybridRetriever._point_payload_data(point)
                    )
                except ValidationError:
                    continue

                valid_count += 1
                researcher_id = payload.basic_info.researcher_id
                if researcher_id and researcher_id not in seen_ids:
                    seen_ids.add(researcher_id)
                    candidate_ids.append(researcher_id)

            branch_counts[f"{branch}:{path}"] = valid_count

        return candidate_ids, branch_counts

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
        # 1. 쿼리 컴파일
        branch_compiled_queries = self.query_builder.build_branch_queries(query, plan)
        keyword_compiled_queries = self.query_builder.build_keyword_branch_queries(query, plan)
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        logger.info(
            "검색 쿼리 컴파일 완료: mode=%s retrieval_keywords=%d semantic_query=%s bundle_ids=%d",
            RETRIEVAL_MODE,
            len(retrieval_keywords),
            bool(plan.semantic_query),
            len(plan.bundle_ids),
        )

        # L3 캐시 키 준비
        compiled_json = json.dumps(
            {
                "retrieval_mode": RETRIEVAL_MODE,
                "branch_queries": {
                    b: {"s": q.stable, "e": q.expanded}
                    for b, q in branch_compiled_queries.items()
                },
                "keyword_branch_queries": {
                    b: {"s": q.stable, "e": q.expanded}
                    for b, q in keyword_compiled_queries.items()
                },
            },
            sort_keys=True,
        )
        filter_json = str(query_filter)
        snapshot_id = self.settings.qdrant_collection_release_id

        # 2. L3 캐시 조회
        if self.l3_cache and self.settings.cache_enabled:
            cached_data = self.l3_cache.get(compiled_json, filter_json, snapshot_id)
            if cached_data:
                hits = [SearchHit.model_validate(h) for h in cached_data]
                logger.info(
                    "검색 캐시 적중: layer=L3 mode=%s cached_hits=%d snapshot_id=%s",
                    RETRIEVAL_MODE,
                    len(hits),
                    snapshot_id,
                )
                return RetrievalResult(
                    hits=hits,
                    query_payload={
                        "cache": "hit",
                        "l3": True,
                        "retrieval_mode": RETRIEVAL_MODE,
                        "keyword_stage_skipped_reason": "l3_cache_hit",
                        "hybrid_stage_skipped_reason": "l3_cache_hit",
                    },
                    branch_queries=branch_compiled_queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                    cache_hit=True
                )

        # 3. 실행할 브랜치/경로 조합 정의
        tasks_meta = self._build_path_tasks(branch_compiled_queries)
        keyword_tasks_meta = self._build_path_tasks(keyword_compiled_queries)

        logger.info(
            "1차 키워드 검색 시작: mode=%s paths=%d query_filter=%s retrieval_keywords=%d",
            RETRIEVAL_MODE,
            len(keyword_tasks_meta),
            query_filter is not None,
            len(retrieval_keywords),
        )
        keyword_tasks = [
            self._execute_single_path_keyword_query_with_path(branch, path, q_text, query_filter)
            for branch, path, q_text in keyword_tasks_meta
        ]

        with Timer() as keyword_timer:
            keyword_results = await asyncio.gather(*keyword_tasks, return_exceptions=True)

        keyword_candidate_ids, keyword_branch_counts = self._collect_keyword_candidate_pool(
            keyword_results
        )
        logger.info(
            "1차 키워드 검색 완료: elapsed_ms=%.2f unique_candidates=%d branch_counts=%s",
            keyword_timer.elapsed_ms,
            len(keyword_candidate_ids),
            keyword_branch_counts,
        )

        if not keyword_candidate_ids:
            logger.warning(
                "2차 하이브리드 검색 스킵: reason=keyword_stage_empty branch_counts=%s",
                keyword_branch_counts,
            )
            return RetrievalResult(
                hits=[],
                query_payload={
                    "retrieval_mode": RETRIEVAL_MODE,
                    "weighted_rrf": False,
                    "keyword_stage_candidate_count": 0,
                    "keyword_stage_branch_counts": keyword_branch_counts,
                    "hybrid_stage_candidate_filter_count": 0,
                    "hybrid_stage_raw_branch_counts": {},
                    "aggregated_candidate_count": 0,
                    "support_pass_count": 0,
                    "support_filtered_count": 0,
                    "hybrid_stage_skipped_reason": "keyword_stage_empty",
                    "support_rules": {
                        "s_min": self.settings.support_rule_stable_min,
                        "e_min": self.settings.support_rule_expanded_min,
                    },
                    "timers": {
                        "keyword_stage_ms": keyword_timer.elapsed_ms,
                        "hybrid_stage_ms": 0.0,
                    },
                },
                branch_queries=branch_compiled_queries,
                retrieval_keywords=retrieval_keywords,
                retrieval_score_traces=[],
            )

        hybrid_query_filter = self._with_candidate_pool_filter(
            query_filter,
            keyword_candidate_ids,
        )
        keyword_candidate_id_set = set(keyword_candidate_ids)

        # 4. 병렬 검색 실행
        logger.info(
            "2차 하이브리드 검색 시작: paths=%d candidate_filter_count=%d query_filter=%s semantic_query=%s",
            len(tasks_meta),
            len(keyword_candidate_ids),
            hybrid_query_filter is not None,
            bool(plan.semantic_query),
        )
        search_tasks = [
            self._execute_single_path_query_with_path(branch, path, q_text, hybrid_query_filter)
            for branch, path, q_text in tasks_meta
        ]
            
        with Timer() as hybrid_timer:
            raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        logger.info("2차 하이브리드 검색 완료: elapsed_ms=%.2f", hybrid_timer.elapsed_ms)

        # 5. 결과 집계 및 Weighted RRF 계산
        aggregator: dict[str, dict[str, Any]] = {}
        hybrid_branch_counts: dict[str, int] = {}
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error("Query task failed: %s", result)
                continue
            
            branch, path, points = result
            hybrid_branch_counts[f"{branch}:{path}"] = self._point_count(points)
            weight = BRANCH_WEIGHTS[branch] * PATH_WEIGHTS[path]
            
            for rank, pt in enumerate(points, start=1):
                pt_id = self._point_id(pt)
                payload_data = self._point_payload_data(pt)
                try:
                    payload = ExpertPayload.model_validate(payload_data)
                except ValidationError: continue
                
                eid = payload.basic_info.researcher_id
                if eid not in keyword_candidate_id_set:
                    continue

                rrf_contribution = weight * (1.0 / (rank + 60))
                
                if eid not in aggregator:
                    aggregator[eid] = {
                        "score": 0.0, "stable_hits": 0, "expanded_hits": 0,
                        "branches": set(), "payload": payload, "point_id": self._point_key(pt_id),
                        "branch_matches": []
                    }
                
                entry = aggregator[eid]
                entry["score"] += rrf_contribution
                entry["branches"].add(branch)
                if path == "stable": entry["stable_hits"] += 1
                else: entry["expanded_hits"] += 1
                
                entry["branch_matches"].append({
                    "branch": branch, "path": path, "rank": rank,
                    "score": float(getattr(pt, "score", 0.0) if hasattr(pt, "score") else pt.get("score", 0.0))
                })

        # 6. Support Rule 필터링 및 결과 생성
        final_hits: list[SearchHit] = []
        filtered_out: list[dict[str, Any]] = []
        
        s_min = self.settings.support_rule_stable_min
        e_min = self.settings.support_rule_expanded_min

        for eid, data in aggregator.items():
            if plan.exclude_orgs:
                organization = data["payload"].basic_info.affiliated_organization or ""
                normalized_org = normalize_org_name(organization) or ""
                if any((normalize_org_name(ex) in normalized_org for ex in plan.exclude_orgs if normalize_org_name(ex))):
                    continue

            is_supported = (data["stable_hits"] >= s_min) or (data["expanded_hits"] >= e_min and len(data["branches"]) >= e_min)
            
            hit = SearchHit(
                expert_id=eid,
                score=data["score"],
                payload=data["payload"],
                data_presence_flags={
                    "basic": False,
                    "art": bool(data["payload"].publications),
                    "pat": bool(data["payload"].intellectual_properties),
                    "pjt": bool(data["payload"].research_projects),
                },
                stable_support_count=data["stable_hits"],
                expanded_support_count=data["expanded_hits"],
                support_branches=list(data["branches"])
            )

            if is_supported:
                final_hits.append(hit)
            else:
                filtered_out.append({
                    "expert_id": eid,
                    "name": data["payload"].basic_info.researcher_name,
                    "stable_hits": data["stable_hits"],
                    "expanded_hits": data["expanded_hits"],
                    "branches": list(data["branches"]),
                    "reason": f"Insufficient support (stable < {s_min} and expanded_branches < {e_min})"
                })

        final_hits = self._sort_hits(final_hits)
        logger.info(
            "검색 집계 완료: raw_branch_counts=%s aggregated_candidates=%d support_pass=%d support_filtered=%d final_hits=%d",
            hybrid_branch_counts,
            len(aggregator),
            len(final_hits),
            len(filtered_out),
            len(final_hits),
        )
        
        if self.l3_cache and self.settings.cache_enabled and final_hits:
            self.l3_cache.set(
                compiled_json, filter_json, snapshot_id,
                [h.model_dump(mode="json") for h in final_hits]
            )

        retrieval_score_traces = [
            self._build_retrieval_score_trace(
                expert_id=hit.expert_id,
                point_id=aggregator[hit.expert_id]["point_id"],
                final_score=hit.score,
                branch_matches=aggregator[hit.expert_id]["branch_matches"]
            ) for hit in final_hits
        ]

        expanded_shadow = [
            SearchHit(expert_id=eid, score=d["score"], payload=d["payload"])
            for eid, d in aggregator.items() if d["stable_hits"] == 0 and d["expanded_hits"] > 0
        ]

        return RetrievalResult(
            hits=final_hits,
            query_payload={
                "retrieval_mode": RETRIEVAL_MODE,
                "weighted_rrf": True,
                "keyword_stage_candidate_count": len(keyword_candidate_ids),
                "keyword_stage_branch_counts": keyword_branch_counts,
                "hybrid_stage_candidate_filter_count": len(keyword_candidate_ids),
                "hybrid_stage_raw_branch_counts": hybrid_branch_counts,
                "aggregated_candidate_count": len(aggregator),
                "support_pass_count": len(final_hits),
                "support_filtered_count": len(filtered_out),
                "support_rules": {"s_min": s_min, "e_min": e_min},
                "timers": {
                    "keyword_stage_ms": keyword_timer.elapsed_ms,
                    "hybrid_stage_ms": hybrid_timer.elapsed_ms,
                },
            },
            branch_queries=branch_compiled_queries,
            retrieval_keywords=retrieval_keywords,
            retrieval_score_traces=retrieval_score_traces,
            expanded_shadow_hits=expanded_shadow,
            filtered_out_candidates=filtered_out
        )

    async def _execute_single_path_query(
        self, 
        branch: str, 
        query_text: str, 
        query_filter: models.Filter | None
    ) -> tuple[str, str, Any]:
        """단일 브랜치/경로에 대해 Qdrant 하이브리드 검색을 수행합니다."""
        processed_query = query_text
        if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
            instruct_prefix = (
                "Instruct: Find experts whose profile, papers, patents, or projects "
                "match the given query.\nQuery: "
            )
            processed_query = f"{instruct_prefix}{query_text}"
        
        dense_query = self.dense_encoder.embed(processed_query)

        # Sparse 임베딩 생성 (커스텀 인코더 우선 활용)
        sparse_query = self._build_sparse_query(query_text)
        
        payload = self._build_branch_query_payload(
            branch=branch,
            dense_query=dense_query,
            sparse_query=sparse_query,
            query_filter=query_filter,
        )
        
        result = await asyncio.to_thread(self.client.query_points, **payload)
        points = getattr(result, "points", result)
        
        return branch, "", points

    async def _execute_single_path_query_with_path(
        self, 
        branch: str, 
        path: str,
        query_text: str, 
        query_filter: models.Filter | None
    ) -> tuple[str, str, Any]:
        _, _, points = await self._execute_single_path_query(branch, query_text, query_filter)
        return branch, path, points

    async def _execute_single_path_keyword_query(
        self,
        branch: str,
        query_text: str,
        query_filter: models.Filter | None,
    ) -> tuple[str, str, Any]:
        """Run the first-stage sparse keyword query for one branch."""
        sparse_query = self._build_sparse_query(query_text)
        payload = self._build_keyword_query_payload(
            branch=branch,
            sparse_query=sparse_query,
            query_filter=query_filter,
        )

        result = await asyncio.to_thread(self.client.query_points, **payload)
        points = getattr(result, "points", result)

        return branch, "", points

    async def _execute_single_path_keyword_query_with_path(
        self,
        branch: str,
        path: str,
        query_text: str,
        query_filter: models.Filter | None,
    ) -> tuple[str, str, Any]:
        _, _, points = await self._execute_single_path_keyword_query(
            branch,
            query_text,
            query_filter,
        )
        return branch, path, points
