"""
Hybrid Qdrant retriever for NTIS expert search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pprint
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
    """
    Qdrant 검색 엔진의 최종 검색 결과를 담는 데이터 구조입니다.
    검색된 전문가 목록(hits)과 검색에 사용된 쿼리 메타데이터, 추적용 로그 정보(traces)들을 포괄합니다.
    """
    hits: list[SearchHit]
    query_payload: dict[str, Any]
    branch_queries: dict[str, CompiledBranchQueries]
    retrieval_keywords: list[str]
    retrieval_score_traces: list[dict[str, Any]]
    expanded_shadow_hits: list[SearchHit] = field(default_factory=list)
    filtered_out_candidates: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False


# 가중치 설정 (아키텍처 가이드라인 준수)
RRF_K = 60.0
RETRIEVAL_QUERY_SCHEMA_VERSION = "v3_branch_instruct_prefix"

# multilingual-e5-large-instruct 브랜치별 asymmetric instruct prefix
# - query side: "Instruct: <task>\nQuery: <text>"
# - document side: 접두어 없음 (인덱싱 시 raw text 그대로)
# - branch hint를 instruct task 문장에 통합하여 임베딩 방향을 브랜치 특화
BRANCH_INSTRUCT_PREFIXES: dict[str, str] = {
    "basic": (
        "Instruct: Find Korean research experts whose academic background, "
        "institutional affiliation, and technical expertise match the given query.\nQuery: "
    ),
    "art": (
        "Instruct: Find Korean research experts whose academic publications, "
        "papers, or research articles are about the given topic.\nQuery: "
    ),
    "pat": (
        "Instruct: Find Korean research experts whose patents or inventions "
        "are related to the given technology.\nQuery: "
    ),
    "pjt": (
        "Instruct: Find Korean research experts who have conducted government-funded "
        "research projects related to the given topic.\nQuery: "
    ),
}


class QdrantHybridRetriever:
    """
    Qdrant 벡터 데이터베이스를 사용하여 전문가를 검색하는 핵심 하이브리드(Hybrid) 검색기입니다.
    Dense(의미 기반) 임베딩과 Sparse(키워드 기반, BM25) 임베딩을 결합하여, 
    논문(art), 특허(pat), 과제(pjt) 등 각 데이터 브랜치별로 병렬 검색을 수행하고 결과를 집계합니다.
    """
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
    def _has_distinct_expanded_query(compiled: CompiledBranchQueries) -> bool:
        return compiled.expanded_differs()

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
        sparse_query: models.Document | models.SparseVector,
        query_filter: models.Filter | None,
    ) -> dict[str, Any]:
        keyword_prefetch = models.Prefetch(
            query=sparse_query,
            using=self.registry.sparse_vector_by_branch[branch],
            limit=self.settings.keyword_prefetch_limit,
            filter=query_filter,
        )
        
        return {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": [
                # Dense branch: narrow down candidates first using keyword_prefetch, then apply dense search
                models.Prefetch(
                    prefetch=[keyword_prefetch],
                    query=dense_query,
                    using=self.registry.dense_vector_by_branch[branch],
                    limit=self.settings.branch_prefetch_limit,
                ),
                # Sparse branch: narrow down candidates first using keyword_prefetch, then apply sparse search
                models.Prefetch(
                    prefetch=[keyword_prefetch],
                    query=sparse_query,
                    using=self.registry.sparse_vector_by_branch[branch],
                    limit=self.settings.branch_prefetch_limit,
                ),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit": self.settings.branch_output_limit,
            "with_payload": True,
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
        """
        플래너가 추출한 검색 의도(plan)를 바탕으로 여러 브랜치(기본정보, 논문, 특허, 과제)와 
        경로(기본 쿼리, 확장 쿼리)에 대해 병렬로 Qdrant 검색을 실행합니다.
        검색된 여러 결과들의 점수를 하나로 합산(Aggregating)하고 필터링하여 최종 후보자 목록을 반환합니다.
        """
        # 1. 쿼리 컴파일
        branch_compiled_queries = self.query_builder.build_branch_queries(query, plan)
        retrieval_keywords = self.query_builder.normalize_keywords(plan.retrieval_core)

        # L3 캐시 키 준비
        compiled_json = json.dumps(
            {
                "schema_version": RETRIEVAL_QUERY_SCHEMA_VERSION,
                "branches": {
                    branch: {
                        "stable": compiled.stable,
                        "expanded": compiled.expanded,
                        "stable_dense": compiled.stable_dense,
                        "stable_sparse": compiled.stable_sparse,
                        "expanded_dense": compiled.expanded_dense,
                        "expanded_sparse": compiled.expanded_sparse,
                        "dense_base_source": compiled.dense_base_source,
                        "sparse_base_source": compiled.sparse_base_source,
                    }
                    for branch, compiled in branch_compiled_queries.items()
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
                logger.info("Retriever cache hit (L3)")
                hits = [SearchHit.model_validate(h) for h in cached_data]
                return RetrievalResult(
                    hits=hits,
                    query_payload={"cache": "hit", "l3": True},
                    branch_queries=branch_compiled_queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                    cache_hit=True
                )

        # 3. 실행할 브랜치/경로 조합 정의
        tasks_meta: list[tuple[str, str, str, str]] = []
        for branch in BRANCHES:
            compiled = branch_compiled_queries[branch]
            tasks_meta.append(
                (
                    branch,
                    "stable",
                    compiled.stable_dense,
                    compiled.stable_sparse,
                )
            )
            if self._has_distinct_expanded_query(compiled):
                tasks_meta.append(
                    (
                        branch,
                        "expanded",
                        compiled.expanded_dense,
                        compiled.expanded_sparse,
                    )
                )

        # 4. 병렬 검색 실행
        search_tasks = [
            self._execute_single_path_query_with_path(
                branch,
                path,
                dense_text,
                sparse_text,
                query_filter,
            )
            for branch, path, dense_text, sparse_text in tasks_meta
        ]
            
        with Timer() as timer:
            raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        logger.info("Multi-path parallel search finished: elapsed_ms=%.2f", timer.elapsed_ms)

        # 5. 결과 집계 및 점수 산정 (Max Embedding Score)
        aggregator: dict[str, dict[str, Any]] = {}
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error("Query task failed: %s", result)
                continue
            
            branch, path, points = result
            
            for rank, pt in enumerate(points, start=1):
                pt_id = getattr(pt, "id", None) or (pt.get("id") if isinstance(pt, dict) else None)
                payload_data = getattr(pt, "payload", {}) or (pt.get("payload", {}) if isinstance(pt, dict) else {})
                try:
                    payload = ExpertPayload.model_validate(payload_data)
                except ValidationError: continue
                
                eid = payload.basic_info.researcher_id
                raw_score = float(getattr(pt, "score", 0.0) if hasattr(pt, "score") else pt.get("score", 0.0))
                
                if raw_score < self.settings.retrieval_score_cutoff:
                    continue
                    
                branch_weight = self.settings.branch_weights.get(branch, 1.0)
                score = raw_score * branch_weight
                
                if eid not in aggregator:
                    aggregator[eid] = {
                        "score": score, "stable_hits": 0, "expanded_hits": 0,
                        "branches": set(), "payload": payload, "point_id": self._point_key(pt_id),
                        "branch_matches": []
                    }
                else:
                    aggregator[eid]["score"] = max(aggregator[eid]["score"], score)
                
                entry = aggregator[eid]
                entry["branches"].add(branch)
                if path == "stable": entry["stable_hits"] += 1
                else: entry["expanded_hits"] += 1
                
                entry["branch_matches"].append({
                    "branch": branch, "path": path, "rank": rank,
                    "score": score
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

            # 소속 미상(organization missing)인 경우 약간의 페널티 부여
            if not data["payload"].basic_info.affiliated_organization:
                data["score"] *= 0.98


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
                "retrieve_mode": "sparse_then_dense_max_score",
                "expanded_path_policy": "distinct_expanded_all_branches",
                "query_schema_version": RETRIEVAL_QUERY_SCHEMA_VERSION,
                "support_rules": {"s_min": s_min, "e_min": e_min},
                "branch_query_sources": {
                    branch: {
                        "dense_base_source": compiled.dense_base_source,
                        "sparse_base_source": compiled.sparse_base_source,
                    }
                    for branch, compiled in branch_compiled_queries.items()
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
        dense_text: str,
        sparse_text: str,
        query_filter: models.Filter | None
    ) -> tuple[str, str, Any]:
        """
        단일 브랜치/경로에 대해 Qdrant에 실제 하이브리드 검색(Dense + Sparse) 요청을 보냅니다.
        E5-Instruct 임베딩 모델의 특성을 살려 브랜치별 작업 지시어(Instruct prefix)를 쿼리에 추가함으로써, 
        검색 의도에 맞는 방향으로 벡터 임베딩이 생성되도록 유도합니다.
        """
        # multilingual-e5-large-instruct 계열: 브랜치별 task-specific instruct prefix 적용.
        # 브랜치 컨텍스트를 Instruct 문장에 통합하여 쿼리 임베딩 방향을 특화.
        processed_query = dense_text
        if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
            prefix = BRANCH_INSTRUCT_PREFIXES.get(branch, (
                "Instruct: Find Korean research experts whose profile matches the given query.\nQuery: "
            ))
            processed_query = f"{prefix}{dense_text}"

        dense_query = self.dense_encoder.embed(processed_query)

        # Sparse 임베딩 생성 (커스텀 인코더 우선 활용)
        if self.sparse_encoder:
            sparse_map = self.sparse_encoder.embed(sparse_text)
            sparse_query = models.SparseVector(
                indices=list(sparse_map.keys()),
                values=list(sparse_map.values())
            )
        else:
            sparse_model_name = (
                self.sparse_runtime.active_model_name
                if self.sparse_runtime is not None
                else self.settings.sparse_model_name
            )
            sparse_query = models.Document(
                text=sparse_text,
                model=sparse_model_name,
            )
        
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
        dense_text: str,
        sparse_text: str,
        query_filter: models.Filter | None
    ) -> tuple[str, str, Any]:
        _, _, points = await self._execute_single_path_query(
            branch,
            dense_text,
            sparse_text,
            query_filter,
        )
        return branch, path, points
