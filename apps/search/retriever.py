"""
Hybrid Qdrant retriever for NTIS expert search.

[Architecture Overview]
이 모듈은 Qdrant 벡터 데이터베이스를 활용하여 전문가 추천 후보군을 추출하는 핵심 검색 엔진입니다.
검색 파이프라인은 속도와 재현율(Recall), 그리고 정확도(Precision)를 모두 잡기 위해 2단계(Two-Stage)로 구성됩니다.

* Stage 1 (Keyword Prefetch): 
  - 추출된 핵심 키워드를 기반으로 Sparse 벡터(SPLADE/BM25) 전용 고속 검색을 수행합니다.
  - 이 단계에서 넓은 후보군(Candidate Pool)을 확보합니다 (`branch_prefetch_limit` 사용).

* Stage 2 (Hybrid Search): 
  - 1단계에서 확보된 후보군 ID(Candidate IDs)만을 대상으로 필터를 걸고, 
  - Dense 벡터(문장/의미 매칭)와 Sparse 벡터(단어 매칭)를 혼합한 하이브리드 검색을 수행합니다 (`branch_output_limit` 사용).

* Stage 3 (RRF & Support Filtering):
  - 각 브랜치(논문, 특허, 과제, 기본)에서 반환된 후보들의 등수(Rank)를 RRF(Reciprocal Rank Fusion) 알고리즘으로 합산합니다.
  - 브랜치별 가중치(`BRANCH_WEIGHTS`)와 질의 경로별 가중치(`PATH_WEIGHTS`)가 반영됩니다.
  - 최종적으로 교차 증거 기준(Support Rules)을 통과한 후보만 반환됩니다.
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


# =========================================================================
# 하이브리드 검색 가중치 (Architecture Guidelines)
# =========================================================================
# 브랜치별 중요도 가중치: 특정 메타데이터가 전문가 추천에 미치는 상대적 중요도
BRANCH_WEIGHTS = {
    "basic": 0.6,  # 기본 정보 (상대적으로 비중이 낮음)
    "art": 1.0,    # 논문 실적 (매우 중요)
    "pat": 0.8,    # 특허 실적 (중요)
    "pjt": 1.0,    # 국가 R&D 과제 실적 (매우 중요)
}

# 쿼리 경로별 가중치: 사용자 질의를 어떻게 해석했느냐에 따른 신뢰도
PATH_WEIGHTS = {
    "stable": 1.0,     # 원본 질의나 핵심 키워드 중심의 안정적인 검색
    "expanded": 0.25,  # 동의어/확장 어휘를 포함한 넓은 검색 (가중치 패널티 적용)
}

RETRIEVAL_MODE = "keyword_pool_then_hybrid"
RETRIEVAL_MODE_WEIGHTED = "keyword_pool_then_weighted_hybrid"

# /search/candidates 전용 가중치 (similarity 0.6, keyword 0.4)
WEIGHTED_HYBRID_DENSE = 0.6
WEIGHTED_HYBRID_SPARSE = 0.4


def _minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    """결과 리스트 내에서 점수를 0~1로 min-max 정규화합니다.

    sparse(SPLADE)와 dense(cosine/dot)의 스케일이 달라 직접 가중 합이 불가하므로
    동일 결과 집합 안에서 상대적인 점수만 추출해 가중 합산에 사용합니다.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {key: 1.0 for key in scores}
    span = hi - lo
    return {key: (value - lo) / span for key, value in scores.items()}


_KEYWORD_FILTER_TEXT_FIELDS = (
    "basic_info",
    "researcher_profile",
    "publications",
    "intellectual_properties",
    "research_projects",
    "technical_classifications",
)


def _collect_payload_text_blob(payload: ExpertPayload) -> str:
    """후보 페이로드에서 키워드 포함 필터링에 사용할 텍스트만 추출해 결합합니다."""
    parts: list[str] = []
    basic = payload.basic_info
    parts.extend(filter(None, [
        basic.affiliated_organization,
        basic.affiliated_organization_exact,
        basic.department,
        basic.position_title,
    ]))
    profile = payload.researcher_profile
    if profile.major_field:
        parts.append(profile.major_field)
    for pub in payload.publications:
        parts.extend(filter(None, [
            pub.publication_title,
            pub.abstract,
            pub.journal_name,
        ]))
        parts.extend(pub.korean_keywords)
        parts.extend(pub.english_keywords)
    for ip in payload.intellectual_properties:
        if ip.intellectual_property_title:
            parts.append(ip.intellectual_property_title)
    for proj in payload.research_projects:
        parts.extend(filter(None, [
            proj.project_title_korean,
            proj.project_title_english,
            proj.research_objective_summary,
            proj.research_content_summary,
            proj.performing_organization,
            proj.managing_agency,
        ]))
    parts.extend(payload.technical_classifications)
    return " ".join(parts).lower()


def _payload_contains_any_keyword(payload: ExpertPayload, keywords: list[str]) -> tuple[bool, list[str]]:
    """페이로드 텍스트에 핵심 키워드가 1개 이상 포함되어 있는지 확인합니다."""
    if not keywords:
        return True, []
    blob = _collect_payload_text_blob(payload)
    matched = [kw for kw in keywords if kw and kw.lower() in blob]
    return bool(matched), matched


def _preview_values(values: list[str], *, limit: int = 5) -> list[str]:
    return values[:limit]


def _compiled_query_summary(
    compiled_queries: dict[str, CompiledBranchQueries],
) -> dict[str, dict[str, str]]:
    return {
        branch: {
            "stable": query.stable,
            "expanded": query.expanded,
        }
        for branch, query in compiled_queries.items()
    }


class QdrantHybridRetriever:
    """
    Qdrant 데이터베이스와 통신하여 2단계 하이브리드 검색을 수행하는 검색 오케스트레이터입니다.
    비동기(Asyncio) 패턴을 적극 활용하여 여러 브랜치에 대한 검색을 병렬로 처리합니다.
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
                logger.error(
                    "Keyword query task failed: %s",
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
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
        """
        [메인 검색 오케스트레이션 함수]
        주어진 질의와 LLM이 분석한 계획(PlannerOutput)을 바탕으로 전체 검색 파이프라인을 실행합니다.
        
        Args:
            query (str): 사용자의 원본 질의어
            plan (PlannerOutput): LLM Planner가 분석한 검색 의도, 키워드, 브랜치별 서브쿼리 등
            query_filter (models.Filter | None): 사용자가 요청한 소속기관 배제 등 사전 필터링 조건
            
        Returns:
            RetrievalResult: 최종 RRF 합산이 완료되고 필터링된 전문가 후보군 객체
        """
        # =========================================================================
        # 1. 쿼리 컴파일 (Query Compilation)
        # =========================================================================
        # LLM Planner가 만든 계획을 바탕으로, 실제 Qdrant에 날릴 브랜치별 쿼리 문자열을 조립합니다.
        branch_compiled_queries = self.query_builder.build_branch_queries(query, plan)
        keyword_compiled_queries = self.query_builder.build_keyword_branch_queries(query, plan)
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        tasks_meta = self._build_path_tasks(branch_compiled_queries)
        keyword_tasks_meta = self._build_path_tasks(keyword_compiled_queries)
        logger.info(
            "검색 쿼리 컴파일 완료: mode=%s retrieval_keywords=%s semantic_query=%r bundle_ids=%s keyword_queries=%s hybrid_queries=%s keyword_paths=%d hybrid_paths=%d limits={prefetch:%d, hybrid:%d, retrieval:%d}",
            RETRIEVAL_MODE,
            retrieval_keywords,
            plan.semantic_query,
            plan.bundle_ids,
            _compiled_query_summary(keyword_compiled_queries),
            _compiled_query_summary(branch_compiled_queries),
            len(keyword_tasks_meta),
            len(tasks_meta),
            self.settings.branch_prefetch_limit,
            self.settings.branch_output_limit,
            self.settings.retrieval_limit,
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
                        "retrieval_keywords": retrieval_keywords,
                        "semantic_query": plan.semantic_query,
                        "keyword_stage_queries": _compiled_query_summary(
                            keyword_compiled_queries
                        ),
                        "hybrid_stage_queries": _compiled_query_summary(
                            branch_compiled_queries
                        ),
                        "keyword_stage_skipped_reason": "l3_cache_hit",
                        "hybrid_stage_skipped_reason": "l3_cache_hit",
                    },
                    branch_queries=branch_compiled_queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                    cache_hit=True
                )

        # =========================================================================
        # 3. 1차 검색 실행 (Stage 1: Keyword Pool Generation)
        # =========================================================================
        # 하이브리드 검색은 연산 비용이 높으므로, 먼저 초고속 Sparse 벡터(키워드) 검색만으로 
        # 후보자 풀을 대량 확보(`branch_prefetch_limit`)합니다.
        logger.info(
            "1차 키워드 검색 시작: mode=%s paths=%d query_filter=%s retrieval_keywords=%s keyword_queries=%s",
            RETRIEVAL_MODE,
            len(keyword_tasks_meta),
            query_filter is not None,
            retrieval_keywords,
            _compiled_query_summary(keyword_compiled_queries),
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
            "1차 키워드 검색 완료: elapsed_ms=%.2f unique_candidates=%d branch_counts=%s candidate_preview=%s",
            keyword_timer.elapsed_ms,
            len(keyword_candidate_ids),
            keyword_branch_counts,
            _preview_values(keyword_candidate_ids),
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
                    "retrieval_keywords": retrieval_keywords,
                    "semantic_query": plan.semantic_query,
                    "keyword_stage_queries": _compiled_query_summary(
                        keyword_compiled_queries
                    ),
                    "hybrid_stage_queries": _compiled_query_summary(
                        branch_compiled_queries
                    ),
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

        # 1차 검색 결과에서 중복을 제거한 고유 후보자 ID(expert_id) 목록을 추출합니다.
        hybrid_query_filter = self._with_candidate_pool_filter(
            query_filter,
            keyword_candidate_ids,
        )
        keyword_candidate_id_set = set(keyword_candidate_ids)

        # =========================================================================
        # 4. 2차 하이브리드 병렬 검색 실행 (Stage 2: Dense + Sparse Hybrid Search)
        # =========================================================================
        # 1단계에서 얻은 후보자들(`hybrid_query_filter`)만을 대상으로,
        # 문장 단위 의미 분석(Dense)과 키워드(Sparse) 매칭을 동시에 수행하는 하이브리드 쿼리를 날립니다.
        logger.info(
            "2차 하이브리드 검색 시작: paths=%d candidate_filter_count=%d query_filter=%s semantic_query=%r hybrid_queries=%s candidate_preview=%s",
            len(tasks_meta),
            len(keyword_candidate_ids),
            hybrid_query_filter is not None,
            plan.semantic_query,
            _compiled_query_summary(branch_compiled_queries),
            _preview_values(keyword_candidate_ids),
        )
        search_tasks = [
            self._execute_single_path_query_with_path(branch, path, q_text, hybrid_query_filter)
            for branch, path, q_text in tasks_meta
        ]
            
        with Timer() as hybrid_timer:
            raw_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        hybrid_failure_count = sum(
            1 for result in raw_results if isinstance(result, Exception)
        )
        logger.info(
            "2차 하이브리드 검색 완료: elapsed_ms=%.2f raw_results=%d failures=%d",
            hybrid_timer.elapsed_ms,
            len(raw_results),
            hybrid_failure_count,
        )

        # =========================================================================
        # 5. 결과 집계 및 RRF(Reciprocal Rank Fusion) 계산
        # =========================================================================
        # 병렬로 실행된 각 브랜치/경로의 결과를 모아서, 점수가 아닌 "등수(Rank)" 기반으로 융합합니다.
        aggregator: dict[str, dict[str, Any]] = {}
        hybrid_branch_counts: dict[str, int] = {}
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error(
                    "Query task failed: %s",
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue
            
            branch, path, points = result
            hybrid_branch_counts[f"{branch}:{path}"] = self._point_count(points)
            weight = BRANCH_WEIGHTS[branch] * PATH_WEIGHTS[path]
            
            for rank, pt in enumerate(points, start=1):
                pt_id = self._point_id(pt)
                payload_data = self._point_payload_data(pt)
                try:
                    payload = ExpertPayload.model_validate(payload_data)
                except ValidationError as e:
                    logger.warning(
                        "Hybrid stage ValidationError: branch=%s path=%s point_id=%s err=%s",
                        branch, path, pt_id, str(e).replace('\n', ' ')[:200]
                    )
                    continue
                
                eid = payload.basic_info.researcher_id
                if eid not in keyword_candidate_id_set:
                    continue

                # RRF 핵심 공식: 가중치 * (1 / (현재 등수 + 스무딩 상수 60))
                # 여러 브랜치에서 자주 등장할수록 점수가 안정적으로 누적됩니다.
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

        # =========================================================================
        # 6. 교차 증거 기준(Support Rule) 필터링 및 최종 결과 정렬
        # =========================================================================
        # 설정(config.py)에 명시된 support_rule_stable_min / expanded_min 값을 기반으로,
        # 후보자가 충분한 교차 검증(여러 브랜치에서 발견됨)을 거쳤는지 확인합니다.
        # (현재 환경에서는 누락 방지를 위해 0으로 설정되어 있어 전부 통과됩니다.)
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
        if self.settings.retrieval_limit > 0:
            final_hits = final_hits[: self.settings.retrieval_limit]

        logger.info(
            "검색 집계 완료: raw_branch_counts=%s aggregated_candidates=%d support_pass=%d support_filtered=%d final_hits=%d filtered_preview=%s",
            hybrid_branch_counts,
            len(aggregator),
            len(final_hits) + len(filtered_out),
            len(filtered_out),
            len(final_hits),
            [
                item.get("expert_id")
                for item in filtered_out[:5]
                if item.get("expert_id")
            ],
        )
        
        if self.l3_cache and self.settings.cache_enabled and final_hits:
            self.l3_cache.set(
                compiled_json, filter_json, snapshot_id,
                [h.model_dump(mode="json") for h in final_hits]
            )
            logger.info(
                "검색 캐시 저장 완료: layer=L3 hits=%d snapshot_id=%s",
                len(final_hits),
                snapshot_id,
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
                "retrieval_keywords": retrieval_keywords,
                "semantic_query": plan.semantic_query,
                "keyword_stage_queries": _compiled_query_summary(
                    keyword_compiled_queries
                ),
                "hybrid_stage_queries": _compiled_query_summary(
                    branch_compiled_queries
                ),
                "keyword_stage_candidate_count": len(keyword_candidate_ids),
                "keyword_stage_branch_counts": keyword_branch_counts,
                "hybrid_stage_candidate_filter_count": len(keyword_candidate_ids),
                "hybrid_stage_raw_branch_counts": hybrid_branch_counts,
                "aggregated_candidate_count": len(aggregator),
                "support_pass_count": len(final_hits),
                "support_filtered_count": len(filtered_out),
                "support_rules": {"s_min": s_min, "e_min": e_min},
                "search_limits": {
                    "prefetch_limit": self.settings.branch_prefetch_limit,
                    "hybrid_limit": self.settings.branch_output_limit,
                    "retrieval_limit": self.settings.retrieval_limit,
                },
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

    async def _execute_single_vector_query(
        self,
        *,
        branch: str,
        path: str,
        vector_kind: str,
        query_text: str,
        query_filter: models.Filter | None,
    ) -> tuple[str, str, str, Any]:
        """단일 named vector(dense 또는 sparse)에 대해서만 검색합니다.

        가중 하이브리드(/search/candidates)에서는 dense/sparse를 분리 발사한 뒤
        결과 점수를 정규화해 가중 합산하기 위해 사용합니다.
        """
        if vector_kind == "dense":
            processed_query = query_text
            if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
                processed_query = (
                    "Instruct: Find experts whose profile, papers, patents, or projects "
                    "match the given query.\nQuery: "
                    f"{query_text}"
                )
            query_vector: Any = self.dense_encoder.embed(processed_query)
            using = self.registry.dense_vector_by_branch[branch]
        elif vector_kind == "sparse":
            query_vector = self._build_sparse_query(query_text)
            using = self.registry.sparse_vector_by_branch[branch]
        else:
            raise ValueError(f"Unsupported vector_kind: {vector_kind}")

        payload = {
            "collection_name": self.settings.qdrant_collection_name,
            "query": query_vector,
            "using": using,
            "query_filter": query_filter,
            "limit": self.settings.branch_output_limit,
            "with_payload": True,
            "with_vectors": False,
        }

        result = await asyncio.to_thread(self.client.query_points, **payload)
        points = getattr(result, "points", result)
        return branch, path, vector_kind, points

    async def search_weighted(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        """[/search/candidates 전용] 가중치 하이브리드 검색 + 키워드 포함 별도 필터.

        키워드 prefetch 풀링 단계가 없고, 컬렉션 전체(또는 사용자 query_filter 범위)에
        대해 단일 하이브리드 검색 단계만 수행한다.

        - Stage 1 (하이브리드 검색):
            브랜치/경로별로 dense, sparse 쿼리를 직접 발사한 뒤 각 결과를 min-max
            정규화하여 ``dense * 0.6 + sparse * 0.4`` 로 점수를 합산한다.
            이후 ``BRANCH_WEIGHTS × PATH_WEIGHTS`` 로 브랜치 간 가중 집계한다.
        - Stage 2 (키워드 포함 별도 필터):
            ``retrieval_core / core_keywords`` 중 1개 이상이 payload 텍스트에
            substring 매칭되지 않은 후보를 제거한다.
        - Stage 3 (support rule + 정렬 + 캐시 저장).
        """
        branch_compiled_queries = self.query_builder.build_branch_queries(query, plan)
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        tasks_meta = self._build_path_tasks(branch_compiled_queries)
        logger.info(
            "[weighted] 검색 쿼리 컴파일 완료: mode=%s retrieval_keywords=%s semantic_query=%r bundle_ids=%s hybrid_paths=%d weights={dense:%.2f,sparse:%.2f}",
            RETRIEVAL_MODE_WEIGHTED,
            retrieval_keywords,
            plan.semantic_query,
            plan.bundle_ids,
            len(tasks_meta),
            WEIGHTED_HYBRID_DENSE,
            WEIGHTED_HYBRID_SPARSE,
        )

        compiled_json = json.dumps(
            {
                "retrieval_mode": RETRIEVAL_MODE_WEIGHTED,
                "weights": {
                    "dense": WEIGHTED_HYBRID_DENSE,
                    "sparse": WEIGHTED_HYBRID_SPARSE,
                },
                "branch_queries": {
                    b: {"s": q.stable, "e": q.expanded}
                    for b, q in branch_compiled_queries.items()
                },
                "keyword_filter": retrieval_keywords,
            },
            sort_keys=True,
        )
        filter_json = str(query_filter)
        snapshot_id = self.settings.qdrant_collection_release_id

        if self.l3_cache and self.settings.cache_enabled:
            cached_data = self.l3_cache.get(compiled_json, filter_json, snapshot_id)
            if cached_data:
                hits = [SearchHit.model_validate(h) for h in cached_data]
                logger.info(
                    "[weighted] 검색 캐시 적중: layer=L3 mode=%s cached_hits=%d snapshot_id=%s",
                    RETRIEVAL_MODE_WEIGHTED,
                    len(hits),
                    snapshot_id,
                )
                return RetrievalResult(
                    hits=hits,
                    query_payload={
                        "cache": "hit",
                        "l3": True,
                        "retrieval_mode": RETRIEVAL_MODE_WEIGHTED,
                        "retrieval_keywords": retrieval_keywords,
                        "semantic_query": plan.semantic_query,
                        "weights": {
                            "dense": WEIGHTED_HYBRID_DENSE,
                            "sparse": WEIGHTED_HYBRID_SPARSE,
                        },
                        "hybrid_stage_queries": _compiled_query_summary(
                            branch_compiled_queries
                        ),
                        "hybrid_stage_skipped_reason": "l3_cache_hit",
                    },
                    branch_queries=branch_compiled_queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                    cache_hit=True,
                )

        # =========================================================================
        # Stage 1: 하이브리드 검색 (브랜치/경로별 dense + sparse 동시 발사)
        # =========================================================================
        # 사전 키워드 풀링 없이 컬렉션 전체에 대해 dense, sparse 쿼리를 직접 실행한다.
        weighted_tasks: list[Any] = []
        for branch, path, q_text in tasks_meta:
            weighted_tasks.append(
                self._execute_single_vector_query(
                    branch=branch,
                    path=path,
                    vector_kind="dense",
                    query_text=q_text,
                    query_filter=query_filter,
                )
            )
            weighted_tasks.append(
                self._execute_single_vector_query(
                    branch=branch,
                    path=path,
                    vector_kind="sparse",
                    query_text=q_text,
                    query_filter=query_filter,
                )
            )

        logger.info(
            "[weighted] Stage 1 하이브리드 검색 시작: dense_paths=%d sparse_paths=%d query_filter=%s",
            len(tasks_meta),
            len(tasks_meta),
            query_filter is not None,
        )
        with Timer() as hybrid_timer:
            raw_results = await asyncio.gather(*weighted_tasks, return_exceptions=True)

        # =========================================================================
        # Stage 1 (계속): 결과 정규화 + 가중치 합산 + 브랜치 집계
        # =========================================================================
        # 경로별로 (dense, sparse) 결과를 모은다.
        per_path_scores: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        hybrid_branch_counts: dict[str, int] = {}
        for result in raw_results:
            if isinstance(result, Exception):
                logger.error(
                    "[weighted] Query task failed: %s",
                    result,
                    exc_info=(type(result), result, result.__traceback__),
                )
                continue

            branch, path, vector_kind, points = result
            hybrid_branch_counts[f"{branch}:{path}:{vector_kind}"] = self._point_count(points)
            bucket = per_path_scores.setdefault((branch, path), {"dense": {}, "sparse": {}, "payloads": {}, "point_ids": {}, "ranks": {}})

            for rank, pt in enumerate(points, start=1):
                pt_id = self._point_id(pt)
                payload_data = self._point_payload_data(pt)
                try:
                    payload = ExpertPayload.model_validate(payload_data)
                except ValidationError as exc:
                    logger.warning(
                        "[weighted] Hybrid stage ValidationError: branch=%s path=%s vector_kind=%s point_id=%s err=%s",
                        branch,
                        path,
                        vector_kind,
                        pt_id,
                        str(exc).replace("\n", " ")[:200],
                    )
                    continue

                eid = payload.basic_info.researcher_id
                raw_score = float(
                    getattr(pt, "score", 0.0)
                    if hasattr(pt, "score")
                    else (pt.get("score", 0.0) if isinstance(pt, dict) else 0.0)
                )
                bucket[vector_kind][eid] = raw_score
                bucket["payloads"].setdefault(eid, payload)
                bucket["point_ids"].setdefault(eid, self._point_key(pt_id))
                bucket["ranks"].setdefault(eid, {})[vector_kind] = rank

        aggregator: dict[str, dict[str, Any]] = {}
        for (branch, path), bucket in per_path_scores.items():
            dense_norm = _minmax_normalize(bucket["dense"])
            sparse_norm = _minmax_normalize(bucket["sparse"])
            path_weight = BRANCH_WEIGHTS[branch] * PATH_WEIGHTS[path]

            candidate_eids = set(dense_norm) | set(sparse_norm)
            for eid in candidate_eids:
                d_norm = dense_norm.get(eid, 0.0)
                s_norm = sparse_norm.get(eid, 0.0)
                combined = WEIGHTED_HYBRID_DENSE * d_norm + WEIGHTED_HYBRID_SPARSE * s_norm
                contribution = path_weight * combined
                payload = bucket["payloads"].get(eid)
                if payload is None:
                    continue

                if eid not in aggregator:
                    aggregator[eid] = {
                        "score": 0.0,
                        "stable_hits": 0,
                        "expanded_hits": 0,
                        "branches": set(),
                        "payload": payload,
                        "point_id": bucket["point_ids"].get(eid, eid),
                        "branch_matches": [],
                    }
                entry = aggregator[eid]
                entry["score"] += contribution
                entry["branches"].add(branch)
                if path == "stable":
                    entry["stable_hits"] += 1
                else:
                    entry["expanded_hits"] += 1

                rank_info = bucket["ranks"].get(eid, {})
                primary_rank = min(rank_info.values()) if rank_info else 0
                entry["branch_matches"].append(
                    {
                        "branch": branch,
                        "path": path,
                        "rank": primary_rank,
                        "score": round(combined, 6),
                        "dense_norm": round(d_norm, 6),
                        "sparse_norm": round(s_norm, 6),
                    }
                )

        logger.info(
            "[weighted] Stage 1 하이브리드 검색 완료: elapsed_ms=%.2f raw_branch_counts=%s aggregated=%d",
            hybrid_timer.elapsed_ms,
            hybrid_branch_counts,
            len(aggregator),
        )

        # =========================================================================
        # Stage 2: 핵심 키워드 포함 post-filter (별도 필터)
        # =========================================================================
        keyword_inclusion_dropped: list[dict[str, Any]] = []
        if retrieval_keywords:
            survivors: dict[str, dict[str, Any]] = {}
            for eid, data in aggregator.items():
                matched_keywords: list[str]
                has_match, matched_keywords = _payload_contains_any_keyword(
                    data["payload"], retrieval_keywords
                )
                if has_match:
                    data["matched_keywords"] = matched_keywords
                    survivors[eid] = data
                else:
                    keyword_inclusion_dropped.append(
                        {
                            "expert_id": eid,
                            "name": data["payload"].basic_info.researcher_name,
                            "reason": "no_core_keyword_in_payload_text",
                        }
                    )
            aggregator = survivors
        logger.info(
            "[weighted] Stage 2 키워드 포함 필터 적용: kept=%d dropped=%d keywords=%s",
            len(aggregator),
            len(keyword_inclusion_dropped),
            retrieval_keywords,
        )

        # =========================================================================
        # Stage 3: support rule + exclude_orgs + 정렬 + 캐시
        # =========================================================================
        final_hits: list[SearchHit] = []
        filtered_out: list[dict[str, Any]] = []
        s_min = self.settings.support_rule_stable_min
        e_min = self.settings.support_rule_expanded_min

        for eid, data in aggregator.items():
            if plan.exclude_orgs:
                organization = data["payload"].basic_info.affiliated_organization or ""
                normalized_org = normalize_org_name(organization) or ""
                if any(
                    (normalize_org_name(ex) in normalized_org for ex in plan.exclude_orgs if normalize_org_name(ex))
                ):
                    continue

            is_supported = (data["stable_hits"] >= s_min) or (
                data["expanded_hits"] >= e_min and len(data["branches"]) >= e_min
            )

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
                support_branches=list(data["branches"]),
            )

            if is_supported:
                final_hits.append(hit)
            else:
                filtered_out.append(
                    {
                        "expert_id": eid,
                        "name": data["payload"].basic_info.researcher_name,
                        "stable_hits": data["stable_hits"],
                        "expanded_hits": data["expanded_hits"],
                        "branches": list(data["branches"]),
                        "reason": f"Insufficient support (stable < {s_min} and expanded_branches < {e_min})",
                    }
                )

        final_hits = self._sort_hits(final_hits)
        if self.settings.retrieval_limit > 0:
            final_hits = final_hits[: self.settings.retrieval_limit]

        if self.l3_cache and self.settings.cache_enabled and final_hits:
            self.l3_cache.set(
                compiled_json,
                filter_json,
                snapshot_id,
                [h.model_dump(mode="json") for h in final_hits],
            )

        retrieval_score_traces = [
            self._build_retrieval_score_trace(
                expert_id=hit.expert_id,
                point_id=aggregator[hit.expert_id]["point_id"],
                final_score=hit.score,
                branch_matches=aggregator[hit.expert_id]["branch_matches"],
            )
            for hit in final_hits
        ]

        expanded_shadow = [
            SearchHit(expert_id=eid, score=d["score"], payload=d["payload"])
            for eid, d in aggregator.items()
            if d["stable_hits"] == 0 and d["expanded_hits"] > 0
        ]

        return RetrievalResult(
            hits=final_hits,
            query_payload={
                "retrieval_mode": RETRIEVAL_MODE_WEIGHTED,
                "weighted_rrf": False,
                "weights": {
                    "dense": WEIGHTED_HYBRID_DENSE,
                    "sparse": WEIGHTED_HYBRID_SPARSE,
                },
                "retrieval_keywords": retrieval_keywords,
                "semantic_query": plan.semantic_query,
                "hybrid_stage_queries": _compiled_query_summary(
                    branch_compiled_queries
                ),
                "hybrid_stage_raw_branch_counts": hybrid_branch_counts,
                "aggregated_candidate_count": len(aggregator) + len(keyword_inclusion_dropped),
                "support_pass_count": len(final_hits),
                "support_filtered_count": len(filtered_out),
                "keyword_inclusion_dropped_count": len(keyword_inclusion_dropped),
                "keyword_inclusion_dropped": keyword_inclusion_dropped[:50],
                "support_rules": {"s_min": s_min, "e_min": e_min},
                "search_limits": {
                    "hybrid_limit": self.settings.branch_output_limit,
                    "retrieval_limit": self.settings.retrieval_limit,
                },
                "timers": {
                    "hybrid_stage_ms": hybrid_timer.elapsed_ms,
                },
            },
            branch_queries=branch_compiled_queries,
            retrieval_keywords=retrieval_keywords,
            retrieval_score_traces=retrieval_score_traces,
            expanded_shadow_hits=expanded_shadow,
            filtered_out_candidates=filtered_out,
        )

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
