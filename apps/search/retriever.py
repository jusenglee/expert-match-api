"""
Hybrid Qdrant retriever (flat chunk 모델, v2.1) — query_points_groups 기반.

[Architecture]
페이로드가 청킹 + row 단위(1 chunk = 1 Point)이므로 브랜치 단위 검색은 폐기. 단일 dense(vector_e5i)
+ 단일 sparse(vector_splade) 하이브리드(FusionQuery.RRF)를 한 번 실행하고 group_by="researcher_id"로
연구자별 top-K chunk를 모은다(query_points_groups). 2단계 keyword-pool / 앱단 수동 집계 / weighted
fan-out은 모두 제거.

후보(연구자) 점수 = 그룹 chunk의 Qdrant 융합 RRF 점수를 앱단에서 누적: Σ chunk_score × doc_type_prior
(doc_type별 chunk cap으로 다작 독식 방지). 그룹의 chunk가 곧 후보의 evidence 풀이다.

HARD: chunk 융합은 Qdrant equal RRF(FusionQuery.RRF) 고정. doc_type prior는 앱단 랭크 누적 가중(기본 equal,
Qdrant 가중 RRF 아님). recency OR(min_should)는 filters.py. evidence 참조 = chunk_id. LLM no-rerank.
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
from apps.domain.models import ChunkHit, ChunkPayload, PlannerOutput, ResearcherCandidate
from apps.search.doc_types import DOC_TYPE_TO_FAMILY
from apps.search.encoders import DenseEncoder, SparseEncoder
from apps.search.query_builder import CompiledQueries, QueryTextBuilder
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import SparseRuntimeConfig
from apps.search.text_utils import normalize_org_name

logger = logging.getLogger(__name__)

RETRIEVAL_MODE = "grouped_hybrid_rrf"

# 도메인 org가 들어있는 doc_attrs 키(교차-chunk 배제 후보).
_ORG_ATTR_KEYS = ("performing_organization", "managing_agency")


@dataclass(slots=True)
class RetrievalResult:
    hits: list[ResearcherCandidate]
    query_payload: dict[str, Any]
    queries: CompiledQueries
    retrieval_keywords: list[str]
    retrieval_score_traces: list[dict[str, Any]]
    expanded_shadow_hits: list[ResearcherCandidate] = field(default_factory=list)
    filtered_out_candidates: list[dict[str, Any]] = field(default_factory=list)
    cache_hit: bool = False


def _candidate_breakdown(hits: list[ResearcherCandidate], limit: int = 10) -> list[dict[str, Any]]:
    """상위 후보의 집계 내역 요약(researcher_id, 집계 점수, doc_type별 chunk 수) — 로깅용."""
    out: list[dict[str, Any]] = []
    for cand in hits[:limit]:
        dist: dict[str, int] = {}
        for hit in cand.chunks:
            dist[hit.doc_type] = dist.get(hit.doc_type, 0) + 1
        out.append(
            {
                "rid": cand.researcher_id,
                "name": cand.researcher_name,
                "score": round(cand.group_score, 6),
                "n_chunks": len(cand.chunks),
                "doc_types": dist,
            }
        )
    return out


class QdrantHybridRetriever:
    """단일 grouped 하이브리드 검색(query_points_groups) + 앱단 RRF 누적 오케스트레이터."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        dense_encoder: DenseEncoder,
        query_builder: QueryTextBuilder,
        sparse_encoder: SparseEncoder | None = None,
        sparse_runtime: SparseRuntimeConfig | None = None,
        l3_cache: RetrievalResultCache | None = None,
        registry: Any = None,  # 하위호환(미사용)
    ) -> None:
        self.client = client
        self.settings = settings
        self.dense_encoder = dense_encoder
        self.query_builder = query_builder
        self.sparse_encoder = sparse_encoder
        self.sparse_runtime = sparse_runtime
        self.l3_cache = l3_cache

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _sort_hits(hits: list[ResearcherCandidate]) -> list[ResearcherCandidate]:
        return sorted(
            hits,
            key=lambda c: (
                -c.group_score,
                " ".join((c.researcher_name or "").split()),
                c.researcher_id,
            ),
        )

    @staticmethod
    def _point_payload_data(point: Any) -> dict[str, Any]:
        return getattr(point, "payload", {}) or (
            point.get("payload", {}) if isinstance(point, dict) else {}
        )

    @staticmethod
    def _point_score(point: Any) -> float:
        if hasattr(point, "score"):
            return float(getattr(point, "score", 0.0) or 0.0)
        if isinstance(point, dict):
            return float(point.get("score", 0.0) or 0.0)
        return 0.0

    def _doc_type_prior(self, doc_type: str) -> float:
        priors = self.settings.doc_type_priors
        if not priors:
            return 1.0  # equal (기본). HARD: Qdrant 가중 RRF 아님.
        return float(priors.get(doc_type, 1.0))

    def _retrieval_doc_type_filter(self) -> models.FieldCondition | None:
        whitelist = self.settings.retrieval_doc_types
        if not whitelist:
            return None
        return models.FieldCondition(key="doc_type", match=models.MatchAny(any=list(whitelist)))

    @staticmethod
    def _validate_chunk(payload_data: dict[str, Any]) -> ChunkPayload | None:
        try:
            return ChunkPayload.model_validate(payload_data)
        except ValidationError:
            return None

    # --------------------------------------------------------- query payloads
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
        return models.Document(text=query_text, model=sparse_model_name)

    def _build_dense_query(self, query_text: str) -> list[float]:
        processed_query = query_text
        if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
            processed_query = (
                "Instruct: Find experts whose profile, papers, patents, or projects "
                "match the given query.\nQuery: "
                f"{query_text}"
            )
        return self.dense_encoder.embed(processed_query)

    @staticmethod
    def _merge_filters(*filters: models.Filter | models.Condition | None) -> models.Filter | None:
        must: list[Any] = []
        should: Any = None
        min_should: Any = None
        must_not: Any = None
        for f in filters:
            if f is None:
                continue
            if isinstance(f, models.Filter):
                if f.must:
                    must.extend(f.must if isinstance(f.must, list) else [f.must])
                should = should or f.should
                min_should = min_should or f.min_should
                must_not = must_not or f.must_not
            else:  # bare condition
                must.append(f)
        if not must and should is None and min_should is None and must_not is None:
            return None
        return models.Filter(
            must=must or None, should=should, min_should=min_should, must_not=must_not
        )

    # --------------------------------------------------------- aggregation
    def _candidate_from_group(self, group: Any) -> ResearcherCandidate | None:
        """query_points_groups의 PointGroup → ResearcherCandidate (앱단 RRF 누적)."""
        raw_hits = getattr(group, "hits", None)
        if raw_hits is None and isinstance(group, dict):
            raw_hits = group.get("hits")
        raw_hits = raw_hits or []

        researcher_id = str(getattr(group, "id", None) or (group.get("id") if isinstance(group, dict) else "") or "")
        identity: ChunkPayload | None = None
        chunks: list[ChunkHit] = []
        score_sum = 0.0
        doc_type_contrib: dict[str, int] = {}
        cap = self.settings.doc_type_chunk_cap

        for rank, point in enumerate(raw_hits, start=1):
            payload = self._validate_chunk(self._point_payload_data(point))
            if payload is None:
                continue
            if not researcher_id:
                researcher_id = payload.researcher_id
            if identity is None:
                identity = payload
            point_score = self._point_score(point)
            chunks.append(ChunkHit(score=point_score, payload=payload, rank=rank))
            # 점수 기여는 (researcher, doc_type)당 상위 cap개 chunk까지만 (다작 독식 방지).
            contributed = doc_type_contrib.get(payload.doc_type, 0)
            if cap <= 0 or contributed < cap:
                score_sum += point_score * self._doc_type_prior(payload.doc_type)
                doc_type_contrib[payload.doc_type] = contributed + 1

        if not chunks or not researcher_id or identity is None:
            return None

        chunks.sort(key=lambda h: -h.score)
        return ResearcherCandidate(
            researcher_id=researcher_id,
            researcher_name=identity.researcher_name,
            affiliated_organization=identity.affiliated_organization,
            highest_degree=identity.highest_degree,
            counts=identity.counts(),
            group_score=score_sum,
            rank_score=score_sum,
            chunks=chunks,
        )

    def _excluded_by_org(self, candidate: ResearcherCandidate, exclude_orgs: list[str]) -> bool:
        if not exclude_orgs:
            return False
        normalized_excludes = [n for ex in exclude_orgs if (n := normalize_org_name(ex))]
        if not normalized_excludes:
            return False
        candidate_orgs = [candidate.affiliated_organization or ""]
        for hit in candidate.chunks:
            for key in _ORG_ATTR_KEYS:
                val = hit.payload.doc_attrs.get(key)
                if isinstance(val, str):
                    candidate_orgs.append(val)
        for org in candidate_orgs:
            normalized_org = normalize_org_name(org) or ""
            if normalized_org and any(ex in normalized_org for ex in normalized_excludes):
                return True
        return False

    def _score_traces(self, hits: list[ResearcherCandidate]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for cand in hits:
            matches: list[dict[str, Any]] = []
            dt_contrib: dict[str, float] = {}
            fam_contrib: dict[str, float] = {}
            for hit in cand.chunks:
                contribution = hit.score * self._doc_type_prior(hit.doc_type)
                dt_contrib[hit.doc_type] = round(dt_contrib.get(hit.doc_type, 0.0) + contribution, 6)
                family = DOC_TYPE_TO_FAMILY.get(hit.doc_type, "?")
                fam_contrib[family] = round(fam_contrib.get(family, 0.0) + contribution, 6)
            for hit in cand.chunks[:10]:
                matches.append(
                    {
                        "doc_type": hit.doc_type,
                        "chunk_id": hit.chunk_id,
                        "rank": hit.rank,
                        "score": round(hit.score, 6),
                        "contribution": round(hit.score * self._doc_type_prior(hit.doc_type), 6),
                    }
                )
            traces.append(
                {
                    "expert_id": cand.researcher_id,
                    "final_score": round(cand.group_score, 6),
                    "doc_types": cand.doc_types_present,
                    "doc_type_contributions": dt_contrib,
                    "family_contributions": fam_contrib,
                    "matches": matches,
                }
            )
        return traces

    # --------------------------------------------------------------- search
    async def search(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        """단일 grouped 하이브리드 검색(query_points_groups) → 연구자별 chunk 그룹 → 앱단 RRF 누적."""
        queries = self.query_builder.build_queries(query, plan)
        query_text = queries.stable
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        base_filter = self._merge_filters(query_filter, self._retrieval_doc_type_filter())

        logger.info(
            "검색 쿼리 컴파일: mode=%s keywords=%s query=%r limits={prefetch:%d,group_size:%d,groups:%d}",
            RETRIEVAL_MODE,
            retrieval_keywords,
            query_text,
            self.settings.prefetch_limit,
            self.settings.group_size,
            self.settings.retrieval_limit,
        )

        compiled_json = json.dumps({"mode": RETRIEVAL_MODE, "q": query_text}, sort_keys=True)
        filter_json = str(base_filter)
        snapshot_id = self.settings.qdrant_collection_release_id

        if self.l3_cache and self.settings.cache_enabled:
            cached = self.l3_cache.get(compiled_json, filter_json, snapshot_id)
            if cached:
                hits = [ResearcherCandidate.model_validate(h) for h in cached]
                logger.info("검색 캐시 적중: layer=L3 mode=%s hits=%d", RETRIEVAL_MODE, len(hits))
                return RetrievalResult(
                    hits=hits,
                    query_payload={"cache": "hit", "l3": True, "retrieval_mode": RETRIEVAL_MODE,
                                   "retrieval_keywords": retrieval_keywords},
                    queries=queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=self._score_traces(hits),
                    cache_hit=True,
                )

        dense_query = self._build_dense_query(query_text)
        sparse_query = self._build_sparse_query(query_text)
        grouped_payload = {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": [
                models.Prefetch(
                    query=dense_query, using=DENSE_VECTOR_NAME,
                    limit=self.settings.prefetch_limit, filter=base_filter,
                ),
                models.Prefetch(
                    query=sparse_query, using=SPARSE_VECTOR_NAME,
                    limit=self.settings.prefetch_limit, filter=base_filter,
                ),
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "group_by": "researcher_id",
            "group_size": self.settings.group_size,
            "limit": self.settings.retrieval_limit,
            "query_filter": base_filter,
            "with_payload": True,
            "with_vectors": False,
        }

        with Timer() as search_timer:
            try:
                result = await asyncio.to_thread(self.client.query_points_groups, **grouped_payload)
            except Exception as exc:  # noqa: BLE001 — 검색 실패는 빈 결과로 강등(파이프라인 보호)
                logger.error("query_points_groups failed: %s", exc, exc_info=exc)
                return RetrievalResult(
                    hits=[],
                    query_payload={"retrieval_mode": RETRIEVAL_MODE, "error": str(exc),
                                   "retrieval_keywords": retrieval_keywords},
                    queries=queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                )

        groups = getattr(result, "groups", None)
        if groups is None and isinstance(result, dict):
            groups = result.get("groups")
        groups = groups or []

        candidates: list[ResearcherCandidate] = []
        for group in groups:
            candidate = self._candidate_from_group(group)
            if candidate is not None:
                candidates.append(candidate)

        filtered_out: list[dict[str, Any]] = []
        survivors: list[ResearcherCandidate] = []
        for candidate in candidates:
            if self._excluded_by_org(candidate, plan.exclude_orgs):
                filtered_out.append(
                    {"expert_id": candidate.researcher_id, "name": candidate.researcher_name,
                     "reason": "excluded_org"}
                )
                continue
            survivors.append(candidate)

        final_hits = self._sort_hits(survivors)

        logger.info(
            "검색 집계 완료: elapsed_ms=%.2f groups=%d candidates=%d org_filtered=%d final_hits=%d",
            search_timer.elapsed_ms, len(groups), len(candidates), len(filtered_out), len(final_hits),
        )
        logger.info(
            "검색 후보군 상위 %d/%d: %s",
            min(10, len(final_hits)), len(final_hits), _candidate_breakdown(final_hits),
        )

        if self.l3_cache and self.settings.cache_enabled and final_hits:
            self.l3_cache.set(
                compiled_json, filter_json, snapshot_id,
                [h.model_dump(mode="json") for h in final_hits],
            )

        return RetrievalResult(
            hits=final_hits,
            query_payload={
                "retrieval_mode": RETRIEVAL_MODE,
                "retrieval_keywords": retrieval_keywords,
                "semantic_query": plan.semantic_query,
                "group_count": len(groups),
                "aggregated_candidate_count": len(candidates),
                "org_filtered_count": len(filtered_out),
                "final_hit_count": len(final_hits),
                "search_limits": {
                    "prefetch_limit": self.settings.prefetch_limit,
                    "group_size": self.settings.group_size,
                    "groups": self.settings.retrieval_limit,
                },
                "timers": {"search_ms": search_timer.elapsed_ms},
            },
            queries=queries,
            retrieval_keywords=retrieval_keywords,
            retrieval_score_traces=self._score_traces(final_hits),
            expanded_shadow_hits=[],
            filtered_out_candidates=filtered_out,
        )

    async def search_weighted(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        """[/search/candidates 전용] grouped RRF로 통일 — search()와 동일 경로(커스텀 가중 폐기)."""
        return await self.search(query=query, plan=plan, query_filter=query_filter)
