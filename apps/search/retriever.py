"""
Hybrid Qdrant retriever (flat chunk 모델, v2.1) — 멀티뷰 flat 검색 + 관련도 재점수.

[Architecture]
페이로드가 청킹 + row 단위(1 chunk = 1 Point)이므로 브랜치 단위 검색은 폐기. 단일 dense(vector_e5i)
+ 단일 sparse(vector_splade)를 사용하되, 같은 질의를 여러 *검색 view*로 분리해 각각 flat query_points로
회수한다:
  · dense_full   — 사용자 원문 의미(질의를 정규식으로 깎지 않음).
  · sparse_raw   — 사용자 원문을 SPLADE로(원문 보존 채널).
  · sparse_focus — 짧은 핵심 명사구(SPLADE 과확장 억제).
  · concept:<id> — 필수 개념별 보조 검색문(AI / 반도체 …).

각 view의 결과는 source 이름·rank·raw score를 보존한 채 payload.chunk_id 기준으로 병합(point_id 아님).
chunk 점수 = Σ_view view_weight × normalized_rank_score(raw score 합산 금지 — relevance.fuse_chunk_score).
chunk는 검색 후 deterministic multi-signal로 concept 태깅(relevance.tag_chunk_concepts).

후보(연구자) 점수 = 관련 chunk를 앱단에서 묶어 capped evidence score(relevance.score_researcher):
  joint(한 chunk가 required 다개념 동시충족) + balance(min 개념별 best, 한쪽 몰빵 억제)
  + concept(개념별 best 합) + capped support(보조 근거 top-k harmonic) − 약근거 감점.
required_concepts가 있으면 전부 충족(coverage joint/separate)한 후보만 main tier, 부분충족(partial)은
fallback tier로 강등(설정에 따라 유지/배제). LLM no-rerank.

query_points_groups(서버 그룹화)는 최종 랭킹에서 제외하되 진단/AB용으로 보존(search_grouped_diagnostic
+ _candidate_from_group; diagnose_retrieval.py가 사용).
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
from apps.domain.chunk_view import parse_year
from apps.domain.models import ChunkHit, ChunkPayload, PlannerOutput, ResearcherCandidate
from apps.search.doc_types import DOC_TYPE_TO_FAMILY
from apps.search.encoders import DenseEncoder, SparseEncoder
from apps.search.query_builder import CompiledQueries, QueryTextBuilder, SearchQueryPlan
from apps.search.relevance import (
    CONCEPT_VIEW_PREFIX,
    ConceptPlan,
    chunk_display_only_concepts,
    fuse_chunk_score,
    has_operation_marker,
    resolve_concept_plan,
    score_researcher,
    tag_chunk_concepts,
)
from apps.search.schema_registry import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from apps.search.sparse_runtime import SparseRuntimeConfig
from apps.search.text_utils import normalize_org_name

logger = logging.getLogger(__name__)

RETRIEVAL_MODE = "multiview_flat_relevance"
HYBRID_MODE = "hybrid_dense_sparse_rrf"
KEYWORD_SIMILARITY_MODE = "keyword_then_dense_similarity"
GROUPED_DIAGNOSTIC_MODE = "grouped_hybrid_rrf"

# 요청 search_mode 토큰(클라이언트 노출) → 내부 retrieval_mode 라벨.
# multiview=기존 멀티뷰(기본), hybrid=dense+sparse 단순 RRF, keyword_similarity=키워드 1차→dense 재정렬.
SEARCH_MODE_MULTIVIEW = "multiview"
SEARCH_MODE_HYBRID = "hybrid"
SEARCH_MODE_KEYWORD_SIMILARITY = "keyword_similarity"
SEARCH_MODE_TO_RETRIEVAL_MODE = {
    SEARCH_MODE_MULTIVIEW: RETRIEVAL_MODE,
    SEARCH_MODE_HYBRID: HYBRID_MODE,
    SEARCH_MODE_KEYWORD_SIMILARITY: KEYWORD_SIMILARITY_MODE,
}
# 프로필 보강(hydration)은 비핵심 enrichment다. Qdrant 클라이언트 전역 timeout(20s)을 그대로 물리면
# 일시적 불통 시 핵심 추천 경로를 20s 막으므로, 보강 조회만 짧게 바운드해 빠르게 강등한다(정상은 ~50ms).
PROFILE_HYDRATION_TIMEOUT_S = 5.0

# 도메인 org가 들어있는 doc_attrs 키(교차-chunk 배제 후보).
_RELEVANCE_GATE_VERSION = "v1"

# grouped 진단 경로의 concept gate도 production search()와 동일하게 relevance.ConceptPlan +
# tag_chunk_concepts(evidence 확정)를 사용한다. grouped 전용 하드코딩 도메인 사전은 제거됨.


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


@dataclass(slots=True)
class _ChunkMerge:
    """여러 view에서 회수된 같은 chunk(payload.chunk_id 기준 병합)."""

    payload: ChunkPayload
    view_best_rank0: dict[str, int] = field(default_factory=dict)  # source -> 최소 0-base rank
    concept_hits: set[str] = field(default_factory=set)  # concept:<id> view가 잡은 base concept

    def observe(self, source: str, rank0: int, concept: str | None) -> None:
        prev = self.view_best_rank0.get(source)
        if prev is None or rank0 < prev:
            self.view_best_rank0[source] = rank0
        if concept:
            self.concept_hits.add(concept)


@dataclass(slots=True)
class _GroupAggregation:
    candidate: ResearcherCandidate | None
    kept_chunks: int = 0
    dropped_chunks: int = 0
    matched_concepts: set[str] = field(default_factory=set)
    missing_concepts: set[str] = field(default_factory=set)
    drop_reason: str | None = None
    researcher_id: str = ""
    researcher_name: str = ""


def _candidate_breakdown(hits: list[ResearcherCandidate], limit: int = 10) -> list[dict[str, Any]]:
    """상위 후보 집계 요약(점수/커버리지/개념/doc_type 분포) — 로깅용."""
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
                "cov": cand.coverage_type,
                "concepts": cand.matched_concepts,
                "n_chunks": len(cand.chunks),
                "doc_types": dist,
            }
        )
    return out


class QdrantHybridRetriever:
    """멀티뷰 flat 하이브리드 검색 + 관련도 기반 앱단 재점수 오케스트레이터."""

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
        self._sparse_idf, self._sparse_idf_default = self._load_sparse_idf(settings)
        self._sparse_idf_ref = max(1e-6, float(getattr(settings, "sparse_idf_ref", 3.5)))
        self._sparse_idf_hard_floor = float(getattr(settings, "sparse_idf_hard_floor", 0.0))
        self._operation_markers = frozenset(
            m.casefold() for m in getattr(settings, "operation_evidence_markers", []) if m
        )
        self._operation_factor = float(getattr(settings, "operation_evidence_factor", 1.0))

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
    @staticmethod
    def _load_sparse_idf(settings: Settings) -> tuple[dict[int, float] | None, float]:
        """query-side IDF 보정 사전 로드(sparse_stopwords.py --dump-idf 산출물).

        미설정/미존재/파싱실패면 (None, 0.0) → IDF 보정 비활성(현행 sparse 동작 유지).
        """
        path = getattr(settings, "sparse_idf_path", None)
        if not path:
            return None, 0.0
        try:
            with open(path, encoding="utf-8") as fp:
                data = json.load(fp)
            idf = {int(k): float(v) for k, v in data["idf"].items()}
            default = float(data.get("default_idf", 0.0))
            logger.info("sparse IDF 보정 활성: %d tokens, default=%.3f (%s)", len(idf), default, path)
            return idf, default
        except Exception as exc:  # noqa: BLE001 - 보정 사전 문제로 검색이 죽으면 안 됨.
            logger.warning("sparse IDF 로드 실패(%s): %s — IDF 보정 비활성", path, exc)
            return None, 0.0

    def _idf_factor(self, idf: float) -> float:
        """downweight-only 계수 ∈ [0, 1]. 1.0을 넘지 않아(boost 금지) 희귀 토큰을 증폭하지 않고
        빈출 토큰(상세/detail 등)만 누른다. idf<=hard_floor면 0(하드 마스크)."""
        if idf <= self._sparse_idf_hard_floor:
            return 0.0
        return min(1.0, idf / self._sparse_idf_ref)

    def _build_sparse_query(self, query_text: str) -> models.Document | models.SparseVector:
        if self.sparse_encoder:
            sparse_map = self.sparse_encoder.embed(query_text)
            if self._sparse_idf is not None:
                # downweight-only IDF: 빈출 토큰만 누르고 희귀 토큰은 보존(증폭 금지). 점수=Σ(q×factor)×d, 재색인 불필요.
                sparse_map = {
                    tid: w * self._idf_factor(self._sparse_idf.get(tid, self._sparse_idf_default))
                    for tid, w in sparse_map.items()
                }
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

    def _dense_prompt(self, query_text: str) -> str:
        """dense 인코딩에 실제로 투입되는 텍스트(instruct 모델이면 영문 instruct 프리픽스 부여).

        진단툴이 동일 임베딩을 재현하도록 프리픽스 구성을 이 한 곳에서만 정의(단일 출처).
        """
        if "instruct" in getattr(self.dense_encoder, "model_name", "").lower():
            return (
                "Instruct: Find experts whose profile, papers, patents, or projects "
                "match the given query.\nQuery: "
                f"{query_text}"
            )
        return query_text

    def _build_dense_query(self, query_text: str) -> list[float]:
        return self.dense_encoder.embed(self._dense_prompt(query_text))

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

    @staticmethod
    def _search_query_plan_trace(plan: SearchQueryPlan) -> dict[str, Any]:
        return {
            "raw_query": plan.raw_query,
            "dense_query": plan.dense_query,
            "sparse_joint_query": plan.sparse_joint_query,
            "sparse_concept_queries": dict(plan.sparse_concept_queries),
            "required_concepts": list(plan.required_concepts),
            "optional_concepts": list(plan.optional_concepts),
            "drop_terms_for_sparse": list(plan.drop_terms_for_sparse),
        }

    # ------------------------------------------------------ multiview helpers
    def _resolve_concept_plan(self, query: str, plan: PlannerOutput) -> ConceptPlan:
        """동적 Concept Evidence Plan 확정(planner concept_specs 우선 → query_exact 합성 폴백).

        concept·alias·검색문은 planner가 질의마다 산출하고, app-layer는 검증 규칙으로만 쓴다.
        하드코딩 도메인 사전은 더 이상 사용하지 않는다(relevance.resolve_concept_plan 단일 출처).
        """
        return resolve_concept_plan(plan, query)

    def _build_view_queries(
        self,
        search_query_plan: SearchQueryPlan,
        concept_plan: ConceptPlan,
        *,
        include_concept_views: bool = True,
    ) -> list[tuple[str, str, Any]]:
        """검색 view 목록: (source, using, query_value). 동일 sparse 텍스트는 1회만(중복 제거).

        include_concept_views=False면 concept:<id> sparse 뷰를 만들지 않는다(query_exact 노이즈 억제).
        """
        views: list[tuple[str, str, Any]] = [
            ("dense_full", DENSE_VECTOR_NAME, self._build_dense_query(search_query_plan.dense_query))
        ]
        seen_sparse: set[str] = set()

        def add_sparse(source: str, text: str | None) -> None:
            normalized = " ".join((text or "").split())
            if not normalized:
                return
            key = normalized.casefold()
            if key in seen_sparse:
                return
            seen_sparse.add(key)
            views.append((source, SPARSE_VECTOR_NAME, self._build_sparse_query(normalized)))

        add_sparse("sparse_raw", search_query_plan.raw_query)
        add_sparse("sparse_focus", search_query_plan.sparse_joint_query)
        # concept view = concept_plan.specs의 query_terms 중심(recall). evidence_terms는 검색에
        # 과투입하지 않는다(SPLADE recall 과확장 방지) — 태깅(확정)에서만 evidence를 쓴다.
        if include_concept_views:
            for key, text in concept_plan.concept_queries.items():
                add_sparse(f"{CONCEPT_VIEW_PREFIX}{key}", text)
        return views

    def _build_hybrid_view_queries(
        self, search_query_plan: SearchQueryPlan
    ) -> list[tuple[str, str, Any]]:
        """[hybrid 모드] dense_full + sparse_raw 2뷰만(단순 하이브리드). focus/concept 뷰 미사용."""
        views: list[tuple[str, str, Any]] = [
            ("dense_full", DENSE_VECTOR_NAME, self._build_dense_query(search_query_plan.dense_query))
        ]
        raw = " ".join((search_query_plan.raw_query or "").split())
        if raw:
            views.append(("sparse_raw", SPARSE_VECTOR_NAME, self._build_sparse_query(raw)))
        return views

    def _view_plan_for_mode(
        self, search_mode: str, search_query_plan: SearchQueryPlan, concept_plan: ConceptPlan
    ) -> tuple[list[tuple[str, str, Any]], dict[str, float]]:
        """view 기반 모드(multiview/hybrid)의 (view_queries, view_weights).

        multiview에서 required concept이 없으면(query_exact/generic) dense 의미신호가 순위를 주도하도록
        dense-우세 가중(multiview_generic_view_weights)을 쓰고, query_exact 합성 concept의 sparse 뷰는
        흔한 토큰 substring 노이즈를 키우므로 만들지 않는다. required concept 질의(gate 활성)는 기존 동작 유지.
        """
        if search_mode == SEARCH_MODE_HYBRID:
            return self._build_hybrid_view_queries(search_query_plan), self.settings.hybrid_view_weights

        has_required = bool(concept_plan.required)
        include_concept_views = not (
            concept_plan.source == "query_exact"
            and getattr(self.settings, "multiview_drop_query_exact_concept_views", True)
        )
        views = self._build_view_queries(
            search_query_plan, concept_plan, include_concept_views=include_concept_views
        )
        weights = (
            self.settings.search_view_weights
            if has_required
            else self.settings.multiview_generic_view_weights
        )
        return views, weights

    @staticmethod
    def _concept_of_source(source: str, concept_plan: ConceptPlan) -> str | None:
        """concept:<key> source → base concept id. 'semiconductor_experience'→'semiconductor'."""
        if not source.startswith(CONCEPT_VIEW_PREFIX):
            return None
        key = source[len(CONCEPT_VIEW_PREFIX):]
        for concept in concept_plan.all_concepts:
            if key == concept or key.startswith(concept):
                return concept
        return key

    async def _run_views(
        self, view_queries: list[tuple[str, str, Any]], base_filter: models.Filter | None
    ) -> list[tuple[str, list[Any]]]:
        """각 view를 flat query_points로 동시 실행. 실패 view는 빈 결과로 강등."""

        async def run_one(source: str, using: str, query_value: Any) -> tuple[str, list[Any]]:
            try:
                response = await asyncio.to_thread(
                    self.client.query_points,
                    collection_name=self.settings.qdrant_collection_name,
                    query=query_value,
                    using=using,
                    limit=self.settings.prefetch_limit,
                    query_filter=base_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001 — view 단위 실패는 격리
                logger.error("query_points view=%s failed: %s", source, exc, exc_info=exc)
                return source, []
            points = getattr(response, "points", None)
            if points is None and isinstance(response, dict):
                points = response.get("points")
            return source, points or []

        return await asyncio.gather(*(run_one(*view) for view in view_queries))

    def _merge_views(
        self, view_results: list[tuple[str, list[Any]]], concept_plan: ConceptPlan
    ) -> dict[str, _ChunkMerge]:
        """view 결과를 payload.chunk_id 기준 병합(point_id 아님). source별 최소 rank·concept hit 보존."""
        merge: dict[str, _ChunkMerge] = {}
        for source, points in view_results:
            concept = self._concept_of_source(source, concept_plan)
            for rank0, point in enumerate(points):
                payload = self._validate_chunk(self._point_payload_data(point))
                if payload is None:
                    continue
                entry = merge.get(payload.chunk_id)
                if entry is None:
                    entry = _ChunkMerge(payload=payload)
                    merge[payload.chunk_id] = entry
                entry.observe(source, rank0, concept)
        return merge

    def _fuse_chunks_to_researchers(
        self,
        merge: dict[str, _ChunkMerge],
        *,
        view_weights: dict[str, float],
        rrf_k: int,
        query: str,
        concept_plan: ConceptPlan,
    ) -> dict[str, list[ChunkHit]]:
        """[view 기반 모드] 병합 chunk → 융합점수(Σ view_weight × rank score) + concept 태깅 → researcher 그룹."""
        # 운영성/교육/행정 과제 근거 하향은 '질의 자체가 운영/교육을 찾는' 경우엔 끈다(역효과 방지).
        op_query = " ".join(query.split()).casefold().replace(" ", "")
        op_penalty_active = (
            self._operation_factor < 1.0
            and bool(self._operation_markers)
            and not any(marker in op_query for marker in self._operation_markers)
        )
        by_researcher: dict[str, list[ChunkHit]] = {}
        for entry in merge.values():
            fused = fuse_chunk_score(entry.view_best_rank0, view_weights=view_weights, rrf_k=rrf_k)
            # 운영성/교육/행정 과제(프로그램 운영비)는 근거 가치 하향(설계 실적 아님).
            if op_penalty_active and has_operation_marker(entry.payload, self._operation_markers):
                fused *= self._operation_factor
            concepts = tag_chunk_concepts(
                entry.payload, view_concept_hits=entry.concept_hits, concept_plan=concept_plan
            )
            hit = ChunkHit(
                score=fused,
                payload=entry.payload,
                concepts=concepts,
                sources=sorted(entry.view_best_rank0),
                display_only_concepts=chunk_display_only_concepts(entry.payload, concept_plan),
            )
            by_researcher.setdefault(entry.payload.researcher_id, []).append(hit)
        return by_researcher

    async def _retrieve_keyword_then_similarity(
        self,
        *,
        search_query_plan: SearchQueryPlan,
        concept_plan: ConceptPlan,
        base_filter: models.Filter | None,
        query: str,
    ) -> tuple[dict[str, list[ChunkHit]], dict[str, int], int]:
        """[keyword_similarity 모드] SPLADE 1차 후보 풀 → 그 안에서 dense 유사도로만 재정렬.

        1) sparse_raw(원문) SPLADE 검색으로 1차 후보 point 풀을 회수(keyword_first_stage_limit).
        2) 1차 point id 집합으로 한정(HasIdCondition)해 dense 검색 → chunk 점수 = dense 유사도(raw score).
        3) chunk_id로 중복 제거(최상위 dense rank 보존) + concept 태깅 후 researcher 그룹으로 반환.

        반환: (by_researcher, view_counts, merged_chunk_count). 융합/순위 RRF는 쓰지 않는다.
        """
        keyword_text = " ".join((search_query_plan.raw_query or "").split()) or " ".join(
            (search_query_plan.sparse_joint_query or "").split()
        )
        empty: tuple[dict[str, list[ChunkHit]], dict[str, int], int] = (
            {},
            {"sparse_keyword": 0, "dense_rerank": 0},
            0,
        )
        if not keyword_text:
            return empty

        # --- stage 1: SPLADE 키워드 1차 후보 풀(point id만 수집) ---
        try:
            stage1 = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.settings.qdrant_collection_name,
                query=self._build_sparse_query(keyword_text),
                using=SPARSE_VECTOR_NAME,
                limit=self.settings.keyword_first_stage_limit,
                query_filter=base_filter,
                with_payload=False,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 — 1차 실패는 빈 결과로 강등.
            logger.error("keyword_similarity stage1(sparse) failed: %s", exc, exc_info=exc)
            return empty
        stage1_points = getattr(stage1, "points", None)
        if stage1_points is None and isinstance(stage1, dict):
            stage1_points = stage1.get("points")
        point_ids: list[Any] = []
        for point in stage1_points or []:
            pid = getattr(point, "id", None)
            if pid is None and isinstance(point, dict):
                pid = point.get("id")
            if pid is not None:
                point_ids.append(pid)
        if not point_ids:
            return empty

        # --- stage 2: 1차 후보 한정 dense 유사도 재정렬 ---
        id_filter = self._merge_filters(base_filter, models.HasIdCondition(has_id=point_ids))
        try:
            stage2 = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.settings.qdrant_collection_name,
                query=self._build_dense_query(search_query_plan.dense_query),
                using=DENSE_VECTOR_NAME,
                limit=self.settings.prefetch_limit,
                query_filter=id_filter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 — 2차 실패는 빈 결과로 강등.
            logger.error("keyword_similarity stage2(dense) failed: %s", exc, exc_info=exc)
            return empty
        stage2_points = getattr(stage2, "points", None)
        if stage2_points is None and isinstance(stage2, dict):
            stage2_points = stage2.get("points")

        by_researcher: dict[str, list[ChunkHit]] = {}
        seen_chunk_ids: set[str] = set()
        for point in stage2_points or []:
            payload = self._validate_chunk(self._point_payload_data(point))
            if payload is None or payload.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(payload.chunk_id)
            # chunk 점수 = dense 유사도(raw). 사용자 선택형 cascade는 순위 RRF 융합을 쓰지 않는다.
            hit = ChunkHit(
                score=self._point_score(point),
                payload=payload,
                concepts=tag_chunk_concepts(payload, concept_plan=concept_plan),
                sources=["dense_similarity"],
                display_only_concepts=chunk_display_only_concepts(payload, concept_plan),
            )
            by_researcher.setdefault(payload.researcher_id, []).append(hit)
        view_counts = {"sparse_keyword": len(point_ids), "dense_rerank": len(seen_chunk_ids)}
        return by_researcher, view_counts, len(seen_chunk_ids)

    def _score_generic(self, chunks: list[ChunkHit]) -> float:
        """concept 미감지 질의용 폴백 점수: 융합 점수를 doc_type cap+harmonic으로 capped 누적."""
        cap = self.settings.doc_type_chunk_cap
        quality = self.settings.doc_type_quality_weight
        contrib: dict[str, int] = {}
        total = 0.0
        for hit in sorted(chunks, key=lambda h: -h.score):
            used = contrib.get(hit.doc_type, 0)
            if cap > 0 and used >= cap:
                continue
            total += hit.score * quality.get(hit.doc_type, 1.0) / (used + 1)
            contrib[hit.doc_type] = used + 1
        return round(total, 6)

    def _build_candidate(self, researcher_id: str, chunks: list[ChunkHit], concept_plan: ConceptPlan) -> ResearcherCandidate:
        """병합된 chunk → ResearcherCandidate.

        required 개념이 있으면 capped evidence(coverage gate) 점수를 쓴다. required 개념이 없으면
        (query_exact optional-only 또는 concept 미감지) coverage 점수가 0으로 붕괴해 전 후보가 동점→이름순
        으로 정렬되는 사고가 있었으므로, 검색 융합 관련도(_score_generic)로 순위를 매긴다. optional concept
        확정분은 표시/증거선별용으로 matched_concepts에 보존한다.
        """
        chunks.sort(key=lambda h: -h.score)
        for index, hit in enumerate(chunks, start=1):
            hit.rank = index
        identity = chunks[0].payload

        if concept_plan.required:
            scored = score_researcher(
                chunks,
                concept_plan,
                weights=self.settings.researcher_score_weights,
                doc_type_quality=self.settings.doc_type_quality_weight,
                support_top_k=self.settings.researcher_support_top_k,
                weak_evidence_floor=self.settings.weak_evidence_floor,
                require_distinct_tokens_for_joint=self.settings.joint_requires_distinct_tokens,
            )
            return ResearcherCandidate(
                researcher_id=researcher_id,
                researcher_name=identity.researcher_name,
                affiliated_organization=identity.affiliated_organization,
                highest_degree=identity.highest_degree,
                counts=identity.counts(),
                group_score=scored.score,
                rank_score=scored.score,
                chunks=chunks,
                matched_concepts=scored.matched_concepts,
                missing_concepts=sorted(
                    set(concept_plan.required) - set(scored.matched_concepts)
                ),
                coverage_type=scored.coverage_type,
                evidence_by_concept=scored.evidence_by_concept,
                score_breakdown=scored.breakdown,
            )

        # required 개념 없음: 융합 관련도로 순위(동점→이름순 붕괴 방지). optional 확정분은 표시용 보존.
        generic_score = self._score_generic(chunks)
        matched_concepts = sorted({c for hit in chunks for c in (hit.concepts or [])})
        return ResearcherCandidate(
            researcher_id=researcher_id,
            researcher_name=identity.researcher_name,
            affiliated_organization=identity.affiliated_organization,
            highest_degree=identity.highest_degree,
            counts=identity.counts(),
            group_score=generic_score,
            rank_score=generic_score,
            chunks=chunks,
            matched_concepts=matched_concepts,
        )

    # --------------------------------------------------------- aggregation
    def _candidate_from_group(
        self,
        group: Any,
        *,
        concept_plan: ConceptPlan,
    ) -> _GroupAggregation:
        """[진단 전용] query_points_groups의 PointGroup → ResearcherCandidate (앱단 RRF 누적).

        concept 태깅·gate는 production search()와 동일하게 relevance.tag_chunk_concepts(evidence
        확정)와 concept_plan.required를 사용한다. grouped 전용 도메인 사전은 더 이상 없다.
        """
        required = set(concept_plan.required)
        raw_hits = getattr(group, "hits", None)
        if raw_hits is None and isinstance(group, dict):
            raw_hits = group.get("hits")
        raw_hits = raw_hits or []

        researcher_id = str(getattr(group, "id", None) or (group.get("id") if isinstance(group, dict) else "") or "")
        identity: ChunkPayload | None = None
        chunks: list[ChunkHit] = []
        score_sum = 0.0
        doc_type_contrib: dict[str, int] = {}
        kept_chunks = 0
        dropped_chunks = 0
        matched_concepts: set[str] = set()
        cap = self.settings.doc_type_chunk_cap

        for rank, point in enumerate(raw_hits, start=1):
            payload = self._validate_chunk(self._point_payload_data(point))
            if payload is None:
                continue
            if not researcher_id:
                researcher_id = payload.researcher_id
            if identity is None:
                identity = payload
            chunk_concepts = set(tag_chunk_concepts(payload, concept_plan=concept_plan))
            if required and not chunk_concepts:
                dropped_chunks += 1
                continue
            matched_concepts.update(chunk_concepts)
            point_score = self._point_score(point)
            chunks.append(ChunkHit(score=point_score, payload=payload, rank=rank, concepts=sorted(chunk_concepts)))
            kept_chunks += 1
            # 점수 기여는 (researcher, doc_type)당 상위 cap개 chunk까지만 (다작 독식 방지).
            contributed = doc_type_contrib.get(payload.doc_type, 0)
            if cap <= 0 or contributed < cap:
                decay = 1.0 / (contributed + 1)
                score_sum += point_score * self._doc_type_prior(payload.doc_type) * decay
                doc_type_contrib[payload.doc_type] = contributed + 1

        if required:
            missing_concepts = required - matched_concepts
            if missing_concepts:
                return _GroupAggregation(
                    candidate=None,
                    kept_chunks=kept_chunks,
                    dropped_chunks=dropped_chunks,
                    matched_concepts=matched_concepts,
                    missing_concepts=missing_concepts,
                    drop_reason="relevance_concepts_missing",
                    researcher_id=researcher_id,
                    researcher_name=identity.researcher_name if identity else "",
                )

        if not chunks or not researcher_id or identity is None:
            return _GroupAggregation(
                candidate=None,
                kept_chunks=kept_chunks,
                dropped_chunks=dropped_chunks,
                matched_concepts=matched_concepts,
                drop_reason="invalid_or_empty_group",
                researcher_id=researcher_id,
                researcher_name=identity.researcher_name if identity else "",
            )

        chunks.sort(key=lambda h: -h.score)
        return _GroupAggregation(
            candidate=ResearcherCandidate(
                researcher_id=researcher_id,
                researcher_name=identity.researcher_name,
                affiliated_organization=identity.affiliated_organization,
                highest_degree=identity.highest_degree,
                counts=identity.counts(),
                group_score=score_sum,
                rank_score=score_sum,
                chunks=chunks,
                matched_concepts=sorted(matched_concepts),
                missing_concepts=[],
                coverage_type="separate" if required else "",
            ),
            kept_chunks=kept_chunks,
            dropped_chunks=dropped_chunks,
            matched_concepts=matched_concepts,
            researcher_id=researcher_id,
            researcher_name=identity.researcher_name,
        )

    @staticmethod
    def _candidate_affiliation_matches(
        candidate: ResearcherCandidate, orgs: list[str]
    ) -> bool:
        """root affiliated_organization만 기관 필터 대상으로 사용한다."""
        normalized_targets = [n for org in orgs if (n := normalize_org_name(org))]
        if not normalized_targets:
            return False
        normalized_affiliation = (
            normalize_org_name(candidate.affiliated_organization or "") or ""
        )
        if not normalized_affiliation:
            return False
        return any(
            target in normalized_affiliation or normalized_affiliation in target
            for target in normalized_targets
        )

    def _included_by_org(
        self, candidate: ResearcherCandidate, include_orgs: list[str]
    ) -> bool:
        if not include_orgs:
            return True
        return self._candidate_affiliation_matches(candidate, include_orgs)

    def _excluded_by_org(
        self, candidate: ResearcherCandidate, exclude_orgs: list[str]
    ) -> bool:
        if not exclude_orgs:
            return False
        return self._candidate_affiliation_matches(candidate, exclude_orgs)

    def _score_traces(self, hits: list[ResearcherCandidate]) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for cand in hits:
            matches: list[dict[str, Any]] = []
            fam_contrib: dict[str, float] = {}
            for hit in cand.chunks:
                family = DOC_TYPE_TO_FAMILY.get(hit.doc_type, "?")
                fam_contrib[family] = round(fam_contrib.get(family, 0.0) + hit.score, 6)
            for hit in cand.chunks[:10]:
                matches.append(
                    {
                        "doc_type": hit.doc_type,
                        "chunk_id": hit.chunk_id,
                        "rank": hit.rank,
                        "fused_score": round(hit.score, 6),
                        "concepts": hit.concepts,
                        "display_only_concepts": hit.display_only_concepts,
                        "sources": hit.sources,
                    }
                )
            traces.append(
                {
                    "expert_id": cand.researcher_id,
                    "final_score": round(cand.group_score, 6),
                    "coverage_type": cand.coverage_type,
                    "matched_concepts": cand.matched_concepts,
                    "score_breakdown": cand.score_breakdown,
                    "doc_types": cand.doc_types_present,
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
        search_mode: str = SEARCH_MODE_MULTIVIEW,
    ) -> RetrievalResult:
        """검색 모드 선택형 진입점(요청 search_mode로 선택, 기본=multiview).

        · multiview(기본): dense_full + sparse_raw/focus + concept 멀티뷰를 순위 RRF로 융합(현행).
        · hybrid: dense_full + sparse_raw 2뷰만 동일 RRF로 융합(단순 하이브리드, focus/concept 뷰 미사용).
        · keyword_similarity: SPLADE 1차 후보 풀 → 그 집합 안에서만 dense 유사도로 재정렬(2단계 cascade).
        공통: chunk_id 병합 → concept 태깅/관련도 게이트 → capped evidence 재점수 → org 필터 → 정렬. LLM no-rerank.
        """
        mode = search_mode or SEARCH_MODE_MULTIVIEW
        if mode not in SEARCH_MODE_TO_RETRIEVAL_MODE:
            logger.warning("알 수 없는 search_mode=%r → multiview 폴백", mode)
            mode = SEARCH_MODE_MULTIVIEW
        retrieval_mode = SEARCH_MODE_TO_RETRIEVAL_MODE[mode]

        concept_plan = self._resolve_concept_plan(query, plan)
        search_query_plan = self.query_builder.build_search_query_plan(query, plan, concept_plan)
        queries = CompiledQueries(
            stable=search_query_plan.dense_query,
            expanded=search_query_plan.dense_query,
        )
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        base_filter = self._merge_filters(query_filter, self._retrieval_doc_type_filter())
        required_active = self.settings.relevance_gate_enabled and bool(concept_plan.required)

        logger.info(
            "검색 컴파일: search_mode=%s mode=%s required=%s(source=%s) optional=%s "
            "limits={prefetch:%d} gate=%s",
            mode,
            retrieval_mode,
            concept_plan.required,
            concept_plan.source,
            concept_plan.optional,
            self.settings.prefetch_limit,
            required_active,
        )

        compiled_json = json.dumps(
            {
                # retrieval_mode + search_mode를 키에 포함해 모드별 캐시를 분리한다(서로 다른 검색 결과).
                "mode": retrieval_mode,
                "search_mode": mode,
                "dense_query": search_query_plan.dense_query,
                "raw_query": search_query_plan.raw_query,
                "sparse_concept_queries": search_query_plan.sparse_concept_queries,
                "required_concepts": sorted(concept_plan.required),
                "optional_concepts": sorted(concept_plan.optional),
                "concept_source": concept_plan.source,
                # 소속 기관 include/exclude는 base_filter(Qdrant)에 들어가지 않고 앱단 post-filter로만
                # 적용된다(org_survivors). 따라서 캐시 키에 직접 넣지 않으면, 무필터 질의가 캐시한
                # 결과를 다른 org 필터 요청이 그대로 재사용해 include/exclude가 침묵 우회된다(governance 사고).
                # org를 키에 포함해 org 조합별로 캐시를 분리한다.
                "include_orgs": sorted(plan.include_orgs),
                "exclude_orgs": sorted(plan.exclude_orgs),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        filter_json = str(base_filter)
        snapshot_id = self.settings.qdrant_collection_release_id

        if self.l3_cache and self.settings.cache_enabled:
            cached = self.l3_cache.get(compiled_json, filter_json, snapshot_id)
            if cached:
                hits = [ResearcherCandidate.model_validate(h) for h in cached]
                logger.info("검색 캐시 적중: layer=L3 mode=%s hits=%d", retrieval_mode, len(hits))
                return RetrievalResult(
                    hits=hits,
                    query_payload={
                        "cache": "hit",
                        "l3": True,
                        "retrieval_mode": retrieval_mode,
                        "search_mode": mode,
                        "retrieval_keywords": retrieval_keywords,
                        "search_query_plan": self._search_query_plan_trace(search_query_plan),
                        "relevance_gate_active_concepts": sorted(concept_plan.required) if required_active else [],
                    },
                    queries=queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=self._score_traces(hits),
                    cache_hit=True,
                )

        rrf_k = self.settings.view_rrf_k
        with Timer() as search_timer:
            if mode == SEARCH_MODE_KEYWORD_SIMILARITY:
                # 2단계 cascade: SPLADE 1차 풀 → dense 유사도 재정렬(뷰 융합 미사용).
                by_researcher, view_counts, merged_chunk_count = (
                    await self._retrieve_keyword_then_similarity(
                        search_query_plan=search_query_plan,
                        concept_plan=concept_plan,
                        base_filter=base_filter,
                        query=query,
                    )
                )
                view_sources = ["sparse_keyword", "dense_rerank"]
                view_weights_used: dict[str, float] = {}
                search_limits = {
                    "keyword_first_stage_limit": self.settings.keyword_first_stage_limit,
                    "prefetch_limit": self.settings.prefetch_limit,
                }
            else:
                view_queries, view_weights_used = self._view_plan_for_mode(
                    mode, search_query_plan, concept_plan
                )
                view_sources = [source for source, _, _ in view_queries]
                view_results = await self._run_views(view_queries, base_filter)
                view_counts = {source: len(points) for source, points in view_results}
                merge = self._merge_views(view_results, concept_plan)
                by_researcher = self._fuse_chunks_to_researchers(
                    merge,
                    view_weights=view_weights_used,
                    rrf_k=rrf_k,
                    query=query,
                    concept_plan=concept_plan,
                )
                merged_chunk_count = len(merge)
                search_limits = {
                    "prefetch_limit": self.settings.prefetch_limit,
                    "views": len(view_queries),
                }

        main_tier: list[ResearcherCandidate] = []
        fallback_tier: list[ResearcherCandidate] = []
        for researcher_id, chunks in by_researcher.items():
            candidate = self._build_candidate(researcher_id, chunks, concept_plan)
            if required_active and candidate.coverage_type == "partial":
                fallback_tier.append(candidate)
            else:
                main_tier.append(candidate)

        filtered_out: list[dict[str, Any]] = []
        org_filtered_count = 0

        def org_survivors(candidates: list[ResearcherCandidate]) -> list[ResearcherCandidate]:
            nonlocal org_filtered_count
            survivors: list[ResearcherCandidate] = []
            for candidate in candidates:
                if not self._included_by_org(candidate, plan.include_orgs):
                    org_filtered_count += 1
                    filtered_out.append(
                        {"expert_id": candidate.researcher_id, "name": candidate.researcher_name,
                         "reason": "include_org_mismatch"}
                    )
                    continue
                if self._excluded_by_org(candidate, plan.exclude_orgs):
                    org_filtered_count += 1
                    filtered_out.append(
                        {"expert_id": candidate.researcher_id, "name": candidate.researcher_name,
                         "reason": "excluded_org"}
                    )
                    continue
                survivors.append(candidate)
            return survivors

        main_tier = org_survivors(main_tier)
        fallback_tier = org_survivors(fallback_tier)

        fallback_kept: list[ResearcherCandidate] = []
        if self.settings.relevance_fallback_tier:
            fallback_kept = fallback_tier
        else:
            for candidate in fallback_tier:
                filtered_out.append(
                    {
                        "expert_id": candidate.researcher_id,
                        "name": candidate.researcher_name,
                        "reason": "relevance_concepts_missing",
                        "matched_concepts": sorted(candidate.matched_concepts),
                        "missing_concepts": sorted(set(concept_plan.required) - set(candidate.matched_concepts)),
                    }
                )

        final_hits = self._sort_hits(main_tier) + self._sort_hits(fallback_kept)

        logger.info(
            "검색 집계: mode=%s elapsed_ms=%.2f view_counts=%s merged_chunks=%d main=%d fallback=%d "
            "org_filtered=%d final=%d",
            retrieval_mode,
            search_timer.elapsed_ms,
            view_counts,
            merged_chunk_count,
            len(main_tier),
            len(fallback_kept),
            org_filtered_count,
            len(final_hits),
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
                "retrieval_mode": retrieval_mode,
                "search_mode": mode,
                "retrieval_keywords": retrieval_keywords,
                "search_query_plan": self._search_query_plan_trace(search_query_plan),
                "semantic_query": plan.semantic_query,
                "concept_plan": {
                    "required": concept_plan.required,
                    "optional": concept_plan.optional,
                    "source": concept_plan.source,
                    "view_sources": view_sources,
                },
                "relevance_gate_enabled": required_active,
                "relevance_gate_active_concepts": sorted(concept_plan.required) if required_active else [],
                "view_counts": view_counts,
                "merged_chunk_count": merged_chunk_count,
                "main_count": len(main_tier),
                "fallback_count": len(fallback_kept),
                "org_filtered_count": org_filtered_count,
                "final_hit_count": len(final_hits),
                "weights": {
                    "view": dict(view_weights_used),
                    "researcher": dict(self.settings.researcher_score_weights),
                    "view_rrf_k": rrf_k,
                },
                "search_limits": search_limits,
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
        search_mode: str = SEARCH_MODE_MULTIVIEW,
    ) -> RetrievalResult:
        """[/search/candidates 전용] search()로 위임(선택 search_mode 전달)."""
        return await self.search(
            query=query, plan=plan, query_filter=query_filter, search_mode=search_mode
        )

    async def hydrate_profile_evidence(
        self,
        researcher_ids: list[str],
        *,
        per_doc_type_cap: int = 3,
        fetch_limit: int = 4000,
    ) -> dict[str, list[ChunkPayload]]:
        """shortlist 후보의 researcher_id로 대표 실적 chunk를 보강 조회한다(질의 매칭 아님, 표시/맥락용).

        단일 scroll(filter=researcher_id IN[...])로 가져와 researcher×doc_type당 최근순 cap개로 제한한다.
        점수/랭킹에는 쓰지 않는다(검색 결과 evidence와 분리). 실패/빈 입력이면 {} 반환.
        """
        ids = [rid for rid in dict.fromkeys(researcher_ids) if rid]
        if not ids:
            return {}
        scroll_filter = self._merge_filters(
            models.Filter(
                must=[models.FieldCondition(key="researcher_id", match=models.MatchAny(any=ids))]
            ),
            self._retrieval_doc_type_filter(),
        )
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.scroll,
                    collection_name=self.settings.qdrant_collection_name,
                    scroll_filter=scroll_filter,
                    limit=max(1, fetch_limit),
                    with_payload=True,
                    with_vectors=False,
                ),
                timeout=PROFILE_HYDRATION_TIMEOUT_S,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # 보강만 빠르게 강등(검색은 별도 경로). Qdrant 클라이언트 20s를 기다리지 않는다.
            logger.warning(
                "hydrate_profile_evidence timed out after %.1fs (non-critical; continuing without hydration): researchers=%d",
                PROFILE_HYDRATION_TIMEOUT_S,
                len(ids),
            )
            return {}
        except Exception as exc:  # noqa: BLE001 — 보강 실패는 격리(주 추천 흐름은 계속).
            logger.error("hydrate_profile_evidence scroll failed: %s", exc, exc_info=exc)
            return {}
        points = response[0] if isinstance(response, tuple) else getattr(response, "points", response)
        grouped: dict[str, dict[str, list[ChunkPayload]]] = {}
        for point in points or []:
            payload = self._validate_chunk(self._point_payload_data(point))
            if payload is None:
                continue
            grouped.setdefault(payload.researcher_id, {}).setdefault(payload.doc_type, []).append(payload)
        out: dict[str, list[ChunkPayload]] = {}
        for researcher_id, by_doc in grouped.items():
            chunks: list[ChunkPayload] = []
            for items in by_doc.values():
                items.sort(key=lambda p: (parse_year(p.doc_date) or 0), reverse=True)
                chunks.extend(items[: max(0, per_doc_type_cap)])
            out[researcher_id] = chunks
        return out

    async def search_grouped_diagnostic(
        self,
        *,
        query: str,
        plan: PlannerOutput,
        query_filter: models.Filter | None,
    ) -> RetrievalResult:
        """[진단/AB 전용] query_points_groups(group_by=researcher_id) 단일 콜 + 앱단 RRF 누적.

        최종 랭킹 경로 아님(search() 사용). 서버 그룹화 결과를 검사하기 위한 보조 경로다.
        """
        concept_plan = self._resolve_concept_plan(query, plan)
        search_query_plan = self.query_builder.build_search_query_plan(query, plan, concept_plan)
        queries = CompiledQueries(
            stable=search_query_plan.dense_query, expanded=search_query_plan.dense_query
        )
        retrieval_keywords = self.query_builder.normalize_keywords(
            plan.retrieval_core or plan.core_keywords
        )
        base_filter = self._merge_filters(query_filter, self._retrieval_doc_type_filter())

        dense_query = self._build_dense_query(search_query_plan.dense_query)
        sparse_prefetches = [
            models.Prefetch(
                query=self._build_sparse_query(sparse_text),
                using=SPARSE_VECTOR_NAME,
                limit=self.settings.prefetch_limit,
                filter=base_filter,
            )
            for sparse_text in search_query_plan.sparse_queries().values()
        ]
        grouped_payload = {
            "collection_name": self.settings.qdrant_collection_name,
            "prefetch": [
                models.Prefetch(
                    query=dense_query, using=DENSE_VECTOR_NAME,
                    limit=self.settings.prefetch_limit, filter=base_filter,
                ),
                *sparse_prefetches,
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
            except Exception as exc:  # noqa: BLE001
                logger.error("query_points_groups failed: %s", exc, exc_info=exc)
                return RetrievalResult(
                    hits=[],
                    query_payload={"retrieval_mode": GROUPED_DIAGNOSTIC_MODE, "error": str(exc)},
                    queries=queries,
                    retrieval_keywords=retrieval_keywords,
                    retrieval_score_traces=[],
                )

        groups = getattr(result, "groups", None)
        if groups is None and isinstance(result, dict):
            groups = result.get("groups")
        groups = groups or []

        filtered_out: list[dict[str, Any]] = []
        candidates: list[ResearcherCandidate] = []
        kept = dropped = filtered = 0
        for group in groups:
            aggregation = self._candidate_from_group(group, concept_plan=concept_plan)
            kept += aggregation.kept_chunks
            dropped += aggregation.dropped_chunks
            if aggregation.candidate is not None:
                candidates.append(aggregation.candidate)
            elif aggregation.drop_reason == "relevance_concepts_missing":
                filtered += 1
                filtered_out.append(
                    {
                        "expert_id": aggregation.researcher_id,
                        "name": aggregation.researcher_name,
                        "reason": aggregation.drop_reason,
                        "matched_concepts": sorted(aggregation.matched_concepts),
                        "missing_concepts": sorted(aggregation.missing_concepts),
                    }
                )

        org_filtered = 0
        survivors: list[ResearcherCandidate] = []
        for candidate in candidates:
            if not self._included_by_org(candidate, plan.include_orgs):
                org_filtered += 1
                filtered_out.append(
                    {
                        "expert_id": candidate.researcher_id,
                        "name": candidate.researcher_name,
                        "reason": "include_org_mismatch",
                    }
                )
                continue
            if self._excluded_by_org(candidate, plan.exclude_orgs):
                org_filtered += 1
                filtered_out.append(
                    {
                        "expert_id": candidate.researcher_id,
                        "name": candidate.researcher_name,
                        "reason": "excluded_org",
                    }
                )
                continue
            survivors.append(candidate)
        final_hits = self._sort_hits(survivors)

        return RetrievalResult(
            hits=final_hits,
            query_payload={
                "retrieval_mode": GROUPED_DIAGNOSTIC_MODE,
                "retrieval_keywords": retrieval_keywords,
                "search_query_plan": self._search_query_plan_trace(search_query_plan),
                "relevance_gate_active_concepts": sorted(concept_plan.required),
                "relevance_kept_chunk_count": kept,
                "relevance_dropped_chunk_count": dropped,
                "relevance_filtered_candidate_count": filtered,
                "group_count": len(groups),
                "aggregated_candidate_count": len(candidates),
                "org_filtered_count": org_filtered,
                "final_hit_count": len(final_hits),
                "timers": {"search_ms": search_timer.elapsed_ms},
            },
            queries=queries,
            retrieval_keywords=retrieval_keywords,
            retrieval_score_traces=self._score_traces(final_hits),
            filtered_out_candidates=filtered_out,
        )
