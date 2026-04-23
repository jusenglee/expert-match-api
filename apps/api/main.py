"""
NTIS 전문가 추천 시스템의 메인 API 엔트리포인트입니다.
이 모듈은 FastAPI 애플리케이션을 초기화하고, 검색/추천 서비스의 런타임 의존성을 설정하며,
주요 API 엔드포인트(추천, 검색, 헬스체크, 피드백 등)를 정의합니다.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from qdrant_client import QdrantClient
import uvicorn

from apps.api.playground import PLAYGROUND_HTML
from apps.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ReadinessResponse,
    RecommendationRequest,
    RecommendationResponse,
    SearchCandidateItem,
    SearchCandidatesRequest,
    SearchCandidatesResponse,
)
from apps.core.cache import PlanCache, RetrievalResultCache
from apps.core.config import Settings, get_settings
from apps.core.feedback_store import FeedbackStore
from apps.core.logging import configure_logging, request_id_ctx, captured_logs_ctx
from apps.core.runtime_validation import (
    RuntimeDependencyValidator,
    validate_runtime_settings,
)
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.evidence_selector import KeywordEvidenceSelector
from apps.recommendation.planner import HeuristicPlanner, OpenAICompatPlanner
from apps.recommendation.reasoner import (
    OpenAICompatReasonGenerator,
    PassThroughReasonGenerator,
)
from apps.recommendation.service import RecommendationService
from apps.search.encoders import (
    HashingDenseEncoder, 
    OpenAIEmbeddingEncoder, 
    SpladeSparseEncoder
)
from apps.search.filters import QdrantFilterCompiler
from apps.search.live_validator import LiveContractValidator
from apps.search.qdrant_bootstrap import QdrantBootstrapper
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.sparse_runtime import (
    prepare_sparse_runtime_environment,
    resolve_sparse_runtime,
)

logger = logging.getLogger(__name__)


def _normalize_query_text(query: str) -> str:
    return ", ".join(line.strip() for line in query.splitlines() if line.strip())


def build_dense_encoder(settings: Settings):
    """
    설정에 따라 적절한 Dense Encoder(임베딩 엔진)를 생성합니다.
    """
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingEncoder(
            model_name=settings.embedding_model_name,
            vector_size=settings.embedding_vector_size,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
    elif settings.embedding_backend == "local":
        from apps.search.encoders import LocalSentenceTransformerEncoder

        return LocalSentenceTransformerEncoder(
            model_name=settings.embedding_model_name,
            vector_size=settings.embedding_vector_size,
        )
    return HashingDenseEncoder(
        model_name=settings.embedding_model_name,
        vector_size=settings.embedding_vector_size,
    )


async def build_app_runtime(
    settings: Settings,
) -> tuple[RecommendationService, LiveContractValidator]:
    """
    애플리케이션 구동에 필요한 핵심 런타임 객체(의존성)들을 초기화하는 팩토리 함수입니다.
    - 데이터베이스 클라이언트(Qdrant)
    - 캐시 매니저(L1 플랜 캐시, L3 검색 결과 캐시)
    - 모델 컴포넌트(Planner, Reasoner, Encoder 등)
    - 최종 RecommendationService 조립
    """
    logger.info(
        "앱 런타임 초기화 시작: app_env=%s app_name=%r",
        settings.app_env,
        settings.app_name,
    )
    if settings.hf_hub_offline:
        logger.info("HuggingFace Hub 오프라인 모드 강제 설정 (HF_HUB_OFFLINE=1)")

    validate_runtime_settings(settings)

    registry = SearchSchemaRegistry.default()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
        timeout=20.0,
    )
    # 로컬 Sparse 인코딩 기능 활성화 (fastembed 필요 모델용)
    cache_dir = prepare_sparse_runtime_environment(settings)

    dense_encoder = build_dense_encoder(settings)
    sparse_runtime, sparse_encoder = resolve_sparse_runtime(
        client=client,
        settings=settings,
        cache_dir=cache_dir,
        sparse_encoder_factory=SpladeSparseEncoder,
    )

    # L1 캐시 초기화 (Canonical Plan Cache)
    plan_cache = PlanCache(cache_dir=settings.runtime_dir / "cache" / "planner")
    
    # L3 캐시 초기화 (Retrieval Result Cache)
    retrieval_cache = RetrievalResultCache(cache_dir=settings.runtime_dir / "cache" / "retrieval")

    planner = (
        HeuristicPlanner()
        if settings.llm_backend == "heuristic"
        else OpenAICompatPlanner(settings, cache=plan_cache)
    )
    reason_generator = (
        PassThroughReasonGenerator()
        if settings.llm_backend == "heuristic"
        else OpenAICompatReasonGenerator(settings)
    )

    feedback_store = FeedbackStore(settings.feedback_db_path, settings.feedback_table)
    feedback_store.initialize()

    bootstrapper = QdrantBootstrapper(
        client=client,
        settings=settings,
        registry=registry,
        sparse_runtime=sparse_runtime,
    )
    bootstrapper.ensure_collection(recreate=settings.seed_allow_recreate_collection)

    service = RecommendationService(
        planner=planner,
        retriever=QdrantHybridRetriever(
            client=client,
            settings=settings,
            registry=registry,
            dense_encoder=dense_encoder,
            sparse_encoder=sparse_encoder,
            sparse_runtime=sparse_runtime,
            query_builder=QueryTextBuilder(),
            l3_cache=retrieval_cache,
        ),
        filter_compiler=QdrantFilterCompiler(),
        card_builder=CandidateCardBuilder(),
        evidence_selector=KeywordEvidenceSelector(),
        reason_generator=reason_generator,
        feedback_store=feedback_store,
    )

    validator = LiveContractValidator(
        client=client,
        settings=settings,
        registry=registry,
        dependency_validator=RuntimeDependencyValidator(settings),
        sparse_runtime=sparse_runtime,
    )

    return service, validator


def create_app(
    settings: Settings | None = None,
    service: RecommendationService | None = None,
    validator: LiveContractValidator | None = None,
) -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고 생명주기(Lifespan) 및 주요 API 엔드포인트(라우트)를 설정합니다.
    """
    configure_logging()
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = active_settings
        app.state.recommendation_service = service
        app.state.live_validator = validator
        app.state.startup_error = None
        if app.state.recommendation_service is None:
            try:
                built_service, built_validator = await build_app_runtime(
                    active_settings
                )
            except Exception as exc:
                app.state.startup_error = f"Application startup failed: {exc}"
                logger.exception(
                    "Application startup failed; readiness will report degraded state"
                )
            else:
                app.state.recommendation_service = built_service
                app.state.live_validator = built_validator
        yield

    app = FastAPI(title=active_settings.app_name, lifespan=lifespan)

    @app.middleware("http")
    async def log_request_middleware(request: Request, call_next):
        """요청마다 고유 ID를 부여하고 처리 시간과 상태를 로그로 남깁니다."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        token = request_id_ctx.set(request_id)
        # 로그 캡처 버퍼 초기화
        log_token = captured_logs_ctx.set([])

        t0 = time.monotonic()
        try:
            logger.info("--> 요청 시작: [%s] %s", request.method, request.url.path)
            response = await call_next(request)
            dt_ms = (time.monotonic() - t0) * 1000

            response.headers["X-Request-ID"] = request_id

            logger.info(
                "<-- 요청 완료: [%s] %s | 상태 코드: %d | 소요 시간: %.2fms",
                request.method,
                request.url.path,
                response.status_code,
                dt_ms,
            )
            return response
        finally:
            request_id_ctx.reset(token)
            captured_logs_ctx.reset(log_token)

    def get_startup_error() -> str | None:
        return getattr(app.state, "startup_error", None)

    def build_readiness_response(
        *,
        ready: bool,
        checks: dict[str, bool],
        issues: list[str],
        sample_point_id: str | None = None,
    ) -> ReadinessResponse:
        return ReadinessResponse(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=active_settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

    def get_service() -> RecommendationService:
        service = app.state.recommendation_service
        if service is None:
            startup_error = get_startup_error()
            if startup_error is not None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Recommendation service is not ready: {startup_error}",
                )
            raise HTTPException(
                status_code=503, detail="Recommendation service is not ready"
            )
        return service

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "collection_name": active_settings.qdrant_collection_name,
            "searched_branches": list(BRANCHES),
        }

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    @app.get("/playground", include_in_schema=False, response_class=HTMLResponse)
    def playground() -> HTMLResponse:
        return HTMLResponse(PLAYGROUND_HTML)

    @app.get("/health/ready", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse | JSONResponse:
        startup_error = get_startup_error()
        if startup_error is not None:
            report = build_readiness_response(
                ready=False,
                checks={"startup_runtime_initialized": False},
                issues=[startup_error],
            )
            return JSONResponse(status_code=503, content=report.model_dump(mode="json"))

        validator = app.state.live_validator
        if validator is None:
            report = build_readiness_response(
                ready=False,
                checks={"validator_initialized": False},
                issues=["Live validator is not ready"],
            )
            return JSONResponse(status_code=503, content=report.model_dump(mode="json"))

        try:
            report = validator.validate()
        except Exception as exc:
            logger.exception("Readiness validation crashed")
            degraded_report = build_readiness_response(
                ready=False,
                checks={"readiness_check_completed": False},
                issues=[f"Readiness validation crashed: {exc}"],
            )
            return JSONResponse(
                status_code=503, content=degraded_report.model_dump(mode="json")
            )
        if not report.ready:
            payload = ReadinessResponse.model_validate(report.to_dict())
            return JSONResponse(
                status_code=503, content=payload.model_dump(mode="json")
            )
        return ReadinessResponse.model_validate(report.to_dict())

    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest) -> RecommendationResponse:
        """
        [2단계 최종 추천 API]
        사용자 질의를 바탕으로 전문가를 검색하고 증거를 필터링한 뒤, 
        LLM을 사용하여 전문가별 맞춤형 추천 사유(Reasoning)가 포함된 최종 추천 결과를 반환합니다.
        """
        service = get_service()
        normalized_query = _normalize_query_text(request.query)
        result = await service.recommend(
            query=normalized_query,
            filters_override=request.filters_override,
            include_orgs=request.include_orgs,
            exclude_orgs=request.exclude_orgs,
            top_k=request.top_k,
        )
        # 캡처된 로그 주입
        logs = captured_logs_ctx.get()
        if logs is not None and "trace" in result:
            result["trace"]["server_logs"] = logs

        return RecommendationResponse.model_validate(result)

    @app.post("/recommend/stream")
    async def recommend_stream(request: RecommendationRequest) -> StreamingResponse:
        """
        [2단계 최종 추천 스트리밍 API]
        SSE(Server-Sent Events)를 통해 추천 생성 과정을 실시간으로 클라이언트에게 전송합니다.
        """
        service = get_service()
        normalized_query = _normalize_query_text(request.query)

        async def event_generator():
            try:
                async for event_chunk in service.recommend_stream(
                    query=normalized_query,
                    filters_override=request.filters_override,
                    include_orgs=request.include_orgs,
                    exclude_orgs=request.exclude_orgs,
                    top_k=request.top_k,
                ):
                    yield event_chunk
            except Exception as e:
                import json
                logger.exception("Error during recommendation stream")
                yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/search/candidates", response_model=SearchCandidatesResponse)
    async def search_candidates(
        request: SearchCandidatesRequest,
    ) -> SearchCandidatesResponse:
        """
        [1단계 단순 검색 API]
        LLM 추천 사유 생성 및 엄격한 증거 필터링 관문을 거치지 않고, 
        Qdrant 벡터 데이터베이스의 순수 하이브리드 검색 후보군(Candidates) 목록을 빠르게 반환합니다.
        """
        service = get_service()
        normalized_query = _normalize_query_text(request.query)
        result = await service.search_candidates(
            query=normalized_query,
            filters_override=request.filters_override,
            include_orgs=request.include_orgs,
            exclude_orgs=request.exclude_orgs,
            top_k=request.top_k,
        )

        # 캡처된 로그 주입
        logs = captured_logs_ctx.get()

        candidate_items = []
        for card in result["candidates"]:
            # hits_with_support에서 매칭되는 원본 히트를 찾아 지지 정보 추출
            support_hit = next((h for h in result.get("hits_with_support", []) if h.expert_id == card.expert_id), None)
            
            item = SearchCandidateItem(
                expert_id=card.expert_id,
                name=card.name,
                organization=card.organization,
                branch_presence_flags=card.branch_presence_flags,
                counts=card.counts,
                data_gaps=card.data_gaps,
                risks=card.risks,
                shortlist_score=card.shortlist_score,
                stable_hits=support_hit.stable_support_count if support_hit else 0,
                expanded_hits=support_hit.expanded_support_count if support_hit else 0,
                support_branches=support_hit.support_branches if support_hit else [],
            )
            candidate_items.append(item)

        planner_payload = result["planner"].model_dump(mode="json")

        trace = {
            "planner": planner_payload,
            "planner_trace": result.get("planner_trace") or {},
            "raw_query": result.get("raw_query", normalized_query),
            "planner_keywords": (
                (result.get("planner_trace") or {}).get("planner_keywords") or []
            ),
            "retrieval_keywords": result.get("retrieval_keywords") or [],
            "bundle_ids": planner_payload.get("bundle_ids", []),
            "expanded_shadow_hits": result.get("expanded_shadow_hits") or [],
            "removed_role_terms": (
                (result.get("planner_trace") or {}).get("removed_role_terms") or []
            ),
            "cache": {
                "canonical_plan": (result.get("planner_trace") or {}).get("cache", {}).get("canonical_plan", "miss"),
                "retrieval": "hit" if result.get("cache_hit") else "miss"
            },
            "planner_retry_count": (
                (result.get("planner_trace") or {}).get("planner_retry_count", 0)
            ),
            "support_rule_applied": True,
            "filtered_out_count": len(result.get("filtered_out_candidates") or []),
            "filtered_out_candidates": result.get("filtered_out_candidates") or [],
            "retrieval_skipped_reason": result.get("retrieval_skipped_reason"),
            "branch_queries": result["branch_queries"],
            "include_orgs": result["planner"].include_orgs,
            "exclude_orgs": result["planner"].exclude_orgs,
            "candidate_ids": [card.expert_id for card in result["candidates"]],
            "candidate_support_info": [
                {
                    "expert_id": hit.expert_id,
                    "stable_hits": hit.stable_support_count,
                    "expanded_hits": hit.expanded_support_count,
                    "support_branches": hit.support_branches,
                }
                for hit in result.get("hits_with_support", [])
            ],
            "retrieval_score_traces": result.get("retrieval_score_traces") or [],
            "final_sort_policy": result.get("final_sort_policy"),
            "query_payload": service._serialize_query_payload(result["query_payload"]),
            "timers": result.get("timers") or {},
        }
        if logs is not None:
            trace["server_logs"] = logs

        return SearchCandidatesResponse(
            intent_summary=result["planner"].intent_summary,
            applied_filters=result["planner"].hard_filters,
            searched_branches=list(BRANCHES),
            retrieved_count=result["retrieved_count"],
            candidates=candidate_items,
            trace=trace,
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        service = get_service()
        feedback_id = service.save_feedback(
            query=request.query,
            selected_expert_ids=request.selected_expert_ids,
            rejected_expert_ids=request.rejected_expert_ids,
            notes=request.notes,
            metadata=request.metadata,
        )
        return FeedbackResponse(feedback_id=feedback_id)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011, access_log=False)
