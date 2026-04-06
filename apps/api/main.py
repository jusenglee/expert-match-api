"""
NTIS 전문가 추천 시스템의 메인 API 엔트리포인트입니다.
이 모듈은 FastAPI 애플리케이션을 초기화하고, 검색/추천 서비스의 런타임 의존성을 설정하며,
주요 API 엔드포인트(추천, 검색, 헬스체크, 피드백 등)를 정의합니다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from qdrant_client import QdrantClient

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
from apps.core.config import Settings, get_settings
from apps.core.feedback_store import FeedbackStore
from apps.core.logging import configure_logging
from apps.core.runtime_validation import RuntimeDependencyValidator, validate_runtime_settings
from apps.recommendation.cards import CandidateCardBuilder
from apps.recommendation.judge import HeuristicJudge, OpenAICompatJudge
from apps.recommendation.planner import HeuristicPlanner, OpenAICompatPlanner
from apps.recommendation.service import RecommendationService
from apps.search.encoders import HashingDenseEncoder, OpenAIEmbeddingEncoder
from apps.search.filters import QdrantFilterCompiler
from apps.search.live_validator import LiveContractValidator
from apps.search.qdrant_bootstrap import QdrantBootstrapper
from apps.search.query_builder import QueryTextBuilder
from apps.search.retriever import QdrantHybridRetriever
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.seed_runner import SeedRunner

logger = logging.getLogger(__name__)


def build_dense_encoder(settings: Settings):
    """
    설정에 따라 적절한 Dense Encoder(임베딩 엔진)를 생성합니다.
    
    - openai: 실제 OpenAI API를 사용하여 의미론적 임베딩을 생성합니다.
    - local: 로컬 SentenceTransformer 모델을 사용하여 임베딩을 생성합니다.
    - 그 외: 테스트나 로컬 개발용 deterministic hashing encoder를 사용합니다.
    """
    if settings.embedding_backend == "openai":
        # 운영 환경에서 실제 임베딩 서버를 붙일 때 쓰는 경로다.
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
        # 로컬 기본값은 deterministic hashing encoder다.
        # 실제 의미 임베딩을 대체하는 것은 아니고, seed/tests를 외부 모델 없이 돌리기 위한 장치다.
        model_name=settings.embedding_model_name,
        vector_size=settings.embedding_vector_size,
    )


async def build_app_runtime(settings: Settings) -> tuple[RecommendationService, LiveContractValidator]:
    """
    애플리케이션 구동에 필요한 핵심 런타임 객체들을 초기화합니다.
    Qdrant 클라이언트, 임베딩 엔진, LLM Planner/Judge, 피드백 저장소 등을 준비합니다.
    """
    # 서비스 초기화 시점에 Qdrant 계약, feedback 저장소, seed를 함께 준비한다.
    # 이렇게 해두면 요청 처리 경로는 검색/추천 자체에만 집중할 수 있다.
    validate_runtime_settings(settings)

    registry = SearchSchemaRegistry.default()
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        cloud_inference=settings.qdrant_cloud_inference,
    )
    dense_encoder = build_dense_encoder(settings)

    # LLM 백엔드 설정에 따라 Heuristic(규칙 기반) 또는 OpenAI(LLM 기반) Planner/Judge를 선택합니다.
    planner = HeuristicPlanner() if settings.llm_backend == "heuristic" else OpenAICompatPlanner(settings)
    judge = HeuristicJudge() if settings.llm_backend == "heuristic" else OpenAICompatJudge(settings)

    # 사용자 피드백을 저장할 SQLite 저장소를 초기화합니다.
    feedback_store = FeedbackStore(settings.feedback_db_path, settings.feedback_table)
    feedback_store.initialize()

    # Qdrant 컬렉션이 존재하는지 확인하고 필요시 생성합니다.
    bootstrapper = QdrantBootstrapper(client=client, settings=settings, registry=registry)
    bootstrapper.ensure_collection(recreate=settings.seed_allow_recreate_collection)

    # 추천 서비스(핵심 비즈니스 로직)를 조립합니다.
    service = RecommendationService(
        planner=planner,
        retriever=QdrantHybridRetriever(
            client=client,
            settings=settings,
            registry=registry,
            dense_encoder=dense_encoder,
            query_builder=QueryTextBuilder(),
        ),
        filter_compiler=QdrantFilterCompiler(),
        card_builder=CandidateCardBuilder(),
        judge=judge,
        feedback_store=feedback_store,
        shortlist_limit=settings.shortlist_limit,
    )
    
    # 실시간 시스템 무결성 검증을 위한 Validator를 생성합니다.
    validator = LiveContractValidator(
        client=client,
        settings=settings,
        registry=registry,
        dependency_validator=RuntimeDependencyValidator(settings),
    )

    # 시작 시 데이터 시딩(Seeding)이 설정되어 있으면 실행합니다.
    if settings.seed_on_startup:
        seed_runner = SeedRunner(
            client=client,
            settings=settings,
            registry=registry,
            dense_encoder=dense_encoder,
        )
        logger.info("Seeding development data into Qdrant...")
        await seed_runner.seed()

    return service, validator


def create_app(
    settings: Settings | None = None,
    service: RecommendationService | None = None,
    validator: LiveContractValidator | None = None,
) -> FastAPI:
    """
    FastAPI 애플리케이션 인스턴스를 생성하고 라우트를 설정합니다.
    """
    configure_logging()
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        애플리케이션 생명주기 관리(시작/종료 시 작업 정의).
        비동기로 런타임 서비스를 초기화하여 app.state에 저장합니다.
        """
        app.state.settings = active_settings
        app.state.recommendation_service = service
        app.state.live_validator = validator
        app.state.startup_error = None
        if app.state.recommendation_service is None:
            try:
                built_service, built_validator = await build_app_runtime(active_settings)
            except Exception as exc:
                app.state.startup_error = f"Application startup failed: {exc}"
                logger.exception("Application startup failed; readiness will report degraded state")
            else:
                app.state.recommendation_service = built_service
                app.state.live_validator = built_validator
        yield

    app = FastAPI(title=active_settings.app_name, lifespan=lifespan)

    def get_startup_error() -> str | None:
        """시작 과정에서 발생한 에러를 가져옵니다."""
        return getattr(app.state, "startup_error", None)

    def build_readiness_response(
        *,
        ready: bool,
        checks: dict[str, bool],
        issues: list[str],
        sample_point_id: str | None = None,
    ) -> ReadinessResponse:
        """준비 상태 응답 객체를 생성합니다."""
        return ReadinessResponse(
            ready=ready,
            checks=checks,
            issues=issues,
            collection_name=active_settings.qdrant_collection_name,
            sample_point_id=sample_point_id,
        )

    def get_service() -> RecommendationService:
        """app.state에서 추천 서비스를 가져오거나 준비되지 않은 경우 예외를 발생시킵니다."""
        service = app.state.recommendation_service
        if service is None:
            startup_error = get_startup_error()
            if startup_error is not None:
                raise HTTPException(
                    status_code=503,
                    detail=f"Recommendation service is not ready: {startup_error}",
                )
            raise HTTPException(status_code=503, detail="Recommendation service is not ready")
        return service

    @app.get("/health")
    def health() -> dict[str, object]:
        """기본적인 서비스 헬스체크 엔드포인트입니다."""
        return {
            "status": "ok",
            "collection_name": active_settings.qdrant_collection_name,
            "searched_branches": list(BRANCHES),
        }

    @app.get("/playground", include_in_schema=False, response_class=HTMLResponse)
    def playground() -> HTMLResponse:
        """로컬 테스트를 위한 웹 UI(Playground)를 제공합니다."""
        return HTMLResponse(PLAYGROUND_HTML)

    @app.get("/health/ready", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse | JSONResponse:
        """
        서비스 상세 준비 상태를 확인합니다.
        Qdrant 연결, 데이터 스키마 일치 여부 등을 실시간으로 검증합니다.
        """
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
            return JSONResponse(status_code=503, content=degraded_report.model_dump(mode="json"))
        if not report.ready:
            payload = ReadinessResponse.model_validate(report.to_dict())
            return JSONResponse(status_code=503, content=payload.model_dump(mode="json"))
        return ReadinessResponse.model_validate(report.to_dict())

    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest) -> RecommendationResponse:
        """
        자연어 질의를 기반으로 최종 전문가 추천 목록을 산출합니다.
        질의 분석(Planner) -> 검색(Retriever) -> 다차원 평가(Judge) 과정을 거칩니다.
        """
        service = get_service()
        result = await service.recommend(
            query=request.query,
            filters_override=request.filters_override,
            exclude_orgs=request.exclude_orgs,
            top_k=request.top_k,
        )
        return RecommendationResponse.model_validate(result)

    @app.post("/search/candidates", response_model=SearchCandidatesResponse)
    async def search_candidates(request: SearchCandidatesRequest) -> SearchCandidatesResponse:
        """
        최종 추천 단계(Judge)를 거치지 않고, 검색 필터가 적용된 전문가 후보 목록만 조회합니다.
        """
        service = get_service()
        result = await service.search_candidates(
            query=request.query,
            filters_override=request.filters_override,
            exclude_orgs=request.exclude_orgs,
            top_k=request.top_k,
        )
        candidate_items = [
            SearchCandidateItem(
                expert_id=card.expert_id,
                name=card.name,
                organization=card.organization,
                branch_coverage=card.branch_coverage,
                counts=card.counts,
                data_gaps=card.data_gaps,
                risks=card.risks,
                shortlist_score=card.shortlist_score,
            )
            for card in result["candidates"]
        ]
        return SearchCandidatesResponse(
            intent_summary=result["planner"].intent_summary,
            applied_filters=result["planner"].hard_filters,
            searched_branches=list(BRANCHES),
            retrieved_count=result["retrieved_count"],
            candidates=candidate_items,
            trace={
                "planner": result["planner"].model_dump(mode="json"),
                "branch_queries": result["branch_queries"],
                "exclude_orgs": result["planner"].exclude_orgs,
                "candidate_ids": [card.expert_id for card in result["candidates"]],
                "query_payload": service._serialize_query_payload(result["query_payload"]),
            },
        )

    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        """
        추천 결과에 대한 사용자의 피드백을 저장합니다.
        선택/거절된 전문가 ID와 관리자 메모 등을 기록합니다.
        """
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
