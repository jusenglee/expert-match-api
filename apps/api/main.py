from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

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
    if settings.embedding_backend == "openai":
        # 운영 환경에서 실제 임베딩 서버를 붙일 때 쓰는 경로다.
        return OpenAIEmbeddingEncoder(
            model_name=settings.embedding_model_name,
            vector_size=settings.embedding_vector_size,
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
        )
    return HashingDenseEncoder(
        # 로컬 기본값은 deterministic hashing encoder다.
        # 실제 의미 임베딩을 대체하는 것은 아니고, seed/tests를 외부 모델 없이 돌리기 위한 장치다.
        model_name=settings.embedding_model_name,
        vector_size=settings.embedding_vector_size,
    )


async def build_app_runtime(settings: Settings) -> tuple[RecommendationService, LiveContractValidator]:
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

    planner = HeuristicPlanner() if settings.llm_backend == "heuristic" else OpenAICompatPlanner(settings)
    judge = HeuristicJudge() if settings.llm_backend == "heuristic" else OpenAICompatJudge(settings)

    feedback_store = FeedbackStore(settings.feedback_db_path, settings.feedback_table)
    feedback_store.initialize()

    bootstrapper = QdrantBootstrapper(client=client, settings=settings, registry=registry)
    bootstrapper.ensure_collection(recreate=settings.seed_allow_recreate_collection)

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
    validator = LiveContractValidator(
        client=client,
        settings=settings,
        registry=registry,
        dependency_validator=RuntimeDependencyValidator(settings),
    )

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
    configure_logging()
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = active_settings
        app.state.recommendation_service = service
        app.state.live_validator = validator
        if app.state.recommendation_service is None:
            built_service, built_validator = await build_app_runtime(active_settings)
            app.state.recommendation_service = built_service
            app.state.live_validator = built_validator
        yield

    app = FastAPI(title=active_settings.app_name, lifespan=lifespan)

    def get_service() -> RecommendationService:
        service = app.state.recommendation_service
        if service is None:
            raise HTTPException(status_code=503, detail="Recommendation service is not ready")
        return service

    def get_validator() -> LiveContractValidator:
        validator = app.state.live_validator
        if validator is None:
            raise HTTPException(status_code=503, detail="Live validator is not ready")
        return validator

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "collection_name": active_settings.qdrant_collection_name,
            "searched_branches": list(BRANCHES),
        }

    @app.get("/health/ready", response_model=ReadinessResponse)
    def readiness() -> ReadinessResponse:
        report = get_validator().validate()
        if not report.ready:
            raise HTTPException(status_code=503, detail=report.to_dict())
        return ReadinessResponse.model_validate(report.to_dict())

    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest) -> RecommendationResponse:
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
