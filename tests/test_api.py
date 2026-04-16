from __future__ import annotations

import apps.api.main as main_module
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from types import SimpleNamespace

from apps.api.main import create_app
from apps.api.schemas import RecommendationResponse, ReadinessResponse
from apps.core.config import Settings
from apps.domain.models import RecommendationDecision
from apps.search.sparse_runtime import ONLINE_PIXIE_SPLADE_MODEL, QDRANT_BM25_MODEL


class FakeRecommendationService:
    def __init__(self):
        self.last_recommend_query = None
        self.last_search_query = None

    async def recommend(self, *, query, filters_override, include_orgs, exclude_orgs, top_k):
        self.last_recommend_query = query
        return {
            "intent_summary": query,
            "applied_filters": filters_override,
            "searched_branches": ["basic", "art", "pat", "pjt"],
            "retrieved_count": 4,
            "recommendations": [
                RecommendationDecision(
                    rank=1,
                    expert_id="1",
                    name="Hong Gildong",
                    fit="높음",
                    recommendation_reason="Publication evidence is available.",
                    evidence=[{"type": "paper", "title": "Test paper"}],
                    risks=[],
                )
            ],
            "data_gaps": [],
            "not_selected_reasons": [],
            "trace": {
                "planner": {},
                "planner_trace": {},
                "reason_generation_trace": {},
                "branch_queries": {
                    "basic": "basic",
                    "art": "paper",
                    "pat": "patent",
                    "pjt": "project",
                },
                "exclude_orgs": [],
                "candidate_ids": ["1"],
                "recommendation_ids": ["1"],
                "retrieval_score_traces": [
                    {
                        "expert_id": "1",
                        "point_id": "1_basic",
                        "point_branch_hint": "basic",
                        "final_score": 17.0,
                        "primary_branch": "basic",
                        "branch_matches": [
                            {"branch": "basic", "rank": 1, "score": 17.0}
                        ],
                    }
                ],
                "final_sort_policy": "rrf_score_desc_name_asc",
                "top_k_used": top_k or 1,
                "query_payload": {},
            },
        }

    async def search_candidates(self, *, query, filters_override, include_orgs, exclude_orgs, top_k):
        self.last_search_query = query
        class Planner(SimpleNamespace):
            def model_dump(self, mode="json"):
                return {
                    "intent_summary": self.intent_summary,
                    "hard_filters": self.hard_filters,
                    "include_orgs": self.include_orgs,
                    "exclude_orgs": self.exclude_orgs,
                }

        class Card:
            expert_id = "1"
            name = "Hong Gildong"
            organization = "Test Institute"
            branch_presence_flags = {
                "basic": True,
                "art": True,
                "pat": False,
                "pjt": True,
            }
            counts = {"article_cnt": 2, "scie_cnt": 1, "patent_cnt": 0, "project_cnt": 3}
            data_gaps = ["Patent evidence is missing."]
            risks = []
            shortlist_score = 17.0

        return {
            "planner": Planner(
                intent_summary=query,
                hard_filters=filters_override,
                include_orgs=include_orgs,
                exclude_orgs=exclude_orgs,
            ),
            "planner_trace": {"planner_keywords": ["semiconductor"]},
            "retrieved_count": 1,
            "candidates": [Card()],
            "branch_queries": {
                "basic": "basic",
                "art": "paper",
                "pat": "patent",
                "pjt": "project",
            },
            "retrieval_keywords": ["semiconductor"],
            "retrieval_score_traces": [
                {
                    "expert_id": "1",
                    "point_id": "1_basic",
                    "point_branch_hint": "basic",
                    "final_score": 17.0,
                    "primary_branch": "basic",
                    "branch_matches": [
                        {"branch": "basic", "rank": 1, "score": 17.0}
                    ],
                }
            ],
            "query_payload": {"prefetch": [], "query_filter": None, "query": "rrf"},
            "raw_query": query,
            "retrieval_skipped_reason": None,
            "final_sort_policy": "rrf_score_desc_name_asc",
            "timers": {},
        }

    def save_feedback(self, *, query, selected_expert_ids, rejected_expert_ids, notes, metadata):
        return 1

    @staticmethod
    def _serialize_query_payload(payload):
        return payload


class FakeValidator:
    def __init__(self, ready: bool = True):
        self._ready = ready

    def validate(self):
        from apps.search.live_validator import LiveValidationReport

        return LiveValidationReport(
            ready=self._ready,
            checks={"collection_exists": self._ready},
            issues=[] if self._ready else ["Collection is not ready."],
            collection_name="researcher_recommend_proto",
            sample_point_id="1" if self._ready else None,
        )


def test_recommend_endpoint_contract():
    service = FakeRecommendationService()
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=service,
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post("/recommend", json={"query": "Recommend AI semiconductor reviewers"})

    assert response.status_code == 200
    parsed = RecommendationResponse.model_validate(response.json())
    assert parsed.searched_branches == ["basic", "art", "pat", "pjt"]
    assert response.json()["trace"]["retrieval_score_traces"][0]["primary_branch"] == "basic"
    assert response.json()["recommendations"][0]["fit"] == "높음"
    assert (
        response.json()["recommendations"][0]["recommendation_reason"]
        == "Publication evidence is available."
    )
    assert response.json()["recommendations"][0]["reasons"] == [
        "Publication evidence is available."
    ]


def test_recommend_endpoint_normalizes_multiline_query_with_commas():
    service = FakeRecommendationService()
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=service,
        validator=FakeValidator(),
    )
    multiline_query = "인공지능 모델 개발\n벡터DB 구축\n파인튜닝\n학습데이터 구축"

    with TestClient(app) as client:
        response = client.post("/recommend", json={"query": multiline_query})

    assert response.status_code == 200
    assert service.last_recommend_query == (
        "인공지능 모델 개발, 벡터DB 구축, 파인튜닝, 학습데이터 구축"
    )
    assert response.json()["intent_summary"] == service.last_recommend_query


def test_playground_route_serves_local_chat_ui():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.get("/playground")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="chatForm"' in response.text
    assert "/recommend" in response.text
    assert "/search/candidates" in response.text
    assert "Primary branch:" in response.text
    assert "Readiness details" in response.text


def test_search_candidates_endpoint_includes_retrieval_score_trace():
    service = FakeRecommendationService()
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=service,
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/search/candidates",
            json={"query": "Recommend AI semiconductor reviewers"},
        )

    assert response.status_code == 200
    assert response.json()["trace"]["retrieval_score_traces"][0]["primary_branch"] == "basic"


def test_search_candidates_endpoint_normalizes_multiline_query_with_commas():
    service = FakeRecommendationService()
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=service,
        validator=FakeValidator(),
    )
    multiline_query = "인공지능 모델 개발\n벡터DB 구축\n파인튜닝\n학습데이터 구축"

    with TestClient(app) as client:
        response = client.post("/search/candidates", json={"query": multiline_query})

    assert response.status_code == 200
    assert service.last_search_query == (
        "인공지능 모델 개발, 벡터DB 구축, 파인튜닝, 학습데이터 구축"
    )
    assert response.json()["intent_summary"] == service.last_search_query


def test_feedback_endpoint_contract():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/feedback",
            json={
                "query": "Recommend AI semiconductor reviewers",
                "selected_expert_ids": ["1"],
                "rejected_expert_ids": [],
                "metadata": {"operator": "tester"},
            },
        )

    assert response.status_code == 200
    assert response.json()["feedback_id"] == 1


def test_readiness_endpoint_returns_report_when_ready():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(ready=True),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    parsed = ReadinessResponse.model_validate(response.json())
    assert parsed.ready is True


def test_readiness_endpoint_returns_503_with_same_shape_when_not_ready():
    app = create_app(
        settings=Settings(app_env="test", strict_runtime_validation=False),
        service=FakeRecommendationService(),
        validator=FakeValidator(ready=False),
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    parsed = ReadinessResponse.model_validate(response.json())
    assert parsed.ready is False
    assert "detail" not in response.json()


def test_startup_failure_is_reported_by_readiness_and_service_endpoints(monkeypatch):
    original_build_app_runtime = main_module.build_app_runtime

    async def fail_build_app_runtime(settings):
        raise RuntimeError("Qdrant bootstrap failed")

    monkeypatch.setattr(main_module, "build_app_runtime", fail_build_app_runtime)
    try:
        app = create_app(settings=Settings(app_env="test", strict_runtime_validation=True))
        with TestClient(app) as client:
            health_response = client.get("/health")
            ready_response = client.get("/health/ready")
            recommend_response = client.post("/recommend", json={"query": "Test query"})
    finally:
        monkeypatch.setattr(main_module, "build_app_runtime", original_build_app_runtime)

    assert health_response.status_code == 200
    assert ready_response.status_code == 503
    ready_payload = ReadinessResponse.model_validate(ready_response.json())
    assert ready_payload.ready is False
    assert ready_payload.checks == {"startup_runtime_initialized": False}
    assert ready_payload.issues == ["Application startup failed: Qdrant bootstrap failed"]

    assert recommend_response.status_code == 503
    assert "Qdrant bootstrap failed" in recommend_response.json()["detail"]


def _patch_runtime_build_dependencies(monkeypatch, *, sparse_encoder_factory):
    dense_encoder = object()
    registry = object()

    class FakeQdrantClient:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.set_sparse_model_calls = []
            FakeQdrantClient.last_instance = self

        def set_sparse_model(self, **kwargs):
            self.set_sparse_model_calls.append(kwargs)

    class FakeFeedbackStore:
        def __init__(self, db_path, table_name):
            self.db_path = db_path
            self.table_name = table_name
            self.initialized = False

        def initialize(self):
            self.initialized = True

    class FakeBootstrapper:
        def __init__(self, client, settings, registry, sparse_runtime=None):
            self.client = client
            self.settings = settings
            self.registry = registry
            self.sparse_runtime = sparse_runtime
            self.recreate = None

        def ensure_collection(self, recreate=False):
            self.recreate = recreate

    monkeypatch.setattr(main_module, "validate_runtime_settings", lambda settings: None)
    monkeypatch.setattr(
        main_module.SearchSchemaRegistry,
        "default",
        staticmethod(lambda: registry),
    )
    monkeypatch.setattr(main_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(main_module, "build_dense_encoder", lambda settings: dense_encoder)
    monkeypatch.setattr(main_module, "SpladeSparseEncoder", sparse_encoder_factory)
    monkeypatch.setattr(main_module, "PlanCache", lambda cache_dir: ("plan", cache_dir))
    monkeypatch.setattr(
        main_module, "RetrievalResultCache", lambda cache_dir: ("retrieval", cache_dir)
    )
    monkeypatch.setattr(main_module, "HeuristicPlanner", lambda: "planner")
    monkeypatch.setattr(main_module, "PassThroughReasonGenerator", lambda: "reasoner")
    monkeypatch.setattr(main_module, "FeedbackStore", FakeFeedbackStore)
    monkeypatch.setattr(main_module, "QdrantBootstrapper", FakeBootstrapper)
    monkeypatch.setattr(
        main_module, "RecommendationService", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        main_module, "LiveContractValidator", lambda **kwargs: SimpleNamespace(**kwargs)
    )

    return dense_encoder, registry, FakeQdrantClient


@pytest.mark.anyio
async def test_build_app_runtime_selects_local_pixie_custom_encoder(monkeypatch):
    model_dir = Path(__file__).resolve().parents[1] / "models" / "PIXIE-Splade-v1.0"
    test_workspace = Path(__file__).resolve().parents[1] / "runtime" / "test_build_app_runtime_local"
    sparse_calls = []

    def sparse_encoder_factory(model_name, local_files_only=False):
        sparse_calls.append((model_name, local_files_only))
        resolved_name = str(Path(model_name).resolve()) if Path(model_name).exists() else model_name
        return SimpleNamespace(model_name=resolved_name)

    dense_encoder, _, fake_qdrant_client = _patch_runtime_build_dependencies(
        monkeypatch,
        sparse_encoder_factory=sparse_encoder_factory,
    )

    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_backend="heuristic",
        sparse_model_name=str(model_dir),
        sparse_cache_dir=str(test_workspace / "models-cache"),
        runtime_dir=test_workspace / "runtime",
    )

    service, validator = await main_module.build_app_runtime(settings)

    assert service.retriever.dense_encoder is dense_encoder
    assert service.retriever.sparse_encoder.model_name == str(model_dir.resolve())
    assert service.retriever.sparse_runtime.backend == "custom_splade"
    assert service.retriever.sparse_runtime.active_model_name == str(model_dir.resolve())
    assert service.retriever.sparse_runtime.used_fallback is False
    assert validator.sparse_runtime.active_model_name == str(model_dir.resolve())
    assert sparse_calls == [(str(model_dir), False)]
    assert fake_qdrant_client.last_instance is not None
    assert fake_qdrant_client.last_instance.set_sparse_model_calls == []


@pytest.mark.anyio
async def test_build_app_runtime_falls_back_to_online_pixie_before_bm25(monkeypatch):
    model_dir = Path(__file__).resolve().parents[1] / "runtime" / "missing-local-pixie"
    test_workspace = Path(__file__).resolve().parents[1] / "runtime" / "test_build_app_runtime_online"
    sparse_calls = []

    def sparse_encoder_factory(model_name, local_files_only=False):
        sparse_calls.append((model_name, local_files_only))
        if model_name == str(model_dir):
            raise FileNotFoundError("configured local pixie is missing")
        return SimpleNamespace(model_name=model_name)

    _, _, fake_qdrant_client = _patch_runtime_build_dependencies(
        monkeypatch,
        sparse_encoder_factory=sparse_encoder_factory,
    )

    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_backend="heuristic",
        sparse_model_name=str(model_dir),
        sparse_cache_dir=str(test_workspace / "models-cache"),
        runtime_dir=test_workspace / "runtime",
    )

    service, validator = await main_module.build_app_runtime(settings)

    assert service.retriever.sparse_encoder.model_name == ONLINE_PIXIE_SPLADE_MODEL
    assert service.retriever.sparse_runtime.backend == "custom_splade"
    assert service.retriever.sparse_runtime.active_model_name == ONLINE_PIXIE_SPLADE_MODEL
    assert service.retriever.sparse_runtime.used_fallback is True
    assert validator.sparse_runtime.active_model_name == ONLINE_PIXIE_SPLADE_MODEL
    assert sparse_calls == [
        (str(model_dir), False),
        (ONLINE_PIXIE_SPLADE_MODEL, False),
    ]
    assert fake_qdrant_client.last_instance.set_sparse_model_calls == []


@pytest.mark.anyio
async def test_build_app_runtime_falls_back_to_bm25_when_pixie_attempts_fail(monkeypatch):
    model_dir = Path(__file__).resolve().parents[1] / "runtime" / "missing-local-pixie"
    test_workspace = Path(__file__).resolve().parents[1] / "runtime" / "test_build_app_runtime_bm25"
    sparse_calls = []

    def sparse_encoder_factory(model_name, local_files_only=False):
        sparse_calls.append((model_name, local_files_only))
        raise RuntimeError(f"cannot load sparse model: {model_name}")

    _, _, fake_qdrant_client = _patch_runtime_build_dependencies(
        monkeypatch,
        sparse_encoder_factory=sparse_encoder_factory,
    )

    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_backend="heuristic",
        sparse_model_name=str(model_dir),
        sparse_cache_dir=str(test_workspace / "models-cache"),
        runtime_dir=test_workspace / "runtime",
    )

    service, validator = await main_module.build_app_runtime(settings)

    assert service.retriever.sparse_encoder is None
    assert service.retriever.sparse_runtime.backend == "fastembed_builtin"
    assert service.retriever.sparse_runtime.active_model_name == QDRANT_BM25_MODEL
    assert service.retriever.sparse_runtime.used_fallback is True
    assert validator.sparse_runtime.active_model_name == QDRANT_BM25_MODEL
    assert sparse_calls == [
        (str(model_dir), False),
        (ONLINE_PIXIE_SPLADE_MODEL, False),
    ]
    assert fake_qdrant_client.last_instance.set_sparse_model_calls == [
        {
            "embedding_model_name": QDRANT_BM25_MODEL,
            "cache_dir": str(test_workspace / "models-cache"),
        }
    ]


@pytest.mark.anyio
async def test_build_app_runtime_skips_online_pixie_when_offline_and_uses_bm25(monkeypatch):
    model_dir = Path(__file__).resolve().parents[1] / "runtime" / "missing-local-pixie"
    test_workspace = Path(__file__).resolve().parents[1] / "runtime" / "test_build_app_runtime_offline_bm25"
    sparse_calls = []

    def sparse_encoder_factory(model_name, local_files_only=False):
        sparse_calls.append((model_name, local_files_only))
        raise RuntimeError(f"cannot load sparse model: {model_name}")

    _, _, fake_qdrant_client = _patch_runtime_build_dependencies(
        monkeypatch,
        sparse_encoder_factory=sparse_encoder_factory,
    )

    settings = Settings(
        app_env="test",
        strict_runtime_validation=False,
        llm_backend="heuristic",
        sparse_model_name=str(model_dir),
        sparse_cache_dir=str(test_workspace / "models-cache"),
        runtime_dir=test_workspace / "runtime",
        hf_hub_offline=True,
    )

    service, _ = await main_module.build_app_runtime(settings)

    assert service.retriever.sparse_encoder is None
    assert service.retriever.sparse_runtime.active_model_name == QDRANT_BM25_MODEL
    assert sparse_calls == [(str(model_dir), False)]
    assert fake_qdrant_client.last_instance.set_sparse_model_calls == [
        {
            "embedding_model_name": QDRANT_BM25_MODEL,
            "cache_dir": str(test_workspace / "models-cache"),
            "local_files_only": True,
        }
    ]
