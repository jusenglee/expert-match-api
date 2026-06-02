from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.core.config import Settings
from apps.core.runtime_validation import (
    BackendCheckResult,
    RuntimeDependencyValidator,
    validate_runtime_settings,
)
from apps.search.live_validator import LiveContractValidator
from apps.search.schema_registry import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
)
from apps.search.sparse_runtime import SparseRuntimeConfig


# ===========================================================================
# flat chunk payload 표본 (live_validator가 기대하는 v2.1 평면 구조)
# ===========================================================================
def build_valid_sample_payload() -> dict[str, object]:
    """live_validator의 _REQUIRED_ROOT_FIELDS + 유효 doc_type을 만족하는 flat 표본."""
    return {
        "researcher_id": "11008395",
        "researcher_name": "홍길동",
        "doc_type": "paper",
        "doc_id": "paper_100000045256",
        "chunk_id": "paper_100000045256_c000",
        "chunk_text": "Test paper chunk text",
        "doc_date": "2024-09-01",
        "affiliated_organization": "Test Organization",
        "highest_degree": "박사",
        "publication_count": 3,
        "scie_publication_count": 1,
        "intellectual_property_count": 0,
        "research_project_count": 1,
        "researcher_assessor_activity_count": 0,
        "doc_attrs": {"indexing_database": "SCIE"},
    }


# ===========================================================================
# validate_runtime_settings (payload-agnostic strict 검증)
# ===========================================================================
def test_strict_runtime_settings_reject_fallback_backends():
    settings = Settings(
        llm_backend="heuristic",
        embedding_backend="hashing",
        strict_runtime_validation=True,
    )

    with pytest.raises(RuntimeError):
        validate_runtime_settings(settings)


def test_strict_runtime_settings_reject_seed_on_startup():
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="local",
        seed_on_startup=True,
        strict_runtime_validation=True,
    )

    with pytest.raises(RuntimeError):
        validate_runtime_settings(settings)


def test_strict_runtime_settings_accept_valid_production_config():
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="local",
        seed_on_startup=False,
        strict_runtime_validation=True,
    )

    # 유효한 운영 설정은 예외 없이 통과해야 한다.
    assert validate_runtime_settings(settings) is None


def test_non_strict_runtime_settings_allow_fallback_backends():
    settings = Settings(
        llm_backend="heuristic",
        embedding_backend="hashing",
        seed_on_startup=True,
        strict_runtime_validation=False,
    )

    # strict가 꺼져 있으면 어떤 백엔드 조합도 허용된다.
    assert validate_runtime_settings(settings) is None


# ===========================================================================
# RuntimeDependencyValidator (백엔드 연결성 — OpenAI 클라이언트를 stub로 대체)
# ===========================================================================
class _FakeModel:
    def __init__(self, model_id: str) -> None:
        self.id = model_id


class _FakeModelsList:
    def __init__(self, model_ids: list[str]) -> None:
        self.data = [_FakeModel(mid) for mid in model_ids]


class _FakeModelsEndpoint:
    def __init__(self, model_ids: list[str] | None, raise_exc: Exception | None) -> None:
        self._model_ids = model_ids
        self._raise_exc = raise_exc

    def list(self):
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeModelsList(self._model_ids or [])


class _FakeOpenAIClient:
    """OpenAI 호환 클라이언트 stub — 네트워크 호출 없이 models.list만 흉내낸다."""

    def __init__(self, *, base_url: str, api_key: str, model_ids, raise_exc):
        self.base_url = base_url
        self.api_key = api_key
        self.models = _FakeModelsEndpoint(model_ids, raise_exc)


def _patch_openai(monkeypatch, *, model_ids=None, raise_exc=None):
    """runtime_validation 모듈이 사용하는 OpenAI 심볼을 stub로 교체."""

    def _factory(*, base_url, api_key):
        return _FakeOpenAIClient(
            base_url=base_url,
            api_key=api_key,
            model_ids=model_ids,
            raise_exc=raise_exc,
        )

    monkeypatch.setattr("apps.core.runtime_validation.OpenAI", _factory)


def test_dependency_validator_reports_ok_when_model_present(monkeypatch):
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="openai",
        llm_model_name="my-llm",
        embedding_model_name="my-embed",
    )
    _patch_openai(monkeypatch, model_ids=["my-llm", "my-embed"])

    results = RuntimeDependencyValidator(settings).validate_backends()

    by_name = {r.name: r for r in results}
    assert by_name["llm_backend"].ok is True
    assert by_name["embedding_backend"].ok is True
    assert all(isinstance(r, BackendCheckResult) for r in results)


def test_dependency_validator_local_embedding_skips_network(monkeypatch):
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="local",
        llm_model_name="my-llm",
    )
    _patch_openai(monkeypatch, model_ids=["my-llm"])

    results = RuntimeDependencyValidator(settings).validate_backends()

    by_name = {r.name: r for r in results}
    assert by_name["llm_backend"].ok is True
    # local 임베딩은 네트워크 점검 없이 항상 ok.
    assert by_name["embedding_backend"].ok is True
    assert "로컬" in by_name["embedding_backend"].detail


def test_dependency_validator_reports_failure_when_backend_unreachable(monkeypatch):
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="local",
        llm_model_name="my-llm",
    )
    _patch_openai(monkeypatch, raise_exc=ConnectionError("boom"))

    results = RuntimeDependencyValidator(settings).validate_backends()

    llm_result = next(r for r in results if r.name == "llm_backend")
    assert llm_result.ok is False
    assert "boom" in llm_result.detail


def test_dependency_validator_reports_missing_model(monkeypatch):
    settings = Settings(
        llm_backend="openai_compat",
        embedding_backend="local",
        llm_model_name="expected-model",
    )
    _patch_openai(monkeypatch, model_ids=["some-other-model"])

    results = RuntimeDependencyValidator(settings).validate_backends()

    llm_result = next(r for r in results if r.name == "llm_backend")
    assert llm_result.ok is False
    assert "expected-model" in llm_result.detail


# ===========================================================================
# LiveContractValidator (flat 규약 — 단일 벡터 / flat payload / registry 제거)
# ===========================================================================
class FakeDependencyValidator:
    def validate_backends(self):
        return [
            BackendCheckResult(name="llm_backend", ok=True, detail="ok"),
            BackendCheckResult(name="embedding_backend", ok=True, detail="ok"),
        ]


class FakeQdrantClient:
    """flat 단일 벡터 컬렉션을 흉내내는 Qdrant 클라이언트 stub."""

    def __init__(
        self,
        *,
        missing_vector: bool = False,
        missing_sparse: bool = False,
        missing_index: bool = False,
        no_points: bool = False,
        malformed_payload: bool = False,
        sparse_modifier: object = None,
        sample_payloads: list[object] | None = None,
    ):
        dense_vectors = {DENSE_VECTOR_NAME: SimpleNamespace(size=1024)}
        if missing_vector:
            dense_vectors.pop(DENSE_VECTOR_NAME)

        sparse_vectors = {
            SPARSE_VECTOR_NAME: SimpleNamespace(modifier=sparse_modifier)
        }
        if missing_sparse:
            sparse_vectors.pop(SPARSE_VECTOR_NAME)

        payload_schema = {
            field_name: SimpleNamespace(data_type=schema_name)
            for field_name, schema_name in PAYLOAD_INDEX_FIELDS
        }
        if missing_index:
            payload_schema.pop("doc_date")

        self.collection_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=dense_vectors,
                    sparse_vectors=sparse_vectors,
                )
            ),
            payload_schema=payload_schema,
        )
        self.no_points = no_points
        self.malformed_payload = malformed_payload
        payloads = sample_payloads or [build_valid_sample_payload()]
        self.sample_records = [
            SimpleNamespace(id=str(11008395 + index), payload=payload)
            for index, payload in enumerate(payloads)
        ]

    def get_collection(self, collection_name):
        return self.collection_info

    def scroll(self, **kwargs):
        if self.no_points:
            return [], None
        if self.malformed_payload:
            return [SimpleNamespace(id="11008395", payload=["not", "an", "object"])], None

        offset = kwargs.get("offset")
        limit = kwargs.get("limit", 1)
        start = int(offset) if offset is not None else 0
        end = start + limit
        records = self.sample_records[start:end]
        next_offset = end if end < len(self.sample_records) else None
        return records, next_offset


def test_live_validator_reports_ready_when_contract_is_satisfied():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["collection_exists"] is True
    assert report.checks["dense_vectors_present"] is True
    assert report.checks["sparse_vectors_present"] is True
    assert report.checks["payload_indexes_present"] is True
    assert report.checks["sample_payload_valid"] is True
    assert report.checks["sample_root_fields"] is True
    assert report.checks["sample_doc_type_valid"] is True
    assert report.sample_point_id == "11008395"
    assert report.collection_name == settings.qdrant_collection_name


def test_live_validator_scans_for_most_complete_sample_point():
    settings = Settings()
    # 첫 표본은 doc_type이 유효 5종이 아니라 완전성 점수가 낮다 → 두 번째 표본이 선택돼야 한다.
    incomplete_payload = build_valid_sample_payload()
    incomplete_payload["doc_type"] = "unknown_doc_type"
    validator = LiveContractValidator(
        client=FakeQdrantClient(
            sample_payloads=[incomplete_payload, build_valid_sample_payload()]
        ),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.sample_point_id == "11008396"
    assert report.checks["sample_doc_type_valid"] is True


def test_live_validator_treats_doc_attrs_and_doc_date_as_optional():
    settings = Settings()
    # doc_attrs / doc_date가 비어 있어도 필수 루트 필드와 doc_type만 유효하면 ready.
    payload = build_valid_sample_payload()
    payload["doc_attrs"] = None
    payload["doc_date"] = "NONE"
    validator = LiveContractValidator(
        client=FakeQdrantClient(sample_payloads=[payload]),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["sample_doc_attrs_present"] is False
    assert report.checks["sample_doc_date_present"] is False


def test_live_validator_accepts_modifier_like_object_with_idf_name():
    class ModifierLike:
        name = "IDF"

    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(sparse_modifier=ModifierLike()),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
        sparse_runtime=SparseRuntimeConfig(
            backend="fastembed_builtin",
            active_model_name="Qdrant/bm25",
            requires_idf_modifier=True,
        ),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["sparse_vectors_idf"] is True


def test_live_validator_accepts_none_modifier_for_splade_runtime():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(sparse_modifier=None),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
        sparse_runtime=SparseRuntimeConfig(
            backend="custom_splade",
            active_model_name="telepix/PIXIE-Splade-v1.0",
            requires_idf_modifier=False,
        ),
    )

    report = validator.validate()

    assert report.ready is True
    assert report.checks["sparse_vectors_idf"] is True


def test_live_validator_flags_idf_mismatch_for_splade_runtime():
    class ModifierLike:
        name = "IDF"

    settings = Settings()
    # SPLADE 런타임은 modifier가 None이어야 하는데 IDF가 설정돼 있으면 불일치.
    validator = LiveContractValidator(
        client=FakeQdrantClient(sparse_modifier=ModifierLike()),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
        sparse_runtime=SparseRuntimeConfig(
            backend="custom_splade",
            active_model_name="telepix/PIXIE-Splade-v1.0",
            requires_idf_modifier=False,
        ),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["sparse_vectors_idf"] is False


def test_settings_default_local_embedding_model_path_points_to_repo_bundle():
    settings = Settings()

    assert settings.embedding_model_name.endswith("multilingual-e5-large-instruct")
    assert settings.embedding_backend == "local"


def test_settings_default_local_sparse_model_path_points_to_repo_bundle():
    settings = Settings()
    model_path = Path(settings.sparse_model_name)
    expected_path = Path(__file__).resolve().parents[1] / "models" / "PIXIE-Splade-v1.0"

    assert model_path == expected_path
    assert model_path.parent.name == "models"


def test_live_validator_reports_missing_vector_and_index():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(missing_vector=True, missing_index=True),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["dense_vectors_present"] is False
    assert report.checks["payload_indexes_present"] is False


def test_live_validator_reports_missing_sparse_vector():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(missing_sparse=True),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["sparse_vectors_present"] is False


def test_live_validator_reports_no_sample_points():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(no_points=True),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["sample_point_exists"] is False


def test_live_validator_reports_malformed_sample_payload_without_raising():
    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(malformed_payload=True),
        settings=settings,
        dependency_validator=FakeDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["sample_payload_valid"] is False
    assert any("JSON" in issue for issue in report.issues)


def test_live_validator_reports_backend_disconnection_in_issues():
    class FailingDependencyValidator:
        def validate_backends(self):
            return [
                BackendCheckResult(name="llm_backend", ok=False, detail="llm down"),
                BackendCheckResult(name="embedding_backend", ok=True, detail="ok"),
            ]

    settings = Settings()
    validator = LiveContractValidator(
        client=FakeQdrantClient(),
        settings=settings,
        dependency_validator=FailingDependencyValidator(),
    )

    report = validator.validate()

    assert report.ready is False
    assert report.checks["llm_backend_connected"] is False
    assert report.checks["embedding_backend_connected"] is True
    assert "llm down" in report.issues
