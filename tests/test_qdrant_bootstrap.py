from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import models

from apps.core.config import Settings
from apps.search.qdrant_bootstrap import QdrantBootstrapper
from apps.search.schema_registry import (
    BRANCHES,
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    PAYLOAD_INDEX_FIELDS_V2,
    SPARSE_VECTOR_NAME,
    SearchSchemaRegistry,
)
from apps.search.sparse_runtime import SparseRuntimeConfig


class FakeBootstrapClient:
    def __init__(self, *, sparse_modifier: object = None) -> None:
        self.registry = SearchSchemaRegistry.default()
        self.collection_info = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    sparse_vectors={
                        self.registry.sparse_vector_by_branch[branch]: SimpleNamespace(modifier=sparse_modifier)
                        for branch in BRANCHES
                    }
                )
            )
        )
        self.created_collection = None
        self.updated_collection = None
        self.payload_indexes: list[tuple[str, object, bool]] = []

    def delete_collection(self, collection_name: str) -> None:
        return None

    def get_collection(self, collection_name: str):
        return self.collection_info

    def create_collection(self, **kwargs) -> None:
        self.created_collection = kwargs

    def update_collection(self, **kwargs) -> None:
        self.updated_collection = kwargs

    def create_payload_index(self, *, collection_name: str, field_name: str, field_schema: object, wait: bool) -> None:
        self.payload_indexes.append((field_name, field_schema, wait))


def test_bootstrapper_repairs_sparse_vector_modifiers_for_bm25_runtime():
    client = FakeBootstrapClient(sparse_modifier=None)
    bootstrapper = QdrantBootstrapper(
        client=client,
        settings=Settings(),
        registry=SearchSchemaRegistry.default(),
        sparse_runtime=SparseRuntimeConfig(
            backend="fastembed_builtin",
            active_model_name="Qdrant/bm25",
            requires_idf_modifier=True,
            used_fallback=True,
        ),
    )

    bootstrapper.ensure_collection()

    assert client.created_collection is None
    assert client.updated_collection is not None
    updated_sparse_vectors = client.updated_collection["sparse_vectors_config"]
    assert set(updated_sparse_vectors.keys()) == {
        SearchSchemaRegistry.default().sparse_vector_by_branch[branch] for branch in BRANCHES
    }
    assert all(params.modifier == models.Modifier.IDF for params in updated_sparse_vectors.values())
    assert len(client.payload_indexes) == len(PAYLOAD_INDEX_FIELDS)


def test_bootstrapper_clears_idf_modifier_for_splade_runtime():
    client = FakeBootstrapClient(sparse_modifier=models.Modifier.IDF)
    bootstrapper = QdrantBootstrapper(
        client=client,
        settings=Settings(),
        registry=SearchSchemaRegistry.default(),
        sparse_runtime=SparseRuntimeConfig(
            backend="custom_splade",
            active_model_name="telepix/PIXIE-Splade-v1.0",
            requires_idf_modifier=False,
        ),
    )

    bootstrapper.ensure_collection()

    assert client.created_collection is None
    assert client.updated_collection is not None
    updated_sparse_vectors = client.updated_collection["sparse_vectors_config"]
    assert all(params.modifier is None for params in updated_sparse_vectors.values())


# ---------------------------------------------------------------------------
# WO-B: v2.0 단일 벡터 부트스트랩 + blue/green capability flag
# ---------------------------------------------------------------------------
class RecordingClient:
    """create/update/delete/index 호출을 기록하고 컬렉션 존재를 추적하는 fake."""

    def __init__(self, *, exists: bool = False, sparse_vectors: dict | None = None) -> None:
        self.exists = exists
        self._sparse_vectors = sparse_vectors or {}
        self.created_collection = None
        self.updated_collection = None
        self.deleted: list[str] = []
        self.payload_indexes: list[tuple[str, object, bool]] = []

    def get_collection(self, collection_name: str):
        if not self.exists:
            raise RuntimeError("collection not found")
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(sparse_vectors=self._sparse_vectors))
        )

    def create_collection(self, **kwargs) -> None:
        self.created_collection = kwargs
        self.exists = True
        svc = kwargs.get("sparse_vectors_config") or {}
        self._sparse_vectors = {
            name: SimpleNamespace(modifier=getattr(params, "modifier", None))
            for name, params in svc.items()
        }

    def update_collection(self, **kwargs) -> None:
        self.updated_collection = kwargs

    def create_payload_index(self, *, collection_name, field_name, field_schema, wait) -> None:
        self.payload_indexes.append((field_name, field_schema, wait))

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.exists = False


def _v2_bootstrapper(client, *, collection="ntis_researcher_chunks", idf=False):
    backend = "fastembed_builtin" if idf else "custom_splade"
    model = "Qdrant/bm25" if idf else "telepix/PIXIE-Splade-v1.0"
    return QdrantBootstrapper(
        client=client,
        settings=Settings(qdrant_collection_name=collection),
        registry=SearchSchemaRegistry.default(),
        sparse_runtime=SparseRuntimeConfig(
            backend=backend,
            active_model_name=model,
            requires_idf_modifier=idf,
            used_fallback=idf,
        ),
    )


def test_v2_creates_single_dense_and_sparse_vectors_with_v2_indexes():
    client = RecordingClient(exists=False)
    _v2_bootstrapper(client).ensure_collection()

    assert client.created_collection is not None
    assert set(client.created_collection["vectors_config"].keys()) == {DENSE_VECTOR_NAME}
    sparse = client.created_collection["sparse_vectors_config"]
    assert set(sparse.keys()) == {SPARSE_VECTOR_NAME}
    assert sparse[SPARSE_VECTOR_NAME].modifier is None  # SPLADE → modifier 없음

    # v2.0 payload 인덱스 세트가 생성됨(nested 잔재 없음)
    assert len(client.payload_indexes) == len(PAYLOAD_INDEX_FIELDS_V2)
    fields = {field for field, _, _ in client.payload_indexes}
    assert {"researcher_id", "doc_type", "event_year"} <= fields
    assert not any("[]" in field for field in fields)


def test_v2_bm25_fallback_uses_idf_modifier():
    client = RecordingClient(exists=False)
    _v2_bootstrapper(client, idf=True).ensure_collection()
    sparse = client.created_collection["sparse_vectors_config"]
    assert sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


def test_legacy_collection_is_not_recreated_blue_green_guard():
    client = RecordingClient(exists=True, sparse_vectors={})
    _v2_bootstrapper(client, collection="researcher_recommend_proto").ensure_collection(recreate=True)
    assert client.deleted == []  # 레거시 v1.x 컬렉션은 삭제 금지
    assert client.created_collection is None  # 이미 존재 → 재생성 안 함
    # 레거시 경로는 v1.x 인덱스 세트 사용
    assert len(client.payload_indexes) == len(PAYLOAD_INDEX_FIELDS)


def test_v2_collection_recreate_is_allowed():
    client = RecordingClient(exists=True, sparse_vectors={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=None)})
    _v2_bootstrapper(client).ensure_collection(recreate=True)
    assert client.deleted == ["ntis_researcher_chunks"]
    assert client.created_collection is not None
