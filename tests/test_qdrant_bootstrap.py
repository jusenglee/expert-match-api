from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import models

from apps.core.config import Settings
from apps.search.qdrant_bootstrap import (
    LEGACY_V1X_COLLECTIONS,
    QdrantBootstrapper,
)
from apps.search.schema_registry import (
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
)
from apps.search.sparse_runtime import SparseRuntimeConfig


class RecordingClient:
    """create/update/delete/index 호출을 기록하고 컬렉션 존재를 추적하는 fake QdrantClient."""

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


def _bootstrapper(client, *, collection: str = "ntis_researcher_chunks", idf: bool = False):
    backend = "fastembed_builtin" if idf else "custom_splade"
    model = "Qdrant/bm25" if idf else "telepix/PIXIE-Splade-v1.0"
    return QdrantBootstrapper(
        client=client,
        settings=Settings(qdrant_collection_name=collection),
        sparse_runtime=SparseRuntimeConfig(
            backend=backend,
            active_model_name=model,
            requires_idf_modifier=idf,
            used_fallback=idf,
        ),
    )


# ---------------------------------------------------------------------------
# 컬렉션 생성: 단일 dense(vector_e5i) + 단일 sparse(vector_splade), flat 인덱스
# ---------------------------------------------------------------------------
def test_creates_single_dense_and_sparse_vectors_with_flat_indexes():
    client = RecordingClient(exists=False)
    _bootstrapper(client).ensure_collection()

    assert client.created_collection is not None
    # dense named vector는 vector_e5i 단 하나
    assert set(client.created_collection["vectors_config"].keys()) == {DENSE_VECTOR_NAME}
    dense_params = client.created_collection["vectors_config"][DENSE_VECTOR_NAME]
    assert dense_params.size == 1024
    assert dense_params.distance == models.Distance.COSINE

    # sparse named vector는 vector_splade 단 하나
    sparse = client.created_collection["sparse_vectors_config"]
    assert set(sparse.keys()) == {SPARSE_VECTOR_NAME}
    # SPLADE 런타임 → IDF modifier 없음
    assert sparse[SPARSE_VECTOR_NAME].modifier is None

    # flat payload 인덱스 세트가 PAYLOAD_INDEX_FIELDS와 1:1 정합 (nested 잔재 없음)
    assert len(client.payload_indexes) == len(PAYLOAD_INDEX_FIELDS)
    fields = {field for field, _, _ in client.payload_indexes}
    assert {"researcher_id", "doc_type", "doc_date"} <= fields
    assert "vector" not in fields  # 벡터 named 키가 인덱스로 섞이지 않음
    assert not any("[]" in field for field in fields)


def test_flat_indexes_use_expected_schema_types():
    client = RecordingClient(exists=False)
    _bootstrapper(client).ensure_collection()

    recorded = {field: schema for field, schema, _ in client.payload_indexes}
    assert recorded["researcher_id"] == models.PayloadSchemaType.KEYWORD
    assert recorded["doc_type"] == models.PayloadSchemaType.KEYWORD
    assert recorded["publication_count"] == models.PayloadSchemaType.INTEGER
    assert recorded["researcher_assessor_activity_count"] == models.PayloadSchemaType.INTEGER
    assert recorded["doc_date"] == models.PayloadSchemaType.DATETIME
    assert recorded["doc_attrs.indexing_database"] == models.PayloadSchemaType.KEYWORD
    # 모든 인덱스 생성은 wait=True
    assert all(wait is True for _, _, wait in client.payload_indexes)


def test_bm25_fallback_uses_idf_modifier_on_create():
    client = RecordingClient(exists=False)
    _bootstrapper(client, idf=True).ensure_collection()

    sparse = client.created_collection["sparse_vectors_config"]
    assert set(sparse.keys()) == {SPARSE_VECTOR_NAME}
    assert sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


# ---------------------------------------------------------------------------
# sparse modifier 보수(repair): 기존 컬렉션의 modifier를 런타임에 맞춰 갱신
# ---------------------------------------------------------------------------
def test_existing_collection_repairs_sparse_modifier_for_bm25_runtime():
    # 이미 존재하지만 sparse modifier가 None → BM25 런타임이면 IDF로 교정해야 함
    client = RecordingClient(
        exists=True,
        sparse_vectors={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=None)},
    )
    _bootstrapper(client, idf=True).ensure_collection()

    assert client.created_collection is None  # 이미 존재 → 재생성 없음
    assert client.updated_collection is not None
    updated_sparse = client.updated_collection["sparse_vectors_config"]
    assert set(updated_sparse.keys()) == {SPARSE_VECTOR_NAME}
    assert updated_sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF
    # 보수 후에도 flat payload 인덱스는 보장된다
    assert len(client.payload_indexes) == len(PAYLOAD_INDEX_FIELDS)


def test_existing_collection_clears_idf_modifier_for_splade_runtime():
    # 기존 컬렉션이 IDF인데 SPLADE 런타임 → modifier None으로 교정
    client = RecordingClient(
        exists=True,
        sparse_vectors={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=models.Modifier.IDF)},
    )
    _bootstrapper(client, idf=False).ensure_collection()

    assert client.created_collection is None
    assert client.updated_collection is not None
    updated_sparse = client.updated_collection["sparse_vectors_config"]
    assert updated_sparse[SPARSE_VECTOR_NAME].modifier is None


def test_no_modifier_update_when_already_consistent():
    # 이미 IDF인데 BM25 런타임 → 추가 update_collection 호출 없음
    client = RecordingClient(
        exists=True,
        sparse_vectors={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=models.Modifier.IDF)},
    )
    _bootstrapper(client, idf=True).ensure_collection()

    assert client.created_collection is None
    assert client.updated_collection is None  # 일관 → 갱신 불필요


# ---------------------------------------------------------------------------
# blue/green 가드: 레거시 v1.x 컬렉션은 절대 재생성/삭제/인덱스 생성하지 않는다
# ---------------------------------------------------------------------------
def test_legacy_collection_is_a_total_no_op_blue_green_guard():
    assert "researcher_recommend_proto" in LEGACY_V1X_COLLECTIONS
    client = RecordingClient(exists=True, sparse_vectors={})
    _bootstrapper(client, collection="researcher_recommend_proto").ensure_collection(recreate=True)

    assert client.deleted == []  # 레거시 v1.x 삭제 금지
    assert client.created_collection is None  # 재생성 금지
    assert client.updated_collection is None  # modifier 보수 금지
    assert client.payload_indexes == []  # 인덱스 생성 금지


def test_v2_collection_recreate_is_allowed():
    client = RecordingClient(
        exists=True,
        sparse_vectors={SPARSE_VECTOR_NAME: SimpleNamespace(modifier=None)},
    )
    _bootstrapper(client).ensure_collection(recreate=True)

    assert client.deleted == ["ntis_researcher_chunks"]
    assert client.created_collection is not None
    assert set(client.created_collection["vectors_config"].keys()) == {DENSE_VECTOR_NAME}


# ---------------------------------------------------------------------------
# settings 기반 idf 판정(런타임 미주입 시): sparse_model_name으로 결정
# ---------------------------------------------------------------------------
def test_idf_inferred_from_settings_when_no_runtime_provided():
    # sparse_runtime 없이 BM25 모델명 → IDF modifier
    client = RecordingClient(exists=False)
    QdrantBootstrapper(
        client=client,
        settings=Settings(
            qdrant_collection_name="ntis_researcher_chunks",
            sparse_model_name="Qdrant/bm25",
        ),
    ).ensure_collection()

    sparse = client.created_collection["sparse_vectors_config"]
    assert sparse[SPARSE_VECTOR_NAME].modifier == models.Modifier.IDF


def test_no_idf_inferred_from_settings_for_splade_model():
    client = RecordingClient(exists=False)
    QdrantBootstrapper(
        client=client,
        settings=Settings(
            qdrant_collection_name="ntis_researcher_chunks",
            sparse_model_name="telepix/PIXIE-Splade-v1.0",
        ),
    ).ensure_collection()

    sparse = client.created_collection["sparse_vectors_config"]
    assert sparse[SPARSE_VECTOR_NAME].modifier is None
