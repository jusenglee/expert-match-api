"""Qdrant 컬렉션 초기화 부트스트래퍼 (flat 단일 벡터 모델, v2.1).

단일 dense(vector_e5i) + 단일 sparse(vector_splade) 컬렉션 + flat payload 인덱스를 생성한다.
doc_type별 named vector는 두지 않는다(doc_type은 payload 필터). 레거시 v1.x 컬렉션
(researcher_recommend_proto)은 절대 재생성/변형하지 않는다(blue/green 가드).
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.search.schema_registry import (
    DENSE_VECTOR_NAME,
    DENSE_VECTOR_SIZE,
    PAYLOAD_INDEX_FIELDS,
    SPARSE_VECTOR_NAME,
)
from apps.search.sparse_runtime import SparseRuntimeConfig, model_requires_idf_modifier

logger = logging.getLogger(__name__)


FIELD_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "datetime": models.PayloadSchemaType.DATETIME,
}

# blue/green 가드: 레거시 v1.x 컬렉션은 부트스트래퍼가 절대 재생성/스키마 변형하지 않는다.
LEGACY_V1X_COLLECTIONS: frozenset[str] = frozenset({"researcher_recommend_proto"})


class QdrantBootstrapper:
    """Qdrant 저장소 초기 설정(컬렉션/벡터/인덱스) 담당."""

    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        sparse_runtime: SparseRuntimeConfig | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.sparse_runtime = sparse_runtime

    def _requires_idf_modifier(self) -> bool:
        if self.sparse_runtime is not None:
            return self.sparse_runtime.requires_idf_modifier
        return model_requires_idf_modifier(self.settings.sparse_model_name)

    def _is_v2_collection(self, collection_name: str) -> bool:
        """flat 단일 벡터 스키마 적용 대상인지. 레거시 v1.x 컬렉션은 보호한다."""
        return collection_name not in LEGACY_V1X_COLLECTIONS

    def ensure_collection(self, recreate: bool = False) -> None:
        """flat 단일 벡터 컬렉션(vector_e5i + vector_splade) + payload 인덱스를 보장한다.

        레거시 v1.x 컬렉션을 가리키면 아무 것도 하지 않는다(blue/green 가드).
        """
        collection_name = self.settings.qdrant_collection_name

        if not self._is_v2_collection(collection_name):
            logger.warning(
                "Refusing to bootstrap legacy v1.x collection %s (blue/green guard); no-op",
                collection_name,
            )
            return

        if recreate:
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception:
                logger.info("Collection %s did not exist before recreate", collection_name)

        if not self._collection_exists(collection_name):
            requires_idf_modifier = self._requires_idf_modifier()
            sparse_modifier = models.Modifier.IDF if requires_idf_modifier else None
            logger.info(
                "Creating collection %s (single vector) sparse_modifier=%s (requires_idf=%s)",
                collection_name, sparse_modifier, requires_idf_modifier,
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: models.VectorParams(
                        size=self.settings.embedding_vector_size or DENSE_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=sparse_modifier)
                },
            )

        self.ensure_sparse_vector_modifiers()
        self.ensure_payload_indexes()

    @staticmethod
    def _modifier_is_idf(modifier: object) -> bool:
        if modifier is None:
            return False
        candidates = [modifier, getattr(modifier, "value", None), getattr(modifier, "name", None)]
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            if normalized == "idf" or "modifier.idf" in normalized:
                return True
        return False

    def ensure_sparse_vector_modifiers(self) -> None:
        if not self._is_v2_collection(self.settings.qdrant_collection_name):
            return
        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection_name)
        except Exception as exc:
            logger.warning("Skipping sparse modifier repair (collection lookup failed): %s", exc)
            return

        sparse_config = getattr(collection_info.config.params, "sparse_vectors", None) or {}
        if not isinstance(sparse_config, dict):
            return

        target_modifier = models.Modifier.IDF if self._requires_idf_modifier() else None
        params = sparse_config.get(SPARSE_VECTOR_NAME)
        current_modifier = getattr(params, "modifier", None)
        needs_update = (
            not self._modifier_is_idf(current_modifier)
            if target_modifier == models.Modifier.IDF
            else current_modifier is not None
        )
        if not needs_update:
            return
        try:
            self.client.update_collection(
                collection_name=self.settings.qdrant_collection_name,
                sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=target_modifier)},
            )
            logger.info("Updated sparse vector modifier to %s for %s", target_modifier, self.settings.qdrant_collection_name)
        except Exception as exc:
            logger.warning("Failed to update sparse vector modifier: %s", exc)

    def ensure_payload_indexes(self) -> None:
        """flat payload 인덱스(PAYLOAD_INDEX_FIELDS)를 생성한다."""
        if not self._is_v2_collection(self.settings.qdrant_collection_name):
            return
        for field_name, schema_name in PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.qdrant_collection_name,
                    field_name=field_name,
                    field_schema=FIELD_SCHEMA_MAP[schema_name],
                    wait=True,
                )
            except Exception as exc:
                logger.warning("Skipping payload index for %s: %s", field_name, exc)

    def _collection_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False
