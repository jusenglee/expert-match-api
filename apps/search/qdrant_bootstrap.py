"""
Qdrant 컬렉션을 초기화하고 스키마에 맞춰 인덱스를 설정하는 부트스트래퍼 모듈입니다.
데이터 브랜치별 벡터 설정, 스파스 벡터 수식(IDF), 페이로드 인덱스 등을 관리합니다.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.search.schema_registry import (
    BRANCHES,
    DENSE_VECTOR_NAME,
    PAYLOAD_INDEX_FIELDS,
    PAYLOAD_INDEX_FIELDS_V2,
    SPARSE_VECTOR_NAME,
    SearchSchemaRegistry,
)
from apps.search.sparse_runtime import SparseRuntimeConfig, model_requires_idf_modifier

logger = logging.getLogger(__name__)


# 필드 스키마 타입 매핑
FIELD_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "datetime": models.PayloadSchemaType.DATETIME,
}

# blue/green 가드: 레거시 v1.x 컬렉션은 부트스트래퍼가 절대 재생성/스키마 변형하지 않는다.
# 그 외 컬렉션명(기본 ntis_researcher_chunks 또는 override)은 v2.0 단일 벡터 스키마로 처리.
LEGACY_V1X_COLLECTIONS: frozenset[str] = frozenset({"researcher_recommend_proto"})


class QdrantBootstrapper:
    """
    Qdrant 저장소의 초기 설정을 담당하는 클래스입니다.
    컬렉션 생성, 벡터 설정, 인덱스 관리를 수행합니다.
    """

    def __init__(
        self,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        sparse_runtime: SparseRuntimeConfig | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry
        self.sparse_runtime = sparse_runtime

    def _requires_idf_modifier(self) -> bool:
        if self.sparse_runtime is not None:
            return self.sparse_runtime.requires_idf_modifier
        return model_requires_idf_modifier(self.settings.sparse_model_name)

    def _is_v2_collection(self, collection_name: str) -> bool:
        """v2.0 chunk 스키마(단일 벡터) 적용 대상인지. 레거시 v1.x 컬렉션은 보호한다(blue/green)."""
        return collection_name not in LEGACY_V1X_COLLECTIONS

    def _sparse_vector_names(self, collection_name: str) -> list[str]:
        if self._is_v2_collection(collection_name):
            return [SPARSE_VECTOR_NAME]
        return [self.registry.sparse_vector_by_branch[branch] for branch in BRANCHES]

    def _payload_index_fields(self, collection_name: str):
        if self._is_v2_collection(collection_name):
            return PAYLOAD_INDEX_FIELDS_V2
        return PAYLOAD_INDEX_FIELDS

    def ensure_collection(self, recreate: bool = False) -> None:
        """
        필요한 컬렉션이 존재하는지 확인하고, 없으면 생성합니다.

        v2.0 컬렉션(예: ntis_researcher_chunks)은 단일 dense_e5i + 단일 sparse_splade로,
        레거시 v1.x 컬렉션(researcher_recommend_proto)은 기존 4브랜치 스키마로 생성한다.
        recreate=True는 v2.0 컬렉션에만 허용되며, 레거시 컬렉션은 절대 삭제하지 않는다(blue/green 가드).
        """
        collection_name = self.settings.qdrant_collection_name
        is_v2 = self._is_v2_collection(collection_name)

        if recreate:
            if not is_v2:
                logger.warning(
                    "Refusing to recreate legacy v1.x collection %s (blue/green guard); no-op",
                    collection_name,
                )
            else:
                try:
                    self.client.delete_collection(collection_name=collection_name)
                except Exception:
                    logger.info(
                        "Collection %s did not exist before recreate", collection_name
                    )

        if not self._collection_exists(collection_name):
            # Sparse 모델 종류에 따라 적절한 Modifier 설정.
            # SPLADE 계열은 모델이 직접 가중치를 계산하므로 IDF modifier 없음, bm25 fallback이면 IDF.
            requires_idf_modifier = self._requires_idf_modifier()
            sparse_modifier = models.Modifier.IDF if requires_idf_modifier else None

            logger.info(
                "Creating collection %s (v2=%s) with sparse_modifier=%s (requires_idf=%s)",
                collection_name,
                is_v2,
                sparse_modifier,
                requires_idf_modifier,
            )

            if is_v2:
                # v2.0: 단일 dense_e5i + 단일 sparse_splade. doc_type은 named vector가 아니라 payload 필터.
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=self.settings.embedding_vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            modifier=sparse_modifier,
                        )
                    },
                )
            else:
                # v1.x: 4브랜치 named vector (레거시 컬렉션 호환 — 보존)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        self.registry.dense_vector_by_branch[branch]: models.VectorParams(
                            size=self.settings.embedding_vector_size,
                            distance=models.Distance.COSINE,
                        )
                        for branch in BRANCHES
                    },
                    sparse_vectors_config={
                        self.registry.sparse_vector_by_branch[
                            branch
                        ]: models.SparseVectorParams(
                            modifier=sparse_modifier,
                        )
                        for branch in BRANCHES
                    },
                )

        # 키워드 가중치 수식(Modifier) 확인 및 인덱스 설정
        self.ensure_sparse_vector_modifiers()
        self.ensure_payload_indexes()

    @staticmethod
    def _modifier_is_idf(modifier: object) -> bool:
        """해당 객체가 Qdrant의 IDF 수식 설정을 의미하는지 확인합니다."""
        if modifier is None:
            return False

        candidates = [
            modifier,
            getattr(modifier, "value", None),
            getattr(modifier, "name", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            normalized = str(candidate).strip().lower()
            # IDF 설정 여부를 유연하게 체크
            if normalized == "idf" or "modifier.idf" in normalized:
                return True
        return False

    def ensure_sparse_vector_modifiers(self) -> None:
        """
        기존 컬렉션의 스파스 벡터 수정자가 설정과 일치하는지 확인하고,
        다를 경우 수정을 시도합니다.
        """
        try:
            collection_info = self.client.get_collection(
                self.settings.qdrant_collection_name
            )
        except Exception as exc:
            logger.warning(
                "Skipping sparse vector modifier repair because collection lookup failed: %s",
                exc,
            )
            return

        sparse_config = (
            getattr(collection_info.config.params, "sparse_vectors", None) or {}
        )
        if not isinstance(sparse_config, dict):
            logger.warning(
                "Skipping sparse vector modifier repair because sparse vector config is unavailable"
            )
            return

        target_modifier = (
            models.Modifier.IDF if self._requires_idf_modifier() else None
        )

        updates: dict[str, models.SparseVectorParams] = {}
        for vector_name in self._sparse_vector_names(self.settings.qdrant_collection_name):
            params = sparse_config.get(vector_name)
            current_modifier = getattr(params, "modifier", None)
            
            # 현재 설정이 목표와 다르면 업데이트 목록에 추가
            if target_modifier == models.Modifier.IDF:
                if not self._modifier_is_idf(current_modifier):
                    updates[vector_name] = models.SparseVectorParams(modifier=target_modifier)
            else:
                if current_modifier is not None:
                    updates[vector_name] = models.SparseVectorParams(modifier=None)

        if not updates:
            return

        try:
            self.client.update_collection(
                collection_name=self.settings.qdrant_collection_name,
                sparse_vectors_config=updates,
            )
            logger.info(
                "Updated sparse vector modifiers to %s for collection %s: %s",
                target_modifier,
                self.settings.qdrant_collection_name,
                sorted(updates.keys()),
            )
        except Exception as exc:
            logger.warning(
                "Failed to update sparse vector modifiers for %s: %s",
                self.settings.qdrant_collection_name,
                exc,
            )

    def ensure_payload_indexes(self) -> None:
        """정의된 필드들에 대해 검색 성능 향상을 위한 인덱스를 생성합니다.

        v2.0 컬렉션은 PAYLOAD_INDEX_FIELDS_V2(chunk 3층 payload), 레거시 v1.x는 기존 nested 인덱스.
        """
        for field_name, schema_name in self._payload_index_fields(
            self.settings.qdrant_collection_name
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.qdrant_collection_name,
                    field_name=field_name,
                    field_schema=FIELD_SCHEMA_MAP[schema_name],
                    wait=True,
                )
            except Exception as exc:
                # 이미 존재하거나 생성 중인 경우 스킵
                logger.warning("Skipping payload index for %s: %s", field_name, exc)

    def _collection_exists(self, collection_name: str) -> bool:
        """컬렉션 존재 여부를 확인합니다."""
        try:
            self.client.get_collection(collection_name=collection_name)
            return True
        except Exception:
            return False
