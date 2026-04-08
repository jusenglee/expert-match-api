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
    PAYLOAD_INDEX_FIELDS,
    SearchSchemaRegistry,
)

logger = logging.getLogger(__name__)


# 필드 스키마 타입 매핑
FIELD_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "datetime": models.PayloadSchemaType.DATETIME,
}


class QdrantBootstrapper:
    """
    Qdrant 저장소의 초기 설정을 담당하는 클래스입니다.
    컬렉션 생성, 벡터 설정, 인덱스 관리를 수행합니다.
    """

    def __init__(
        self, client: QdrantClient, settings: Settings, registry: SearchSchemaRegistry
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry

    def ensure_collection(self, recreate: bool = False) -> None:
        """
        필요한 컬렉션이 존재하는지 확인하고, 없으면 생성합니다.
        recreate가 True인 경우 기존 컬렉션을 삭제하고 새로 만듭니다.
        """
        collection_name = self.settings.qdrant_collection_name
        if recreate:
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception:
                logger.info(
                    "Collection %s did not exist before recreate", collection_name
                )

        if not self._collection_exists(collection_name):
            # 컬렉션 생성 (다중 Dense 벡터 및 Sparse 벡터 설정 포함)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    self.registry.dense_vector_by_branch[branch]: models.VectorParams(
                        size=self.settings.embedding_vector_size,
                        distance=models.Distance.COSINE,
                    )
                    for branch in BRANCHES
                },
                # BM25 기반 키워드 검색을 위한 Sparse 벡터 설정
                sparse_vectors_config={
                    self.registry.sparse_vector_by_branch[
                        branch
                    ]: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
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
        기존 컬렉션의 스파스 벡터 수정자가 IDF로 설정되어 있는지 확인하고,
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

        updates: dict[str, models.SparseVectorParams] = {}
        for branch in BRANCHES:
            vector_name = self.registry.sparse_vector_by_branch[branch]
            params = sparse_config.get(vector_name)
            modifier = getattr(params, "modifier", None)
            # IDF가 아니면 업데이트 목록에 추가
            if not self._modifier_is_idf(modifier):
                updates[vector_name] = models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )

        if not updates:
            return

        try:
            self.client.update_collection(
                collection_name=self.settings.qdrant_collection_name,
                sparse_vectors_config=updates,
            )
            logger.info(
                "Updated sparse vector modifiers to IDF for collection %s: %s",
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
        """정의된 필드들에 대해 검색 성능 향상을 위한 인덱스를 생성합니다."""
        for field_name, schema_name in PAYLOAD_INDEX_FIELDS:
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
