from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry

logger = logging.getLogger(__name__)


FIELD_SCHEMA_MAP = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "integer": models.PayloadSchemaType.INTEGER,
    "datetime": models.PayloadSchemaType.DATETIME,
}


class QdrantBootstrapper:
    def __init__(self, client: QdrantClient, settings: Settings, registry: SearchSchemaRegistry) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry

    def ensure_collection(self, recreate: bool = False) -> None:
        collection_name = self.settings.qdrant_collection_name
        if recreate:
            try:
                self.client.delete_collection(collection_name=collection_name)
            except Exception:
                logger.info("Collection %s did not exist before recreate", collection_name)

        if not self._collection_exists(collection_name):
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
                    self.registry.sparse_vector_by_branch[branch]: models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                    for branch in BRANCHES
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
        try:
            collection_info = self.client.get_collection(self.settings.qdrant_collection_name)
        except Exception as exc:
            logger.warning("Skipping sparse vector modifier repair because collection lookup failed: %s", exc)
            return

        sparse_config = getattr(collection_info.config.params, "sparse_vectors", None) or {}
        if not isinstance(sparse_config, dict):
            logger.warning("Skipping sparse vector modifier repair because sparse vector config is unavailable")
            return

        updates: dict[str, models.SparseVectorParams] = {}
        for branch in BRANCHES:
            vector_name = self.registry.sparse_vector_by_branch[branch]
            params = sparse_config.get(vector_name)
            modifier = getattr(params, "modifier", None)
            if not self._modifier_is_idf(modifier):
                updates[vector_name] = models.SparseVectorParams(modifier=models.Modifier.IDF)

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
            logger.warning("Failed to update sparse vector modifiers for %s: %s", self.settings.qdrant_collection_name, exc)

    def ensure_payload_indexes(self) -> None:
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
