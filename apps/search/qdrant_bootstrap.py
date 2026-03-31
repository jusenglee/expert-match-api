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
        self.ensure_payload_indexes()

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
