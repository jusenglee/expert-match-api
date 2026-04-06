from __future__ import annotations

from types import SimpleNamespace

from qdrant_client import models

from apps.core.config import Settings
from apps.search.qdrant_bootstrap import QdrantBootstrapper
from apps.search.schema_registry import BRANCHES, PAYLOAD_INDEX_FIELDS, SearchSchemaRegistry


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


def test_bootstrapper_repairs_sparse_vector_modifiers_on_existing_collection():
    client = FakeBootstrapClient(sparse_modifier=None)
    bootstrapper = QdrantBootstrapper(
        client=client,
        settings=Settings(),
        registry=SearchSchemaRegistry.default(),
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
