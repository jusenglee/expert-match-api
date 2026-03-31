from __future__ import annotations

import asyncio

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import SeedExpertRecord
from apps.search.encoders import DenseEncoder
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.seed_data import build_canonical_payload_fixture, build_synthetic_records


class SeedRunner:
    def __init__(
        self,
        *,
        client: QdrantClient,
        settings: Settings,
        registry: SearchSchemaRegistry,
        dense_encoder: DenseEncoder,
    ) -> None:
        self.client = client
        self.settings = settings
        self.registry = registry
        self.dense_encoder = dense_encoder

    async def seed(self) -> list[SeedExpertRecord]:
        # seed는 외부 샘플 파일에 의존하지 않고, 코드 내부 canonical payload에서 시작한다.
        canonical = build_canonical_payload_fixture()
        records = build_synthetic_records(canonical)
        points = [self._to_point(record) for record in records]
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        return records

    def _to_point(self, record: SeedExpertRecord) -> models.PointStruct:
        vectors = {}
        dense_texts = {
            "basic": record.basic_text,
            "art": record.art_text,
            "pat": record.pat_text,
            "pjt": record.pjt_text,
        }
        for branch in BRANCHES:
            vectors[self.registry.dense_vector_by_branch[branch]] = self.dense_encoder.embed(dense_texts[branch])
            vectors[self.registry.sparse_vector_by_branch[branch]] = models.Document(
                text=dense_texts[branch],
                model="qdrant/bm25",
            )
        return models.PointStruct(
            id=record.point_id,
            vector=vectors,
            payload=record.payload.to_payload_dict(),
        )
