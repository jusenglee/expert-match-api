"""
생성된 가상 전문가(Seed) 데이터를 실제 Qdrant 컬렉션에 삽입하는 실행기 모듈입니다.
텍스트 데이터를 임베딩(Vectorize)하고 Qdrant의 포인트(Point) 구조로 변환하여 저장합니다.
"""

from __future__ import annotations

import asyncio

from qdrant_client import QdrantClient, models

from apps.core.config import Settings
from apps.domain.models import SeedExpertRecord
from apps.search.encoders import DenseEncoder
from apps.search.schema_registry import BRANCHES, SearchSchemaRegistry
from apps.search.seed_data import build_canonical_payload_fixture, build_synthetic_records


class SeedRunner:
    """
    Seed 데이터를 처리하여 Qdrant에 Upsert하는 역할을 수행합니다.
    """
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
        """
        내부 테스트 데이터를 생성하고 Qdrant 컬렉션에 동기화합니다.
        """
        # 1. 가상 데이터 생성
        canonical = build_canonical_payload_fixture()
        records = build_synthetic_records(canonical)
        
        # 2. Qdrant 포인트 구조로 변환 (벡터화 포함)
        points = [self._to_point(record) for record in records]
        
        # 3. Qdrant 서버로 대량 삽입 (Upsert)
        # QdrantClient의 upsert는 동기 함수이므로 별도 스레드에서 실행
        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.settings.qdrant_collection_name,
            points=points,
            wait=True,
        )
        return records

    def _to_point(self, record: SeedExpertRecord) -> models.PointStruct:
        """
        SeedExpertRecord를 Qdrant의 PointStruct로 변환합니다.
        각 브랜치(기본, 논문, 특허, 과제)에 대해 Dense 및 Sparse 벡터를 생성합니다.
        """
        vectors = {}
        dense_texts = {
            "basic": record.basic_text,
            "art": record.art_text,
            "pat": record.pat_text,
            "pjt": record.pjt_text,
        }
        
        # 각 데이터 브랜치별로 멀티 벡터 구성 (Hybrid Search 대응)
        for branch in BRANCHES:
            # Dense Vector: 의미론적 임베딩 생성
            vectors[self.registry.dense_vector_by_branch[branch]] = self.dense_encoder.embed(dense_texts[branch])
            # Sparse Vector: BM25 키워드 인덱싱용 텍스트 할당 (로컬 인코딩)
            vectors[self.registry.sparse_vector_by_branch[branch]] = models.Document(
                text=dense_texts[branch],
                model=self.settings.bm25_model_name,
            )
            
        return models.PointStruct(
            id=record.point_id,
            vector=vectors,
            payload=record.payload.to_payload_dict(), # 분석용 페이로드 저장
        )
