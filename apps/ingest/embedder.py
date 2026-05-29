"""WO-A: dense_e5i + sparse_splade 임베딩 생성 (DATA_MODEL §2 / §6-6).

두 벡터는 **동일 chunk_text 단일 입력**에서 생성한다. 신규 모델/인코더를 만들지 않고
apps.search.encoders / apps.search.sparse_runtime의 기존 인코더를 주입받아 재사용한다.

§6-6 보장: 동일 chunk_text 문자열을 두 인코더에 그대로 전달하고, 입력 sha256을 ChunkEmbedding에
부착한다. validate_chunks가 이 해시로 dense·sparse 입력 동일성을 검증한다.

VPN: 로컬 PIXIE(models/PIXIE-Splade-v1.0) + 로컬 dense 모델이면 VPN 불필요. 임베딩 서버 호출
(OpenAIEmbeddingEncoder) 시에만 VPN 필요(이 단계만 분리 실행). 본 모듈은 인코더를 주입받으므로
어떤 백엔드를 쓸지는 호출부(WO-B/운영)가 결정한다.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from apps.domain.models import ChunkPayload


def chunk_text_sha256(chunk_text: str) -> str:
    return hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChunkEmbedding:
    chunk_id: str
    dense_e5i: list[float]
    sparse_splade: dict[int, float]
    input_sha256: str  # = sha256(chunk_text). dense·sparse 동일 입력 보장(§6-6)


class ChunkEmbedder:
    """주입된 dense/sparse 인코더(`.embed(text)`)를 재사용해 chunk 임베딩을 생성한다."""

    def __init__(self, *, dense_encoder, sparse_encoder) -> None:
        self.dense_encoder = dense_encoder
        self.sparse_encoder = sparse_encoder

    def embed_chunk(self, chunk: ChunkPayload) -> ChunkEmbedding:
        text = chunk.chunk_text
        # 동일 문자열 객체를 두 인코더에 전달(§6-6). 직렬화 후 변형 금지.
        dense = self.dense_encoder.embed(text)
        sparse = self.sparse_encoder.embed(text)
        return ChunkEmbedding(
            chunk_id=chunk.chunk_id,
            dense_e5i=list(dense),
            sparse_splade=dict(sparse),
            input_sha256=chunk_text_sha256(text),
        )

    def embed_chunks(self, chunks) -> list[ChunkEmbedding]:
        return [self.embed_chunk(chunk) for chunk in chunks]
