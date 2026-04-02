from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from apps.search.text_utils import stable_unit_vector


class DenseEncoder(Protocol):
    model_name: str
    vector_size: int

    def embed(self, text: str) -> list[float]:
        ...


@dataclass(slots=True)
class HashingDenseEncoder:
    model_name: str
    vector_size: int

    def embed(self, text: str) -> list[float]:
        # 테스트 전용 encoder다.
        # 실운영에서는 사용하지 않고, 외부 임베딩 서버 없이 검색 조립만 검증할 때만 쓴다.
        return stable_unit_vector(text, self.vector_size)


@dataclass(slots=True)
class OpenAIEmbeddingEncoder:
    model_name: str
    vector_size: int
    base_url: str
    api_key: str

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        # 실운영에서는 이 경로를 통해 실제 임베딩 서버에 질의한다.
        response = self._client.embeddings.create(model=self.model_name, input=text)
        vector = list(response.data[0].embedding)
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.vector_size}, got {len(vector)}"
            )
        return vector

@dataclass(slots=True)
class LocalSentenceTransformerEncoder:
    model_name: str
    vector_size: int

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text).tolist()
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.vector_size}, got {len(vector)}"
            )
        return vector
