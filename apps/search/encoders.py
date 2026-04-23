"""
텍스트를 고차원 벡터로 변환하는 Dense 및 Sparse Encoder들을 정의하는 모듈입니다.
OpenAI API, 로컬 SentenceTransformer, 그리고 커스텀 SPLADE 인코더 등을 지원합니다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from openai import OpenAI

from apps.search.text_utils import stable_unit_vector

logger = logging.getLogger(__name__)


class DenseEncoder(Protocol):
    """텍스트 임베딩을 위한 인코더 인터페이스 정의입니다."""
    model_name: str     # 사용할 모델의 이름 또는 경로
    vector_size: int    # 생성될 벡터의 차원 수

    def embed(self, text: str) -> list[float]:
        """텍스트를 입력받아 숫자 리스트(벡터)로 변환합니다."""
        ...


class SparseEncoder(Protocol):
    """희소 벡터(Sparse Vector) 생성을 위한 인코더 인터페이스 정의입니다."""
    model_name: str

    def embed(self, text: str) -> dict[int, float]:
        """텍스트를 {token_id: weight} 형태의 희소 벡터로 변환합니다."""
        ...


@dataclass(slots=True)
class SpladeSparseEncoder:
    """
    Transformers 라이브러리를 사용하여 SPLADE(Sparse Lexical and Semantic) 기반 희소 벡터(Sparse Vector)를 생성합니다.
    fastembed 등 외부 라이브러리에서 공식 지원하지 않는 커스텀 모델(예: PIXIE-Splade)을 로컬 환경에서 구동하기 위해 작성되었습니다.
    문맥을 고려한 키워드 확장 및 가중치 계산을 수행합니다.
    """
    model_name: str
    local_files_only: bool = False
    _tokenizer: Any = field(init=False, repr=False)
    _model: Any = field(init=False, repr=False)
    _device: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            resolved_model_name, load_kwargs = _prepare_model_reference(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self.model_name = resolved_model_name
            self._tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name, **load_kwargs
            )
            self._model = AutoModelForMaskedLM.from_pretrained(
                resolved_model_name, **load_kwargs
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("SpladeSparseEncoder initialized on %s: %s", self._device, self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize SpladeSparseEncoder: %s", exc)
            raise

    def embed(self, text: str) -> dict[int, float]:
        """
        입력된 텍스트를 모델에 통과시켜 SPLADE 희소 벡터 가중치 맵(dictionary 형태)으로 변환합니다.
        가중치가 0이 아닌 유효한 토큰(Token) ID와 그에 해당하는 점수를 반환합니다.
        """
        if not text or not text.strip():
            return {}

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self._device)
        
        with torch.no_grad():
            logits = self._model(**inputs).logits
            
        # SPLADE log(1 + ReLU(logits)) * attention_mask
        weights = torch.log(1 + torch.relu(logits))
        # Max pooling across tokens
        input_mask = inputs.attention_mask.unsqueeze(-1).expand_as(weights)
        sparse_vector = torch.max(weights * input_mask, dim=1).values[0]
        
        # 0이 아닌 가중치만 추출
        indices = torch.nonzero(sparse_vector).flatten()
        values = sparse_vector[indices]
        
        return {int(idx): float(val) for idx, val in zip(indices, values)}


@dataclass(slots=True)
class HashingDenseEncoder:
    """테스트용 해싱 기반 인코더입니다."""
    model_name: str
    vector_size: int

    def embed(self, text: str) -> list[float]:
        return stable_unit_vector(text, self.vector_size)


@dataclass(slots=True)
class OpenAIEmbeddingEncoder:
    """
    OpenAI 호환 API(vLLM, TGI 등)를 사용하여 텍스트를 밀집 벡터(Dense Vector)로 임베딩하는 인코더입니다.
    원격 서버로 API 요청을 보내 임베딩 결과를 받아옵니다.
    """
    model_name: str
    vector_size: int
    base_url: str
    api_key: str
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(model=self.model_name, input=text)
        vector = list(response.data[0].embedding)
        if len(vector) != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: expected {self.vector_size}, got {len(vector)}")
        return vector


@dataclass(slots=True)
class LocalSentenceTransformerEncoder:
    """
    HuggingFace SentenceTransformer 라이브러리를 사용하여 로컬 환경(GPU/CPU)에서 
    직접 텍스트를 밀집 벡터(Dense Vector)로 변환하는 인코더입니다.
    네트워크 지연 없이 빠른 로컬 임베딩이 필요할 때 사용됩니다.
    """
    model_name: str
    vector_size: int
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_model_name, load_kwargs = _prepare_model_reference(
            self.model_name,
            required_files=("modules.json", "1_Pooling/config.json"),
        )
        self.model_name = resolved_model_name
        self._model = SentenceTransformer(resolved_model_name, **load_kwargs)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text).tolist()
        if len(vector) != self.vector_size:
            raise ValueError(f"Embedding dimension mismatch: expected {self.vector_size}, got {len(vector)}")
        return vector


def _resolve_local_model_path(model_name: str) -> Path | None:
    path = Path(model_name).expanduser()
    return path.resolve() if path.exists() else None


def _looks_like_local_model_reference(model_name: str) -> bool:
    path = Path(model_name).expanduser()
    if path.is_absolute() or model_name.startswith((".", "~")) or "\\" in model_name:
        return True

    parts = [part for part in path.parts if part not in ("", ".")]
    if len(parts) > 2:
        return True

    return bool(parts) and Path(parts[0]).exists()


def _validate_local_bundle(model_path: Path, required_files: tuple[str, ...]) -> None:
    for relative_path in required_files:
        required_path = model_path / relative_path
        if not required_path.exists():
            raise FileNotFoundError(
                f"Local model bundle is missing required file: {required_path}"
            )


def _prepare_model_reference(
    model_name: str,
    *,
    local_files_only: bool = False,
    required_files: tuple[str, ...] = (),
) -> tuple[str, dict[str, Any]]:
    local_model_path = _resolve_local_model_path(model_name)
    if local_model_path is not None:
        _validate_local_bundle(local_model_path, required_files)
        return str(local_model_path), {"local_files_only": True}

    if _looks_like_local_model_reference(model_name):
        requested_path = Path(model_name).expanduser().resolve(strict=False)
        raise FileNotFoundError(f"Local model path does not exist: {requested_path}")

    return model_name, {"local_files_only": local_files_only}
