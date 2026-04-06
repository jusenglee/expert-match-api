"""
텍스트를 고차원 벡터로 변환하는 Dense Encoder들을 정의하는 모듈입니다.
OpenAI API, 로컬 SentenceTransformer, 그리고 테스트용 Hashing Encoder 등
다양한 벡터화 방식을 지원합니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from apps.search.text_utils import stable_unit_vector


class DenseEncoder(Protocol):
    """텍스트 임베딩을 위한 인코더 인터페이스 정의입니다."""
    model_name: str     # 사용할 모델의 이름 또는 경로
    vector_size: int    # 생성될 벡터의 차원 수

    def embed(self, text: str) -> list[float]:
        """텍스트를 입력받아 숫자 리스트(벡터)로 변환합니다."""
        ...


def _resolve_local_model_path(model_name: str) -> Path | None:
    """모델 이름이 로컬 파일 경로인지 확인하고 절대 경로를 반환합니다."""
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        return None
    return model_path.resolve()


def _validate_local_sentence_transformer_bundle(model_path: Path) -> None:
    """로컬에 저장된 SentenceTransformer 모델의 구성 파일들이 온전한지 검증합니다."""
    modules_path = model_path / "modules.json"
    if not modules_path.is_file():
        return

    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    missing_module_dirs = [
        str(model_path / module_path)
        for module in modules
        if (module_path := module.get("path")) and not (model_path / module_path).is_dir()
    ]
    if missing_module_dirs:
        missing_dirs_text = ", ".join(missing_module_dirs)
        raise FileNotFoundError(
            "Local sentence-transformer bundle is incomplete. "
            f"Missing module directories: {missing_dirs_text}"
        )


@dataclass(slots=True)
class HashingDenseEncoder:
    """
    결정론적 해싱(Deterministic Hashing) 기반의 테스트용 인코더입니다.
    실제 의미론적 분석은 수행하지 않으며, 외부 API 없이 시스템 통합 테스트를 할 때 사용합니다.
    """
    model_name: str
    vector_size: int

    def embed(self, text: str) -> list[float]:
        # 테스트 전용 encoder다.
        # 실운영에서는 사용하지 않고, 외부 임베딩 서버 없이 검색 조립만 검증할 때만 쓴다.
        return stable_unit_vector(text, self.vector_size)


@dataclass(slots=True)
class OpenAIEmbeddingEncoder:
    """
    OpenAI 호환 임베딩 API를 사용하는 인코더입니다.
    운영 환경에서 가장 높은 검색 품질을 보장합니다.
    """
    model_name: str
    vector_size: int
    base_url: str
    api_key: str
    _client: OpenAI = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """OpenAI 클라이언트를 초기화합니다."""
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def embed(self, text: str) -> list[float]:
        """OpenAI API를 호출하여 텍스트 임베딩을 생성합니다."""
        # 실운영에서는 이 경로를 통해 실제 임베딩 서버에 질의한다.
        response = self._client.embeddings.create(model=self.model_name, input=text)
        vector = list(response.data[0].embedding)
        
        # 설정된 벡터 차원과 실제 응답 차원이 일치하는지 확인
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.vector_size}, got {len(vector)}"
            )
        return vector


@dataclass(slots=True)
class LocalSentenceTransformerEncoder:
    """
    로컬 CPU/GPU 자원을 사용하여 SentenceTransformer 모델을 실행하는 인코더입니다.
    인터넷 연결이 제한된 환경이나 비용 절감이 필요한 경우에 적합합니다.
    """
    model_name: str
    vector_size: int
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """SentenceTransformer 모델을 로드합니다."""
        from sentence_transformers import SentenceTransformer

        # 로컬 경로에 모델이 있는지 먼저 확인
        local_model_path = _resolve_local_model_path(self.model_name)
        if local_model_path is not None:
            _validate_local_sentence_transformer_bundle(local_model_path)
            self._model = SentenceTransformer(str(local_model_path), local_files_only=True)
            return

        # 로컬에 없으면 HuggingFace Hub 등에서 다운로드 (또는 이미 다운로드된 캐시 사용)
        self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        """로컬 모델을 사용하여 텍스트 임베딩을 추론합니다."""
        vector = self._model.encode(text).tolist()
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.vector_size}, got {len(vector)}"
            )
        return vector
