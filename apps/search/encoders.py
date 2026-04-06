from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from apps.search.text_utils import stable_unit_vector


class DenseEncoder(Protocol):
    model_name: str
    vector_size: int

    def embed(self, text: str) -> list[float]:
        ...


def _resolve_local_model_path(model_name: str) -> Path | None:
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        return None
    return model_path.resolve()


def _validate_local_sentence_transformer_bundle(model_path: Path) -> None:
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
    _client: OpenAI = field(init=False, repr=False)

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
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        local_model_path = _resolve_local_model_path(self.model_name)
        if local_model_path is not None:
            _validate_local_sentence_transformer_bundle(local_model_path)
            self._model = SentenceTransformer(str(local_model_path), local_files_only=True)
            return

        self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text).tolist()
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch for {self.model_name}: "
                f"expected {self.vector_size}, got {len(vector)}"
            )
        return vector
