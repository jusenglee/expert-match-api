from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from apps.core.config import Settings


@dataclass(slots=True)
class BackendCheckResult:
    name: str
    ok: bool
    detail: str


def validate_runtime_settings(settings: Settings) -> None:
    # 운영전용 모드에서는 fallback 경로를 허용하지 않는다.
    # 즉, heuristic/hash 기반 보조 구현은 테스트나 로컬 샘플 검증에서만 사용한다.
    if settings.strict_runtime_validation:
        if settings.llm_backend != "openai_compat":
            raise RuntimeError("운영전용 모드에서는 NTIS_LLM_BACKEND=openai_compat 이어야 합니다.")
        if settings.embedding_backend != "openai":
            raise RuntimeError("운영전용 모드에서는 NTIS_EMBEDDING_BACKEND=openai 이어야 합니다.")
        if settings.seed_on_startup:
            raise RuntimeError("운영전용 모드에서는 seed_on_startup을 사용할 수 없습니다.")


class RuntimeDependencyValidator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_backends(self) -> list[BackendCheckResult]:
        return [
            self._validate_openai_compatible_backend(
                name="llm_backend",
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model_name=self.settings.llm_model_name,
            ),
            self._validate_openai_compatible_backend(
                name="embedding_backend",
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
                model_name=self.settings.embedding_model_name,
            ),
        ]

    def _validate_openai_compatible_backend(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model_name: str,
    ) -> BackendCheckResult:
        # OpenAI 호환 서버는 보통 /models를 지원하므로,
        # 여기서는 실제 목록 조회를 수행해 "연결 가능 + 모델 노출 여부"를 함께 검증한다.
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            models_response = client.models.list()
            model_ids = [item.id for item in models_response.data]
            if model_name in model_ids or not model_ids:
                return BackendCheckResult(name=name, ok=True, detail=f"{name} 연결 확인")
            return BackendCheckResult(
                name=name,
                ok=False,
                detail=f"{name} 연결은 되었지만 모델 {model_name!r} 이(가) 목록에 없습니다.",
            )
        except Exception as exc:
            return BackendCheckResult(name=name, ok=False, detail=f"{name} 연결 실패: {exc}")
