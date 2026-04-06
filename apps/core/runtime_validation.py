"""
애플리케이션 실행 전 시스템 환경 및 외부 종속성(LLM, Embedding 서버 등)의 상태를 검증하는 모듈입니다.
운영 환경에서의 필수 설정 준수 여부와 백엔드 서비스의 가용성을 체크합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from apps.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackendCheckResult:
    """백엔드 서비스 점검 결과를 담는 데이터 클래스입니다."""
    name: str    # 서비스 이름 (예: llm_backend)
    ok: bool      # 점검 성공 여부
    detail: str   # 결과 상세 메시지


def validate_runtime_settings(settings: Settings) -> None:
    """
    운영 환경(strict_runtime_validation=True)에서 필수적인 설정들이 올바르게 구성되었는지 검증합니다.
    테스트용 백엔드(Heuristic/Hashing)가 운영 환경에서 사용되는 것을 방지합니다.
    """
    if settings.strict_runtime_validation:
        # LLM 백엔드 검증 (반드시 실제 서버 사용)
        if settings.llm_backend != "openai_compat":
            raise RuntimeError("운영 환경(strict)에서는 NTIS_LLM_BACKEND가 반드시 'openai_compat'이어야 합니다.")
        
        # 임베딩 백엔드 검증
        if settings.embedding_backend not in ("openai", "local"):
            raise RuntimeError("운영 환경(strict)에서는 NTIS_EMBEDDING_BACKEND가 'openai' 또는 'local'이어야 합니다.")
        
        # 초기 데이터 시딩 방지
        if settings.seed_on_startup:
            raise RuntimeError("운영 환경(strict)에서는 seed_on_startup 옵션을 비활성화해야 합니다.")


class RuntimeDependencyValidator:
    """
    외부 서비스(LLM, Embedding API 등)와의 연결성을 실제 호출을 통해 검증하는 클래스입니다.
    """
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_backends(self) -> list[BackendCheckResult]:
        """설정된 모든 백엔드 서비스에 대해 연결 테스트를 수행합니다."""
        results = [
            # 1. LLM 서버 연결 확인
            self._validate_openai_compatible_backend(
                name="llm_backend",
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model_name=self.settings.llm_model_name,
            )
        ]
        
        # 2. 임베딩 백엔드 연결 확인
        if self.settings.embedding_backend == "openai":
            results.append(
                self._validate_openai_compatible_backend(
                    name="embedding_backend",
                    base_url=self.settings.embedding_base_url,
                    api_key=self.settings.embedding_api_key,
                    model_name=self.settings.embedding_model_name,
                )
            )
        elif self.settings.embedding_backend == "local":
            results.append(
                BackendCheckResult(
                    name="embedding_backend",
                    ok=True,
                    detail="로컬 SentenceTransformer 모델 사용 중",
                )
            )
        return results

    def _validate_openai_compatible_backend(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        model_name: str,
    ) -> BackendCheckResult:
        """
        OpenAI 호환 API 서버의 /models 엔드포인트를 호출하여 연결성과 모델 존재 여부를 확인합니다.
        """
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            # 모델 리스트 조회 시도 (가장 기본적인 연결 테스트)
            models_response = client.models.list()
            model_ids = [item.id for item in models_response.data]
            
            # 설정된 모델명이 실제 서버 목록에 포함되어 있는지 확인
            if model_name in model_ids or not model_ids:
                return BackendCheckResult(name=name, ok=True, detail=f"{name} 연결이 확인되었습니다.")
            
            return BackendCheckResult(
                name=name,
                ok=False,
                detail=f"{name} 연결은 성공했으나, 모델 '{model_name}'을 찾을 수 없습니다.",
            )
        except Exception as exc:
            logger.warning("%s connectivity check failed during readiness validation", name, exc_info=True)
            return BackendCheckResult(name=name, ok=False, detail=f"{name} 연결 실패: {exc}")
