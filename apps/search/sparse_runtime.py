"""
희소 벡터(Sparse Vector) 모델 환경을 구성하고 관리하는 런타임 모듈입니다.

[Architecture Overview]
Qdrant의 Sparse 벡터 검색을 지원하기 위해, 어떤 인코더를 사용할지 런타임에 결정합니다.
1. Custom SPLADE (PIXIE-Splade-v1.0): 
   - 한국어 도메인에 특화된 모델로, 단어의 문맥적 중요도를 가중치로 추출합니다.
   - 로컬 모델 캐시 경로를 최우선으로 시도하며, 실패 시 HuggingFace Hub 다운로드를 시도합니다.
2. FastEmbed BM25 (Fallback):
   - SPLADE 로드에 최종 실패하거나, 오프라인 환경에서 로컬 파일이 없을 경우
   - 시스템이 중단되지 않고 Qdrant 내장 BM25 혹은 FastEmbed BM25로 안전하게 Fallback 하도록 보장합니다.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from apps.core.config import Settings
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

ONLINE_PIXIE_SPLADE_MODEL = "telepix/PIXIE-Splade-v1.0"
QDRANT_BM25_MODEL = "Qdrant/bm25"


@dataclass(slots=True, frozen=True)
class SparseRuntimeConfig:
    """
    현재 런타임에 활성화된 희소 벡터(Sparse) 인코더의 상태 정보입니다.
    """
    backend: Literal["custom_splade", "fastembed_builtin"]  # 활성화된 백엔드 종류
    active_model_name: str                                  # 실제로 로드된 모델의 이름 또는 경로
    requires_idf_modifier: bool                             # BM25 Fallback 시 IDF 수정자 필요 여부
    used_fallback: bool = False                             # 최우선 모델(SPLADE) 로드 실패로 Fallback을 사용했는지 여부
    attempt_log: tuple[str, ...] = field(default_factory=tuple) # 로드 시도 및 실패 내역(로깅용)

    @property
    def uses_custom_encoder(self) -> bool:
        return self.backend == "custom_splade"


SparseEncoderFactory = Callable[..., Any]


def sparse_online_fallback_allowed(settings: "Settings") -> bool:
    return not (settings.hf_hub_offline or settings.sparse_local_files_only)


def prepare_sparse_runtime_environment(settings: "Settings") -> Path:
    cache_dir = Path(settings.sparse_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    if settings.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    return cache_dir


def resolve_sparse_runtime(
    *,
    client: "QdrantClient",
    settings: "Settings",
    cache_dir: Path,
    sparse_encoder_factory: SparseEncoderFactory,
) -> tuple[SparseRuntimeConfig, Any | None]:
    """
    시스템 시작 시 Sparse 인코더를 결정하고 로드하는 핵심 함수입니다.
    
    [Fallback 순서]
    1. 로컬 경로 (settings.sparse_model_name) 의 SPLADE 모델 로드 시도
    2. (허용 시) 온라인 HuggingFace Hub의 "telepix/PIXIE-Splade-v1.0" 로드 시도
    3. (허용 시) FastEmbed BM25 ("Qdrant/bm25") 로드 시도 (최후의 수단)
    
    Returns:
        현재 런타임 구성 객체(SparseRuntimeConfig)와 실제 로드된 Encoder 인스턴스
    """
    attempt_log: list[str] = []
    primary_model_name = settings.sparse_model_name
    bm25_local_files_only = settings.sparse_local_files_only or settings.hf_hub_offline

    def _select_custom_splade(
        *,
        step: str,
        model_name: str,
        local_files_only: bool,
        used_fallback: bool,
    ) -> tuple[SparseRuntimeConfig, Any]:
        encoder = sparse_encoder_factory(
            model_name=model_name,
            local_files_only=local_files_only,
        )
        runtime = SparseRuntimeConfig(
            backend="custom_splade",
            active_model_name=encoder.model_name,
            requires_idf_modifier=False,
            used_fallback=used_fallback,
            attempt_log=tuple(attempt_log),
        )
        logger.info(
            "Sparse backend selected: backend=%s model=%s step=%s fallback=%s",
            runtime.backend,
            runtime.active_model_name,
            step,
            runtime.used_fallback,
        )
        return runtime, encoder

    try:
        return _select_custom_splade(
            step="configured_pixie",
            model_name=primary_model_name,
            local_files_only=settings.sparse_local_files_only,
            used_fallback=False,
        )
    except Exception as exc:
        message = (
            "Sparse backend attempt failed: step=configured_pixie "
            f"backend=custom_splade model={primary_model_name} reason={exc}"
        )
        attempt_log.append(message)
        logger.warning(message)

    if ONLINE_PIXIE_SPLADE_MODEL == primary_model_name:
        skip_message = (
            "Sparse backend fallback skipped: step=online_pixie reason=configured model "
            "already matched the online fallback model"
        )
        attempt_log.append(skip_message)
        logger.info(skip_message)
    elif not sparse_online_fallback_allowed(settings):
        skip_reason = (
            "HF_HUB_OFFLINE=1"
            if settings.hf_hub_offline
            else "NTIS_SPARSE_LOCAL_FILES_ONLY=true"
        )
        skip_message = (
            "Sparse backend fallback skipped: step=online_pixie "
            f"reason={skip_reason}"
        )
        attempt_log.append(skip_message)
        logger.info(skip_message)
    else:
        try:
            return _select_custom_splade(
                step="online_pixie",
                model_name=ONLINE_PIXIE_SPLADE_MODEL,
                local_files_only=False,
                used_fallback=True,
            )
        except Exception as exc:
            message = (
                "Sparse backend attempt failed: step=online_pixie "
                f"backend=custom_splade model={ONLINE_PIXIE_SPLADE_MODEL} reason={exc}"
            )
            attempt_log.append(message)
            logger.warning(message)

    bm25_kwargs: dict[str, object] = {}
    if bm25_local_files_only:
        bm25_kwargs["local_files_only"] = True

    try:
        client.set_sparse_model(
            embedding_model_name=QDRANT_BM25_MODEL,
            cache_dir=str(cache_dir),
            **bm25_kwargs,
        )
        runtime = SparseRuntimeConfig(
            backend="fastembed_builtin",
            active_model_name=QDRANT_BM25_MODEL,
            requires_idf_modifier=True,
            used_fallback=True,
            attempt_log=tuple(attempt_log),
        )
        logger.warning(
            "Sparse backend selected via fallback: backend=%s model=%s fallback=%s",
            runtime.backend,
            runtime.active_model_name,
            runtime.used_fallback,
        )
        return runtime, None
    except Exception as exc:
        message = (
            "Sparse backend attempt failed: step=bm25_fallback "
            f"backend=fastembed_builtin model={QDRANT_BM25_MODEL} reason={exc}"
        )
        attempt_log.append(message)
        logger.error(message)
        raise RuntimeError(
            "Sparse backend initialization failed after fallback chain: "
            + " | ".join(attempt_log)
        ) from exc


def model_requires_idf_modifier(model_name: str) -> bool:
    return "splade" not in model_name.lower()
