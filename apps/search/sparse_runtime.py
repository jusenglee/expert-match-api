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
    backend: Literal["custom_splade", "fastembed_builtin"]
    active_model_name: str
    requires_idf_modifier: bool
    used_fallback: bool = False
    attempt_log: tuple[str, ...] = field(default_factory=tuple)

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
