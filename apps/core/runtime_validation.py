from __future__ import annotations

import logging
from dataclasses import dataclass

from openai import OpenAI

from apps.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackendCheckResult:
    name: str
    ok: bool
    detail: str


def validate_runtime_settings(settings: Settings) -> None:
    # Production runtime must use the real backend paths instead of test fallbacks.
    if settings.strict_runtime_validation:
        if settings.llm_backend != "openai_compat":
            raise RuntimeError("NTIS_LLM_BACKEND must be openai_compat when strict runtime validation is enabled.")
        if settings.embedding_backend not in ("openai", "local"):
            raise RuntimeError("NTIS_EMBEDDING_BACKEND must be openai or local when strict runtime validation is enabled.")
        if settings.seed_on_startup:
            raise RuntimeError("seed_on_startup must remain disabled when strict runtime validation is enabled.")


class RuntimeDependencyValidator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def validate_backends(self) -> list[BackendCheckResult]:
        results = [
            self._validate_openai_compatible_backend(
                name="llm_backend",
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                model_name=self.settings.llm_model_name,
            )
        ]
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
                    detail="local sentence-transformer model in use",
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
        # OpenAI-compatible servers usually expose /models, so this verifies both
        # basic connectivity and that the configured model can be listed.
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            models_response = client.models.list()
            model_ids = [item.id for item in models_response.data]
            if model_name in model_ids or not model_ids:
                return BackendCheckResult(name=name, ok=True, detail=f"{name} connectivity confirmed")
            return BackendCheckResult(
                name=name,
                ok=False,
                detail=f"{name} connected but model {model_name!r} was not returned by /models.",
            )
        except Exception as exc:
            logger.warning("%s connectivity check failed during readiness validation", name, exc_info=True)
            return BackendCheckResult(name=name, ok=False, detail=f"{name} connectivity failed: {exc}")
