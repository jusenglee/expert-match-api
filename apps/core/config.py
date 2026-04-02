from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NTIS_", env_file=".env", extra="ignore")

    app_name: str = "NTIS Evaluator Recommendation API"
    app_env: Literal["dev", "test", "prod"] = "prod"
    api_prefix: str = ""
    strict_runtime_validation: bool = True

    qdrant_url: str = "http://203.250.234.159:8005"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "expert_master"
    qdrant_cloud_inference: bool = False

    llm_backend: Literal["heuristic", "openai_compat"] = "openai_compat"
    llm_base_url: str = "http://203.250.234.159:8010/v1"
    llm_api_key: str = "EMPTY"
    llm_model_name: str = "/model"

    embedding_backend: Literal["hashing", "openai", "local"] = "local"
    embedding_base_url: str = "http://203.250.234.159:8011/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model_name: str = "intfloat/multilingual-e5-large-instruct"
    embedding_vector_size: int = 1024

    branch_prefetch_limit: int = 80
    branch_output_limit: int = 50
    retrieval_limit: int = 40
    shortlist_limit: int = 10
    final_recommendation_min: int = 3
    final_recommendation_max: int = 5

    runtime_dir: Path = Field(default_factory=lambda: Path("runtime"))
    feedback_db_path: Path = Field(default_factory=lambda: Path("runtime") / "feedback.db")
    feedback_table: str = "feedback_events"

    seed_on_startup: bool = False
    seed_allow_recreate_collection: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    return settings
