"""
애플리케이션의 모든 설정 정보를 중앙 관리하는 모듈입니다.
Pydantic Settings를 사용하여 환경 변수(.env)로부터 설정을 로드하며,
데이터베이스 연결 정보, LLM API 설정, 검색 엔진 파라미터 등을 정의합니다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # pydantic v1 또는 pydantic-settings가 없는 환경을 위한 하위 호환성 처리
    from pydantic.v1 import BaseSettings, Field

    SettingsConfigDict = None


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정 클래스입니다.
    모든 환경 변수는 'NTIS_' 접두사를 사용하여 재정의할 수 있습니다.
    예) NTIS_QDRANT_URL=http://localhost:6333
    """
    if SettingsConfigDict is not None:
        model_config = SettingsConfigDict(env_prefix="NTIS_", env_file=".env", extra="ignore")
    else:
        class Config:
            env_prefix = "NTIS_"
            env_file = ".env"
            extra = "ignore"

    # 기본 앱 정보
    app_name: str = "NTIS Evaluator Recommendation API"
    app_env: Literal["dev", "test", "prod"] = "prod"
    api_prefix: str = ""
    strict_runtime_validation: bool = True

    # Qdrant 벡터 데이터베이스 설정
    qdrant_url: str = "http://203.250.234.159:8005"
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "researcher_recommend_test"
    qdrant_cloud_inference: bool = False

    # LLM (의도의 분석 및 심사) 백엔드 설정
    llm_backend: Literal["heuristic", "openai_compat"] = "openai_compat"
    llm_base_url: str = "http://203.250.234.159:8010/v1"
    llm_api_key: str = "EMPTY"
    llm_model_name: str = "/model"

    # 임베딩 (밀집 벡터 생성) 백엔드 설정
    embedding_backend: Literal["hashing", "openai", "local"] = "local"
    embedding_base_url: str = "http://203.250.234.159:8011/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model_name: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "multilingual-e5-large-instruct")
    )
    embedding_vector_size: int = 1024

    # 검색 및 추천 오케스트레이션 파라미터
    branch_prefetch_limit: int = 80   # 각 브랜치별 초기 검색 수
    branch_output_limit: int = 50     # RRF 결합 후 브랜치별 출력 수
    retrieval_limit: int = 40          # 최종 리트리벌 결과 수
    shortlist_limit: int = 10          # 심사에 전달할 후보자 수
    final_recommendation_min: int = 3  # 최소 추천 인원
    final_recommendation_max: int = 5  # 최대 추천 인원

    # 런타임 저장소 및 피드백 DB 설정
    runtime_dir: Path = Field(default_factory=lambda: Path("runtime"))
    feedback_db_path: Path = Field(default_factory=lambda: Path("runtime") / "feedback.db")
    feedback_table: str = "feedback_events"

    # 서버 시작 시 데이터 초기화(Seeding) 옵션
    seed_on_startup: bool = False
    seed_allow_recreate_collection: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    설정 객체를 싱글톤 패턴으로 반환합니다.
    최초 호출 시 필요한 런타임 디렉토리를 생성합니다.
    """
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    return settings
