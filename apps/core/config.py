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
        model_config = SettingsConfigDict(
            env_prefix="NTIS_", env_file=".env", extra="ignore"
        )
    else:

        class Config:
            env_prefix = "NTIS_"
            env_file = ".env"
            extra = "ignore"

    # =========================================================================
    # 1. 기본 앱 정보 (Application Metadata & Network)
    # =========================================================================
    app_name: str = "NTIS Evaluator Recommendation API"
    app_env: Literal["dev", "test", "prod"] = (
        "prod"  # 스타트업 로그에 기록됨 (개발, 테스트, 운영 환경 구분)
    )
    app_host: str = "0.0.0.0"  # 컨테이너 환경을 위해 모든 IP의 접근을 허용
    app_port: int = 8011  # API 서버 구동 포트
    strict_runtime_validation: bool = (
        True  # 시작 시 LLM/Embedding/Qdrant 등 외부 의존성 정상 연결 여부 강제 확인
    )

    # =========================================================================
    # 2. Qdrant 벡터 데이터베이스 설정 (Vector Database Settings)
    # =========================================================================
    qdrant_url: str = "http://203.250.234.159:8005"  # Qdrant 서버 주소
    qdrant_api_key: str | None = None  # 인증이 필요한 경우 설정

    # [핵심] 검색을 수행할 대상 컬렉션 이름입니다.
    # Proto(테스트) 환경: "researcher_recommend_proto"
    # (주의: 이 컬렉션은 Sparse 벡터 이름이 _splade 형식으로 구성되어 있어야 정상 동작합니다.)
    qdrant_collection_name: str = "researcher_recommend_proto"

    qdrant_cloud_inference: bool = False  # Qdrant Cloud의 내장 추론 모델 사용 여부
    qdrant_collection_release_id: str = (
        "v0.3.0"  # L3 캐시(검색 결과 캐싱) 무효화 및 데이터 스냅샷 버전 관리용 키워드
    )

    # =========================================================================
    # 3. 아키텍처 Support Rule 설정 (Cross-Validation Rules)
    # =========================================================================
    # 2차 하이브리드 검색을 거친 후보자가 실제로 "추천될 자격"이 있는지 검증하는 최소 교차 증거 기준입니다.
    # (매우 중요) 값을 0으로 설정하면 엔진의 순수 RRF(Reciprocal Rank Fusion) 검색 점수를 100% 신뢰하여,
    # 특정 브랜치(논문, 특허 등)에서 실적이 없다는 이유로 유망한 후보를 임의 탈락(Omission)시키지 않습니다.
    # - stable_min: 핵심 브랜치(예: 프로젝트 실적 등)에서 최소 1회 이상 교차 검색되어야 하는가
    # - expanded_min: 확장 브랜치(예: 특허, 논문 등)를 포함하여 최소 N개 이상의 개별 브랜치에서 증거가 나와야 하는가
    support_rule_stable_min: int = 0
    support_rule_expanded_min: int = 0

    # =========================================================================
    # 4. 성능 최적화용 캐시 설정 (Caching Optimization)
    # =========================================================================
    # API 응답 속도 개선을 위해 동일한 검색/프롬프트 결과를 메모리 혹은 Redis에 보관하는 TTL(만료 시간) 설정
    cache_enabled: bool = True
    cache_ttl_l1_planner: int = (
        86400  # [L1 캐시] 사용자 질의 -> 의도 분석 및 키워드 추출 결과 보관 (24시간)
    )
    cache_ttl_l2_branch: int = (
        86400  # [L2 캐시] 브랜치별 서브쿼리 생성 결과 보관 (24시간)
    )
    cache_ttl_l3_retrieval: int = (
        1800  # [L3 캐시] Qdrant 하이브리드 검색 결과 객체 자체 보관 (30분)
    )

    # =========================================================================
    # 5. 백엔드 AI 모델 연결 설정 (LLM & Embeddings)
    # =========================================================================
    # 5-1. LLM (심사관 모델 - 의도 분석, 키워드 추출, 최종 평가)
    llm_backend: Literal["heuristic", "openai_compat"] = "openai_compat"
    llm_base_url: str = "http://203.250.234.159:8010/v1"
    llm_api_key: str = "EMPTY"  # 사설 모델 사용 시 API Key 불필요 (vLLM 호환)
    llm_model_name: str = "/model"

    # 5-2. Dense Embedding (밀집 벡터 - 문장/문단의 의미론적 벡터 추출기)
    embedding_backend: Literal["hashing", "openai", "local"] = "local"
    embedding_base_url: str = "http://203.250.234.159:8011/v1"
    embedding_api_key: str = "EMPTY"
    embedding_model_name: str = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2] / "multilingual-e5-large-instruct"
        )
    )
    embedding_vector_size: int = 1024  # E5 모델의 벡터 차원 수

    # 5-3. Sparse Embedding (희소 벡터 - BM25/SPLADE 기반 키워드 중요도 추출기)
    sparse_model_name: str = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2] / "models" / "PIXIE-Splade-v1.0"
        )
    )
    sparse_cache_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "models")
    )
    sparse_local_files_only: bool = False
    hf_hub_offline: bool = False

    # =========================================================================
    # 6. 오케스트레이션 파라미터 (Retrieval Pipeline Limits)
    # =========================================================================
    # 검색 파이프라인의 각 단계(Stage)별로 데이터 처리량을 조절하여 품질과 속도의 균형을 맞춥니다.

    # [1단계] 1차 키워드 검색 시 (Sparse), 각 브랜치(논문, 특허, 기본 등)별로 가져오는 최대 후보자 수.
    # 넓은 키워드 매칭 풀(Pool)을 먼저 확보하여 2차 하이브리드 검색이 이 풀 안에서만 이루어지도록 강제합니다.
    branch_prefetch_limit: int = 100

    # [2단계] 2차 하이브리드 검색 시 (Dense+Sparse), 1단계 모수 내에서 각 브랜치별로 상위 N명을 가져옵니다.
    # 추출된 이 결과들이 모두 모여서 최종 RRF(Reciprocal Rank Fusion) 가중합산 점수가 계산됩니다.
    branch_output_limit: int = 40

    # [3단계] 브랜치별 결과를 모두 합산(RRF)하고 Support Rule 필터링을 거친 후,
    # LLM(평가위원 심사기)에게 넘기기 전에 최종적으로 추려내는 최대 후보자의 수입니다.
    retrieval_limit: int = 80

    # [최종 단계] LLM이 최종 판단하여 추천할 인원의 상한선과 하한선입니다.
    final_recommendation_min: int = 1  # 최소 추천 인원 (LLM이 이 인원수 미만 도출 시 품질 미달로 간주하여 재시도/오류 처리)
    final_recommendation_max: int = (
        20  # 최대 추천 인원 (API가 최종 응답하는 배열의 최대 길이)
    )

    # =========================================================================
    # 7. LLM 심사 (Judge) 상세 및 기타 시스템 설정
    # =========================================================================
    llm_judge_batch_size: int = 10  # 한 번의 LLM 프롬프트에 동시 심사할 후보자 수
    llm_judge_max_concurrency: int = 10  # 비동기 LLM 호출의 최대 동시 연결 제한 수
    use_map_reduce_judging: bool = True  # 병렬 심사(Map-Reduce) 아키텍처 활성화 여부

    # 런타임 저장소 및 피드백 DB 설정 (운영 로깅용)
    runtime_dir: Path = Field(default_factory=lambda: Path("runtime"))
    feedback_db_path: Path = Field(
        default_factory=lambda: Path("runtime") / "feedback.db"
    )
    feedback_table: str = "feedback_events"

    # 서버 시작 시 데이터 초기화(Seeding) 옵션 (주로 Dev 환경용)
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
