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

    # [핵심] 검색 대상 컬렉션. flat chunk 모델(1 chunk = 1 Point) 단일 벡터 컬렉션.
    # 단일 named vector: vector_e5i(dense) + vector_splade(sparse). doc_type은 payload 필터.
    qdrant_collection_name: str = "researcher_recommend_v1"

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
    # query-side IDF 보정(기본 ON). _build_sparse_query가 query 토큰 가중치에 downweight-only
    # 계수를 곱해 코퍼스 편재 토큰(상세/detail 등 SPLADE 확장 artifact)을 억제한다.
    # 계수 = 0 if idf<=hard_floor else min(1, idf/ref) — 1.0을 넘지 않아(boost 금지) 희귀 토큰은
    # 그대로 두고 빈출 토큰만 누른다. 기본 경로의 idf.json(sparse_stopwords.py --dump-idf 산출)이
    # 있으면 자동 활성, 없으면(CI/타 체크아웃) 무동작(현행 sparse 유지). 끄려면 sparse_idf_path="".
    sparse_idf_path: str | None = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[2] / "models" / "PIXIE-Splade-v1.0" / "sparse_idf.json"
        )
    )
    sparse_idf_ref: float = 3.5         # 이 idf 이상이면 계수 1.0(미변경). 빈출 억제 강도 조절.
    sparse_idf_hard_floor: float = 0.0  # idf<=이 값이면 계수 0(하드 마스크). 0=비활성(soft only).

    # =========================================================================
    # 6. 오케스트레이션 파라미터 (Retrieval Pipeline Limits)
    # =========================================================================
    # 검색 파이프라인의 각 단계(Stage)별로 데이터 처리량을 조절하여 품질과 속도의 균형을 맞춥니다.

    # [Prefetch] 그룹 하이브리드 검색의 prefetch 단계에서 dense/sparse가 각각 가져오는 chunk 풀 크기.
    # 넓게 확보할수록 group_by=researcher_id 그룹이 더 많이 형성된다.
    prefetch_limit: int = 256

    # [Group size] researcher 그룹당 회수할 최대 chunk 수(query_points_groups.group_size).
    # 이 chunk들이 곧 후보의 evidence 풀이며, 앱단 RRF 누적(Σ chunk score × doc_type prior)의 입력이다.
    group_size: int = 10

    # [그룹 수] query_points_groups가 반환할 최대 연구자(그룹) 수 = LLM에 넘기기 전 최종 후보 상한.
    retrieval_limit: int = 80

    # [최종 단계] LLM이 최종 판단하여 추천할 인원의 상한선과 하한선입니다.
    final_recommendation_min: int = 1  # 최소 추천 인원 (LLM이 이 인원수 미만 도출 시 품질 미달로 간주하여 재시도/오류 처리)
    final_recommendation_max: int = (
        15  # 최대 추천 인원 (API가 최종 응답하는 배열의 최대 길이)
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

    # =========================================================================
    # 8. v2.0 chunk 모델 설정 (WO-C C1)
    # =========================================================================
    # 전부 additive·비파괴 — 소비 코드(retriever/evidence/main 등)가 v2.0으로 바뀌기 전까지
    # 이 설정들은 미사용이며 v1.x 동작에 영향이 없다. (branch_* 개명·support_rule 제거·컬렉션
    # default 변경 등 BREAKING 정리는 해당 소비 코드 전환 슬라이스에서 함께 수행한다.)

    # doc_type 집계 prior. 미설정(None)=equal. HARD: equal이 기본값(ADR 0005 §2).
    # 이는 앱단 랭크 누적 가중일 뿐 Qdrant 가중 RRF/score 가중합이 아니다.
    doc_type_priors: dict[str, float] | None = None

    # 후보(연구자) cross-encoder 리랭커. 기본 off. band=score 동률 밴드 내 재배열만
    # (탈락/생성 금지). 후보 순위 척추는 RRF 고정. (ADR 0005 §3)
    candidate_reranker: Literal["off", "band"] = "off"

    # 검색 대상 doc_type 화이트리스트. 미설정(None)=전체 5종(paper/patent/project/assessor_activity/specialty).
    # planner는 doc_type on/off를 결정하지 않으며, 축소는 운영 화이트리스트로만. (ADR 0003)
    retrieval_doc_types: list[str] | None = None

    # 한 연구자의 동일 doc_type에서 점수 누적에 기여하는 최대 chunk 수(다작 독식 방지).
    doc_type_chunk_cap: int = 3

    # evidence family별 top-N cap (grounding 선별 한정 — 후보 순위 영향 0).
    evidence_family_cap: dict[str, int] = Field(
        default_factory=lambda: {
            "achievement": 10,
            "assessment": 6,
            "expertise": 6,
            "identity": 1,
        }
    )

    # evidence 리랭커 백엔드. 모델 부재 시 기본 lexical 강등. cross_encoder는 모델/서빙 설정 필요.
    evidence_reranker_backend: Literal["cross_encoder", "lexical"] = "lexical"
    cross_encoder_model_name: str | None = None
    cross_encoder_base_url: str | None = None
    cross_encoder_api_key: str | None = None
    # CrossEncoderEvidenceSelector 생성자 인자와 1:1 매핑(WO-0 스텁/WO-C 구현).
    ce_relevance_floor: float = 0.30
    ce_pregate_per_type: int = 20
    ce_max_pairs_per_request: int = 256
    ce_top_n_per_type: int = 5

    # =========================================================================
    # 9. 관련도 기반 검색·정렬 (v2.1 multi-view + capped evidence scoring)
    # =========================================================================
    # per-chunk 멀티뷰 융합: raw score 합산 금지 — source별 normalized rank score × view_weight.
    # 개념별 sparse view(sparse_concept)는 required_concepts마다 1개씩이며 동일 가중을 쓴다.
    search_view_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "dense_full": 1.0,
            "sparse_raw": 0.25,
            "sparse_focus": 0.7,
            "sparse_concept": 0.5,
        }
    )
    view_rrf_k: int = 60  # view_rank_score = 1.0 / (view_rrf_k + rank0)

    # 사용자 선택형 검색 모드별 파라미터(요청 search_mode로 선택, 기본=multiview).
    # · multiview(기본): search_view_weights로 dense_full + sparse_raw/focus + concept 뷰 융합(현행).
    # · hybrid: dense_full + sparse_raw 2뷰만 동일 RRF로 융합(단순 하이브리드). concept/focus 뷰 미사용.
    #   기본 가중을 균등(1.0/1.0)으로 둬 multiview(sparse_raw=0.25 보조채널)와 의미 있게 구분한다.
    # · keyword_similarity: SPLADE 희소로 1차 후보 풀을 회수한 뒤 그 집합 안에서만 dense 유사도로
    #   재정렬(2단계 cascade). chunk 점수 = dense 유사도(raw), 뷰 융합 미사용.
    hybrid_view_weights: dict[str, float] = Field(
        default_factory=lambda: {"dense_full": 1.0, "sparse_raw": 1.0}
    )
    # keyword_similarity 1차(SPLADE) 후보 풀 상한. 이 풀로 dense 재정렬 대상을 한정(HasIdCondition).
    keyword_first_stage_limit: int = 512

    # researcher capped evidence score 가중(단순 합산 금지).
    # joint=한 chunk가 required 다개념 동시충족 / balance=min(개념별 best) / concept=개념별 best 합 / support=보조 근거.
    researcher_score_weights: dict[str, float] = Field(
        default_factory=lambda: {"joint": 1.5, "balance": 1.2, "concept": 0.8, "support": 0.3}
    )
    researcher_support_top_k: int = 3  # best 외 보조 근거 cap(volume-bias 방지)
    # 개념 충족을 '서로 다른 토큰'으로만 인정(True). 붙은 단일 합성어(예: '인공지능반도체대학원')가 두 개념을
    # 한 토큰에서 동시 confirm해 만드는 거짓 joint + concept/balance 중복 계상을 차단한다. 같은 토큰값으로만
    # 확정되는 개념은 독립 근거가 없어 partial(→fallback)로 강등. off면 기존 동작(개념별 독립 best, 중복 허용).
    joint_requires_distinct_tokens: bool = True

    # doc_type별 근거 품질 가중(실적 ≥ 선언 > 활동).
    doc_type_quality_weight: dict[str, float] = Field(
        default_factory=lambda: {
            "specialty": 1.0, "project": 1.0, "paper": 0.9, "patent": 0.8, "assessor_activity": 0.55,
        }
    )

    # 운영성/교육/행정 과제(연구자 본인 설계 실적이 아닌 '프로그램 운영비': 대학원 운영·인력양성 등)는
    # 근거 가치를 낮춘다. 마커(도메인 무관 generic 행정어)가 chunk 본문/제목에 있으면 융합점수에 factor를
    # 곱한다. factor=1.0이면 무효. 실데이터 분포에 맞춰 마커/계수 튜닝 대상.
    operation_evidence_markers: list[str] = Field(
        default_factory=lambda: [
            "대학원", "인력양성", "전문인력양성", "부트캠프", "교육과정", "양성사업",
            "센터운영", "사업단운영", "운영지원",
        ]
    )
    operation_evidence_factor: float = 0.5

    # shortlist 확정 후 researcher_id로 대표 실적을 추가 조회해 '참고 프로필'로 보강(질의 매칭 evidence와 별도).
    # 점수/랭킹에는 영향 없고 표시/맥락용. researcher×doc_type당 cap개, 단일 scroll로 fetch_limit까지.
    profile_hydration_enabled: bool = True
    profile_evidence_per_doc_type_cap: int = 3
    profile_hydration_fetch_limit: int = 4000

    # 후보 1명당 LLM/표시에 들어가는 질의-매칭 evidence 총량 상한(doc_type별 family cap 이후 전역 적용).
    # joint→required 개념별 best→doc_type 다양성 우선으로 선택. 0=무제한(기존 family cap만).
    candidate_evidence_budget: int = 12

    # required concept gate: main top-K는 required_concepts 전부 충족. 부분 충족은 fallback tier로 분리.
    relevance_gate_enabled: bool = True
    relevance_fallback_tier: bool = True
    # weak evidence: 융합 관련도가 이 값 이하인 근거만 가진 후보는 감점(0=비활성, 튜닝).
    weak_evidence_floor: float = 0.0
    recency_bonus_weight: float = 0.0  # 최근 doc_date 가산(0=비활성, 튜닝)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    설정 객체를 싱글톤 패턴으로 반환합니다.
    최초 호출 시 필요한 런타임 디렉토리를 생성합니다.
    """
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    return settings
