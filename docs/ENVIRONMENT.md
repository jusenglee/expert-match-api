# 환경 변수 (Environment Variables)

모든 환경 변수는 `NTIS_` 접두사를 사용합니다.

## 핵심 설정 (Core)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_APP_NAME` | `NTIS 전문가 추천 API` | FastAPI 앱 제목 |
| `NTIS_APP_ENV` | `prod` | 런타임 환경 (prod, dev 등) |
| `NTIS_APP_HOST` | `0.0.0.0` | FastAPI 서비스 바인딩 주소 |
| `NTIS_APP_PORT` | `8011` | FastAPI 수신 포트 |
| `NTIS_API_PREFIX` | `""` | 예약된 API 접두사 설정 |
| `NTIS_STRICT_RUNTIME_VALIDATION` | `true` | 운영 환경에서 필요한 설정들이 충족되지 않으면 추천 서비스를 비활성화함 |
| `NTIS_RUNTIME_DIR` | `runtime` | 런타임 출력물 저장 디렉토리 |
| `NTIS_FEEDBACK_DB_PATH` | `runtime/feedback.db` | 피드백 저장용 SQLite 경로 |
| `NTIS_FEEDBACK_TABLE` | `feedback_events` | 피드백 테이블 이름 |

## Qdrant 설정

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_QDRANT_URL` | `http://203.250.234.159:8005` | Qdrant 기본 URL |
| `NTIS_QDRANT_API_KEY` | (설정 안 됨) | Qdrant API 키 |
| `NTIS_QDRANT_COLLECTION_NAME` | `researcher_recommend_proto` | 사용할 컬렉션 이름 |
| `NTIS_QDRANT_CLOUD_INFERENCE` | `false` | Qdrant 클라우드 추론 활성화 여부 |

## LLM 백엔드 (LLM Backend)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_LLM_BACKEND` | `openai_compat` | 플래너/판정기용 LLM 백엔드 종류 |
| `NTIS_LLM_BASE_URL` | `http://203.250.234.159:8010/v1` | OpenAI 호환 LLM 서버 URL |
| `NTIS_LLM_API_KEY` | `EMPTY` | OpenAI 호환 API 키 |
| `NTIS_LLM_MODEL_NAME` | `/model` | 플래너/판정기용 모델 이름 |

## 임베딩 백엔드 (Embedding Backend)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_EMBEDDING_BACKEND` | `local` | 임베딩 추출용 백엔드 종류 |
| `NTIS_EMBEDDING_BASE_URL` | `http://203.250.234.159:8011/v1` | OpenAI 호환 임베딩 서버 URL |
| `NTIS_EMBEDDING_API_KEY` | `EMPTY` | OpenAI 호환 API 키 |
| `NTIS_EMBEDDING_MODEL_NAME` | `<repo>/multilingual-e5-large-instruct` | 로컬 번들 경로 또는 원격 모델 이름 |
| `NTIS_EMBEDDING_VECTOR_SIZE` | `1024` | Dense 벡터 크기 |

로컬 임베딩 번들 경로는 기본적으로 리포지토리 번들인 `multilingual-e5-large-instruct`를 가리킵니다. 해당 번들을 교체할 때 `modules.json`, `1_Pooling/config.json`, `2_Normalize/` 구조를 그대로 유지해야 서버 시작 시 오류가 발생하지 않습니다.

## BM25 및 오프라인 설정 (BM25 & Offline)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_BM25_MODEL_NAME` | `Qdrant/bm25` | Sparse 벡터 생성용 BM25 모델 이름 |
| `NTIS_BM25_CACHE_DIR` | `../../Models/hub/` | BM25 모델 캐시 디렉토리 |
| `NTIS_BM25_LOCAL_FILES_ONLY` | `true` | BM25 모델 로드 시 로컬 파일만 사용 여부 |
| `NTIS_HF_HUB_OFFLINE` | `true` | HuggingFace Hub 오프라인 모드 강제 여부 |

## 검색 제어 (Retrieval Controls)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_BRANCH_PREFETCH_LIMIT` | `80` | 각 데이터 브랜치별 프리페치(Prefetch) 제한 |
| `NTIS_BRANCH_OUTPUT_LIMIT` | `50` | 브랜치 통합(Fusion) 출력 제한 |
| `NTIS_RETRIEVAL_LIMIT` | `40` | 최종 검색 결과 통합 제한 |
| `NTIS_SHORTLIST_LIMIT` | `10` | 판정(Judge) 단계로 넘길 숏리스트 크기 |
| `NTIS_FINAL_RECOMMENDATION_MIN` | `3` | 최소 요구 최종 추천 수 |
| `NTIS_FINAL_RECOMMENDATION_MAX` | `5` | 최대 허용 최종 추천 수 |

## 시드 설정 (Seed Controls)

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `NTIS_SEED_ON_STARTUP` | `false` | 서버 시작 시 개발용 시드 데이터를 컬렉션에 로드할지 여부 |
| `NTIS_SEED_ALLOW_RECREATE_COLLECTION` | `false` | 시드 수행 시 기존 컬렉션을 삭제 후 재생성할지 여부 |

## 운영 참고 사항 (Operational Notes)

- 권장되는 라이브 설정:
  - `NTIS_STRICT_RUNTIME_VALIDATION=true`
  - `NTIS_LLM_BACKEND=openai_compat`
  - `NTIS_EMBEDDING_BACKEND=openai` 또는 `local`
  - `NTIS_SEED_ON_STARTUP=false`
- 엄격한 런타임 검증이 활성화된 경우 라이브 설정은 위와 같이 구성되어야 하며, `NTIS_SEED_ON_STARTUP=false`여야 합니다. 그럼에도 불구하고 플래너와 판정기는 실제 실행 중에 AI 모델 호출이 실패하면 내부적으로 휴리스틱 폴백을 수행합니다.
- 엄격한 검증이나 의존성 초기화에 실패하더라도 프로세스는 시작될 수 있으나, 실제 추천 트래픽을 보내기 전에 반드시 `GET /health/ready` 엔드포인트의 응답을 확인하시기 바랍니다.
