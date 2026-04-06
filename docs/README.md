# NTIS 전문가 추천 서비스 (Evaluator Recommendation Service)

NTIS 심사위원 발굴 및 추천을 위한 FastAPI 기반 서비스입니다. 이 서비스는 Qdrant 하이브리드 검색과 Planner/Judge LLM 단계를 결합하며, 기계용 API와 로컬 브라우저 플레이그라운드를 모두 제공합니다.

## 주요 엔드포인트 (Endpoints)

- `POST /recommend`: 최종 추천 결과, 사유, 근거 및 데이터 결측치를 반환합니다. 근거가 충분한 추천 결과가 없더라도 `200` 상태 코드와 함께 `recommendations=[]` 및 설명 사유를 반환합니다.
- `POST /search/candidates`: 최종 판정(Judge) 단계 전의 숏리스트 후보군을 반환합니다.
- `POST /feedback`: 운영자의 피드백을 SQLite에 저장합니다.
- `GET /health`: 가벼운 프로세스 수준의 헬스체크 응답을 반환합니다.
- `GET /health/ready`: 실행 중인 의존성 및 샘플 데이터에 대한 런타임 준비 상태(Readiness) 응답을 반환합니다.
- `GET /playground`: 준비 상태 점검 및 API 연동 테스트를 위한 로컬 브라우저 UI를 제공합니다.
- `ntis-validate-live`: 동일한 라이브 준비 상태 검증기를 실행하는 CLI 래퍼입니다.

## 준비 상태(Readiness) 동작 방식

- `/health/ready`는 런타임이 완전히 준비되었을 때 `200`을 반환합니다.
- 검증에 실패하면 `503`을 반환하지만, 응답 본문에는 성공 시와 동일한 주요 필드(`ready`, `checks`, `issues`, `collection_name`, `sample_point_id`)를 포함합니다.
- 라이브 검증기는 컬렉션 내의 일정 수의 포인트를 스캔하여 가장 완전한 샘플 페이로드를 선택한 뒤 데이터 구조 이슈를 보고합니다. 로드된 데이터에는 루트 필드 외에 논문/과제 근거와 과제 날짜 필드가 필수적으로 포함되어야 하며, 특허 근거는 선택 사항입니다.
- 라이브 검증기가 활성화되기 전 초기화 단계에서 실패하면 앱은 성능 저하 모드(Degraded Mode)로 시작될 수 있습니다. 이 경우 `/health/ready`는 `checks.startup_runtime_initialized=false`와 함께 `503`을 반환하며, 문제가 해결될 때까지 추천 엔드포인트를 사용할 수 없습니다.

## 관련 문서 (Docs)

- [SERVICE_FLOW.md](/D:/Project/python_project/Ntis_person_API/docs/SERVICE_FLOW.md): 파이프라인 상세 흐름 가이드.
- [CONTRACT.md](/D:/Project/python_project/Ntis_person_API/docs/CONTRACT.md): 요청/응답 및 페이로드 규약(Contract).
- [RUNBOOK.md](/D:/Project/python_project/Ntis_person_API/docs/RUNBOOK.md): 로컬 실행 및 트러블슈팅 체크리스트.
- [ENVIRONMENT.md](/D:/Project/python_project/Ntis_person_API/docs/ENVIRONMENT.md): 환경 변수 및 런타임 기본값 설정.

## 런타임 정보 (Runtime Notes)

- Qdrant 컬렉션: `researcher_recommend_proto`
- 검색 브랜치: `basic`, `art`, `pat`, `pjt`
- 검색 모드: Dense + Sparse + RRF (하이브리드)
- 초기화 시 기존 컬렉션의 Sparse 벡터 수정자(Modifier)를 `IDF`로 자동 복구 시도 후 검증을 진행합니다.
- 검색 엔진은 누락된 선택적 리스트나 숫자 필드가 빈 문자열(`""`)로 저장된 레거시 페이로드를 읽기 시점에 정규화합니다.
- 엄격한 런타임 검증(Strict Validation)이 활성화된 경우, 운영 환경에서 휴리스틱 LLM 백엔드나 해싱 임베딩 백엔드를 사용하는 것을 금지합니다. 단, Planner와 Judge는 실행 중 AI 모델 호출 실패 시 내부적으로 휴리스틱 폴백(Fallback) 처리 로직을 유지합니다.

## Planner / Judge 상세

- OpenAI 호환 Planner는 `PlannerOutput` 규격에 맞는 단일 JSON 객체를 반환해야 합니다.
- Planner 응답이 유효하지 않은 경우 요청을 즉시 실패시키는 대신 휴리스틱 플래너로 폴백합니다.
- `POST /recommend` 호출 시 검색된 후보가 없거나 숏리스트가 비어있으면 판정(Judge) 단계를 건너뛰고 구조화된 빈 결과를 즉시 반환합니다.
- Judge는 LLM 응답에서 첫 번째 JSON 객체를 추출하며, 검증 전 복구 가능한 리스트/문자열 불일치를 정규화합니다. 유효하지 않은 응답의 경우 마찬가지로 휴리스틱 판정기로 폴백합니다.

## 빠른 시작 (Quick Start)

1. `python -m pip install -e .[dev]`
2. Qdrant 서버를 준비하고 앱이 대상 컬렉션을 바라보도록 설정합니다.
3. 필요한 `NTIS_` 환경 변수들을 설정합니다. ([ENVIRONMENT.md](/D:/Project/python_project/Ntis_person_API/docs/ENVIRONMENT.md) 참조)
4. 컬렉션 데이터를 로드하거나 검증합니다.
5. `ntis-validate-live` 명령어로 준비 상태를 확인합니다.
6. `uvicorn apps.api.main:app --reload` 명령어로 서버를 실행합니다.
7. `GET /health/ready`를 확인합니다.
8. `http://127.0.0.1:8000/playground`에 접속합니다.
9. 필요한 경우 `POST /search/candidates` 또는 `POST /recommend` 테스트를 수행합니다.

`uvicorn`은 시작되었으나 `/health/ready`가 `503`을 반환하는 경우, 추천 API를 호출하기 전에 구조화된 응답 내용을 확인하십시오. 브라우저 플레이그라운드의 상세 정보 패널에서도 동일한 정보를 확인할 수 있습니다.
