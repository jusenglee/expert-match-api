# NTIS 전문가 추천 시스템 프로토타입

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![Qdrant](https://img.shields.io/badge/Qdrant-1.10%2B-red)
![OpenAI](https://img.shields.io/badge/OpenAI-1.40%2B-lightgray)

본 프로젝트는 다차원 인물 정보(기본 정보, 논문, 특허, 과제 이력 등)를 분석하여 요구 조건에 가장 부합하는 전문가를 추천하는 지능형 API 시스템입니다. Qdrant를 이용한 하이브리드 검색과 LLM의 판단 능력을 결합하여 신뢰도 높은 추천 결과를 제공합니다.

## 주요 특징

- **다차원 하이브리드 검색 (Multi-branch Hybrid Search)**
  - 인물 정보를 기본 정보, 논문, 특허, 과제 4개 관점으로 분리하여 탐색합니다.
  - 의미론적 검색(Dense)과 키워드 검색(Sparse)을 결합하고 RRF 알고리즘으로 통합 랭킹을 산출합니다.
- **LLM 기반 지능형 파이프라인**
  - **Planner**: 자연어 질의를 분석하여 검색 쿼리와 필터로 변환합니다.
  - **Judge**: 검색된 후보자들의 실적 증빙 자료를 대조하여 최종 추천 사유와 적합도를 생성합니다.
- **실시간 추적 및 가시성 (Observability)**
  - 모든 요청에 Trace ID를 부여하여 플래닝부터 최종 판단까지 전 과정을 모니터링할 수 있습니다.
  - Playground UI를 통해 실시간 서버 로그와 추천 과정을 한눈에 확인할 수 있습니다.

## 시작 가이드

### 1. 환경 준비 및 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### 2. 환경 변수 설정
루트 디렉토리에 `.env` 파일을 생성하고 필수 정보를 입력합니다. 상세 설정은 [환경 변수 가이드](apps/docs/operation/ENVIRONMENT.md)를 참고하세요.
```env
NTIS_APP_ENV=local
NTIS_QDRANT_URL=http://localhost:6333
NTIS_LLM_BASE_URL=https://api.openai.com/v1
NTIS_LLM_API_KEY=your-api-key
```

기본 로컬 모델 경로는 Dense 용 `multilingual-e5-large-instruct`, Sparse 용 `models/PIXIE-Splade-v1.0` 입니다. Sparse backend 는 `로컬 PIXIE -> online telepix/PIXIE-Splade-v1.0 -> Qdrant/bm25` 순서로 fallback 하며, `Qdrant/bm25` 는 FastEmbed/Qdrant builtin sparse 경로를 사용합니다. `NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true` 이면 online PIXIE 단계는 건너뜁니다. `ntis-validate-live` 와 readiness 검증도 같은 resolver 를 사용하므로, bm25 fallback 시 sparse modifier 기대값은 `IDF` 입니다.

### 3. 서버 실행
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload
```

## 문서 안내 (Documentation)

모든 상세 문서는 `apps/docs/` 디렉토리에 체계적으로 정리되어 있습니다. [**전체 문서 인덱스**](apps/docs/INDEX.md)를 먼저 확인해 보세요.

### 📁 [API 및 데이터 규약](apps/docs/api/)
- [API 명세서](apps/docs/api/API_SPECIFICATION.md): 엔드포인트 및 입출력 규격
- [데이터 규약](apps/docs/api/DATA_CONTRACT.md): 내부 컴포넌트 간 데이터 교환 형식

### 📁 [시스템 설계](apps/docs/architecture/)
- [서비스 동작 흐름](apps/docs/architecture/SERVICE_FLOW.md): 추천 시스템의 전체 실행 단계
- [설계 지침서](apps/docs/architecture/DESIGN_GUIDELINES.md): 핵심 원칙 및 제약 사항

### 📁 [운영 및 설정](apps/docs/operation/)
- [환경 변수 설정](apps/docs/operation/ENVIRONMENT.md): 시스템 구동을 위한 설정값 목록
- [운영 가이드 (Runbook)](apps/docs/operation/RUNBOOK.md): 배포 및 상태 점검 절차
- [검증 시나리오](apps/docs/operation/GOLDEN_TESTS.md): 품질 확인을 위한 테스트 케이스

---

## 기술 스택
- **Backend**: FastAPI
- **Vector DB**: Qdrant
- **LLM**: OpenAI 호환 API (Llama 3 등)
- **Data**: Pydantic v2
- **Data**: Pydantic v2

## Recommendation Runtime Notes (2026-04-15)

- Reason generation now runs in sequential batches of up to `5` candidates.
- Candidate-internal relevant evidence is expanded to up to `10` papers, `10` projects, and `10` patents before LLM selection.
- The reasoner now prefers `tool calling` for structured batch output, retries once with a smaller JSON payload, and then falls back to deterministic server-side reasons if the LLM output is unusable.
- Prompt payloads are budget-trimmed. `relevant_*` remains the direct grounding set, while `all_*`, retrieval grounding, and evaluation activities are sent only as compact supporting context.
- No new environment variables or operator runbook steps were introduced by this change set.
- Detailed runtime contract: [apps/docs/api/REASONER_RUNTIME_POLICY.md](apps/docs/api/REASONER_RUNTIME_POLICY.md)
