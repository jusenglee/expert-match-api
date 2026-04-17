# NTIS 전문가 추천 시스템 프로토타입

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)
![Qdrant](https://img.shields.io/badge/Qdrant-1.10%2B-red)
![OpenAI](https://img.shields.io/badge/OpenAI-1.40%2B-lightgray)

NTIS 전문가 추천 시스템은 기본 정보, 논문, 특허, 과제 이력을 바탕으로 질의에 맞는 전문가 후보를 찾고 추천 사유를 생성하는 API다. 현재 런타임은 `planner -> retrieval -> evidence selector -> shortlist gate -> reasoner -> validator` 흐름으로 동작한다.

## 핵심 특징

- Multi-branch hybrid retrieval
  - `basic`, `art`, `pat`, `pjt` 4개 branch를 dense + sparse로 검색하고 RRF로 합친다.
- Meta-term safe planning
  - `평가위원`, `전문가`, `추천` 같은 메타어는 검색어에서 제거하고 trace로만 남긴다.
- Deterministic evidence selection
  - 직접 매치 evidence만 선택하고, `title + year` 기준 dedup과 aspect quota를 적용한다.
- Deterministic shortlist gate
  - direct evidence, aspect coverage, generic-only 여부를 기준으로 후보를 재배치한다.
- Summary-only reasoner
  - LLM은 evidence를 다시 고르지 않고, 서버가 고른 evidence만 요약한다.
- Reason sync validator
  - 다른 후보 이름, 내부 evidence id, 무근거 강한 문장을 탐지하면 서버 fallback 사유로 치환한다.
- Observability
  - planner, selector, shortlist gate, reasoner, validator 단계별 trace와 로그를 남긴다.

## 빠른 시작

### 1. 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

### 2. 환경 변수

`.env` 예시:

```env
NTIS_APP_ENV=local
NTIS_QDRANT_URL=http://localhost:6333
NTIS_LLM_BASE_URL=https://api.openai.com/v1
NTIS_LLM_API_KEY=your-api-key
```

기본 dense 모델 경로는 `multilingual-e5-large-instruct`다.

기본 sparse backend 순서는 다음과 같다.

1. 로컬 `models/PIXIE-Splade-v1.0`
2. online `telepix/PIXIE-Splade-v1.0`
3. `Qdrant/bm25` FastEmbed builtin fallback

`NTIS_HF_HUB_OFFLINE=true` 또는 `NTIS_SPARSE_LOCAL_FILES_ONLY=true`면 online PIXIE 단계는 건너뛴다.

### 3. 실행

```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload
```

## 현재 추천 런타임 정책

- Planner는 `retrieval_core`, `must_aspects`, `generic_terms`를 만든다.
- Query builder는 `retrieval_core -> core_keywords -> raw query` 순서로 검색어를 만든다.
- Evidence selector는 직접 매치 evidence만 후보별 최대 4건까지 고른다.
- Shortlist gate는 점수 가중합 없이 순차 규칙으로만 적용된다.
- Reason generator는 최대 5명씩 batch로 돌고, tool calling 1회 + compact retry 1회를 시도한다.
- Empty reason 또는 validator 실패 시 서버 fallback 사유를 사용한다.

## 문서

- [서비스 동작 흐름](apps/docs/architecture/SERVICE_FLOW.md)
- [데이터 계약](apps/docs/api/DATA_CONTRACT.md)
- [Reasoner Runtime Policy](apps/docs/api/REASONER_RUNTIME_POLICY.md)
- [Golden Tests](apps/docs/operation/GOLDEN_TESTS.md)
- [환경 변수 가이드](apps/docs/operation/ENVIRONMENT.md)
- [운영 Runbook](apps/docs/operation/RUNBOOK.md)
## 2026-04-17 runtime updates

- Retrieval now uses equal-weight RRF. Branch/path weights are not applied.
- Expanded queries follow one rule across all branches and run only when expanded text differs from stable text.
- Meta terms such as `평가위원`, `전문가`, `추천` stay out of retrieval keywords, but contextual phrases such as `과제 평가`, `기술 평가`, `논문 평가` stay in `must_aspects`.
- Reason generation performs one compact retry when a batch returns missing candidates, empty reasons, or any partial output.
- Future-dated project evidence is still allowed and is now surfaced in selector trace/logs via `future_selected_evidence_ids`.
