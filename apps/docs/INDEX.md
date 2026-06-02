# 문서 인덱스 (Documentation Index)

**문서 세트 버전:** v2.1 (flat chunk payload 정렬, 2026-06-02)

이 문서는 `apps/docs/`의 전체 구조와 읽기 순서를 안내한다. v2.0은 데이터 저장 단위를 **연구자 1 Point → chunk 1 Point**로 전환한 재설계이며, 2026-06-02 **flat payload 정렬(v2.1)** 로 실제 적재 데이터(flat chunk payload)에 맞춰 전체 문서를 정정했다. 가장 먼저 [`architecture/DATA_MODEL.md`](architecture/DATA_MODEL.md)를 읽으면 나머지 문서의 전제가 잡힌다.

> **2026-06-02 flat 정렬 요약:** 실제 적재 데이터는 **flat chunk payload**다. ① doc_type은 **5종**(paper / patent / project / assessor_activity / specialty), ② 연구자 공통 메타는 payload **root에 평탄화**(`researcher_meta` 중첩 없음), doc_type별 상세만 `doc_attrs`(구 `domain_attrs`), ③ 단일 날짜 `doc_date`(구 `event_date`/`event_year`), ④ named vector는 `vector_e5i`(dense) + `vector_splade`(sparse), doc_type은 **payload 필터**. 적재(ingestion)는 **외부 제공자 소관**이며, 구 `apps/ingest`는 `legacy_v1x/ingest/`로 격리됨.

## 퀵 링크 (Quick Reference)

| 목적 | 문서 |
|---|---|
| **데이터 모델(chunk 스키마)** ⭐ | [`architecture/DATA_MODEL.md`](architecture/DATA_MODEL.md) |
| **설계 원칙 및 제약** | [`architecture/DESIGN_GUIDELINES.md`](architecture/DESIGN_GUIDELINES.md) |
| **시스템 동작 원리** | [`architecture/SERVICE_FLOW.md`](architecture/SERVICE_FLOW.md) |
| **API 상세 규격** | [`api/API_SPECIFICATION.md`](api/API_SPECIFICATION.md) |
| **내부 데이터 계약** | [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) |
| **외부 응답 변경 이력** | [`api/EXTERNAL_API_CHANGELOG.md`](api/EXTERNAL_API_CHANGELOG.md) |
| **리즈너 런타임 정책** | [`api/REASONER_RUNTIME_POLICY.md`](api/REASONER_RUNTIME_POLICY.md) |
| **환경 설정 및 변수** | [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) |
| **서버 실행 및 운영** | [`operation/RUNBOOK.md`](operation/RUNBOOK.md) |
| **검증 시나리오** | [`operation/GOLDEN_TESTS.md`](operation/GOLDEN_TESTS.md) |
| **의사결정 기록(ADR)** | [`architecture/ADR/`](architecture/ADR/) |
| **전환 계획** | [`plans/MIGRATION_PLAN.md`](plans/MIGRATION_PLAN.md) |
| **원천 payload 규격** | [`전체 샘플 payload (도메인별).txt`](전체%20샘플%20payload%20(도메인별).txt) |

---

## 역할별 읽기 가이드

### 🚀 신규 개발자 (Onboarding)
1. [`architecture/DATA_MODEL.md`](architecture/DATA_MODEL.md) — chunk 스키마(전제)
2. [`architecture/SERVICE_FLOW.md`](architecture/SERVICE_FLOW.md) — 전체 흐름
3. [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) — 단계 간 계약
4. [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) → [`operation/RUNBOOK.md`](operation/RUNBOOK.md) — 설정·실행

### 🛠 API 연동 개발자
1. [`api/API_SPECIFICATION.md`](api/API_SPECIFICATION.md) — 엔드포인트·필드
2. [`api/EXTERNAL_API_CHANGELOG.md`](api/EXTERNAL_API_CHANGELOG.md) — **v2.0 breaking change**
3. [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) — 응답 구조 배경

### ⚙️ 시스템 운영자
1. [`operation/RUNBOOK.md`](operation/RUNBOOK.md) — 배포·점검(신규 컬렉션 readiness)
2. [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) — 환경 변수
3. [`operation/GOLDEN_TESTS.md`](operation/GOLDEN_TESTS.md) — 품질 검증

### 🧭 설계 의사결정 추적
- [`architecture/DESIGN_GUIDELINES.md`](architecture/DESIGN_GUIDELINES.md) (v2.0) + ADR 0002~0005

---

## 주요 문서 분류

### 1. 설계 및 구조 (`architecture/`)
- `DATA_MODEL.md` — ⭐ chunk 컬렉션 스키마(5 doc_type, flat payload, 단일 벡터, 인덱스)
- `DESIGN_GUIDELINES.md` — 고정 설계 원칙·제약 (v2.0)
- `SERVICE_FLOW.md` — 런타임 흐름 + trace 필드
- `ADR/` — 의사결정 기록
  - `0002-chunk-level-point-model.md` — chunk 단위 Point 전환
  - `0003-all-doc-types-searchable.md` — 모든 doc_type 항상 검색
  - `0004-chunk-id-evidence-contract.md` — evidence id를 chunk_id로
  - `0005-fusion-and-reranker-reconsidered.md` — 융합·가중·리랭커 재검토 결정

### 2. API (`api/`)
- `API_SPECIFICATION.md`, `DATA_CONTRACT.md`, `EXTERNAL_API_CHANGELOG.md`, `REASONER_RUNTIME_POLICY.md`

### 3. 운영 및 설정 (`operation/`)
- `ENVIRONMENT.md`, `RUNBOOK.md`, `GOLDEN_TESTS.md`

### 4. 관리 및 계획 (`plans/`)
- `MIGRATION_PLAN.md` — v1.x→v2.0 전환 로드맵 (2026-06-02 flat 정렬로 일부 supersede — 문서 상단 주석 참조)

### 5. 원천 규격 (루트)
- `전체 샘플 payload (도메인별).txt` — 적재 payload 권위 규격(도메인별 예시)
