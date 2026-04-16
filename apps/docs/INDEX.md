# 문서 인덱스 (Documentation Index)

이 문서는 `docs/` 폴더의 전체 문서 구조와 읽기 순서를 안내합니다.

## 퀵 링크 (Quick Reference)

| 목적 | 문서 |
|---|---|
| **시스템 동작 원리 이해** | [`architecture/SERVICE_FLOW.md`](architecture/SERVICE_FLOW.md) |
| **API 상세 규격 확인** | [`api/API_SPECIFICATION.md`](api/API_SPECIFICATION.md) |
| **데이터 입출력 규약** | [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) |
| **외부 응답 변경 이력** | [`api/EXTERNAL_API_CHANGELOG.md`](api/EXTERNAL_API_CHANGELOG.md) |
| **환경 설정 및 변수** | [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) |
| **서버 실행 및 운영** | [`operation/RUNBOOK.md`](operation/RUNBOOK.md) |
| **검증 시나리오** | [`operation/GOLDEN_TESTS.md`](operation/GOLDEN_TESTS.md) |
| **설계 원칙 및 제약** | [`architecture/DESIGN_GUIDELINES.md`](architecture/DESIGN_GUIDELINES.md) |
| **의사결정 기록** | [`architecture/ADR/`](architecture/ADR/) |

---

## 역할별 읽기 가이드

### 🚀 신규 개발자 (Onboarding)
1. [`architecture/SERVICE_FLOW.md`](architecture/SERVICE_FLOW.md) — 전체 흐름 파악
2. [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) — 데이터 규약 이해
3. [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) — 로컬 설정
4. [`operation/RUNBOOK.md`](operation/RUNBOOK.md) — 서버 실행

### 🛠 API 연동 개발자
1. [`api/API_SPECIFICATION.md`](api/API_SPECIFICATION.md) — 엔드포인트 및 필드 설명
2. [`api/DATA_CONTRACT.md`](api/DATA_CONTRACT.md) — 응답 구조 상세
3. [`api/EXTERNAL_API_CHANGELOG.md`](api/EXTERNAL_API_CHANGELOG.md) — breaking change 및 호환 포인트

### ⚙️ 시스템 운영자
1. [`operation/RUNBOOK.md`](operation/RUNBOOK.md) — 배포 및 점검 절차
2. [`operation/ENVIRONMENT.md`](operation/ENVIRONMENT.md) — 환경 변수 최적화
3. [`operation/GOLDEN_TESTS.md`](operation/GOLDEN_TESTS.md) — 품질 검증

---

## 주요 문서 분류

### 1. API (`api/`)
- 외부 시스템과의 인터페이스 및 데이터 구조를 정의합니다.

### 2. 설계 및 구조 (`architecture/`)
- 시스템의 내부 동작 논리, 컴포넌트 간 상호작용, 설계 결정 근거를 포함합니다.

### 3. 운영 및 설정 (`operation/`)
- 설치, 실행, 설정 변경 및 문제 해결을 위한 지침을 제공합니다.

### 4. 관리 및 계획 (`plans/`)
- 프로젝트의 발전 방향, 리팩토링 계획 등을 기록합니다.
