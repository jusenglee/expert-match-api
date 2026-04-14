# 문서 인덱스

이 문서는 `docs/` 폴더의 전체 파일 목록과 독자별 읽기 순서를 안내합니다.

## 빠른 참조

| 목적 | 파일 |
|---|---|
| 시스템이 어떻게 동작하는지 이해 | [`SERVICE_FLOW.md`](SERVICE_FLOW.md) |
| API 엔드포인트 스펙 | [`API_SPECIFICATION.md`](API_SPECIFICATION.md) |
| 데이터 규약 (Planner/Judge 입출력) | [`CONTRACT.md`](CONTRACT.md) |
| 환경 변수 설정 | [`ENVIRONMENT.md`](ENVIRONMENT.md) |
| 서버 실행 · 상태 점검 · 운영 절차 | [`RUNBOOK.md`](RUNBOOK.md) |
| 테스트 시나리오 · 수락 기준 | [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md) |
| 설계 원칙 · 기술 제약 · 로드맵 | [`평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.2.md`](평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.2.md) |
| 아키텍처 의사결정 기록 | [`ADR/`](ADR/) |

---

## 독자별 읽기 순서

### 신규 개발자
1. 이 파일 (`INDEX.md`)
2. [`SERVICE_FLOW.md`](SERVICE_FLOW.md) — 전체 흐름 이해
3. [`CONTRACT.md`](CONTRACT.md) — Planner·Judge I/O 규약
4. [`ENVIRONMENT.md`](ENVIRONMENT.md) — 로컬 환경 설정
5. [`RUNBOOK.md`](RUNBOOK.md) — 서버 실행
6. [`지침서_v1.2.md`](평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.2.md) — 설계 원칙과 제약 이해

### API 연동 개발자
1. [`API_SPECIFICATION.md`](API_SPECIFICATION.md)
2. [`CONTRACT.md`](CONTRACT.md)

### 운영자
1. [`RUNBOOK.md`](RUNBOOK.md)
2. [`GOLDEN_TESTS.md`](GOLDEN_TESTS.md)
3. [`ENVIRONMENT.md`](ENVIRONMENT.md)

### 설계 검토 / 기획
1. [`지침서_v1.2.md`](평가위원_추천_RAG_챗봇_설계구현제약방향성_지침서_v1.2.md)
2. [`ADR/`](ADR/)

---

## 전체 파일 목록

| 파일 | 줄 수 | 역할 | 주요 독자 |
|---|---|---|---|
| `INDEX.md` | — | 이 파일. 문서 진입점 | 전체 |
| `SERVICE_FLOW.md` | ~400 | 서비스 동작 흐름 + 아키텍처 다이어그램 | 개발자 |
| `API_SPECIFICATION.md` | ~200 | API 엔드포인트 명세 | API 연동 |
| `CONTRACT.md` | ~190 | Planner·Judge I/O 규약, 필드 정의 | 개발자 |
| `ENVIRONMENT.md` | ~90 | 환경 변수 전체 목록 및 기본값 | 개발자·운영자 |
| `RUNBOOK.md` | ~100 | 설치·실행·상태점검·로깅 운영 절차 | 운영자 |
| `GOLDEN_TESTS.md` | ~65 | 테스트 시나리오 및 수락 기준 | QA·운영자 |
| `지침서_v1.2.md` | ~200 | 설계 원칙, 기술 제약, 로드맵 (확정 기준선) | 설계·기획·개발자 |
| `ADR/0001-all-branches-on.md` | ~20 | 브랜치 전체 검색 상시 켜기 결정 기록 | 설계 |
| `평가위원(전문가) 추천서비스 데이터 구조 정의_v0.3.xlsx` | — | 데이터 필드 원본 정의 (Excel) | 설계·데이터 |

> **규칙**: 코드와 직접 연결된 규약·스펙 변경 시에는 `CONTRACT.md`, `API_SPECIFICATION.md`, `ENVIRONMENT.md`를 함께 갱신합니다. 설계 원칙의 변경은 `지침서_v1.2.md`와 해당 `ADR`을 함께 업데이트합니다.
