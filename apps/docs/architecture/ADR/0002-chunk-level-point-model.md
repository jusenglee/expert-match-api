# ADR 0002: chunk 단위 Point 모델로 전환

## 상태

승인 (2026-05-28). 이전 기준 "연구자 1명 = 1 Point + nested 배열"을 대체한다.

## 맥락

적재 데이터가 `chunk_id`/`doc_id`/`doc_type`/`chunk_text` 단위(11 doc_type, 3층 payload)로 도착하도록 바뀌었다. 원천 규격은 [`../../전체 샘플 payload (도메인별).txt`](../../전체%20샘플%20payload%20(도메인별).txt). 기존 모델은 연구자 1명을 1 Point로 압축하고 논문/특허/과제를 nested 배열로 담았으며, 표현은 4브랜치 named vector였다.

이 구조는 chunk 단위 데이터와 두 가지로 충돌했다.
1. 도메인별 텍스트를 브랜치 벡터 하나로 합치면(예: 논문 전체를 art_vector 하나로) 개별 실적의 의미가 뭉개진다.
2. evidence가 Point(연구자)와 분리된 배열 인덱스(`paper:0`)로 참조되어 grounding·추적이 약하다.

## 결정

- 저장 단위를 **chunk 1개 = Point 1개**로 전환한다. Point ID = `chunk_id`.
- named vector는 **단일 `dense_e5i` + 단일 `sparse_splade`**(입력 `chunk_text`). doc_type은 payload 필터.
- 연구자 후보는 **검색 시점에 `researcher_id`로 집계**해서 만든다(RRF 누적 + doc_type별 캡 + dedupe).
- `researcher_meta`를 모든 chunk에 비정규화 저장해 chunk 단계에서 연구자 hard filter를 가능하게 한다.

상세 스키마는 [`../DATA_MODEL.md`](../DATA_MODEL.md).

## 결과

- 멱등 적재(중복 Point 방지), evidence 참조 안정성(`chunk_id` 불변), 의미 검색·evidence 선별의 자연스러움 확보.
- 비용: 연구자 집계 로직 신규 필요, `researcher_meta` 적재 일관성 책임이 적재 파이프라인으로 이동.
- 후속: 컬렉션 기본값 `ntis_researcher_chunks`, 외부 API 필드 변경([`../../api/EXTERNAL_API_CHANGELOG.md`](../../api/EXTERNAL_API_CHANGELOG.md)), 전환 계획([`../../plans/MIGRATION_PLAN.md`](../../plans/MIGRATION_PLAN.md)).
