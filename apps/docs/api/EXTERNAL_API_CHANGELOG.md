# 외부 노출 리턴 API 명세 변경기록

## 범위

- 대상 엔드포인트: `/recommend`, `/search/candidates`, `/feedback`, `/health`, `/health/ready`
- 포함 범위: HTTP 상태 코드, 응답 본문 필드, 하위 객체 필드, `trace` 계약, 하위 호환 alias
- 제외 범위: 내부 구현 변경만 있는 커밋, 요청 파라미터만 바뀐 커밋

본 문서는 `git log`와 실제 diff를 기준으로 외부 소비자가 체감하는 응답 계약 변경만 추렸습니다.

## 타임라인

| 날짜 | 커밋 | 영향도 | 변경 내용 |
|---|---|---|---|
| 2026-03-31 | `c4a3cfa` | 기준선 | 공개 응답 스키마 초안이 추가되었습니다. `/recommend`는 `intent_summary`, `applied_filters`, `searched_branches`, `retrieved_count`, `recommendations`, `data_gaps`, `not_selected_reasons`, `trace` 구조를 갖고, `/search/candidates`는 `candidates[*].branch_coverage`를 사용했습니다. |
| 2026-04-06 | `368cb28` | 동작 변경 | `/recommend`가 더 이상 "후보 없음" 또는 "evidence 없음" 때문에 `500`을 내지 않고, `200 OK`와 함께 `recommendations=[]`를 반환하도록 바뀌었습니다. 빈 결과 사유는 `not_selected_reasons`에 남기도록 정리되었습니다. |
| 2026-04-14 | `9b71535` | 경미 | 응답 최상위 스키마 변경은 없지만, 검색/추천 결과의 `trace.include_orgs`가 노출되기 시작했습니다. 같은 커밋에서 `top_k` 상한이 `5 -> 15`로 바뀌었지만 이는 요청 계약 변경입니다. |
| 2026-04-14 | `79328b9` | 확장 | `/search/candidates.trace`에 `planner_trace`, `raw_query`, `retrieval_keywords`, `planner_retry_count`, `retrieval_skipped_reason`, `include_orgs`, `timers`가 추가되었습니다. 디버그 UI 의존성이 있다면 이 시점부터 새 필드를 읽을 수 있습니다. |
| 2026-04-15 | `b9084c4` | 주요 변경 | `/recommend.recommendations[*]`의 핵심 사유 필드가 `reasons: list[str]`에서 `recommendation_reason: str`로 바뀌었습니다. 동시에 `trace.judge_trace`, `trace.evidence_resolution_trace`, `trace.recommendation_evidence_summary`, `trace.final_reduce_*`, `trace.planner_raw_keywords`, `trace.verifier_*`가 제거되고, `trace.reason_generation_trace`, `trace.planner_keywords`, `trace.recommendation_ids`, `trace.final_sort_policy`, `trace.top_k_used`가 추가되었습니다. 또한 `/search/candidates`는 이 시점부터 요청의 `top_k`를 실제 응답 개수 제한에 반영합니다. |
| 2026-04-15 | `8c4b8e7` | 하위 호환 | 구 프런트 호환을 위해 `/recommend.recommendations[*].reasons`가 계산 필드 alias로 다시 제공되었습니다. 기본 계약은 여전히 `recommendation_reason`이며, `reasons`는 구형 소비자 보호용입니다. 같은 시점에 `trace.reason_generation_trace.evidence_selection`이 중첩 추적으로 추가되었습니다. |
| 2026-04-15 | `3c9bceb` | 주요 변경 | `/search/candidates.candidates[*].branch_coverage`가 `branch_presence_flags`로 이름이 바뀌었습니다. `/search/candidates.trace`에는 `retrieval_score_traces`와 `final_sort_policy`가 추가되어, 최종 점수와 branch 매칭 근거를 외부에서 직접 볼 수 있게 되었습니다. |

## 변경 상세

### 1. 빈 추천 결과의 상태 코드 정책 변경

- 변경 전: 일부 empty-result 케이스가 서버 오류로 처리될 수 있었습니다.
- 변경 후: `/recommend`는 빈 결과도 정상 응답으로 간주하며 `200 OK`를 유지합니다.
- 소비자 영향: 프런트는 "추천 없음"을 예외가 아니라 빈 리스트 상태로 렌더링해야 합니다.

### 2. `/recommend` 응답 객체의 의미 체계 변경

- 구 계약: `recommendations[*].reasons` 중심
- 신 계약: `recommendations[*].recommendation_reason` 중심
- 호환 계층: `8c4b8e7`부터 `reasons` alias 복원
- 소비자 영향: 신규 연동은 `recommendation_reason`를 1순위로 읽고, 구형 화면만 `reasons`를 fallback으로 쓰는 편이 안전합니다.

### 3. `/search/candidates` 후보 카드 필드명 변경

- 변경 전: `branch_coverage`
- 변경 후: `branch_presence_flags`
- 소비자 영향: 구형 프런트가 `branch_coverage`를 직접 참조하면 2026-04-15 이후 깨집니다.

### 4. `trace`는 디버그 계약이며, 2026-04-15에 한 번 크게 재편됨

- 제거된 축: `judge_trace`, `evidence_resolution_trace`, `final_reduce_*`, `verifier_*`
- 추가된 축: `reason_generation_trace`, `retrieval_score_traces`, `recommendation_ids`, `final_sort_policy`, `top_k_used`
- 소비자 영향: 운영 UI가 `trace`를 강하게 파싱한다면, 이 변경점을 기준으로 버전 분기 또는 방어 코드를 둬야 합니다.

## 현재 기준 호환성 메모

- `/recommend`의 1차 계약 필드는 `recommendation_reason`입니다.
- `/recommend.recommendations[*].reasons`는 하위 호환 alias로만 취급하는 것이 맞습니다.
- `/search/candidates`의 branch 상태 필드는 `branch_presence_flags`입니다.
- `/recommend`와 `/search/candidates` 모두 `trace`는 존재하지만, 디버그 목적 필드이므로 안정성이 top-level 계약보다 낮습니다.

## 근거 파일

- `apps/api/schemas.py`
- `apps/api/main.py`
- `apps/recommendation/service.py`
- `apps/domain/models.py`
- `apps/docs/api/API_SPECIFICATION.md`
- `apps/docs/api/DATA_CONTRACT.md`
- `apps/docs/api/REASONER_RUNTIME_POLICY.md`
- `apps/docs/operation/GOLDEN_TESTS.md`
