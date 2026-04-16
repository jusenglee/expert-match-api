# Retrospective: 추천 파이프라인 시행착오 기록 (2026-04-06 ~ 2026-04-15)

## 기록 기준

- 이 문서는 `git log`와 `git show`로 확인한 주요 커밋의 실제 diff를 바탕으로 재구성한 회고다.
- 아래 해석은 커밋 메시지와 수정 파일을 근거로 한 추정이며, 당시 의사결정의 모든 맥락을 완전히 대변하지는 않는다.
- 기준으로 본 주요 커밋:
  - `368cb28`, `0d853b7`, `90e6187`, `44d5bab`, `86a346a`
  - `aa5ff87`, `a950760`
  - `9b71535`, `79328b9`
  - `d44586c`, `97d169b`, `78535ad`, `cc5e52f`, `916a4a9`
  - `b9084c4`, `8c4b8e7`, `3c9bceb`
- 실제 diff를 확인한 대표 파일:
  - `apps/recommendation/judge.py`
  - `apps/recommendation/evidence.py`
  - `apps/recommendation/evidence_selector.py`
  - `apps/recommendation/reasoner.py`
  - `apps/recommendation/service.py`
  - `apps/recommendation/planner.py`
  - `apps/search/query_builder.py`
  - `apps/search/retriever.py`
  - `apps/search/seed_runner.py`
  - `apps/search/seed_data.py`
  - `apps/api/schemas.py`

## 한 줄 요약

추천 품질과 설명력을 빠르게 끌어올리기 위해 `judge`, Map-Reduce, evidence resolver, inclusion/consistency 같은 레이어를 계속 얹었지만, 그 과정에서 책임 경계와 계약이 흔들렸다. 최종적으로는 `planner -> retrieval -> reasoner -> evidence_selector`로 다시 단순화하고, 마지막에는 `SeedRunner` 같은 deprecated 보조 실행기까지 제거하는 쪽이 더 안정적이었다.

## 타임라인

| 날짜 | 커밋 | 보인 움직임 | 해석 |
|---|---|---|---|
| 2026-04-06 | `368cb28`, `0d853b7`, `90e6187`, `44d5bab`, `86a346a` | 빈 후보 / 빈 evidence 방어, 프롬프트 수정, 구 파라미터 참조 수정, 확장 준비 | 기능 확장보다 먼저 런타임 안정화와 프롬프트-스키마 정합성 확보가 급했던 시기였다. 초기에 계약이 아직 흔들리고 있었다는 신호다. |
| 2026-04-08 | `aa5ff87`, `44ad2cb` | lightweight Map phase 도입, JSON 추출과 추천 처리 보강 | 속도와 필터링 품질을 올리기 위해 judge 축을 강화하기 시작했다. 다만 판단 로직이 한 레이어로 몰리기 시작했다. |
| 2026-04-09 | `a950760` | Map-Reduce judging 확장, Planner JSON 처리 개선 | planner 출력 안정화와 judge 개선이 동시에 묶였다. 이 시점부터 추천 품질 개선과 계약 보정이 강하게 결합되기 시작했다. |
| 2026-04-09 ~ 2026-04-10 | `d44586c`, `97d169b`, `78535ad` | 오래된 문서 삭제, 누락 문서 hotfix, 다시 outdated 문서 정리 | 코드 구조가 빠르게 바뀌는 동안 문서가 따라가지 못했다. 문서 드리프트가 이미 운영 리스크가 된 구간이다. |
| 2026-04-14 | `9b71535`, `79328b9` | inclusion/LLM consistency 기능 추가 후 deprecated 필드 제거, planner/query builder 정리 | 기능을 더 얹을수록 기존 호환 필드와 혼합 책임이 비용으로 드러났다. 결국 새 기능 추가 직후 구조 정리가 필요해졌다. |
| 2026-04-14 | `cc5e52f`, `916a4a9` | ADR, API, Contract, Environment, Golden Tests 전면 문서화 | 구조를 다시 설명 가능한 상태로 만들기 위해 문서 재작성까지 필요했다. "코드만 정리하면 끝"이 아니었다는 뜻이다. |
| 2026-04-15 | `b9084c4`, `8c4b8e7`, `3c9bceb` | evidence resolver 제거, `judge.py` 제거, `SeedRunner` 제거, trace/API/test 정리 | 최종적으로는 레이어를 더 두는 방향보다 역할을 줄이고 경계를 명확히 하는 방향이 채택되었다. |

## 코드 변경으로 확인한 사실

### 1. Map-Reduce judge는 실제로 `judge.py` 안에서 비대해졌다

`aa5ff87`와 `a950760`의 diff를 보면, `apps/recommendation/judge.py`에 아래가 직접 추가되었다.

- `_build_map_system_prompt()`
- `_serialize_shortlist_for_map()`
- `_parse_map_response()`
- `_invoke_non_stream_with_limit(..., max_tokens_hint=...)`

즉, judge는 단순 최종 판정기가 아니라, 경량 Map 프롬프트, survivor 압축, Reduce 전 게이트, JSON 파싱 보정까지 함께 담당하게 되었다. 특히 `a950760`에서는 Map phase의 `max_tokens_hint`가 `200 -> 400`으로 커졌는데, 이는 이미 중간 판정 단계가 가벼운 보조 로직을 넘어서 별도 튜닝 대상이 되었다는 신호다.

### 2. `include_orgs`와 consistency 보강은 planner와 retrieval 경계를 더 복잡하게 만들었다

`9b71535`의 실제 diff에서는 단순 옵션 추가가 아니라 경계 이동이 보인다.

- `apps/recommendation/planner.py`
  - `plan()` 시그니처가 `include_orgs`를 받도록 확장됨
  - `build_consistency_invoke_kwargs()` 적용
  - API 입력과 LLM 추출 기관 목록을 union으로 병합
- `apps/recommendation/service.py`
  - `search_candidates()`와 `recommend()`가 모두 `include_orgs`를 planner로 전달
- `apps/search/query_builder.py`
  - raw query 대신 `core_keywords` 기반 텍스트를 branch query의 바탕으로 사용
- `apps/search/retriever.py`
  - instruct 모델일 때 dense query prefix를 붙임
  - Python 레벨의 기관 post-filtering이 추가됨

즉, "조금 더 일관된 planner"를 만들려던 수정이 실제로는 planner, query builder, retriever가 동시에 움직이는 변경이었다. 회고에서 말한 "새 기능 추가와 계약 정리가 같은 축에서 섞였다"는 평가는 이 diff로 뒷받침된다.

### 3. 바로 다음 정리 커밋에서 deprecated 필드와 retrieval 보조 개념이 빠르게 걷혔다

`79328b9`의 diff에서는 전날 추가한 보조 개념이 빠르게 단순화되는 흐름이 보인다.

- `apps/search/query_builder.py`
  - `normalize_keywords()` 추가
  - branch별 다른 query를 조합하던 방식에서, 모든 branch에 동일한 `core_keywords` 텍스트를 넘기는 방식으로 축소
- `apps/recommendation/cards.py`
  - 키워드 매칭 점수와 최신성 가산점 기반 정렬 로직이 사라지고, 최신 실적 중심의 단순 정렬로 바뀜
- `apps/recommendation/service.py`
  - `planner_trace`, `retrieval_keywords`, `retrieval_skipped_reason`, `timers`를 응답 trace에 명시적으로 남김

커밋 메시지에 적힌 "deprecated fields 제거"가 실제로는 파일 몇 개 정리 수준이 아니라, 검색 텍스트 생성과 카드 조립 방식 자체를 더 단순한 계약으로 되돌리는 작업이었음을 확인할 수 있다.

### 4. `evidence.py`와 `judge.py` 제거는 이름만 바뀐 리팩터링이 아니라 책임 분리였다

`b9084c4`와 `8c4b8e7`의 diff를 보면 구조 전환이 분명하다.

- `apps/recommendation/evidence.py` 삭제
- `apps/recommendation/judge.py` 삭제
- `apps/recommendation/reasoner.py` 신설 및 확장
- `apps/recommendation/evidence_selector.py` 신설
- `apps/recommendation/service.py`
  - 생성자에서 `judge` 대신 `evidence_selector`, `reason_generator`를 받음
  - `judge_trace` / `evidence_resolution_trace` 대신 `reason_generation_trace`와 `evidence_selection` trace를 조립

특히 `8c4b8e7`에서는 `reasoner`가 `relevant_evidence_by_expert_id`를 직접 입력으로 받고, `service.py`가 추천 사유 생성 전 `evidence_selector.select()`를 호출하도록 바뀐다. 즉, 과거처럼 "judge가 everything"인 구조가 아니라, "증거 추림 -> 사유 생성 -> 최종 조립"으로 역할이 나뉜 것이다.

### 5. 최신 커밋 `3c9bceb`은 SeedRunner 제거로 끝난 것이 아니라, 남은 계약을 최신 구조에 맞췄다

`3c9bceb`의 diff를 보면 커밋 메시지보다 범위가 더 넓다.

- `apps/search/seed_runner.py` 삭제
- `apps/search/seed_data.py`
  - `record_from_payload()` 삭제
  - seed 경로가 `points_from_payload()` 중심으로 수렴
- `apps/search/retriever.py`
  - `RetrievalResult`에 `retrieval_score_traces` 추가
  - branch별 `query_points` 결과를 따로 모아 per-hit branch match trace 생성
  - 최종 정렬이 `score desc -> researcher name asc -> expert_id asc`로 고정
- `apps/recommendation/reasoner.py`
  - `selected_evidence_ids` 필드 추가
  - LLM 입력에 broader raw candidate context와 retrieval trace를 함께 실음
- `apps/recommendation/service.py`
  - selected evidence id 기준으로 실제 evidence를 조립
  - 빈 사유에 대한 server fallback reason과 trace를 남김
- `apps/api/schemas.py`
  - `branch_coverage` 대신 `branch_presence_flags` 등 현재 런타임 용어에 맞게 응답 스키마 정리

즉, 마지막 커밋은 단순히 오래된 실행기 하나를 지운 것이 아니라, seed 적재 경로, retrieval trace, reason/evidence 계약, API 스키마까지 현재 구조에 맞춰 마무리 정렬한 커밋이었다.

## 이번 시행착오에서 반복된 패턴

### 1. 한 레이어에 너무 많은 책임을 몰아넣었다

`judge.py` 축은 실제 diff 기준으로 한동안 다음을 동시에 떠안고 있었다고 볼 수 있다.

- 후보 판정
- Map/Reduce 단계 운영
- 설명 문장 생성 보조
- evidence 해석 또는 정렬 보정
- 프롬프트 변화에 따른 계약 적응

커밋 흐름상 이 구조는 기능을 빠르게 얹기에는 편했지만, 수정할 때마다 planner, service, tests, docs가 함께 흔들리는 비용을 만들었다. `judge.py`가 결국 제거된 것은 단순 삭제가 아니라 "책임을 한곳에 몰아넣는 설계가 오래 버티지 못했다"는 결과로 보는 편이 맞다.

### 2. 프롬프트 수정과 계약 정리가 서로 뒤엉켰다

2026-04-06에 프롬프트 수정과 "구 파라미터 참조 변경"이 연속 커밋으로 나타난다. 이는 LLM 동작을 바꾸는 작업과 런타임 계약 정리가 분리되지 않았다는 신호다.

이 패턴은 이후에도 반복된다.

- planner JSON 처리 개선
- inclusion / consistency 기능 추가
- deprecated 필드 제거

즉, "새 기능을 넣는 일"과 "이전 계약을 정리하는 일"이 같은 축에서 동시에 발생했다. 이런 구조에서는 새 기능이 들어올수록 오히려 정리 비용이 더 커진다.

### 3. 문서는 항상 한 박자 늦게 따라왔다

2026-04-09부터 2026-04-14까지의 문서 커밋 흐름은 꽤 명확하다.

- 오래된 문서를 제거했다.
- 누락된 문서를 hotfix로 다시 추가했다.
- 이후 전면 문서화를 두 번에 걸쳐 정리했다.

이건 단순히 문서를 게을리 썼다는 의미보다, 런타임 구조가 충분히 고정되기 전에 문서를 유지하려 했다는 신호에 가깝다. 결국 구조가 정리된 뒤에야 `CONTRACT.md`, `SERVICE_FLOW.md`, `GOLDEN_TESTS.md`, `API_SPECIFICATION.md`가 현재 상태를 설명할 수 있게 되었다.

### 4. deprecated 레이어를 오래 안고 갈수록 정리 비용이 커졌다

`79328b9`에서 deprecated 필드와 retrieval 보조 개념을 걷어냈고, 바로 다음 날 `b9084c4`, `8c4b8e7`, `3c9bceb`에서 evidence resolver, judge, SeedRunner까지 제거했다. 이 흐름은 호환성 레이어를 오래 유지하는 전략이 여기서는 큰 이득을 주지 못했다는 점을 보여준다.

이 저장소는 오히려 다음 방식이 더 맞았다.

- 입력 계약을 먼저 명확히 줄인다.
- 중간 레이어를 짧게 실험한다.
- 유효성이 입증되지 않으면 빨리 걷어낸다.

## 무엇이 최종적으로 남았나

현재 문서와 코드 기준으로 살아남은 구조는 아래와 같다.

1. planner는 `core_keywords`, `task_terms`, 명시 필터, `top_k`만 다룬다.
2. retrieval은 planner 키워드만 받아 branch 검색과 RRF를 수행하고, 현재는 `retrieval_score_traces`까지 남긴다.
3. `/recommend`는 retrieval 순서를 유지한 채 Top-k 후보만 reason generation 대상으로 넘긴다.
4. evidence는 후보 내부에서 재정렬하고, 선택은 `evidence_selector`가 담당한다.
5. LLM은 후보 재정렬이나 필터링을 하지 않고, 이유 생성과 evidence 선택에만 집중한다.
6. seed 적재 경로도 `SeedRunner`가 아니라 payload-to-points 중심으로 단순화되었다.

`apps/docs/SERVICE_FLOW.md`에도 현재 active runtime path에서 제거된 항목이 명시되어 있다.

- verifier stage
- retrieval views
- branch query hints
- judge map-reduce
- evidence resolver alignment stage

즉, 이번 시행착오의 결론은 "더 똑똑한 중간 단계 추가"보다 "단순한 단계 분리와 역할 고정"이 더 유지보수 가능하다는 점이다.

## 이번 기록에서 남길 교훈

### 교훈 1. 새 LLM 단계를 추가할 때는 책임 경계부터 먼저 문서화해야 한다

프롬프트를 먼저 바꾸고 나중에 계약을 따라가게 하면, 곧바로 deprecated 필드와 보정 로직이 쌓인다. 다음에는 기능 추가 전 아래 두 가지를 먼저 확정하는 편이 낫다.

- 이 단계의 입력은 무엇인가
- 이 단계가 절대 하지 말아야 할 일은 무엇인가

### 교훈 2. "설명 품질 향상"과 "검색 파이프라인 변경"은 같은 커밋 축에서 다루지 않는 편이 낫다

이번 커밋 흐름에서는 retrieval, planner, judge, evidence, docs가 함께 움직였다. 이렇게 되면 어느 개선이 실제 효과를 냈는지 분리하기 어렵다.

가능하면 다음처럼 나누는 편이 좋다.

- retrieval / ranking 변경
- reason generation 변경
- contract / docs 변경

### 교훈 3. deprecated 호환은 짧고 공격적으로 끝내야 한다

호환 레이어를 길게 유지하면 테스트도 이중으로 써야 하고, 문서도 이중 설명이 필요해진다. 이 저장소에서는 호환 유지보다 조기 제거가 더 적합했다.

### 교훈 4. 문서 재작성은 마무리 작업이 아니라 구조 고정의 검증 수단이다

이번에는 문서를 다시 쓰는 과정이 단순 정리가 아니라 "현재 구조를 말로 설명할 수 있는가"를 확인하는 검증 단계로 작동했다. 앞으로도 `SERVICE_FLOW.md`, `CONTRACT.md`, `GOLDEN_TESTS.md`를 같은 변경 세트에서 같이 보는 편이 맞다.

## 다음 변경에서 지킬 원칙

- planner, retrieval, reasoner, evidence selector 중 둘 이상이 동시에 바뀌면 `SERVICE_FLOW.md`와 `CONTRACT.md`를 같은 커밋에서 같이 갱신한다.
- 프롬프트 수정 커밋에는 구 파라미터 호환 로직을 늘리기보다 삭제 가능한 계약 정리를 우선한다.
- 새 중간 레이어를 도입할 때는 "언제 제거할지" 조건까지 미리 적는다.
- 문서 hotfix가 나오기 시작하면 구조가 아직 덜 고정된 것으로 보고, 기능 추가보다 경계 정리를 먼저 한다.
