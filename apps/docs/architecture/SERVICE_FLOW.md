# 서비스 동작 흐름 (Service Flow)

## 런타임 흐름

### 1. Planner

`RecommendationService.search_candidates()`는 planner부터 시작한다.

Planner의 현재 책임은 다음과 같다.

- 자연어 질의를 구조화된 `PlannerOutput`으로 정규화한다.
- 검색용 핵심어는 `retrieval_core`에 넣는다.
- 질의 충족에 필요한 aspect는 `must_aspects`에 넣는다.
- 범용 요청어는 `generic_terms`에 넣는다.
- `평가위원`, `전문가`, `추천` 같은 메타 요청어는 `role_terms` 또는 `action_terms`로만 남기고 검색어에서는 제거한다.
- `core_keywords`는 하위 호환 alias로 유지하되, 실제 검색 기준은 `retrieval_core`다.

Planner가 하지 않는 일은 다음과 같다.

- raw query를 그대로 검색어로 재사용하는 일
- branch별 가중치를 계산하는 일
- 후보 순위를 재조정하는 일

### 2. Query Builder / Retrieval

`QueryTextBuilder`는 다음 순서로 branch query의 안정 경로를 만든다.

1. `retrieval_core`
2. `core_keywords`
3. raw query

즉, planner가 정제한 검색어가 있으면 raw query의 메타어를 다시 검색에 넣지 않는다.

`QdrantHybridRetriever`는 다음을 수행한다.

- `basic`, `art`, `pat`, `pjt` 4개 branch에서 dense + sparse 검색
- branch 내부 RRF 결합
- branch 간 RRF 결합
- `retrieval_score_traces`에 branch별 매칭 근거 기록

### 3. `/search/candidates`

`/search/candidates`는 retrieval 결과를 `CandidateCard`로 직렬화해서 반환한다.

- `top_k`가 있으면 그 개수만큼만 잘라서 반환한다.
- `top_k`가 없으면 retrieval 순서를 유지한 전체 후보를 반환한다.
- 이 단계에서는 evidence selector, shortlist gate, reason generator를 호출하지 않는다.

### 4. `/recommend`

`/recommend`는 search 결과 위에 증거 기반 단계를 추가한다.

#### 4-1. 전체 후보 evidence selection

`search_candidates(..., limit_candidates=False)`로 retrieval 후보 전체를 먼저 확보한다.

그 다음 `KeywordEvidenceSelector`가 후보별로 다음 규칙을 적용한다.

- 직접 매치 evidence만 relevant pool에 포함
- dedup 기준은 `normalized title + year`
- aspect별 1건 우선 선발
- aspect별 최대 2건
- 후보당 전체 evidence 최대 4건

selector는 후보별로 아래 진단값을 남긴다.

- `direct_match_count`
- `aspect_coverage`
- `matched_aspects`
- `matched_generic_terms`
- `generic_only`
- `dedup_dropped_count`

#### 4-2. deterministic shortlist gate

reason generation 전에 서버가 후보를 다시 분류한다.

- gate 1: `direct_match_count == 0` 이면 drop
- gate 2: `aspect_coverage < min(2, len(must_aspects))` 이면 상위 진입 금지
- gate 3: `generic_only == true` 이면 하단 고정

정렬 규칙은 점수 가중합이 아니라 순차 gate다. keep 후보가 먼저 오고, 그 뒤에 low coverage, 마지막에 generic only가 온다.

#### 4-3. reason generation

reason generator는 최대 5명씩 순차 batch로 실행된다.

LLM의 현재 역할은 판단이 아니라 요약이다.

- evidence 선택은 서버가 이미 끝낸 상태다.
- LLM 입력에는 `selected_evidence`와 `do_not_mention`이 포함된다.
- LLM은 `fit`, `recommendation_reason`, `risks`만 생성한다.
- `selected_evidence_ids`는 더 이상 LLM 계약의 일부가 아니다.

#### 4-4. reason sync validator

LLM 출력 뒤에 deterministic validator가 한 번 더 적용된다.

차단 조건은 다음과 같다.

- 다른 후보 이름이 사유에 등장함
- 내부 evidence id 문자열이 사유에 노출됨
- 핵심 aspect가 evidence title에 전혀 닿지 않음
- direct evidence 없이 과한 추천 문장이 생성됨

검증 실패 시 서버 fallback 사유로 치환한다.

### 5. fallback 경로

추천 사유 fallback은 두 경로가 있다.

- `selected_evidence` 기반 fallback: LLM이 사유를 비우거나 후보를 누락한 경우
- `reason_sync_validator` fallback: LLM이 사유를 냈지만 검증을 통과하지 못한 경우

candidate 전체가 gate에서 탈락하면 `recommendations=[]`와 `No candidates satisfied deterministic evidence gates.`를 반환한다.

retrieval keyword가 planner 단계에서 비어 있으면 검색 자체를 건너뛰고 `retrieval_skipped_reason`을 trace에 남긴다.

## Trace와 로깅

### planner trace

- `removed_meta_terms`
- `must_aspects`
- `generic_terms`
- `fallback_terms`

### reason generation trace

- `evidence_selection`
- `shortlist_gate`
- `selected_evidence`
- `server_fallback_reasons`
- `reason_sync_validator`
- `batches`

### 운영 로그

INFO 로그

- planner 완료
- retrieval 완료
- evidence selector 후보별 요약
- shortlist gate 결과
- reason batch 시작/종료
- 최종 추천 완료

WARNING 로그

- retrieval skipped
- gate 이후 후보 0건
- empty reason fallback
- reason sync validator fallback
## 2026-04-17 flow updates

- Retrieval uses equal-weight RRF across branch/path results.
- Expanded queries are evaluated for every branch under one rule: run the expanded path only when its text differs from the stable path.
- Planner keeps contextual evaluation phrases such as `과제 평가`, `기술 평가`, `논문 평가` in `must_aspects` while still stripping meta request terms.
- Reason generation performs one compact retry when a batch returns missing candidates, empty reasons, or partial output.
- Selector and recommendation traces may expose `future_selected_evidence_ids` when future-dated project evidence is selected.
