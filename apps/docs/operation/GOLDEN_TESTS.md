# Golden Tests

## Scenarios

### 1. 메타어 제거

- 입력: `AI 반도체 평가위원 추천`
- 기대값:
  - `평가위원`, `추천`은 `removed_meta_terms`로만 추적된다.
  - `retrieval_core`와 `must_aspects`에는 `AI 반도체`만 남는다.
  - raw query의 메타어가 branch query에 재주입되지 않는다.

### 2. planner fallback의 안전한 broad search

- 입력: 메타어를 제거하면 검색어가 거의 남지 않는 질의
- 기대값:
  - planner는 broad fallback으로 내려갈 수 있다.
  - fallback에서도 제거된 메타어를 다시 검색어로 쓰지 않는다.
  - 남은 토큰이 없으면 retrieval skip으로 끝난다.

### 3. 직접 매치 evidence만 선택

- 입력: 최신이지만 무관한 실적과 오래됐지만 직접 매치되는 실적이 함께 있는 후보
- 기대값:
  - selector는 직접 매치 evidence만 relevant pool에 넣는다.
  - 무관한 최신 실적은 recommendation evidence에 들어가지 않는다.

### 4. evidence dedup과 quota

- 입력: 같은 제목과 같은 연도의 중복 project/paper
- 기대값:
  - dedup key `title + year` 기준으로 중복 제거
  - aspect별 최대 2건
  - 후보당 전체 최대 4건

### 5. shortlist gate

- 입력: direct evidence 0건 후보, low coverage 후보, generic-only 후보가 함께 있는 질의
- 기대값:
  - direct evidence 0건 후보는 drop
  - low coverage 후보는 keep 뒤로 밀림
  - generic-only 후보는 하단 고정
  - 점수 가중합 없이 gate 순서로만 재배치

### 6. `/search/candidates` vs `/recommend`

- 입력: 같은 질의에 대해 `/search/candidates`와 `/recommend` 각각 호출
- 기대값:
  - `/search/candidates`는 retrieval 순서만 보여준다.
  - `/recommend`는 evidence selector와 shortlist gate를 추가로 적용한다.

### 7. reason generator는 요약만 수행

- 입력: 후보별 `selected_evidence`가 이미 확정된 상태
- 기대값:
  - LLM payload에 `selected_evidence`, `do_not_mention`이 포함된다.
  - LLM tool/json schema는 `selected_evidence_ids`를 요구하지 않는다.
  - 최종 `recommendation.evidence`는 서버가 확정한 evidence를 사용한다.

### 8. empty reason fallback

- 입력: LLM이 후보를 반환했지만 `recommendation_reason`을 비운 경우
- 기대값:
  - 서버가 `selected_evidence` 기반 fallback 사유를 생성한다.
  - `server_fallback_reasons.source == "selected_evidence"`가 trace에 남는다.

### 9. reason sync validator

- 입력: 추천 사유에 다른 후보 이름, 내부 evidence id, 무근거 강한 표현이 포함된 경우
- 기대값:
  - validator가 이를 탐지한다.
  - 사유는 서버 fallback으로 치환된다.
  - `reason_sync_validator.fallback_count`와 violation 목록이 trace에 남는다.

### 10. batched reason generation

- 입력: gate를 통과한 최종 shortlist가 6명 이상인 질의
- 기대값:
  - reason generation은 최대 5명씩 순차 batch로 실행된다.
  - 최종 추천 순서는 shortlist 순서를 유지한다.
  - batch trace에 candidate ids가 남는다.

### 11. retrieval skipped on empty keywords

- 입력: planner가 메타어 제거 후 안전한 검색어를 만들지 못하는 질의
- 기대값:
  - retrieval이 스킵된다.
  - `retrieval_skipped_reason`이 trace에 남는다.
  - `recommendations=[]`

## Acceptance Criteria

- 검색어는 `retrieval_core`를 우선 사용하고, 없을 때만 `core_keywords`를 사용한다.
- `평가위원`, `전문가`, `추천` 같은 메타어는 검색어에 남지 않는다.
- evidence selector는 직접 매치 evidence만 선택한다.
- evidence 선택은 dedup과 quota 규칙을 따른다.
- `/recommend`는 retrieval 후보 전체에 먼저 evidence selector를 적용한다.
- shortlist는 deterministic gate로만 재배치된다.
- reason generator는 요약만 수행하고 evidence를 다시 고르지 않는다.
- empty reason과 validator failure는 서버 fallback으로 치환된다.
- trace는 `evidence_selection`, `shortlist_gate`, `selected_evidence`, `server_fallback_reasons`, `reason_sync_validator`를 포함한다.
- 운영 로그에서 planner, evidence selector, shortlist gate, validator fallback을 추적할 수 있다.
## 2026-04-17 additions

- Contextual evaluation phrase case: `AI 기반 의료영상 과제를 평가할 수 있는 전문가를 추천해 주세요`
  - `평가위원`, `전문가`, `추천` are still stripped as meta terms.
  - `과제 평가` remains in `must_aspects`.
- Equal-weight retrieval case:
  - branch/path weights are absent from query payload and trace.
  - expanded search is executed only for distinct expanded text.
- Partial batch retry case:
  - the first reasoner batch may return partial output.
  - one compact retry is executed before server fallback.
