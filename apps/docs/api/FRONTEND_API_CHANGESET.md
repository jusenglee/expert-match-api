# 프론트엔드 전달용 API 변경사항 문서

기준일: 2026-06-15  
대상: 기존 프론트엔드 명세 대비 현재 `Ntis_person_API` 서버 응답/요청 변경사항  
주요 대상 엔드포인트: `POST /search/candidates`, `POST /recommend`, `POST /feedback`, `GET /health`, `GET /health/ready`

## 1. 변경 요약

현재 서버는 기존 `basic/art/pat/pjt` branch 기반 응답이 아니라 `paper/patent/project/assessor_activity/specialty` doc_type 기반 응답을 반환한다.

따라서 단순 값 변경만 있는 것이 아니라 다음 세 종류가 모두 있다.

| 구분 | 의미 | 대표 예 |
|---|---|---|
| 변경 | 필드는 유지되지만 값 또는 의미가 바뀜 | `searched_branches` |
| 삭제/대체 | 기존 필드가 더 이상 내려오지 않고 새 필드로 대체됨 | `branch_presence_flags` -> `doc_types_present` |
| 추가 | 기존 프론트 명세에는 없던 필드가 새로 내려옴 | `profile_evidence`, `fallback_reason_used`, `metadata` |

## 2. 요청 Request 변경사항

### 2.1 공통 요청 모델

적용 엔드포인트:

- `POST /search/candidates`
- `POST /recommend`
- `POST /recommend/stream` 단, 현재 프론트 사용 금지

### 2.2 학위 필터 키 변경

| 구분 | 기존 | 현재 |
|---|---|---|
| 변경 전 | `filters_override.degree_slct_nm` | 원천 데이터의 학위 선택명으로 보이던 기존 프론트 키 |
| 변경 후 | `filters_override.highest_degree` | 현재 서버가 사용하는 최종 학위 필터 키. 예: `"박사"` |

기존 요청:

```json
{
  "filters_override": {
    "degree_slct_nm": "박사"
  }
}
```

현재 요청:

```json
{
  "filters_override": {
    "highest_degree": "박사"
  }
}
```

프론트 영향:

- `degree_slct_nm`은 요청 body 스키마상 object 내부 값이라 422를 만들지 않을 수 있다.
- 하지만 현재 검색 필터 컴파일러가 기대하는 키가 아니므로 학위 필터가 적용되지 않을 수 있다.
- 학위 필터 UI는 반드시 `highest_degree`로 전송해야 한다.

### 2.3 `search_mode` 추가

| 필드 | 타입 | 의미 | 허용값 | 기본값 |
|---|---|---|---|---|
| `search_mode` | string | 검색 전략 선택값. 검색 엔진이 후보를 찾고 정렬하는 방식을 지정한다. | `multiview`, `hybrid`, `keyword_similarity` | `multiview` |

각 값의 의미:

| 값 | 설명 |
|---|---|
| `multiview` | 현재 기본 검색 방식. dense, sparse, concept view를 함께 사용해 후보를 RRF 방식으로 융합한다. |
| `hybrid` | dense 검색과 SPLADE sparse 검색을 단순 RRF 방식으로 결합한다. |
| `keyword_similarity` | SPLADE 키워드 검색으로 1차 후보를 만든 뒤 dense 유사도로 재정렬한다. |

프론트 영향:

- 기존 화면에서 검색 모드 선택 UI가 없다면 생략 가능하다.
- 잘못된 문자열을 보내면 422가 발생한다.

### 2.4 `include_orgs` 추가

| 필드 | 타입 | 의미 |
|---|---|---|
| `include_orgs` | string[] | 검색 결과에 포함할 기관명 목록. 서버는 연구자 root 필드 `affiliated_organization` 기준으로 후보를 후처리 필터링한다. |

예시:

```json
{
  "include_orgs": ["서울대학교", "한국과학기술원"]
}
```

프론트 영향:

- 기관 포함 필터 UI가 없다면 빈 배열 또는 생략 가능하다.
- `exclude_orgs`와 동시에 보낼 수 있으나 UX상 충돌 조건을 만들지 않는 것이 좋다.

### 2.5 `top_k` 제한 강화

| 필드 | 기존 기대 | 현재 서버 |
|---|---|---|
| `top_k` | 5 등 임의 숫자 사용 | 1~15만 허용. 16 이상이면 422 |

프론트 영향:

- 페이지 크기 또는 후보 수 선택 UI는 최대 15로 제한해야 한다.
- 미전송 시 서버가 planner 값 또는 최대 15를 사용한다.

## 3. `/search/candidates` 응답 변경사항

## 3.1 최상위 필드 변경

### `searched_branches`

| 항목 | 기존 | 현재 |
|---|---|---|
| 필드 존재 여부 | 존재 | 존재 |
| 타입 | string[] | string[] |
| 기존 의미 | 검색 대상 branch 목록. 예: `basic`, `art`, `pat`, `pjt` |
| 현재 의미 | 검색 대상 doc_type 목록. 예: `paper`, `patent`, `project`, `assessor_activity`, `specialty` |
| 프론트 영향 | 필드명만 보고 branch로 해석하면 라벨/아이콘/탭 매핑이 잘못된다. |

기존:

```json
{
  "searched_branches": ["basic", "art", "pat", "pjt"]
}
```

현재:

```json
{
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"]
}
```

권장 화면 라벨:

| 현재 값 | 권장 라벨 |
|---|---|
| `paper` | 논문 |
| `patent` | 특허 |
| `project` | 과제 |
| `assessor_activity` | 평가/심사 활동 |
| `specialty` | 전문분야 |

### `keywords`

| 항목 | 내용 |
|---|---|
| 변경 여부 | 유지 |
| 타입 | string[] |
| 의미 | 검색 단계에서 실제 사용된 핵심 키워드 목록. planner가 질의에서 추출한 기술/도메인 중심 키워드다. |
| 프론트 사용 | 검색 결과 상단의 “검색 키워드” 표시 또는 디버그 패널에 사용 가능 |

### `retrieved_count`

| 항목 | 내용 |
|---|---|
| 변경 여부 | 유지 |
| 타입 | integer |
| 의미 | 필터와 검색 집계 후 서버 내부에서 랭킹된 전체 후보 수. 화면에 실제 표시되는 `candidates.length`와 다를 수 있다. |
| 프론트 사용 | “총 N명 검색됨” 표시 가능 |

### `candidates`

| 항목 | 내용 |
|---|---|
| 변경 여부 | 유지, 단 후보 객체 내부 구조 변경 |
| 타입 | object[] |
| 의미 | 추천 사유 생성 전 단계의 초기 후보군 목록. `shortlist_score` 기준으로 정렬된 사용자 노출 후보 목록이다. |
| 프론트 사용 | 후보 목록 카드/테이블의 주 데이터 |

## 3.2 `candidates[*]` 필드 변경

### 유지 필드

#### `expert_id`

| 항목 | 내용 |
|---|---|
| 타입 | string |
| 의미 | 전문가 또는 연구자 고유 식별자 |
| 프론트 사용 | 상세 조회 key, 선택/제외 처리 key, React list key 후보 |

#### `name`

| 항목 | 내용 |
|---|---|
| 타입 | string |
| 의미 | 전문가 성명 |
| 프론트 사용 | 후보 카드 제목 또는 이름 컬럼 |

#### `organization`

| 항목 | 내용 |
|---|---|
| 타입 | string 또는 null |
| 의미 | 전문가 소속 기관명. 서버의 flat payload root `affiliated_organization`에서 온다. |
| 프론트 사용 | 후보 카드의 소속/기관 표시 |
| 주의 | 값이 없을 수 있으므로 `-`, `소속 미상` 같은 fallback 처리 필요 |

#### `data_gaps`

| 항목 | 내용 |
|---|---|
| 타입 | string[] |
| 의미 | 해당 후보에 대해 데이터가 부족하거나 특정 실적이 비어 있다는 서버 판단 메시지 |
| 프론트 사용 | 후보 카드의 주의 배지 또는 보조 안내 |
| 예시 | `["특허 실적 없음"]` |

#### `risks`

| 항목 | 내용 |
|---|---|
| 타입 | string[] |
| 의미 | 데이터 편향, 평가상 주의점, 추천 근거의 약점 등 위험 요소 메시지 |
| 프론트 사용 | 경고 영역, tooltip, 상세 패널 |

#### `shortlist_score`

| 항목 | 내용 |
|---|---|
| 타입 | number |
| 의미 | 검색/RRF 집계 후 부여된 후보 점수. 0~100 정규화 점수지만 절대 적합도는 아니다. |
| 프론트 사용 | 정렬 점수 표시, 디버그 점수 표시 |

### 삭제/대체 필드

#### `branch_presence_flags` 삭제 -> `doc_types_present` 대체

| 항목 | 기존 | 현재 |
|---|---|---|
| 필드 | `branch_presence_flags` | `doc_types_present` |
| 타입 | object/map | string[] |
| 기존 의미 | `art`, `pat`, `pjt` 등 branch별 실적 보유 여부 boolean map |
| 현재 의미 | 이번 검색에서 실제 hit된 chunk의 doc_type 목록 |

기존:

```json
{
  "branch_presence_flags": {
    "art": true,
    "pat": false,
    "pjt": true
  }
}
```

현재:

```json
{
  "doc_types_present": ["paper", "project"]
}
```

프론트 수정 예시:

```ts
const docTypes = new Set(candidate.doc_types_present ?? []);

const branchPresenceFlags = {
  art: docTypes.has("paper"),
  pat: docTypes.has("patent"),
  pjt: docTypes.has("project"),
};
```

주의:

- `basic`은 현재 직접 대응되는 doc_type이 없다.
- `assessor_activity`, `specialty`는 기존 branch 체계에 없던 새 표시 대상이다.

#### `stable_hits` 삭제

| 항목 | 내용 |
|---|---|
| 기존 타입 | integer |
| 기존 의미 | stable branch에서 매칭된 히트 수 |
| 현재 상태 | `candidates[*]`에 내려오지 않음 |
| 프론트 영향 | `candidate.stable_hits` 직접 접근 시 `undefined` |
| 권장 처리 | 해당 UI 제거 또는 디버그 trace 기반으로 별도 재설계 |

#### `expanded_hits` 삭제

| 항목 | 내용 |
|---|---|
| 기존 타입 | integer |
| 기존 의미 | expanded branch에서 매칭된 히트 수 |
| 현재 상태 | `candidates[*]`에 내려오지 않음 |
| 프론트 영향 | `candidate.expanded_hits` 직접 접근 시 `undefined` |
| 권장 처리 | 해당 UI 제거 |

#### `support_branches` 삭제/대체

| 항목 | 내용 |
|---|---|
| 기존 타입 | string[] |
| 기존 의미 | 후보를 뒷받침한 branch 식별자 목록. 예: `art`, `pjt` |
| 현재 상태 | `candidates[*]`에 내려오지 않음 |
| 대체 후보 | `doc_types_present` |
| 권장 처리 | “근거 영역” 표시가 필요하면 `doc_types_present`를 라벨로 변환 |

### 변경 필드: `counts`

`counts`는 필드 자체는 유지되지만 내부 key가 변경되었다.

| 기존 key | 기존 의미 | 현재 key | 현재 의미 |
|---|---|---|---|
| `article_cnt` | 누적 논문 수 | `publication_count` | 연구자 누적 논문 수 |
| `scie_cnt` | 누적 SCIE 논문 수 | `scie_publication_count` | 연구자 누적 SCIE 논문 수 |
| `patent_cnt` | 누적 특허/지식재산 수 | `intellectual_property_count` | 연구자 누적 지식재산 수 |
| `project_cnt` | 누적 과제 수 | `research_project_count` | 연구자 누적 연구과제 수 |
| 없음 | 없음 | `researcher_assessor_activity_count` | 연구자 누적 평가/심사 활동 수 |

기존:

```json
{
  "counts": {
    "article_cnt": 10,
    "scie_cnt": 3,
    "patent_cnt": 0,
    "project_cnt": 4
  }
}
```

현재:

```json
{
  "counts": {
    "publication_count": 10,
    "scie_publication_count": 3,
    "intellectual_property_count": 0,
    "research_project_count": 4,
    "researcher_assessor_activity_count": 1
  }
}
```

프론트 표시 권장 라벨:

| 현재 key | 권장 라벨 |
|---|---|
| `publication_count` | 논문 |
| `scie_publication_count` | SCIE |
| `intellectual_property_count` | 특허/지식재산 |
| `research_project_count` | 과제 |
| `researcher_assessor_activity_count` | 평가/심사 |

## 3.3 `/search/candidates.trace` 변경

`trace`는 디버깅용 필드이므로 프론트 핵심 비즈니스 로직에서 강하게 의존하지 않는 것이 좋다.

### 유지 또는 유사 유지

| 필드 | 타입 | 의미 |
|---|---|---|
| `planner` | object | planner가 생성한 내부 검색 계획 |
| `planner_trace` | object | planner 실행 방식, retry, cache 등 디버그 정보 |
| `raw_query` | string | 서버가 처리한 원 질의 |
| `cache` | object | planner/retrieval cache hit 여부 |
| `planner_keywords` | string[] | planner가 추출한 키워드 |
| `retrieval_keywords` | string[] | 실제 retrieval에 사용된 키워드 |
| `bundle_ids` | string[] | planner 호환용 bundle id 목록 |
| `branch_queries` | object | stable/expanded query 호환 필드. 현재 검색 경로의 핵심 계약으로 보지 않는 것이 좋음 |
| `candidate_ids` | string[] | 응답 후보 ID 목록 |
| `retrieval_score_traces` | object[] | 검색 점수와 evidence 매칭 디버그 정보 |
| `final_sort_policy` | string/null | 최종 정렬 정책 |
| `query_payload` | object | 검색 엔진 호출에 사용된 내부 payload |
| `expanded_shadow_hits` | object[] | 확장 검색 관련 디버그 후보 |
| `filtered_out_candidates` | object[] | 후처리/strict filter에서 제외된 후보 |
| `retrieval_skipped_reason` | string/null | 검색이 스킵된 경우의 이유 |
| `include_orgs` | string[] | 적용된 포함 기관 |
| `exclude_orgs` | string[] | 적용된 제외 기관 |
| `planner_retry_count` | integer | planner 재시도 횟수 |
| `timers` | object | 단계별 소요 시간 |

### 의미 변경 또는 사용 주의

#### `support_rule_applied`

| 항목 | 기존 | 현재 |
|---|---|---|
| 타입 | boolean | boolean |
| 기존 의미 | Support Rule 적용 여부, 기존 명세상 항상 true |
| 현재 의미 | deprecated legacy trace 성격. 현재 서버에서는 false 성격 |
| 프론트 권장 | 화면/비즈니스 로직 조건으로 사용하지 말 것 |

#### `candidate_support_info`

| 항목 | 기존 | 현재 |
|---|---|---|
| 타입 | object[] | object[] |
| 기존 의미 | 후보별 stable/expanded/support branch 상세 |
| 현재 의미 | 후보별 `expert_id`, `doc_types_present` 중심의 간단한 지원 정보 |

기존 기대:

```json
{
  "expert_id": "12345678",
  "stable_hits": 2,
  "expanded_hits": 1,
  "support_branches": ["art", "pjt"]
}
```

현재:

```json
{
  "expert_id": "12345678",
  "doc_types_present": ["paper", "project"]
}
```

### 추가 또는 강조 필드

#### `strict_filter`

| 필드 | 타입 | 의미 |
|---|---|---|
| `strict_filter.enabled` | boolean | required concept gate가 활성화되었는지 여부 |
| `strict_filter.required_concepts` | string[] | 반드시 충족해야 하는 concept 목록 |
| `strict_filter.excluded_candidate_count` | integer | concept 미충족으로 제외된 후보 수 |
| `strict_filter.excluded_reasons` | object[] | 제외 후보별 ID, 이름, 충족/미충족 concept 정보 |

프론트 사용:

- 일반 사용자 화면보다는 디버그/운영자 패널에 적합하다.

#### `top_k_used`

| 항목 | 내용 |
|---|---|
| 타입 | integer |
| 의미 | 서버가 실제 적용한 반환 상한. 요청 `top_k`, planner `top_k`, 최대 15 clamp 결과 |
| 프론트 사용 | “상위 N명 표시” 안내 또는 디버그 표시 |

#### `retrieval_mode`

| 항목 | 내용 |
|---|---|
| 타입 | string/null |
| 의미 | 실제 검색 엔진이 사용한 내부 검색 모드. 예: `multiview_flat_relevance` |
| 프론트 사용 | 디버그 표시 |

#### `weights`

| 항목 | 내용 |
|---|---|
| 타입 | object/null |
| 의미 | 검색 view 또는 단계별 가중치 정보 |
| 프론트 사용 | 디버그 표시 |

#### `keyword_inclusion_dropped_count`

| 항목 | 내용 |
|---|---|
| 타입 | integer |
| 의미 | 키워드 포함 필터 때문에 제외된 결과 수 |
| 프론트 사용 | 디버그 표시 |

## 4. branch 체계에서 doc_type 체계로 변경

### 4.1 값 매핑

| 기존 branch | 현재 doc_type | 의미 | UI 라벨 |
|---|---|---|---|
| `art` | `paper` | 논문 실적 | 논문 |
| `pat` | `patent` | 특허/지식재산 실적 | 특허 |
| `pjt` | `project` | 연구과제 실적 | 과제 |
| `basic` | 직접 대응 없음 | 기존 기본 인적/프로필 branch | 현재는 별도 doc_type이 아니라 root/profile 정보로 취급 |
| 없음 | `assessor_activity` | 평가위원/심사위원 활동 실적 | 평가/심사 |
| 없음 | `specialty` | 전문분야/전문성 정보 | 전문분야 |

### 4.2 프론트 호환 변환 예시

```ts
type CurrentDocType =
  | "paper"
  | "patent"
  | "project"
  | "assessor_activity"
  | "specialty";

function toLegacyBranchFlags(docTypesPresent: CurrentDocType[] = []) {
  const docTypes = new Set(docTypesPresent);
  return {
    art: docTypes.has("paper"),
    pat: docTypes.has("patent"),
    pjt: docTypes.has("project"),
  };
}

function docTypeLabel(docType: string) {
  const labels: Record<string, string> = {
    paper: "논문",
    patent: "특허",
    project: "과제",
    assessor_activity: "평가/심사",
    specialty: "전문분야",
  };
  return labels[docType] ?? docType;
}
```

## 5. `/recommend` 응답 변경사항

`POST /recommend`는 `/search/candidates`와 같은 검색 기반 위에서 추천 사유와 evidence를 추가로 생성한다.

### 5.1 최상위 필드

| 필드 | 타입 | 의미 | 변경 여부 |
|---|---|---|---|
| `intent_summary` | string | planner가 해석한 추천 의도 요약 | 유지 |
| `applied_filters` | object | 실제 적용된 hard filter | 유지 |
| `searched_branches` | string[] | 검색 대상 doc_type 5종. 필드명은 branch지만 값은 doc_type | 의미 변경 |
| `doc_type_coverage` | object | 추천 evidence가 어떤 doc_type을 커버했는지 요약 | 추가/강조 |
| `retrieved_count` | integer | 검색/집계된 전체 후보 수 | 유지 |
| `recommendations` | object[] | 최종 추천 후보 목록 | 유지, 내부 구조 변경 |
| `data_gaps` | string[] | 전체 추천 결과에 대한 공통 데이터 공백 | 유지 |
| `not_selected_reasons` | string[] | 추천 후보가 없는 경우 또는 제외 사유 | 유지 |
| `trace` | object | 추천 파이프라인 디버그 정보 | 변경 |

### 5.2 `recommendations[*]` 추가/변경 필드

#### `organization`

| 항목 | 내용 |
|---|---|
| 타입 | string/null |
| 의미 | 추천 후보의 소속 기관명. 검색 후보의 `organization`과 같은 의미 |
| 변경 | 기존 추천 응답 명세에 없었다면 추가 필드 |
| 프론트 사용 | 추천 카드의 소속 기관 표시 |

#### `recommendation_reason`

| 항목 | 내용 |
|---|---|
| 타입 | string |
| 의미 | 서버 또는 LLM이 생성한 단일 추천 사유 문장 |
| 변경 | 현재 주 필드 |
| 프론트 사용 | 추천 카드 본문에 우선 표시 |

#### `reasons`

| 항목 | 내용 |
|---|---|
| 타입 | string[] |
| 의미 | 구형 프론트 호환용 사유 배열. 현재는 `recommendation_reason`을 배열로 감싼 computed field |
| 변경 | 하위호환 필드 |
| 프론트 사용 | 새 UI는 `recommendation_reason` 우선, 구형 fallback이 필요할 때만 사용 |

#### `match_badges`

| 항목 | 내용 |
|---|---|
| 타입 | string[] |
| 의미 | 추천 후보가 충족한 조건을 짧은 배지 문구로 표현한 목록 |
| 예시 | `["AI 충족", "반도체 충족", "직접 수행 근거 있음"]` |
| 프론트 사용 | 카드 상단 badge/chip |

#### `match_summary`

| 항목 | 내용 |
|---|---|
| 타입 | string |
| 의미 | 이 후보가 질의와 어떻게 맞는지 짧게 요약한 문장 |
| 프론트 사용 | 추천 사유 위의 한 줄 요약 또는 subtitle |

#### `match_details`

| 하위 필드 | 타입 | 의미 |
|---|---|---|
| `matched_concepts` | string[] | 후보 evidence에서 충족된 concept ID 목록 |
| `missing_concepts` | string[] | required concept 중 충족하지 못한 concept ID 목록 |
| `coverage_type` | string | concept 충족 형태. 예: `joint`, `separate`, `partial`, 빈 문자열 |
| `matched_doc_types` | string[] | 추천 근거로 실제 매칭된 doc_type 목록 |
| `direct_evidence_count` | integer | profile이 아닌 직접 evidence 수 |

프론트 사용:

- 추천 상세 펼침 영역, 디버그 패널, 근거 커버리지 표시.

#### `score_explanation`

| 하위 필드 | 타입 | 의미 |
|---|---|---|
| `final_score` | number/null | 내부 최종 검색 점수 |
| `rank_score` | number/null | 사용자 표시용 0~100 정규화 점수 |
| `score_breakdown` | object | concept, joint, support 등 점수 구성 요소 |
| `top_chunks` | object[] | 점수에 기여한 상위 chunk 요약 |

프론트 사용:

- 검색 점수 근거 상세 패널에 적합하다.

#### `evidence_summary`

| 하위 필드 | 타입 | 의미 |
|---|---|---|
| `total_profile_counts` | object | 후보의 누적 실적 count. `counts`와 같은 계열의 값 |
| `matched_evidence_count` | integer | 이번 질의에 매칭된 evidence 총 수 |
| `shown_evidence_count` | integer | 응답 `evidence`에 실제 표시되는 evidence 수 |
| `profile_evidence_count` | integer | 질의 매칭이 아닌 프로필 보강 evidence 수 |

프론트 사용:

- “누적 실적”과 “이번 질의 근거”를 분리해서 표시할 때 사용.

#### `evidence`

| 항목 | 내용 |
|---|---|
| 타입 | object[] |
| 의미 | 추천 사유의 근거가 되는 matched evidence 목록 |
| 변경 | `chunk_id`가 핵심 식별자로 추가/강조됨 |
| 프론트 사용 | 근거 카드 목록 |

`evidence[*]` 구조:

| 필드 | 타입 | 의미 |
|---|---|---|
| `type` | string | evidence의 doc_type. `paper`, `patent`, `project`, `assessor_activity`, `specialty`, `profile` |
| `title` | string | 근거 제목 |
| `date` | string/null | 근거 날짜 |
| `detail` | string/null | 기관, 기간, 구분 등 부가 설명 |
| `snippet` | string/null | 근거 본문 일부 또는 요약 |
| `chunk_id` | string/null | evidence 고유 식별자. 가능하면 UI key와 상세 추적 id로 사용 |
| `evidence_kind` | string | `matched` 또는 `profile` |

#### `profile_evidence`

| 항목 | 내용 |
|---|---|
| 타입 | object[] |
| 의미 | 질의에 직접 매칭된 evidence는 아니지만 후보 프로필 보강을 위해 붙은 참고 evidence |
| 프론트 사용 | 추천 근거와 분리된 “참고 실적” 영역에 표시 가능 |
| 주의 | 점수/랭킹 근거와 혼동하지 않는 것이 좋다 |

#### `data_gaps`

| 항목 | 내용 |
|---|---|
| 타입 | string[] |
| 의미 | 해당 추천 후보 개인의 데이터 공백 |
| 프론트 사용 | 후보별 경고/안내 메시지 |

#### `fallback_reason_used`

| 항목 | 내용 |
|---|---|
| 타입 | boolean |
| 의미 | LLM 추천 사유가 비어 있거나 부적절해서 서버가 fallback 추천 사유를 생성했는지 여부 |
| 프론트 사용 | 운영자 디버그 표시. 일반 사용자 화면에서는 숨겨도 됨 |

## 6. `/feedback` 변경사항

### `metadata` 추가

| 필드 | 타입 | 의미 |
|---|---|---|
| `metadata` | object | 피드백 저장 시 함께 남길 추가 메타데이터. 운영자 ID, 화면 버전, 실험군, 세션 정보 등을 넣을 수 있다. |

예시:

```json
{
  "query": "인공지능 및 반도체 분야 평가위원 추천",
  "selected_expert_ids": ["12345678"],
  "rejected_expert_ids": ["99999999"],
  "notes": "적합한 평가위원이 배정됨",
  "metadata": {
    "operator": "tester",
    "frontend_version": "2026.06.15"
  }
}
```

프론트 영향:

- 기존처럼 생략해도 동작한다.
- 운영/분석 목적이 있으면 화면 버전, 사용자 액션 출처 등을 추가로 보낼 수 있다.

## 7. `/health` 변경사항

### `searched_branches`

| 항목 | 기존 | 현재 |
|---|---|---|
| 타입 | string[] | string[] |
| 기존 의미 | branch 목록 | doc_type 5종 |
| 현재 값 | 해당 없음 | `paper`, `patent`, `project`, `assessor_activity`, `specialty` |

예시:

```json
{
  "status": "ok",
  "collection_name": "researcher_recommend_v1",
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"]
}
```

## 8. `/health/ready` 변경사항

`checks` 내부 key가 기존 문서와 다르다.

| 기존 key | 현재 key | 현재 key 의미 |
|---|---|---|
| `llm_backend` | `llm_backend_connected` | LLM backend 연결 가능 여부 |
| `embedding_backend` | `embedding_backend_connected` | embedding backend 연결 가능 여부 |
| `qdrant_collection_exists` | `collection_exists` | Qdrant 컬렉션 존재 여부 |
| `vectors_present` | `dense_vectors_present` | dense vector 이름이 컬렉션에 존재하는지 |
| `vectors_present` | `sparse_vectors_present` | sparse vector 이름이 컬렉션에 존재하는지 |
| 없음 | `sparse_vectors_idf` | sparse vector modifier가 기대 설정과 맞는지 |
| `payload_indexes_present` | `payload_indexes_present` | 필수 payload index 존재 여부 |
| 없음 | `sample_point_exists` | 검증용 샘플 point가 조회되는지 |
| 없음 | `sample_payload_valid` | 샘플 payload가 JSON 객체로 분석 가능한지 |
| 없음 | `sample_root_fields` | 샘플 payload에 `researcher_id`, `doc_type`, `chunk_id`, `chunk_text`가 있는지 |
| 없음 | `sample_doc_type_valid` | 샘플 `doc_type`이 현재 5종 중 하나인지 |
| 없음 | `sample_doc_attrs_present` | 샘플 `doc_attrs`가 object 형태로 있는지. 선택적 check |
| 없음 | `sample_doc_date_present` | 샘플 `doc_date`가 정규화 가능한지. 선택적 check |

프론트/운영 모니터링 영향:

- readiness 화면 또는 모니터링 코드가 기존 key 이름을 직접 참조하면 실패한다.
- `ready` boolean을 1차 판단값으로 쓰고, `checks`는 동적으로 렌더링하는 방식을 권장한다.

## 9. 사용 금지 엔드포인트

### `POST /recommend/stream`

| 항목 | 내용 |
|---|---|
| 라우트 존재 여부 | 존재 |
| 응답 형식 | `text/event-stream` |
| 현재 상태 | `RecommendationService.recommend_stream` 메서드가 구현되어 있지 않음 |
| 프론트 권장 | 호출하지 말 것. 일반 추천은 `POST /recommend` 사용 |

## 10. 프론트 수정 체크리스트

- `filters_override.degree_slct_nm` 전송 코드를 `filters_override.highest_degree`로 변경한다.
- `searched_branches`를 branch가 아니라 doc_type으로 해석한다.
- `branch_presence_flags` 접근 코드를 제거하고 `doc_types_present` 기반으로 화면 표시값을 계산한다.
- `counts.article_cnt`, `counts.scie_cnt`, `counts.patent_cnt`, `counts.project_cnt` 접근을 현재 count key로 변경한다.
- `stable_hits`, `expanded_hits`, `support_branches` UI는 제거하거나 doc_type 기반 UI로 재설계한다.
- `support_rule_applied`로 화면 조건 분기를 하지 않는다.
- 추천 사유 표시는 `recommendation_reason`을 우선 사용한다.
- evidence 목록 key는 가능하면 `evidence[*].chunk_id`를 사용한다.
- `/health/ready.checks`는 고정 key 참조보다 동적 렌더링을 권장한다.
- `/recommend/stream`은 호출하지 않는다.
