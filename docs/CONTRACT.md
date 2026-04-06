# 규약 (Contract)

## Planner 출력 (Planner Output)

`PlannerOutput` 데이터 구조 예시:

```json
{
  "intent_summary": "AI 반도체 분야 심사위원 추천",
  "hard_filters": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE",
    "art_recent_years": 5,
    "project_cnt_min": 1
  },
  "exclude_orgs": ["A 기관"],
  "soft_preferences": ["최근 성과 중심"],
  "branch_query_hints": {
    "basic": "프로필 중심 질의",
    "art": "논문 중심 질의",
    "pat": "특허 중심 질의",
    "pjt": "과제 중심 질의"
  },
  "top_k": 5
}
```

Planner 동작 방식:

- LLM 플래너는 `PlannerOutput` 규격에 맞는 단일 JSON 객체를 반환해야 합니다.
- `branch_query_hints`는 리스트가 아닌 `basic`, `art`, `pat`, `pjt` 키를 가진 객체여야 합니다.
- LLM 출력이 유효한 `PlannerOutput` 형식이 아니면 시스템은 히우리스틱 플래너로 폴백합니다.

## Judge 출력 (Judge Output)

`JudgeOutput` 데이터 구조 예시:

```json
{
  "recommended": [
    {
      "rank": 1,
      "expert_id": "11008395",
      "name": "홍길동",
      "fit": "높음",
      "reasons": ["논문 실적이 우수합니다."],
      "evidence": [
        {"type": "paper", "title": "예시 논문 제목", "date": "2024-09", "detail": "SCIE"},
        {"type": "project", "title": "예시 과제 제목", "date": "2020-04-06", "detail": "주관연구책임자"}
      ],
      "risks": ["특허 실적이 누락되었습니다."],
      "relevance_score": 95.5
    }
  ],
  "not_selected_reasons": ["다른 후보자들의 실적 범위가 더 넓었습니다."],
  "data_gaps": ["특허 실적이 누락되었습니다."]
}
```

Judge 동작 방식:

- 각 `recommended[]` 항목은 `rank`, `expert_id`, `name`, `fit`, `reasons`, `evidence`, `risks`, `relevance_score` 필드를 포함해야 합니다.
- `reasons`, `risks`, `not_selected_reasons`, `data_gaps`는 문자열 배열이어야 합니다.
- 판정 도중 LLM 응답에서 첫 번째 JSON 객체를 추출하며, 검증 전 가능한 범위 내에서 문자열/리스트 불일치를 정규화합니다.
- `expert_id`가 누락되었으나 `name`이 숏리스트 후보 중 한 명과 정확히 일치하면 해당 전문가의 `expert_id`를 자동으로 보완합니다.
- 정규화된 결과가 여전히 `JudgeOutput` 규격에 맞지 않으면 히우리스틱 판정기로 폴백합니다.

## API 명세

### `POST /recommend`

요청(Request):

```json
{
  "query": "AI 반도체 논문 및 과제 실적이 우수한 심사위원 추천해줘",
  "top_k": 5,
  "filters_override": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE"
  },
  "exclude_orgs": ["A 기관"]
}
```

응답 필드(Response fields):

- `intent_summary`: 질의 분석 요약
- `applied_filters`: 적용된 필터 조건
- `searched_branches`: 검색된 데이터 브랜치 목록
- `retrieved_count`: 검색된 총 히트 수
- `recommendations`: 최종 추천 전문가 리스트
- `data_gaps`: 전체 데이터 결측치 정보
- `not_selected_reasons`: 선택되지 않은 사유 및 보강 제안
- `trace`: 디버깅용 추적 데이터

동작 방식:

- 최종 추천 결과가 없더라도 `POST /recommend`는 `200` 상태 코드를 반환합니다.
- 검색된 후보가 없거나 숏리스트가 비어 있으면 판정 단계를 건너뛰고 구조화된 빈 결과를 즉시 반환하며, `recommendations`는 빈 리스트가 되고 사유는 `not_selected_reasons` 또는 `data_gaps`에 기록됩니다.
- 근거(evidence)가 전혀 없는 추천 결과는 서버 오류를 일으키는 대신 최종 응답에서 제외됩니다.

### `POST /search/candidates`

응답 필드:

- `intent_summary`
- `applied_filters`
- `searched_branches`
- `retrieved_count`
- `candidates`: 숏리스트 후보 카드 목록
- `trace`

### `POST /feedback`

SQLite에 운영자의 선택 결과 및 메모를 저장합니다.

### `GET /health/ready`

응답 필드:

- `ready`: 준비 완료 여부 (bool)
- `checks`: 개별 점검 항목 결과
- `issues`: 발생한 이슈 목록
- `collection_name`: 사용 중인 Qdrant 컬렉션명
- `sample_point_id`: 검증 시 사용된 샘플 데이터 ID

동작 방식:

- `200`은 시스템이 정상적으로 준비되었음을 의미합니다.
- `503`은 준비 상태 확인 중 실패가 발생했음을 의미하며, 실패 시에도 성공 시와 동일한 최상위 필드를 포함한 응답 본문을 반환합니다.
- 실패 응답은 `ReadinessResponse` 규격에 맞춰 직접 반환되며 `detail` 필드에 감싸여 있지 않습니다.
- 초기화 중 문제가 발생하여 라이브 검증기가 실행되기 전이라면 `checks.startup_runtime_initialized=false`와 함께 오류 메시지를 `issues`에 담아 `503`을 반환합니다.

## 필수 페이로드 구조 (Required Payload Structure)

준비 상태 점검 시 선택되는 대표 샘플 데이터는 최소 다음 필드를 포함해야 합니다:

- 루트 필드:
  - `basic_info`
  - `researcher_profile`
- 하위 근거 리스트:
  - `publications[]` (논문)
  - `research_projects[]` (과제)
- 과제 하위 날짜 필드 (모든 `research_projects[]` 항목):
  - `project_start_date`
  - `project_end_date`
  - `reference_year`

준비 상태 검증기는 일정 범위 내의 컬렉션 포인트들을 스캔하여 가장 완전한 샘플을 선택한 뒤 이 요구사항들을 평가합니다.

레거시 페이로드 유의사항:

- Qdrant에 저장된 선택적 리스트나 숫자 필드가 빈 문자열(`""`)로 저장되어 있는 경우, 시스템은 읽기 시점에 이를 자동으로 정규화합니다.
- 필수 식별자나 제목 필드는 자동 복원되지 않으며, 데이터 형식이 잘못된 경우 검증에 실패할 수 있습니다.

## 준비 상태 실패 조건 (Readiness Failure Conditions)

다음 점검 항목 중 하나라도 실패하면 준비 상태는 `false`로 간주됩니다:

- LLM 백엔드 연결성
- 임베딩 백엔드 연결성
- Qdrant 컬렉션 조회
- 필수 Dense 이름 벡터 존재 여부
- 필수 Sparse 이름 벡터 존재 여부
- Sparse 벡터 IDF 수정자(Modifier) 설정 여부
- 필수 페이로드 인덱스 존재 여부
- 대표 샘플 데이터 존재 여부
- 샘플 페이로드 객체 구조 유효성
- `publications[]` 또는 `research_projects[]` 누락 또는 비어있음
- `research_projects[]` 내 과제 날짜 필드 누락
