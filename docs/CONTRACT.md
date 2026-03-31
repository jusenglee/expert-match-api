# 계약 문서

## Planner 출력

`PlannerOutput`

```json
{
  "intent_summary": "AI 반도체 평가위원 추천",
  "hard_filters": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE",
    "art_recent_years": 5,
    "project_cnt_min": 1
  },
  "exclude_orgs": ["A기관"],
  "soft_preferences": ["최근 5년 실적"],
  "branch_query_hints": {
    "basic": "전공 학위 소속유형 중심",
    "art": "논문 키워드와 초록 중심",
    "pat": "특허와 사업화 중심",
    "pjt": "과제명과 전문기관 중심"
  },
  "top_k": 5
}
```

## Judge 출력

`JudgeOutput`

```json
{
  "recommended": [
    {
      "rank": 1,
      "expert_id": "11008395",
      "name": "홍길동",
      "fit": "높음",
      "reasons": ["논문 근거가 확인됨", "과제 수행 근거가 확인됨"],
      "evidence": [
        {"type": "paper", "title": "...", "date": "2024-09", "detail": "Fire"},
        {"type": "project", "title": "...", "date": "2020-04-06", "detail": "중소기업기술정보진흥원"}
      ],
      "risks": ["특허 근거 부족"]
    }
  ],
  "not_selected_reasons": ["상위 추천 대비 근거 다양성 또는 최근성이 상대적으로 약함"],
  "data_gaps": ["특허 근거 부족"]
}
```

## API

### `POST /recommend`

request:

```json
{
  "query": "AI 반도체 논문과 과제 실적이 좋은 평가위원 추천",
  "top_k": 5,
  "filters_override": {
    "degree_slct_nm": "박사",
    "art_sci_slct_nm": "SCIE"
  },
  "exclude_orgs": ["A기관"]
}
```

response 핵심 필드:

- `intent_summary`
- `applied_filters`
- `searched_branches`
- `retrieved_count`
- `recommendations`
- `data_gaps`
- `not_selected_reasons`
- `trace`

### `POST /search/candidates`

response 핵심 필드:

- `intent_summary`
- `applied_filters`
- `searched_branches`
- `retrieved_count`
- `candidates`
- `trace`

### `POST /feedback`

운영자 선택 결과를 sqlite에 저장한다.

### `GET /health/ready`

response 핵심 필드:

- `ready`
- `checks`
- `issues`
- `collection_name`
- `sample_point_id`

`ready=false`면 HTTP 503으로 응답한다.

## Payload 필수 필드

실데이터 sample point는 최소 아래를 포함해야 한다.

- root
  - `doc_id`
  - `blng_org_nm_exact`
  - `degree_slct_nm`
  - `article_cnt`
  - `scie_cnt`
  - `patent_cnt`
  - `project_cnt`
- nested
  - `art[]`
  - `pat[]`
  - `pjt[]`
- 교정된 과제 날짜 필드
  - `pjt[].start_dt`
  - `pjt[].end_dt`
  - `pjt[].stan_yr`

## 운영 실패 조건

아래 중 하나라도 깨지면 운영 readiness 실패로 본다.

- Qdrant 컬렉션 없음
- 8개 named vector 누락
- sparse vector modifier가 IDF가 아님
- 필수 payload index 누락
- sample point 없음
- sample point에 `art[]`, `pat[]`, `pjt[]` 중 하나라도 없음
- sample point에 `blng_org_nm_exact` 없음
- sample point의 과제 날짜 필드 누락
- 추천 결과에 evidence가 비어 있음

