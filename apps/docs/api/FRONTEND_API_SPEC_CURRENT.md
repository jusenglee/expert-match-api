# 현재 시스템 API 명세서 - 프론트 전달용

기준일: 2026-06-15  
대상 서버: `Ntis_person_API` 현재 코드 기준  
변경 표시 기준: 기존 프론트 명세의 branch 기반 `/search/candidates` 계약

## 0. 주요 변경 요약

| 구분 | 기존 프론트 명세 | 현재 서버 계약 | 영향 |
|---|---|---|---|
| 변경 | `filters_override.degree_slct_nm` | `filters_override.highest_degree` | 기존 키는 요청 검증은 통과할 수 있으나 필터가 적용되지 않을 수 있음 |
| 변경 | branch 값 `basic/art/pat/pjt` | doc_type 값 `paper/patent/project/assessor_activity/specialty` | 화면 라벨/분기 로직 수정 필요 |
| 변경 | `searched_branches` 값이 branch 목록 | 필드명은 유지, 값은 doc_type 5종 | 필드명만 보고 기존 branch로 해석하면 오류 |
| 삭제/대체 | `candidates[*].branch_presence_flags` | `candidates[*].doc_types_present` | 기존 직접 접근 시 undefined 오류 가능 |
| 변경 | `counts.article_cnt`, `patent_cnt`, `project_cnt` | `publication_count`, `intellectual_property_count`, `research_project_count` 등 | 카운트 표시 매핑 수정 필요 |
| 삭제/대체 | `stable_hits`, `expanded_hits`, `support_branches` | 후보 객체에 없음. trace의 `candidate_support_info`도 `doc_types_present` 중심 | Support Rule UI는 제거/재설계 필요 |
| 변경 | `support_rule_applied` 현재 항상 true | 현재 서버에서는 deprecated 값으로 false 성격 | 이 값으로 화면 조건 분기 금지 |
| 추가 | `/recommend.recommendations[*].reasons` | 하위호환 computed field | `recommendation_reason`이 1순위, `reasons`는 구형 fallback |
| 추가 | `/recommend.recommendations[*].profile_evidence`, `data_gaps`, `fallback_reason_used` | 현재 직렬화됨 | 필요 시 표시 가능, 기존 UI는 무시 가능 |
| 주의 | `POST /recommend/stream` | 라우트는 있으나 `RecommendationService.recommend_stream` 미구현 | 프론트 사용 금지. 호출 시 SSE error 이벤트 가능 |

## 1. 공통 요청 모델

`POST /recommend`, `POST /search/candidates`, `POST /recommend/stream`은 같은 요청 모델을 사용한다.

```json
{
  "query": "인공지능 및 반도체 분야 대학 교수 추천",
  "top_k": 5,
  "search_mode": "multiview",
  "filters_override": {
    "highest_degree": "박사"
  },
  "include_orgs": [],
  "exclude_orgs": ["A대학교", "B연구소"]
}
```

| 필드 | 타입 | 필수 | 설명 |
|---|---:|---:|---|
| `query` | string | Y | 자연어 질의. 여러 줄은 서버에서 `, `로 합쳐진다. |
| `top_k` | integer/null | N | 1~15. 미지정 시 planner 값 또는 최대 15. 16 이상은 422. |
| `search_mode` | string | N | `multiview` 기본값, `hybrid`, `keyword_similarity`. 잘못된 값은 422. |
| `filters_override` | object | N | hard filter override. 학위 필터는 `highest_degree` 사용. |
| `include_orgs` | string[] | N | 포함 기관. root `affiliated_organization` 기준 post-filter. |
| `exclude_orgs` | string[] | N | 제외 기관. root `affiliated_organization` 기준 post-filter. |

### 1.1 기존 요청 대비 변경

```diff
 {
   "query": "인공지능 및 반도체 분야 대학 교수 추천",
   "top_k": 5,
   "filters_override": {
-    "degree_slct_nm": "박사"
+    "highest_degree": "박사"
   },
   "exclude_orgs": ["A대학교", "B연구소"]
 }
```

## 2. 후보자 목록 조회 - Search Candidates

`POST /search/candidates`

추천 사유 생성 없이 검색/집계/정렬된 초기 후보 목록을 반환한다. 응답 후보 수는 최대 15명이다.

### 2.1 응답 예시

```json
{
  "intent_summary": "AI·반도체 분야 대학 교수 추천",
  "applied_filters": {
    "highest_degree": "박사"
  },
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"],
  "keywords": ["인공지능", "반도체"],
  "retrieved_count": 27,
  "candidates": [
    {
      "expert_id": "12345678",
      "name": "홍길동",
      "organization": "A대학교",
      "doc_types_present": ["paper", "project"],
      "counts": {
        "publication_count": 10,
        "scie_publication_count": 3,
        "intellectual_property_count": 0,
        "research_project_count": 4,
        "researcher_assessor_activity_count": 1
      },
      "data_gaps": ["특허 실적 없음"],
      "risks": [],
      "shortlist_score": 95.8
    }
  ],
  "trace": {
    "planner": {},
    "planner_trace": {},
    "raw_query": "인공지능 및 반도체 분야 대학 교수 추천",
    "planner_keywords": ["인공지능", "반도체"],
    "retrieval_keywords": ["인공지능", "반도체"],
    "bundle_ids": [],
    "expanded_shadow_hits": [],
    "removed_role_terms": ["교수", "추천"],
    "cache": {
      "canonical_plan": "miss",
      "retrieval": "miss"
    },
    "planner_retry_count": 0,
    "support_rule_applied": false,
    "filtered_out_count": 0,
    "filtered_out_candidates": [],
    "strict_filter": {
      "enabled": true,
      "required_concepts": ["ai", "semiconductor"],
      "excluded_candidate_count": 0,
      "excluded_reasons": []
    },
    "retrieval_skipped_reason": null,
    "branch_queries": {
      "stable": "...",
      "expanded": "..."
    },
    "include_orgs": [],
    "exclude_orgs": ["A대학교", "B연구소"],
    "candidate_ids": ["12345678"],
    "candidate_support_info": [
      {
        "expert_id": "12345678",
        "doc_types_present": ["paper", "project"]
      }
    ],
    "retrieval_score_traces": [],
    "final_sort_policy": "rrf_score_desc_name_asc",
    "top_k_used": 5,
    "query_payload": {},
    "retrieval_mode": "multiview_flat_relevance",
    "weights": {},
    "keyword_inclusion_dropped_count": 0,
    "timers": {}
  }
}
```

### 2.2 최상위 응답 필드

| 필드 | 타입 | 변경 표시 | 설명 |
|---|---:|---|---|
| `intent_summary` | string | 유지 | planner가 해석한 질의 의도 요약 |
| `applied_filters` | object | 유지 | 실제 적용된 hard filter |
| `searched_branches` | string[] | 변경 | 필드명은 유지. 값은 `basic/art/pat/pjt`가 아니라 doc_type 5종 |
| `keywords` | string[] | 유지 | 검색에 사용된 핵심 키워드 |
| `retrieved_count` | integer | 유지 | 검색/필터/집계 후 전체 후보 수 |
| `candidates` | object[] | 변경 | 후보 객체 구조 변경 |
| `trace` | object | 변경 | 디버깅용. 하위 키는 안정 계약보다 약함 |

### 2.3 `candidates[*]` 필드

| 필드 | 타입 | 변경 표시 | 설명 |
|---|---:|---|---|
| `expert_id` | string | 유지 | 전문가/연구자 ID |
| `name` | string | 유지 | 성명 |
| `organization` | string/null | 유지 | 소속 기관 |
| `doc_types_present` | string[] | 추가/대체 | 이번 검색에서 hit된 doc_type 목록 |
| `counts` | object | 변경 | flat root 기반 누적 실적 count |
| `data_gaps` | string[] | 유지 | 후보별 데이터 공백 |
| `risks` | string[] | 유지 | 후보별 위험/주의사항 |
| `shortlist_score` | number | 유지 | RRF 기반 정규화 점수 |
| `branch_presence_flags` | 없음 | 삭제/대체 | `doc_types_present`로 대체 |
| `stable_hits` | 없음 | 삭제 | 현재 후보 객체에서 제공하지 않음 |
| `expanded_hits` | 없음 | 삭제 | 현재 후보 객체에서 제공하지 않음 |
| `support_branches` | 없음 | 삭제/대체 | `doc_types_present` 또는 trace를 사용 |

### 2.4 count 필드 매핑

| 기존 프론트 키 | 현재 서버 키 |
|---|---|
| `article_cnt` | `publication_count` |
| `scie_cnt` | `scie_publication_count` |
| `patent_cnt` | `intellectual_property_count` |
| `project_cnt` | `research_project_count` |
| 없음 | `researcher_assessor_activity_count` |

### 2.5 branch/doc_type 매핑

| 기존 branch | 현재 doc_type |
|---|---|
| `art` | `paper` |
| `pat` | `patent` |
| `pjt` | `project` |
| `basic` | 직접 대응 없음 |
| 없음 | `assessor_activity` |
| 없음 | `specialty` |

프론트에서 기존 `branch_presence_flags`가 필요하면 다음처럼 변환한다.

```ts
const docTypes = new Set(candidate.doc_types_present ?? []);
const branchPresenceFlags = {
  art: docTypes.has("paper"),
  pat: docTypes.has("patent"),
  pjt: docTypes.has("project"),
};
```

## 3. 전문가 추천 - Recommend

`POST /recommend`

검색 후보를 대상으로 evidence 선별 및 추천 사유 생성을 수행한 최종 추천 목록을 반환한다. 검색 결과가 없거나 strict filter를 통과한 후보가 없으면 `200 OK`와 `recommendations: []`를 반환한다.

### 3.1 응답 예시

```json
{
  "intent_summary": "AI·반도체 분야 대학 교수 추천",
  "applied_filters": {
    "highest_degree": "박사"
  },
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"],
  "doc_type_coverage": {
    "searched": ["paper", "patent", "project", "assessor_activity", "specialty"],
    "matched": ["paper", "project"],
    "missing": ["patent", "assessor_activity", "specialty"]
  },
  "retrieved_count": 27,
  "recommendations": [
    {
      "rank": 1,
      "expert_id": "12345678",
      "name": "홍길동",
      "organization": "A대학교",
      "fit": "높음",
      "recommendation_reason": "AI 반도체 관련 논문과 과제 수행 근거가 확인됩니다.",
      "reasons": ["AI 반도체 관련 논문과 과제 수행 근거가 확인됩니다."],
      "match_badges": ["AI 충족", "반도체 충족"],
      "match_summary": "AI, 반도체 근거가 확인되었습니다.",
      "match_details": {
        "matched_concepts": ["ai", "semiconductor"],
        "missing_concepts": [],
        "coverage_type": "joint",
        "matched_doc_types": ["paper", "project"],
        "direct_evidence_count": 2
      },
      "score_explanation": {
        "final_score": 0.77,
        "rank_score": 95.8,
        "score_breakdown": {},
        "top_chunks": []
      },
      "evidence_summary": {
        "total_profile_counts": {
          "publication_count": 10,
          "scie_publication_count": 3,
          "intellectual_property_count": 0,
          "research_project_count": 4,
          "researcher_assessor_activity_count": 1
        },
        "matched_evidence_count": 2,
        "shown_evidence_count": 2,
        "profile_evidence_count": 0
      },
      "evidence": [
        {
          "type": "paper",
          "chunk_id": "paper_100000045256_c000",
          "title": "AI 반도체 설계 연구",
          "date": "2023-01-01",
          "detail": null,
          "snippet": "AI 반도체 설계 관련 연구...",
          "evidence_kind": "matched"
        }
      ],
      "profile_evidence": [],
      "risks": [],
      "rank_score": 95.8,
      "data_gaps": [],
      "fallback_reason_used": false
    }
  ],
  "data_gaps": [],
  "not_selected_reasons": [],
  "trace": {}
}
```

### 3.2 `recommendations[*]` 주요 필드

| 필드 | 타입 | 변경 표시 | 설명 |
|---|---:|---|---|
| `rank` | integer | 유지 | 추천 순위 |
| `expert_id` | string | 유지 | 전문가/연구자 ID |
| `name` | string | 유지 | 성명 |
| `organization` | string/null | 추가 | 소속 기관 |
| `fit` | string | 유지 | `높음`, `중간`, `보통` |
| `recommendation_reason` | string | 변경 | 주 추천 사유. 프론트는 이 필드를 우선 사용 |
| `reasons` | string[] | 추가 | 하위호환 alias. `recommendation_reason`을 배열로 변환 |
| `match_badges` | string[] | 추가 | UI 배지 |
| `match_summary` | string | 추가 | 짧은 매칭 요약 |
| `match_details` | object | 추가 | concept/doc_type 매칭 상세 |
| `score_explanation` | object | 추가 | 점수 설명용 상세 |
| `evidence_summary` | object | 추가 | 누적 실적과 표시 evidence 수 요약 |
| `evidence` | object[] | 변경 | `chunk_id` 기반 matched evidence |
| `profile_evidence` | object[] | 추가 | 질의 매칭이 아닌 프로필 보강 evidence |
| `risks` | string[] | 유지 | 위험/주의사항 |
| `rank_score` | number | 유지 | RRF 기반 정규화 점수 |
| `data_gaps` | string[] | 추가 | 후보별 데이터 공백 |
| `fallback_reason_used` | boolean | 추가 | LLM 사유 누락 시 서버 fallback 사용 여부 |

### 3.3 evidence item

| 필드 | 타입 | 설명 |
|---|---:|---|
| `type` | string | `paper`, `patent`, `project`, `assessor_activity`, `specialty`, `profile` |
| `title` | string | 근거 제목 |
| `date` | string/null | 근거 날짜 |
| `detail` | string/null | 부가 설명 |
| `snippet` | string/null | 근거 본문 일부/요약 |
| `chunk_id` | string/null | evidence 식별자. 가능한 경우 이 값을 key로 사용 |
| `evidence_kind` | string | `matched` 또는 `profile` |

## 4. 피드백 - Feedback

`POST /feedback`

### 4.1 요청

```json
{
  "query": "인공지능 및 반도체 분야 평가위원 추천",
  "selected_expert_ids": ["12345678"],
  "rejected_expert_ids": ["99999999"],
  "notes": "적합한 평가위원이 배정됨",
  "metadata": {
    "operator": "tester"
  }
}
```

| 필드 | 타입 | 필수 | 변경 표시 | 설명 |
|---|---:|---:|---|---|
| `query` | string | Y | 유지 | 원 질의 |
| `selected_expert_ids` | string[] | N | 유지 | 선택된 전문가 ID |
| `rejected_expert_ids` | string[] | N | 유지 | 제외된 전문가 ID |
| `notes` | string/null | N | 유지 | 운영자 메모 |
| `metadata` | object | N | 추가 | 추가 메타데이터 |

### 4.2 응답

```json
{
  "feedback_id": 1,
  "stored": true
}
```

## 5. 헬스체크

### 5.1 앱 생존 확인

`GET /health`

```json
{
  "status": "ok",
  "collection_name": "researcher_recommend_v1",
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"]
}
```

변경: `searched_branches` 필드명은 유지되지만 값은 branch가 아니라 doc_type 5종이다.

### 5.2 준비 상태 확인

`GET /health/ready`

성공 시 `200 OK`, 준비 실패 시 `503 Service Unavailable`이며 본문 구조는 동일하다.

```json
{
  "ready": true,
  "checks": {
    "llm_backend_connected": true,
    "embedding_backend_connected": true,
    "collection_exists": true,
    "dense_vectors_present": true,
    "sparse_vectors_present": true,
    "sparse_vectors_idf": true,
    "payload_indexes_present": true,
    "sample_point_exists": true,
    "sample_payload_valid": true,
    "sample_root_fields": true,
    "sample_doc_type_valid": true,
    "sample_doc_attrs_present": true,
    "sample_doc_date_present": true
  },
  "issues": [],
  "collection_name": "researcher_recommend_v1",
  "sample_point_id": "11008395"
}
```

변경: 기존 문서의 `llm_backend`, `embedding_backend`, `qdrant_collection_exists`, `vectors_present` 키가 아니라 현재 구현의 상세 check 키가 내려온다.


## 6. 프론트 수정 체크리스트

- `degree_slct_nm` 요청 키를 `highest_degree`로 교체한다.
- `branch_presence_flags` 직접 참조를 제거하고 `doc_types_present`에서 화면용 flag를 계산한다.
- `counts` 표시 키를 현재 count 필드명으로 교체한다.
- `searched_branches` 값을 branch가 아니라 doc_type으로 표시/해석한다.
- `stable_hits`, `expanded_hits`, `support_branches` 기반 UI를 제거하거나 trace 기반 별도 디버그 UI로 격리한다.
- `support_rule_applied`를 비즈니스 로직 조건으로 사용하지 않는다.
- 추천 사유는 `recommendation_reason`을 우선 사용하고, 구형 호환만 필요할 때 `reasons[0]`를 fallback으로 사용한다.
- `/recommend/stream`은 호출하지 않는다.
