# 외부 노출 리턴 API 명세 변경기록

## 범위

- 대상 엔드포인트: `/recommend`, `/search/candidates`, `/feedback`, `/health`, `/health/ready`
- 포함 범위: HTTP 상태 코드, 응답 본문 필드, 하위 객체 필드, `trace` 계약, 하위 호환 alias
- 제외 범위: 내부 구현 변경만 있는 커밋, 요청 파라미터만 바뀐 커밋

본 문서는 `git log`와 실제 diff를 기준으로 외부 소비자가 체감하는 응답 계약 변경만 추렸습니다.

## 타임라인

| 날짜 | 커밋 | 영향도 | 변경 내용 |
|---|---|---|---|
| 2026-06-04 | `v2.1.1` | **정책 정정** | hard_filter와 payload index 대상을 flat root 필드로 제한했습니다. `doc_attrs.*`는 유동 상세 필드이므로 `journal_class`→`doc_attrs.indexing_database` 매핑과 project `performing_organization`/`managing_agency` 기반 기관 필터를 제거했습니다. 기관 include/exclude는 root `affiliated_organization` 앱단 post-filter만 사용합니다. |
| 2026-03-31 | `c4a3cfa` | 기준선 | 공개 응답 스키마 초안이 추가되었습니다. `/recommend`는 `intent_summary`, `applied_filters`, `searched_branches`, `retrieved_count`, `recommendations`, `data_gaps`, `not_selected_reasons`, `trace` 구조를 갖고, `/search/candidates`는 `candidates[*].branch_coverage`를 사용했습니다. |
| 2026-04-06 | `368cb28` | 동작 변경 | `/recommend`가 더 이상 "후보 없음" 또는 "evidence 없음" 때문에 `500`을 내지 않고, `200 OK`와 함께 `recommendations=[]`를 반환하도록 바뀌었습니다. 빈 결과 사유는 `not_selected_reasons`에 남기도록 정리되었습니다. |
| 2026-04-14 | `9b71535` | 경미 | 응답 최상위 스키마 변경은 없지만, 검색/추천 결과의 `trace.include_orgs`가 노출되기 시작했습니다. 같은 커밋에서 `top_k` 상한이 `5 -> 15`로 바뀌었지만 이는 요청 계약 변경입니다. |
| 2026-04-14 | `79328b9` | 확장 | `/search/candidates.trace`에 `planner_trace`, `raw_query`, `retrieval_keywords`, `planner_retry_count`, `retrieval_skipped_reason`, `include_orgs`, `timers`가 추가되었습니다. 디버그 UI 의존성이 있다면 이 시점부터 새 필드를 읽을 수 있습니다. |
| 2026-04-15 | `b9084c4` | 주요 변경 | `/recommend.recommendations[*]`의 핵심 사유 필드가 `reasons: list[str]`에서 `recommendation_reason: str`로 바뀌었습니다. 동시에 `trace.judge_trace`, `trace.evidence_resolution_trace`, `trace.recommendation_evidence_summary`, `trace.final_reduce_*`, `trace.planner_raw_keywords`, `trace.verifier_*`가 제거되고, `trace.reason_generation_trace`, `trace.planner_keywords`, `trace.recommendation_ids`, `trace.final_sort_policy`, `trace.top_k_used`가 추가되었습니다. 또한 `/search/candidates`는 이 시점부터 요청의 `top_k`를 실제 응답 개수 제한에 반영합니다. |
| 2026-04-15 | `8c4b8e7` | 하위 호환 | 구 프런트 호환을 위해 `/recommend.recommendations[*].reasons`가 계산 필드 alias로 다시 제공되었습니다. 기본 계약은 여전히 `recommendation_reason`이며, `reasons`는 구형 소비자 보호용입니다. 같은 시점에 `trace.reason_generation_trace.evidence_selection`이 중첩 추적으로 추가되었습니다. |
| 2026-04-15 | `3c9bceb` | 주요 변경 | `/search/candidates.candidates[*].branch_coverage`가 `branch_presence_flags`로 이름이 바뀌었습니다. `/search/candidates.trace`에는 `retrieval_score_traces`와 `final_sort_policy`가 추가되어, 최종 점수와 branch 매칭 근거를 외부에서 직접 볼 수 있게 되었습니다. |
| 2026-04-23 | `pending` | 확장 | 검색 동작이 고정 2단계(`keyword_pool_then_hybrid`)로 바뀌었습니다. 외부 최상위 스키마는 유지되지만 `trace.query_payload`에서 `retrieval_mode`, `keyword_stage_candidate_count`, `keyword_stage_branch_counts`, `hybrid_stage_candidate_filter_count`, `hybrid_stage_raw_branch_counts`, `aggregated_candidate_count`, `support_pass_count`, `support_filtered_count`를 확인할 수 있습니다. 같은 변경에서 `trace.server_logs`에는 `trace=<id>`와 `[METHOD /path]`를 포함한 사용자 질의, 플래너, 1차 검색, 2차 검색, 응답 준비 단계별 요약 로그가 포함됩니다. |
| 2026-04-23 | `pending` | 확장 | 운영 로그와 `trace.query_payload`에 실제 키워드 추출 결과와 검색 쿼리 텍스트가 추가되었습니다. `trace.server_logs`는 `retrieval_core`, `core_keywords`, `role_terms`, `action_terms`, `semantic_query`, 실제 `retrieval_keywords`, 1차/2차 검색 텍스트를 값 그대로 보여줍니다. v1.x 확장 사전 기반 `bundle_ids`는 active 검색 경로에서 제거됐습니다. |
| 2026-05-28 | `v2.0` | **주요 변경(BREAKING)** | chunk 데이터 모델 재설계(저장 단위 "연구자 1 Point" → "chunk 1 Point"). 외부 응답에서 다음이 변경됩니다 — `recommendations[*].evidence[*].type`이 4종(`paper/patent/project/profile`)에서 doc_type 문자열로 확장; `evidence[*]`에 **`chunk_id` 추가**; `counts`가 연구자 누적 실적 기반 명칭(`publication_count` 등)으로 변경. 컬렉션 기본값 `researcher_recommend_proto` → `ntis_researcher_chunks`. *(이 시점 설계 문서는 11 doc_type / `searched_doc_types` 리네임 / `doc_type_coverage`(family) 등을 예고했으나, 실제 적재 데이터와 코드는 아래 v2.1에서 flat·5 doc_type으로 확정되었습니다.)* |
| 2026-06-02 | `v2.1` | **주요 변경(BREAKING)** | **flat payload 확정.** 실제 적재 데이터는 평탄 chunk payload(연구자 공통 메타 root 비정규화 + doc_type별 `doc_attrs{}`)이며 doc_type은 **정확히 5종**(`paper`/`patent`/`project`/`assessor_activity`/`specialty`)입니다. `recommendations[*].evidence[*].type`은 이 5종 또는 합성 `profile`이고, 식별자는 `chunk_id`(형식 `<doc_type>_<숫자doc_id>_c<NNN>`)입니다. `/search/candidates.candidates[*]`는 family boolean이 아니라 **`doc_types_present`(hit한 doc_type 문자열 목록)** 를 노출하며, `counts`는 flat root 단일 평가위원 카운트를 포함한 5개 키입니다. hard_filter는 flat root 키(`highest_degree`, root count `*_count_min`, `recent_years`+`recent_doc_types` on `doc_date`)입니다. `doc_attrs.*`는 필터/인덱스 대상이 아닙니다. |
| 2026-06-04 | `pending` | 동작/trace 변경 | 검색 쿼리가 `SearchQueryPlan`으로 분리되었습니다. dense는 사용자 원문 중심 `dense_query`, SPLADE는 짧은 `sparse_joint_query`와 `sparse_<concept>` 보조 쿼리를 사용합니다. `trace.query_payload.search_query_plan`에 `raw_query`, `dense_query`, `sparse_joint_query`, `sparse_concept_queries`, `required_concepts`, `optional_concepts`가 노출됩니다. 검색 모드는 `grouped_hybrid_rrf`이며, 최종 연구자 후보는 `required_concepts` coverage gate를 통과해야 합니다. evidence 식별 기준은 Qdrant Point ID가 아니라 payload root `chunk_id`입니다. |

## v2.1 flat payload 확정 (실데이터 정합, 2026-06-02)

> v2.0이 예고했던 "11 doc_type / `searched_doc_types` 리네임 / `doc_type_coverage`(family 단위) / `matched_doc_types` / `researcher_meta` 중첩 카운트"는 **실제 적재 데이터·코드와 어긋났습니다.** v2.1은 실데이터(flat payload, doc_type 5종)에 외부 계약을 맞춘 확정본입니다. 배경은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md).

### A. doc_type 5종 확정 (BREAKING)

- 검색 대상 doc_type은 **정확히 5종**: `paper` / `patent` / `project` / `assessor_activity` / `specialty`. (구 11종 분기 폐기)
- 실데이터 분포(참고): paper 64.6% / patent 18.7% / assessor_activity 9.7% / project 6.5% / specialty 0.5%.
- `recommendations[*].evidence[*].type`은 위 5종 doc_type 문자열 또는 합성 신원 근거 `profile`. *(v2.0 문서가 예고한 11종 문자열은 실재하지 않습니다.)*

### B. flat payload 계약 (BREAKING)

- payload는 평탄(flat): 연구자 공통 메타(`researcher_id`, `researcher_name`, `affiliated_organization`, `highest_degree`, `publication_count`, `scie_publication_count`, `intellectual_property_count`, `research_project_count`, `researcher_assessor_activity_count`)가 **root에 비정규화 반복** 저장되고, doc_type별 상세만 `doc_attrs{}`에 들어갑니다. `researcher_meta` 중첩은 없습니다.
- 날짜는 단일 `doc_date`(문자열, 결측은 `"NONE"`). `event_date`/`event_year`는 없습니다.
- `doc_attrs` 알려진 키: paper={indexing_database, journal_name, keywords, main_language_title, sub_language_title, is_scie, publication_year_month}, project={project_title_korean, project_title_english, project_period, performing_organization, managing_agency}, patent={intellectual_property_title, intellectual_property_type, application_registration_type, application_country, application_number, application_date, intellectual_property_foreign_type}. **assessor_activity / specialty의 `doc_attrs` 키는 미상(untyped passthrough)** 이며 본 서비스는 임의 키를 만들지 않습니다.
- `counts`(`/search/candidates`)는 flat root 기반 5개 키: `publication_count`, `scie_publication_count`, `intellectual_property_count`, `research_project_count`, `researcher_assessor_activity_count`. 구 분리 평가위원 카운트(`researcher_assessor_count`/`expert_assessor_count`)는 **단일 `researcher_assessor_activity_count`로 병합**됩니다.

### C. 응답 필드 (실제 코드 기준)

- 검색 대상 영역 필드는 응답에서 여전히 **`searched_branches`** 라는 이름으로 노출되며, 값은 doc_type 5종 문자열 목록입니다(`/recommend`, `/search/candidates`, `/health`). *(v2.0 문서가 예고한 `searched_doc_types` 리네임은 코드에 반영되지 않았습니다 — 필드명은 `searched_branches` 유지, 값만 doc_type으로 전환.)*
- `/search/candidates.candidates[*]`는 family boolean 맵(`doc_type_coverage`)이나 `matched_doc_types`가 아니라 **`doc_types_present`**(이번 검색에서 hit한 doc_type 문자열 목록) 단일 필드를 노출합니다.
- `recommendations[*].evidence[*].chunk_id`는 evidence 참조 식별자입니다(불변).
- 컬렉션 기본값 `ntis_researcher_chunks`. named vector는 단일 `vector_e5i`(dense, 1024, Cosine) + 단일 `vector_splade`(sparse). doc_type은 **payload 필터**이며 별도 named vector가 아닙니다.

### D. hard_filter 키 (flat)

- 모든 hard_filter 백엔드 경로는 flat root: `highest_degree`, `*_count_min`(root count, 단일 평가위원 `researcher_assessor_activity_count_min` 포함), `recent_years`+`recent_doc_types`는 root **`doc_date`(datetime)** 기준. `doc_attrs.*`는 유동 상세 필드이므로 필터/인덱스 대상에서 제외한다. (구 `researcher_meta.*` 중첩 경로, `event_year` 폐기)
- 다중 doc_type recency는 `min_should(min_count=1)` **OR**로 묶입니다(AND 0건 회귀 방지).

### 소비자 가이드

- doc_type을 11종으로 가정한 코드는 **5종**(`paper`/`patent`/`project`/`assessor_activity`/`specialty`)으로 좁혀야 합니다.
- `searched_doc_types`/`doc_type_coverage`/`matched_doc_types`를 기대하던 v2.0 예고 기반 코드는 실제 응답의 `searched_branches`/`doc_types_present`로 교체해야 합니다.
- payload에서 `researcher_meta.*` 중첩이나 `event_date`/`event_year`를 읽는 코드는 flat root 필드와 `doc_date`로 교체해야 합니다.
- 근거를 식별·dedupe할 때는 `evidence[*].chunk_id`를 사용합니다.
- trace를 읽는 운영 도구는 `trace.query_payload.search_query_plan`을 우선 사용해야 합니다. SPLADE 디버깅 시 사용자 원문이 아니라 `sparse_joint_query`와 `sparse_concept_queries`가 실제 검색 텍스트입니다.

## v2.0 재설계 (chunk 데이터 모델, 2026-05-28)

> ⚠️ **이 절은 v2.0 설계 시점의 예고이며, 일부 항목은 실데이터·코드와 어긋나 v2.1에서 정정되었습니다.** 외부 계약의 사실 기준은 위 **v2.1 flat payload 확정** 절입니다. 특히 ① doc_type은 11종이 아니라 **5종**, ② 응답 필드는 `searched_doc_types`가 아니라 실제로는 **`searched_branches`(값만 doc_type)**, ③ `doc_type_coverage`(family)/`matched_doc_types`가 아니라 **`doc_types_present`**, ④ payload는 flat(`researcher_meta` 중첩 없음)입니다. 아래는 역사적 기록으로 남깁니다.

> 데이터 적재 단위가 `chunk_id`/`doc_type` 기반으로 바뀌면서, 저장 단위를 "연구자 1 Point"에서 "chunk 1 Point"로 전환했습니다. 배경은 [`../architecture/ADR/0002-chunk-level-point-model.md`](../architecture/ADR/0002-chunk-level-point-model.md).

### A. 필드 이름 변경 (BREAKING)

| 구 필드 | v2.0 예고 | 실제(v2.1) | 위치 |
|---|---|---|---|
| `searched_branches`(값=branch) | `searched_doc_types` | `searched_branches`(필드명 유지, 값=doc_type 5종) | `/recommend`, `/search/candidates`, `/health` |
| `candidates[*].branch_presence_flags` | `doc_type_coverage`(family) + `matched_doc_types` | `candidates[*].doc_types_present`(doc_type 문자열 목록) | `/search/candidates` |
| `trace.query_payload.keyword_stage_branch_counts` | `...keyword_stage_doc_type_counts` | `...keyword_stage_path_counts` | trace |
| `trace.query_payload.hybrid_stage_raw_branch_counts` | `...hybrid_stage_raw_doc_type_counts` | `...hybrid_stage_path_counts` | trace |

### B. evidence 구조 변경 (BREAKING)

- `recommendations[*].evidence[*].type`이 `paper/patent/project/profile`에서 doc_type 문자열로 확장됩니다. *(v2.0 예고는 11종이었으나 실데이터 기준은 v2.1의 5종 — `paper`/`patent`/`project`/`assessor_activity`/`specialty` + 합성 `profile`.)*
- `evidence[*]`에 불변 식별자 **`chunk_id`** 가 추가됩니다.
- 내부 LLM 계약의 `selected_evidence_ids`도 `paper:N` 형식에서 **`chunk_id`** 로 바뀝니다(외부 응답에는 노출되지 않던 내부 계약이지만, trace를 파싱하는 운영 UI는 영향).

### C. 신규 필드

- `/search/candidates.candidates[*]`에 이번 검색에서 hit한 doc_type 목록이 노출됩니다. *(v2.0 예고 명칭 `matched_doc_types`는 실제 코드에서 `doc_types_present`로 확정 — v2.1 참조.)*
- `/recommend.recommendations[*].evidence[*].chunk_id`.

### D. 컬렉션/스키마 변경

- 기본 컬렉션 `researcher_recommend_proto` → `ntis_researcher_chunks`.
- named vector `basic/art/pat/pjt_vector_*` 4쌍 → 단일 dense `vector_e5i`(1024, Cosine) + 단일 sparse `vector_splade`. *(v2.1 확정 명칭. v2.0 초안의 `dense_e5i`/`sparse_splade`는 실제 코드와 다릅니다.)*
- 외부 API 호출 형태(요청 스키마)는 변하지 않습니다.

### 소비자 가이드

- `branch_presence_flags`를 직접 참조하던 프런트는 깨집니다 → 실제 응답은 `candidates[*].doc_types_present`(v2.1, doc_type 문자열 목록)입니다.
- `evidence[*].type`을 4종 enum으로 가정한 코드는 doc_type 5종(+`profile`) 문자열을 허용하도록 확장해야 합니다.
- 근거를 식별·dedupe할 때는 `evidence[*].chunk_id`를 사용하는 것이 안전합니다.

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
- 추가된 축: `reason_generation_trace`, `retrieval_score_traces`, `recommendation_ids`, `final_sort_policy`, `top_k_used`, `query_payload.retrieval_mode`, `query_payload.keyword_stage_candidate_count`, `query_payload.hybrid_stage_raw_branch_counts`, `query_payload.aggregated_candidate_count`, `query_payload.support_pass_count`, `query_payload.support_filtered_count`, `server_logs`
- 소비자 영향: 운영 UI가 `trace`를 강하게 파싱한다면, 이 변경점을 기준으로 버전 분기 또는 방어 코드를 둬야 합니다.

## 현재 기준 호환성 메모 (v2.1 기준)

- `/recommend`의 1차 계약 필드는 `recommendation_reason`입니다(`reasons` alias는 하위 호환 전용).
- 검색 대상 영역 필드는 응답에서 **`searched_branches`** 이며 값은 doc_type 5종 문자열 목록입니다(`paper`/`patent`/`project`/`assessor_activity`/`specialty`). `/search/candidates`의 후보별 보유 doc_type은 **`candidates[*].doc_types_present`**(문자열 목록)입니다. *(family boolean 맵 `doc_type_coverage`나 `matched_doc_types`는 실재하지 않습니다.)*
- 근거 식별자는 `evidence[*].chunk_id`(형식 `<doc_type>_<숫자doc_id>_c<NNN>`)이며, `evidence[*].type`은 doc_type 5종 또는 `profile`입니다.
- payload는 flat입니다(연구자 공통 메타 root 비정규화, doc_type별 `doc_attrs{}`, 단일 `doc_date`). `researcher_meta` 중첩, `event_date`/`event_year`, `domain_attrs`, `tags`, `chunk_text_len`은 존재하지 않습니다.
- 컬렉션 기본값은 `ntis_researcher_chunks`, named vector는 단일 `vector_e5i`(dense, 1024, Cosine) + `vector_splade`(sparse)이며 doc_type은 payload 필터입니다.
- `trace.query_payload.search_query_plan`은 dense/sparse/concept별 실제 검색 텍스트와 `required_concepts`를 담습니다. SPLADE 검색 텍스트는 사용자 원문이 아니라 `sparse_joint_query` 및 `sparse_concept_queries`입니다.
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
