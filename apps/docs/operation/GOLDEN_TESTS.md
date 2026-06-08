# Golden Tests (flat chunk 모델)

**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)

데이터 모델: [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md) · 흐름: [`../architecture/SERVICE_FLOW.md`](../architecture/SERVICE_FLOW.md) · 계약: [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md).

## Scenarios

### 1. 순수 키워드 추출
- Input: 요청 어투가 섞인 자연어 추천 질의
- Expected: planner가 도메인 `core_keywords`만 추출, role/action 어투는 검색 텍스트에서 제외.

### 2. 명시 Top-k 우선
- Input: 질의 + 명시 `top_k`
- Expected: 명시 `top_k`가 자연어 함의보다 우선, `/recommend`는 정확히 그 수만 반환.

### 3. 정렬 안정성
- Input: 동률 RRF 집계 점수가 둘 이상 나오는 질의
- Expected: 최종 순서 `score desc → researcher_name asc → researcher_id asc`.

### 4. search vs recommend 분리
- Input: 명시 한도 없는 질의
- Expected: `/search/candidates`는 집계·정렬된 전체 후보, `/recommend`는 정렬된 Top-k만.

### 5. 사유 생성은 재정렬하지 않음
- Input: 강한 후보가 여럿인 질의
- Expected: 정렬된 Top-k만 LLM에 전달, 반환 추천은 검색 순서 유지.

### 6. 빈 키워드 시 검색 생략
- Input: 안전한 `core_keywords`를 못 만드는 질의
- Expected: Qdrant 검색 생략, `retrieval_skipped_reason` trace 존재, `recommendations=[]`.

### 7. chunk = 1 Point 적재 (NEW)
- Input: 한 연구자에 paper 3건 + project 2건이 적재됨
- Expected: Qdrant에 5개 Point가 존재하고 각 payload root의 `chunk_id`가 전역 유일하며 authoritative evidence id다. 모두 동일 `researcher_id` 및 동일 flat root 공통 메타(`researcher_name`/`affiliated_organization`/`highest_degree` + count 5종)를 갖는다. Point ID는 `chunk_id` 권장이나 런타임 계약은 `payload.chunk_id` 기준이다.

### 8. 연구자 집계와 dedupe (NEW)
- Input: 한 연구자의 여러 doc_type chunk이 동시에 hit
- Expected: 결과에 연구자당 **1건**만 등장, 점수는 그 연구자 chunk hit들의 RRF 누적, `matched_doc_types`에 hit한 doc_type 나열.

### 9. doc_type별 chunk 캡 (NEW)
- Input: paper chunk을 매우 많이 가진 다작 연구자 vs 소수지만 고관련 연구자
- Expected: doc_type별 상위 `chunk_cap`(기본 3)개만 집계에 기여, 다작이 chunk 수만으로 순위를 독식하지 않음.

### 10. 통합 recency는 OR 결합 (NEW, 과거 장애 회귀 방지)
- Input: "최근 3년 활동" 의도(여러 doc_type 대상)
- Expected: `doc_date >= 올해-3년` 조건이 doc_type들에 대해 **OR(min_should, min_count=1)** 로 결합, 세 영역 모두 동시 충족하는 극소수만 남는 AND 회귀가 발생하지 않음. `doc_date`가 `"NONE"`/결측인 chunk은 datetime range에 매칭되지 않아 recency 대상에서 제외된다.

### 10.1 grouped relevance gate prevents sibling leakage
- Input: query asks for both AI and semiconductor expertise, and one researcher group contains an AI assessor chunk plus an unrelated paper chunk.
- Expected: unrelated sibling chunks do not contribute to candidate score or evidence. If the kept chunks do not collectively cover all active concepts, the researcher is filtered with `reason="relevance_concepts_missing"` and trace exposes `relevance_dropped_chunk_count` and `relevance_filtered_candidate_count`.

### 11. flat root 메타 기반 hard filter (NEW)
- Input: `publication_count_min` / `highest_degree` 필터
- Expected: flat root의 `publication_count`/`highest_degree`로 chunk 단계에서 deterministic 필터, 위반 후보 0건.

### 12. 소속 기관 include/exclude
- Input: `exclude_orgs` 지정
- Expected: root `affiliated_organization`만 정규화 비교해 include/exclude를 적용한다. `doc_attrs.performing_organization`/`doc_attrs.managing_agency`는 과제 속성이므로 소속기관 필터에 쓰지 않는다.

### 13. 평가이력 신호 (NEW)
- Input: "평가위원 경험이 풍부한" 의도
- Expected: `assessor_activity` chunk(assessment family)이 검색·evidence에 1급으로 포함, (옵트인 시) assessment family prior가 집계에 반영.

### 14. chunk_id 기반 evidence (NEW)
- Input: 추천 후보가 다수 관련 chunk 보유
- Expected: LLM에 family별 캡이 적용된 chunk 풀 전달, LLM은 풀의 `chunk_id`만 `selected_evidence_ids`로 복사, 최종 `recommendation.evidence`가 그 `chunk_id`로 resolve.

### 15. evidence id 계약과 fallback
- Input: LLM이 풀에 없는/형식이 깨진 id 반환
- Expected: 무효 id는 별도 기록(`invalid_selected_evidence_ids`), trace에 제공 id(`resolver_available_evidence_ids`)와 반환 id 동시 노출, 결정론적 chunk fallback으로 evidence 조립.

### 16. evidence 선별 캡
- Input: 매칭 chunk이 매우 많은 후보
- Expected: family별 캡(`achievement:10` 등) 적용, 후보 **순위는 변하지 않음**(evidence 선별은 grounding 한정).

### 17. channel-specific SearchQueryPlan 고정
- Input: "인공지능 분야 전문성과 반도체 연구개발 또는 반도체 산업 경험을 가진 연구자"
- Expected: `search_query_plan.dense_query`는 사용자 원문이고, `sparse_joint_query`는 `인공지능 반도체 연구개발 산업 경험`처럼 짧은 자연문/명사구다. `sparse_concept_queries`는 `ai`, `semiconductor`, `semiconductor_experience`를 포함하고, `required_concepts=["ai","semiconductor"]`가 post-group coverage gate에 쓰인다.

### 18. 확장 사전 제거
- Input: planner 출력에 legacy `bundle_ids`가 포함된 질의
- Expected: 검색 쿼리에 v1.x `basic`/`art`/`pat`/`pjt` 확장어가 추가되지 않으며, `branch_queries.expanded`는 `branch_queries.stable`과 동일하다.

### 19. 배치 사유 생성 + 서버 fallback
- Input: Top-k>5이고 LLM이 일부 후보를 누락/공란
- Expected: 5명 단위 순차 배치, 검색 순서 유지, 누락·공란 후보는 보수적 서버 fallback 사유, trace에 배치별 후보 id와 fallback 대상 노출.

### 20. 단계별 로깅
- Input: 정상 `/recommend` 또는 `/search/candidates`
- Expected: `trace.server_logs`가 `trace=<id>` + `[METHOD /path]` 한 줄 형식, 요청 시작·질의 수신·플래너·1차/2차 검색·집계·응답 준비 로그 포함, planner/retriever 로그에 실제 키워드·쿼리 텍스트·doc_type 경로 count 포함(벡터/전체 payload 미출력).

## Acceptance Criteria

- 저장 단위는 chunk이며, evidence id는 payload root `chunk_id`다. 한 연구자는 다수 Point.
- 검색은 항상 `grouped_hybrid_rrf`: dense_full + sparse_joint + sparse_concept_queries prefetch → equal RRF → post-group concept coverage gate.
- v1.x 확장 사전은 active 검색 경로에 없으며 `expanded` 쿼리는 별도 확장어를 추가하지 않는다.
- 검색 후 `researcher_id`로 집계해 연구자당 1건, 점수는 RRF 누적, doc_type별 `chunk_cap` 적용.
- hard filter는 시스템이 deterministic 보장(flat root 메타 기준), 다중 doc_type `doc_date` recency는 OR 결합. `doc_attrs.*`는 필터/인덱스 대상이 아니다.
- `/search/candidates`와 `/recommend`는 검색·집계 순서를 유지하고 `/recommend`는 Top-k만 LLM에 전달.
- evidence 선별은 family별 캡을 적용하되 후보 순위를 바꾸지 않는다.
- `recommendation.evidence`는 LLM이 고른 `chunk_id`로 resolve, 무효 시 결정론적 fallback.
- Trace는 `planner_keywords`, `retrieval_keywords`, `retrieval_skipped_reason`, `retrieval_score_traces`, `final_sort_policy`, `top_k_used`, `query_payload.retrieval_mode`, `query_payload.search_query_plan`, `query_payload.group_count`, `query_payload.aggregated_candidate_count`, `query_payload.relevance_gate_active_concepts`, `query_payload.relevance_kept_chunk_count`, `query_payload.relevance_dropped_chunk_count`, `query_payload.relevance_filtered_candidate_count`, `server_logs`, `reason_generation_trace.*`, 후보별 evidence resolution 상세를 노출한다.
- evidence id는 `chunk_id`이며, 구 `paper:N`/`project:N`/`patent:N` 형식은 더 이상 계약에 없다.
- 구 verifier / multi-view retrieval / branch named vector / judge-as-core 구조는 active 계약이 아니다.
