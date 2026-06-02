# 서비스 동작 흐름 (Service Flow) — flat chunk 파이프라인

**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)
**데이터 전제:** [`DATA_MODEL.md`](DATA_MODEL.md) / **설계 원칙:** [`DESIGN_GUIDELINES.md`](DESIGN_GUIDELINES.md)

런타임 파이프라인은 `planner → retrieval(chunk 검색 → 연구자 집계) → evidence_selector → reasoner` 4단계다. `RecommendationService.search_candidates()`가 진입점이며, `/recommend`는 여기에 evidence 선별 + 사유 생성을 더한다.

> **데이터 단위 전제(실데이터 기준).** 1 chunk = 1 Qdrant Point, Point ID == `chunk_id`. payload는 **flat**이다 — 연구자 공통 메타(`researcher_id`, `researcher_name`, `affiliated_organization`, `highest_degree`, `publication_count`, `scie_publication_count`, `intellectual_property_count`, `research_project_count`, `researcher_assessor_activity_count`)가 **root에 비정규화 반복**되고, doc_type별 상세만 `doc_attrs{}`에 들어간다. 날짜는 단일 `doc_date`(문자열, 결측이면 `"NONE"`)다. doc_type은 **paper / patent / project / assessor_activity / specialty 5종**이며, 모든 chunk에 동일하게 존재한다. 적재(ingestion)는 외부 제공자 소관이고 본 시스템은 검색·집계·추천만 담당한다.

---

## 1. 플래너 (Planner)

**역할:**
- 자연어 질의 정규화 (여러 줄은 `, `로 합쳐 단일 질의)
- 순수 도메인 명사 `core_keywords` / `retrieval_core` 추출
- 의미 검색 문장 `semantic_query` 생성
- 요청의 목적·역할 언어를 `task_terms`/`role_terms`/`action_terms`로 분리(검색 텍스트에서 제외)
- 명시 필터, include/exclude 기관, `top_k` 보존
- (선택) intent flags: 최근성 강조, 평가이력 강조 등 — 집계 prior 힌트로만 사용

**하지 않는 것:** doc_type on/off 결정, 검색 재작성 문장 임의 생성, 후보 판단.

> doc_type을 planner가 켜고 끄지 않는다. 모든 doc_type은 항상 검색 가능하며, 중요도는 prior(기본 equal)와 LLM 비교로 반영한다. doc_type 축소가 필요하면 운영 화이트리스트(`NTIS_RETRIEVAL_DOC_TYPES`, 기본 미설정=5종 전체)로만 한다. ([`ADR/0003-all-doc-types-searchable.md`](ADR/0003-all-doc-types-searchable.md))

---

## 2. 검색 및 집계 (Retrieval & Aggregation)

`QueryTextBuilder`는 1단계 sparse 키워드 검색에 `retrieval_core`/`core_keywords` 텍스트를, 2단계 하이브리드에 `semantic_query`(없으면 동일 키워드 텍스트)를 만든다. role/action 용어와 원본 질의는 검색 텍스트로 쓰지 않는다. v1.x 확장 사전은 제거됐으므로 trace 호환용 `expanded` 경로도 별도 확장어 없이 `stable`과 같은 텍스트를 사용한다.

`QdrantHybridRetriever` 동작 (모드 `keyword_pool_then_hybrid` 고정):

1. **1단계 — 키워드 풀:** `vector_splade`(sparse named vector)로 sparse 키워드 검색 → `researcher_id` 후보 풀을 중복 없이 수집. 풀이 비면 2단계 생략, 빈 결과 + `keyword_stage_candidate_count=0`.
2. **2단계 — 하이브리드:** 후보 풀을 `researcher_id MatchAny` 필터로 제한. doc_type(또는 family) 경로별로 `vector_e5i`(dense)+`vector_splade`(sparse) `prefetch` 조립 → `FusionQuery(RRF)`로 chunk hit 산출. doc_type은 named vector가 아니라 **payload 필터**로 분기한다.
3. **3단계 — 연구자 집계:** chunk hit을 `researcher_id`로 묶고 RRF 누적으로 연구자 점수 산출. 한 연구자의 동일 doc_type에서는 상위 N개 chunk만 점수에 기여(`doc_type_chunk_cap`, 기본 3), 집계 prior 적용(기본 equal), 연구자당 1건으로 dedupe.
4. **4단계 — hard filter:** `doc_date` 최근성(여러 doc_type은 OR/min_should), flat root `*_count` 최소 실적, 학위, 제외 기관을 deterministic 적용. 제외 기관은 root `affiliated_organization`(Qdrant 필터) + 매칭 chunk의 `doc_attrs.performing_organization`/`doc_attrs.managing_agency`(앱단 post-filter)로 교차 배제한다.
5. **5단계 — 결정론적 정렬:** score 내림차순 → `researcher_name` 오름차순 → `researcher_id` 오름차순.

각 후보에는 어떤 doc_type/chunk이 어떤 순위로 매칭됐는지 `retrieval_score_traces`로 기록한다.

---

## 3. 후보자 반환 (Candidate Return)

`/search/candidates`는 정렬된 후보 목록을 즉시 반환한다.
- `top_k` 명시 시 그 수만큼 제한, 아니면 전체 반환.
- 각 후보는 family별 보유 여부(`doc_type_coverage`)와 flat root 기반 카운트(`counts`)를 함께 노출.

---

## 4. 추천 결과 생성 (Recommendation Return)

`/recommend`는 검색 후 다음을 추가 수행한다.

1. 정렬된 상위 K명 선택 (검색 순서 유지).
2. **evidence 선별:** 후보별 매칭 chunk을 doc_type별로 모아 `core_keywords`/query 관련도로 재랭크(cross-encoder → 모델 부재 시 lexical 강등). family별 top-N chunk만 LLM 입력 풀로 구성. 각 chunk은 `chunk_id`를 그대로 보존한다(evidence 참조 id == `chunk_id`).
3. 최대 5명 단위 배치로 LLM에 전달. 입력 = 후보 머리(profile/flat 메타/평가이력 요약) + 선별 chunk 풀.
4. LLM은 후보별 `fit`, `recommendation_reason`, `selected_evidence_ids`(=고른 `chunk_id`), `risks`를 반환.
5. **검색 시 원본 순서 유지.** `selected_evidence_ids`로 최종 `recommendation.evidence`를 조립.
6. LLM이 사유를 누락/공란으로 두면 서버가 chunk 근거 기반 보수적 fallback 사유를 결정론적으로 생성.

**LLM이 하지 않는 것:** 후보 재정렬, 후보 탈락, 새 ID(연구자/chunk) 생성.

---

## 5. 빈 결과 / 실패 처리

플래너가 재시도 후에도 빈 `core_keywords`를 내면:
- 검색 단계 생략, `retrieval_skipped_reason`을 trace에 기록
- `/search/candidates`는 빈 목록, `/recommend`는 구조화된 사유와 함께 빈 추천 목록 반환

---

## 6. 추적 기록 (Trace Behavior)

활성 trace 필드:
- `planner` / `planner_trace` — 플래너 출력·실행 상세
- `raw_query`, `planner_keywords`, `retrieval_keywords` — 원본 질의/추출/실제 검색 키워드
- `reason_generation_trace` — 사유 생성 상세
- `retrieval_score_traces` — 후보별 매칭 doc_type/chunk과 순위 근거
- `query_payload.retrieval_mode` — `keyword_pool_then_hybrid` 고정
- `query_payload.retrieval_keywords` / `semantic_query` — 실제 검색 키워드/의미 문장
- `branch_queries.stable` / `branch_queries.expanded` — 1·2차 경로 호환용 검색 텍스트. lexicon 확장 제거 후 두 값은 보통 동일하다.
- `query_payload.keyword_stage_candidate_count` — 1차 sparse가 수집한 `researcher_id` 풀 크기
- `query_payload.hybrid_stage_raw_doc_type_counts` — 2차 doc_type 경로별 raw chunk hit 수
- `query_payload.aggregated_candidate_count` — 연구자 집계 후 후보 수
- `query_payload.support_pass_count` / `support_filtered_count` — hard filter 통과/탈락 수
- `server_logs` — Trace ID + `METHOD /path` 컨텍스트의 단계별 한글 로그
- `timers` — 구간별 실행 시간

> v1.x 대비 변경: `query_payload.keyword_stage_branch_counts`/`hybrid_stage_raw_branch_counts`의 "branch"가 "doc_type"으로 바뀐다(`*_doc_type_counts`). 외부 trace 변경은 [`../api/EXTERNAL_API_CHANGELOG.md`](../api/EXTERNAL_API_CHANGELOG.md) 참조.

---

## 7. 현재 active path에서 제거된 항목 (역사)

다음은 v1.x 반복에서 제거됐고 v2.x에서도 도입하지 않는다: verifier stage, retrieval views, branch query hints, judge map-reduce를 후보 판단 핵심으로 두는 구조, evidence resolver alignment stage. 사유 생성의 Map-Reduce는 토큰 절감 옵션으로만 존재하며 후보 순위를 바꾸지 않는다.
