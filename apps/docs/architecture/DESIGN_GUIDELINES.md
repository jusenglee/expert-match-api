# 평가위원 추천 API
## 설계 · 구현 · 제한 · 방향성 지침서

**기준선:** Qdrant v1.16.0 / Named Vector + Hybrid Retrieval / Deterministic Gate + Summary-only Reasoner
**문서 버전:** v2.0
**기준일:** 2026-04-17

---

## 문서 개요

| 항목 | 내용 |
|---|---|
| 문서 목적 | 평가위원 추천 API 의 확정 설계, 구현 원칙, 제약 사항, 향후 방향을 개발팀·기획팀 기준으로 일관되게 문서화 |
| 고정 기준 | Qdrant v1.16.0 / named vector / prefetch 동시검색 / 가중치 미사용 / equal-weight RRF / evidence 선택은 서버, 사유 요약은 LLM |
| 검색 철학 | Retrieval 은 후보 생성·정렬 보조. Evidence 선택·shortlist gate·validator 는 서버 deterministic. LLM 은 이미 선택된 evidence 의 요약만 담당 |
| 설계 단위 | 전문가 1명 = 1 point, 기본정보(root) + 논문/특허/과제 nested payload |
| 문서 버전 | v2.0 (Map-Reduce Judge 제거, Summary-only Reasoner + Evidence Aspects v0.7.0 + equal-weight RRF 반영, 2026-04-17) |

> **현재 문서에서 확정한 핵심 원칙**
>
> - 별도 리랭커(예: cross-encoder)를 최종 의사결정의 핵심으로 두지 않는다.
> - Evidence 선택은 서버가 결정하고, LLM 은 선택된 evidence 만 요약한다.
> - Qdrant v1.16.0 범위 안에서 `prefetch + equal-weight RRF` 를 사용하며, weighted RRF 는 설계 범위에서 제외한다.
> - 숫자/날짜/기관 제외 같은 hard filter 는 LLM 이 아니라 시스템이 보장한다.
> - 추천 결과에는 반드시 근거, 제외 사유, 데이터 공백을 함께 제시한다.
> - 구현 기본값은 `basic/art/pat/pjt` 전 branch 항상 검색이며, branch 문맥은 query text 가 아닌 retriever 의 branch-specific instruct prefix 가 담당한다.

> **용어 정리**
>
> - **Retrieval:** Qdrant 에서 후보군을 찾는 단계
> - **Evidence Selection:** 검색된 후보에 대해 server 가 직접 매치 evidence 를 고르는 단계
> - **Shortlist Gate:** Evidence 기반으로 후보를 재배치/제외하는 deterministic 게이트
> - **Reason Generation:** 선택된 evidence 를 LLM 이 자연어로 요약하는 단계
> - **Validator:** LLM 사유의 사실성·일관성을 검사하는 deterministic 후처리
> - **Hard filter:** 숫자·날짜·기관 제외처럼 시스템이 강제 보장해야 하는 조건
> - **Evidence card:** LLM 입력용으로 압축한 후보 요약 카드

---

## 1. 문서 목적과 범위

이 문서는 평가위원 추천 API 의 기준 설계를 고정하기 위한 실행 문서이다. 목적은 기술 스택, 데이터 모델, 검색 전략, 사유 생성 규칙, 운영상 제한과 향후 방향을 한 문서에 통합하여 설계 흔들림을 줄이는 데 있다.

- **포함 범위:** 컬렉션 설계, ingestion, retrieval, evidence selection, shortlist gate, reason generation, validator, 필터링, 평가 지표, 운영 가이드, 확장 방향
- **비포함 범위:** 프런트엔드 UI 상세 설계, 인프라 IaC 스크립트, 보안 인증 체계의 세부 구현
- **문서의 우선순위:** 정확한 기준선 수립 > 운영 가능성 > 확장성 > 시각적 완성도

이 문서는 상위 기준선만 정의한다. 계약 수준의 세부는 다음 문서를 따른다.

- 실행 흐름 세부: [`SERVICE_FLOW.md`](SERVICE_FLOW.md)
- 데이터/필드 계약: [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md)
- 사유 생성 정책: [`../api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md)
- 외부 노출 응답 계약 이력: [`../api/EXTERNAL_API_CHANGELOG.md`](../api/EXTERNAL_API_CHANGELOG.md)

---

## 2. 확정된 설계 기준

이 기준은 이후 구현 세부사항보다 상위에 있는 설계 원칙이다. 구현 편의를 위해 가중치나 외부 리랭커를 다시 들여오는 것은 기본선 위반으로 본다.

| 항목 | 최종 결정 | 설계 의미 |
|---|---|---|
| Qdrant 버전 | v1.16.0 고정 | 기능 범위와 제약을 명확히 고정 |
| 검색 방식 | named vector + sparse vector 하이브리드 | 기본정보/논문/특허/과제별 표현 유지 |
| 동시검색 | `prefetch` 사용 | 여러 representation 을 한 요청에서 조합 |
| Fusion | equal-weight RRF | weighted RRF / 직접 score 가중합은 설계 범위 제외 |
| Branch 문맥 | retriever 의 branch-specific e5-instruct prefix | branch hint 를 query text 에 덧붙이지 않음 |
| Evidence 선택 | 서버 deterministic (직접 매치) | LLM 의 evidence hallucination 차단 |
| Shortlist gate | 서버 deterministic (점수 가중합 없음) | 규칙 누락/환각 방지 |
| 사유 생성 | LLM summary-only | 서버가 고른 evidence 를 요약 |
| Validator | deterministic 후처리 | 다른 후보명/내부 id 누설, 무근거 강한 주장 차단 |
| 후보 단위 | 전문가 1명 = 1 point | 검색과 추천 단위 일치 |
| 필터 처리 | 시스템에서 deterministic 보장 | LLM 환각과 조건 누락 방지 |
| 출력 원칙 | 추천 + 근거 + 제외 사유 + 데이터 공백 | 운영자 검토 가능성 확보 |
| 임베딩 정제 | 원문 질의 배제 및 순수 도메인 키워드 조합 | `평가위원` 등 메타어로 인한 임베딩 오염 차단 |

---

## 3. 서비스 목표와 성공 기준

### 3.1 서비스 목표

이 시스템의 목표는 자연어 요청을 입력받아 적합한 평가위원 후보군을 탐색하고, 서버가 고른 evidence 를 근거로 LLM 이 요약 설명을 생성하는 것이다. 일반 QA 형 RAG 와 달리, 핵심은 **정답 문장 회수** 가 아니라 **후보 비교와 추천 근거 제시** 에 있다.

- 사용자 질의의 주제, 맥락, 제외 조건, 최근성 기준을 이해한다.
- 논문·특허·과제·기본 프로필을 함께 반영하여 후보군을 넓게 확보한다.
- Server 가 evidence 를 고르고, shortlist gate 가 후보를 재배치한다.
- LLM 은 선택된 evidence 만 요약하고, validator 가 사유를 검증한다.
- 각 추천 결과에 대해 근거, 주의점, 데이터 공백을 명시한다.

### 3.2 성공 기준

- Hard filter 위반 없이 후보를 반환할 것
- 추천 결과 상위권에 실제로 검토 가능한 전문가가 포함될 것
- 추천 사유가 server-selected evidence 와 일치할 것
- Validator fallback 비율이 운영 지표로 추적 가능할 것
- 운영자가 수작업으로 수정하는 비율이 점차 감소할 것

---

## 4. 전체 아키텍처

```text
[사용자]
   ↓
[Planner]
   - 질의 요약 (intent_summary)
   - 메타어 제거 (평가위원/전문가/추천 → removed_meta_terms)
   - retrieval_core 생성 (한국어, sparse BM25 용)
   - semantic_query 생성 (자연어 한 문장, dense 용)
   - must_aspects 정제 (의미 품질 fallback, 한국어)
   - evidence_aspects 생성 (한국어+영어 이중어, evidence 매칭용)
   - intent_flags.review_context / review_targets 기록
   ↓
[Query Builder]
   - per-branch stable_dense / stable_sparse / expanded_dense / expanded_sparse 조립
   - 확장은 텍스트가 실제로 달라질 때만 실행
   ↓
[Qdrant Hybrid Retriever]
   - basic/art/pat/pjt 4 branch 동시 검색
   - dense prefetch (branch-specific e5-instruct prefix)
   - sparse prefetch (PIXIE-SPLADE / Qdrant/bm25 fallback)
   - branch-local RRF → cross-branch equal-weight RRF
   ↓
[Candidate Card Builder]
   - top-N 후보를 LLM / 응답용 카드로 압축
   ↓
[/search/candidates 분기점]
   - retrieval 후 카드만 반환 (evidence / reason 단계 없음)
   ↓
[Evidence Selector] (/recommend 경로만)
   - 직접 매치 evidence 만 선택 (evidence_aspects 우선, must_aspects → retrieval_core → core_keywords fallback)
   - title+year dedup
   - aspect 당 최대 2건, 후보 당 최대 4건
   - future-dated 과제도 허용, trace 에 future_selected_evidence_ids 노출
   ↓
[Shortlist Gate]
   - direct_match_count == 0 → drop
   - aspect_coverage < min(2, len(target_aspects)) → demote
   - generic_only == true → bottom group
   ↓
[Reasoner] (LLM, summary-only)
   - 배치 크기 5
   - tool calling 1회 + 결과 부족 시 compact retry 1회
   - 입력: selected_evidence + retrieval_grounding (primary_branch / final_score / branch_matches)
   - 출력: expert_id / fit / recommendation_reason / risks (evidence 는 선택하지 않음)
   ↓
[Reason Sync Validator]
   - 다른 후보명 / 내부 evidence id 누설 탐지
   - must_aspects (한국어) 범위 일관성 검사
   - 무근거 강한 주장 탐지
   - 실패 시 server fallback 사유로 치환
   ↓
[응답 생성기]
   - 최종 추천 결과 반환 (빈 결과도 200 OK)
   ↓
[Observability]
   - planner_trace / retrieval / evidence_selection / shortlist_gate / reason_generation_trace / reason_sync_validator
```

| 구성요소 | 역할 | 비고 |
|---|---|---|
| Planner | 질의 요약, retrieval_core/semantic_query/must_aspects/evidence_aspects/intent_flags 생성 | JSON-only 출력 강제, `<thinking>` 태그 미사용 |
| Query Builder | stable/expanded × dense/sparse 조립 | branch hint 는 query text 에 붙이지 않음 |
| Hybrid Retriever | 4 branch 동시 검색, equal-weight RRF | branch-specific e5-instruct prefix 로 dense 문맥 이동 |
| Candidate Card Builder | LLM/응답용 증거 요약 | 토큰 비용과 잡음을 줄이는 단계 |
| Evidence Selector | 직접 매치 evidence 선택 (`evidence_aspects → must_aspects → retrieval_core → core_keywords`) | 이중어 매칭으로 영문 논문·과제 영문 제목 커버 |
| Shortlist Gate | direct match / aspect coverage / generic_only 기준 재배치 | 점수 가중합 없이 순차 규칙 |
| Reasoner | LLM summary-only, batch 5, compact retry 1회 | evidence 는 선택하지 않음 |
| Validator | deterministic 후처리, fallback 생성 | `title + detail + snippet` 범위로 일관성 검사 |
| Observability | planner / selector / gate / reasoner / validator 단계별 trace·로그 | `aspect_source` 등 운영 디버그 필드 노출 |

---

## 5. 데이터 및 컬렉션 설계

### 5.1 컬렉션 단위

권장 컬렉션은 `researcher_recommend_proto` 하나. 한 point 는 한 명의 전문가를 의미한다. 이 구조는 추천 단위와 저장 단위를 일치시켜 retrieval 이후 후처리를 단순화한다.

### 5.2 named vector 구성

| 구분 | 벡터명 | 권장 source text | 사용 목적 |
|---|---|---|---|
| Dense | `basic_vector_e5i` | 이름/소속/직위/학위/전공/기술분류 요약 | 전문가 프로필 의미 검색 |
| Dense | `art_vector_e5i` | 대표 논문명/학술지명/초록/키워드 요약 | 학술 성과 의미 검색 |
| Dense | `pat_vector_e5i` | 발명명/특허 구분/출원·등록 정보 요약 | 지식재산 기반 의미 검색 |
| Dense | `pjt_vector_e5i` | 과제명/목표/내용/기관 요약 | 과제 경험 의미 검색 |
| Sparse | `basic_vector_bm25` | 기본정보 기반 토큰화 텍스트 | exact keyword 보강 |
| Sparse | `art_vector_bm25` | 논문명/학술지/키워드 | 논문 keyword 매칭 |
| Sparse | `pat_vector_bm25` | 발명명/특허 분류/기관명 | 특허 keyword 매칭 |
| Sparse | `pjt_vector_bm25` | 과제명/과제내용/기관명 | 과제 keyword 매칭 |

### 5.3 payload 구조

루트 레벨(`basic_info`, `researcher_profile`) 과 nested 레벨(`publications[]`, `intellectual_properties[]`, `research_projects[]`) 로 구성된다. 전체 필드 목록과 필터 적용 규약, evidence selector 가 각 item 유형별로 탐색하는 title/body 필드는 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md) 를 참조한다.

### 5.4 구현 전 정합성 확인 포인트

> **업로드 설계안 기준 구현 전 확인 필요**
>
> - 정정: 과제 영역 sparse vector 컬럼명은 `pjt_vector_bm25` 를 사용한다.
> - 정정: 과제 날짜 매핑은 `project_start_date = TOT_RSCH_START_DT`, `project_end_date = TOT_RSCH_END_DT`, `reference_year = STAN_YR` 로 구현한다.
> - 권장: 기술분류는 최소한 `basic_vector` source text 에는 반드시 합성한다.
> - 권장: IRIS 계열 데이터는 1차 검색 스키마와 분리하고, candidate card enrichment 용도로 결합한다.

---

## 6. 검색 설계 (Qdrant v1.16.0 기준)

### 6.1 검색 철학

검색 단계의 목적은 **정답 1명을 바로 고르는 것** 이 아니라, **적절한 후보군을 넓게 확보하는 것** 이다. Retrieval 은 recall 우선, evidence/gate/reasoner 단계에서 정확도를 보강한다.

- 기본정보/논문/특허/과제는 서로 다른 representation 으로 유지한다.
- 한 질의에서 `basic/art/pat/pjt` branch 는 모두 항상 검색한다 ([ADR-0001](ADR/0001-all-branches-on.md)).
- 가중치 대신 branch-specific e5-instruct prefix 와 evidence selection 으로 검색 전략을 보정한다.
- 최종 Top-N 는 shortlist gate + reasoner 조합이 결정하므로, retrieval 단계에서는 증거가 풍부한 후보를 충분히 수집하는 것이 중요하다.

### 6.2 Planner 의 역할

Planner 는 `PlannerOutput` 을 반환한다. 각 필드 책임은 다음과 같으며, 상세 계약은 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md) 에 있다.

| 필드 | 언어 | 용도 |
|---|---|---|
| `intent_summary` | 한국어 | 질의 주제 요약 |
| `retrieval_core` | **한국어 전용** | sparse BM25 질의어 |
| `semantic_query` | 자연어 한 문장 | dense 임베딩 질의 텍스트 |
| `must_aspects` | **한국어 전용** | 의미 품질 설명 및 fallback 게이트 (pruned) |
| `evidence_aspects` | **한국어 + 영어 이중어** | evidence 직접 매칭 primary (planner v0.7.0+) |
| `generic_terms` / `role_terms` / `action_terms` | 한국어 | 메타·범용어 분리 |
| `intent_flags.review_context` / `review_targets` | boolean / string[] | `과제 평가`, `기술 평가`, `논문 평가` 같은 리뷰 맥락 |
| `hard_filters` | dict (허용 키 한정) | `degree_slct_nm`, `art_recent_years`, `pat_recent_years`, `pjt_recent_years`, `article_cnt_min`, `patent_cnt_min`, `project_cnt_min`, `art_sci_slct_nm` |
| `exclude_orgs` | string[] | 제외 기관 |
| `top_k` | int | 추천 인원 상한 (1–15) |
| `core_keywords` | 한국어 | `retrieval_core` 의 하위 호환 alias |

Planner 는 ranking, evidence 선택, branch weighting 을 수행하지 않는다.

#### Planner 프롬프트 설계

- **JSON-only 출력 강제**: "첫 글자는 `{`, 마지막 글자는 `}`" 규칙 명시. 설명/마크다운/코드 펜스/`<thinking>` 태그 금지.
- **스키마 + Few-shot 예시**: 실제 질의와 정답 JSON 을 포함하여 출력 형식을 정확히 유도.
- **hard_filters 허용 키 제한**: 허용되지 않은 키 사용 시 Pydantic 검증에서 거부됨.
- **메타어 제거 규칙**: `평가위원`, `전문가`, `추천` 등 메타 요청어는 `retrieval_core` / `must_aspects` / `evidence_aspects` 에 남기지 않고 `planner_trace.removed_meta_terms` 로만 노출.

### 6.3 Retrieval 파이프라인

| 단계 | 설명 | 비고 |
|---|---|---|
| 1차 | branch 별 dense + sparse prefetch → branch-local RRF | e5-instruct prefix 는 branch 별 상이 |
| 2차 | 모든 branch 결과를 equal-weight RRF 로 결합 | `v3_branch_instruct_prefix` 스키마 |
| 3차 | payload filter 적용 | 기관 제외, 날짜, 최소 실적 등 보장 |
| 4차 | top-N 후보 카드 생성 | 응답/Evidence Selector 입력 품질 확보 |
| 5차 | Evidence Selector → Shortlist Gate → Reasoner → Validator | `/recommend` 만 해당 |

확장 경로(expanded)는 해당 modality 의 텍스트가 stable 과 실제로 달라질 때만 실행된다. Branch hint 는 query text 에 append 하지 않고, dense 측 branch 문맥은 retriever 의 e5-instruct prefix 가 담당한다.

### 6.4 필터와 payload index 원칙

- 필터에 자주 쓰는 필드는 반드시 payload index 대상에 포함한다.
- 논문/특허/과제의 배열 객체 조건은 nested filter 로 묶어야 한다.
- exact match 가 필요한 기관명/학위/구분 값은 `keyword` 취급이 안전하다.
- 날짜와 집계값은 임베딩에 녹이지 말고 payload filter 로 처리한다.
- 여러 strict filter 가 겹칠 때는 ACORN 사용 여부를 성능 테스트로 결정한다.

> **권장 인덱스 후보**
>
> - root: `basic_info.researcher_id`, `researcher_profile.highest_degree`, `researcher_profile.publication_count`, `researcher_profile.scie_publication_count`, `researcher_profile.intellectual_property_count`, `researcher_profile.research_project_count`
> - `publications[]`: `journal_index_type`, `publication_year_month`
> - `intellectual_properties[]`: `application_registration_type`, `application_country`, `application_date`, `registration_date`
> - `research_projects[]`: `project_start_date`, `project_end_date`, `reference_year`, `performing_organization`, `managing_agency`

---

## 7. Evidence 선택·게이트·사유 생성 전략

### 7.1 핵심 원칙

- Evidence 는 서버가 deterministic 규칙으로 선택한다. LLM 은 evidence 를 고르지 않는다.
- 추천 사유는 서버가 선택한 evidence 만 근거로 삼는다.
- Evidence 가 불충분하면 shortlist gate 가 drop/demote 하고, 끝내 부족하면 응답은 빈 리스트(200 OK) 를 허용한다.
- Validator 가 LLM 사유를 검사해 실패 시 server fallback 사유로 치환한다.

### 7.2 Evidence Selector (v0.7.0+ 기준)

우선 매칭 필드 순서: `evidence_aspects` → `must_aspects` → `retrieval_core` → `core_keywords`.

- 직접 lexical 매치만 허용 (partial match 규칙은 selector 내부 phrase normalize 기준).
- dedup key: `normalized title + year`.
- aspect 당 최대 2건, 후보 당 총 4건.
- future-dated 과제도 유효 evidence. `future_selected_evidence_ids` 에 trace.

Item 유형별 탐색 범위와 field 매핑 전체 목록은 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md) 의 "Evidence selector search scope by item type" 표 참조.

### 7.3 Shortlist Gate

게이트 순서:

1. `direct_match_count == 0` → drop
2. `aspect_coverage < min(2, len(target_aspects))` → demote
3. `generic_only == true` → bottom group

`aspect_coverage` 는 phrase count 기준이며 token count 가 아니다. `target_aspects` 는 selector 와 동일한 우선순위 체인으로 결정된다.

### 7.4 Reasoner (LLM, summary-only)

- 배치 크기: `5`
- 호출 방식: tool calling 1회 + 결과 부족 시 compact retry 1회
- 입력 per candidate: `expert_id`, `candidate_name`, `selected_evidence`, `selected_evidence_summary`, `retrieval_grounding`, `do_not_mention`
- 출력: `expert_id`, `fit`, `recommendation_reason`, `risks`
- LLM 은 `selected_evidence_ids` 를 반환하거나 선택하지 않는다.

상세 정책은 [`../api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md) 참조.

### 7.5 Validator

- 다른 후보명 leak
- 내부 evidence id leak
- aspect 와 evidence 범위 일관성 (scope: `title + detail + snippet`)
- 무근거 강한 주장

Aspect 일관성 검사는 `must_aspects` (한국어) 범위로 수행된다. 사유 텍스트가 한국어로 생성되므로 `evidence_aspects` 의 영문 용어로는 false positive 가 발생한다.

### 7.6 Candidate Card 설계

| 카드 종류 | 용도 |
|---|---|
| `/search/candidates` 응답 카드 | retrieval 디버깅 / 외부 공개. `branch_presence_flags`, `counts`, `shortlist_score` 포함 |
| Reasoner 입력 카드 | 선택된 evidence 요약 + retrieval_grounding. 토큰 비용 관리 |

I/O 필드 규격은 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md) 의 "Recommendation Contract" / "Evidence Selector Contract" 참조.

### 7.7 프롬프트 가드레일 (LLM 출력 품질 보장)

- 질문에 없는 조건을 임의로 추가하지 말 것
- Hard filter 를 충족하지 않는 후보를 옹호하지 말 것
- 동일한 근거를 여러 번 반복하지 말 것
- 근거가 약한 경우 `추측` 이 아니라 `근거 부족` 으로 표현할 것
- `rank_score` / `final_score` 등 수치는 사유 본문에 인용하지 말 것
- 다른 후보의 이름이나 내부 evidence id 를 사유에 노출하지 말 것

---

## 8. 구현 가이드

### 8.1 Ingestion 파이프라인

| 단계 | 작업 | 비고 |
|---|---|---|
| 원천 적재 | NTIS / IRIS 원천 테이블 수집 | 버전, 수집일, 원천 식별자 관리 |
| 정규화 | 기관명/학위/국가/구분값 표준화 | 필터 오작동 방지 |
| 집계 생성 | 논문수·SCIE수·특허수·과제수 계산 | root payload 에 저장 |
| source text 생성 | `basic/art/pat/pjt` 별 벡터 입력 텍스트 생성 | 필드 그대로 이어붙이지 않고 목적별 요약 |
| 임베딩 생성 | dense + sparse 벡터 생성 | 모델 버전 고정 필요 |
| 적재 | point upsert | 벡터/페이로드 정합성 검증 |
| 검증 | 샘플 질의 기반 스모크 테스트 | 필터, evidence, 추천 결과 동시 확인 |

### 8.2 서비스 레이어 모듈

현재 배치는 다음과 같다.

- `apps/core/config.py`: Settings 정의와 환경 변수 매핑
- `apps/core/json_utils.py`, `apps/core/utils.py`: JSON 추출·머지 유틸
- `apps/core/openai_compat_llm.py`: LLM 호출 wrapper
- `apps/core/runtime_validation.py`: 스타트업 readiness 검증
- `apps/recommendation/planner.py`: LLM / 휴리스틱 Planner
- `apps/recommendation/cards.py`: Candidate card 빌더
- `apps/recommendation/evidence_selector.py`: Evidence Selector + Shortlist Gate
- `apps/recommendation/reasoner.py`: Summary-only Reasoner + Validator 통합
- `apps/recommendation/service.py`: 전체 파이프라인 오케스트레이션 (`/recommend`, `/search/candidates`)
- `apps/search/`: Qdrant 하이브리드 retriever
- `apps/api/main.py`: FastAPI 라우팅 및 health check

### 8.3 현재 외부 API

| API | 설명 | 응답 핵심 |
|---|---|---|
| `POST /recommend` | 질의 기반 추천 | `recommendations[*].recommendation_reason`, `evidence`, `not_selected_reasons`, `data_gaps`, `trace` |
| `POST /search/candidates` | retrieval 디버깅 / 숏리스트 조회 | `candidates[*]`, `branch_presence_flags`, `retrieval_score_traces` |
| `POST /feedback` | 선택/제외 피드백 저장 | `feedback_id`, `stored` |
| `GET /health` | 애플리케이션 생존 확인 | `status`, `collection_name`, `searched_branches` |
| `GET /health/ready` | 의존성 readiness 검사 | `ready`, `checks`, `issues`, `collection_name`, `sample_point_id` |

외부 응답 필드 변경 이력은 [`../api/EXTERNAL_API_CHANGELOG.md`](../api/EXTERNAL_API_CHANGELOG.md) 참조.

### 8.4 운영 로그 (Trace ID 기반)

모든 로그는 Trace ID(Request ID) 를 포함한다. Planner / Retriever / Selector / Gate / Reasoner / Validator 단계의 주요 로깅 포인트와 실제 예시는 [`../operation/RUNBOOK.md`](../operation/RUNBOOK.md) 의 각 섹션 체크리스트를 참조한다.

---

## 9. 제한 사항 및 기술 제약

### 9.1 요약 표

| 구분 | 제한 사항 | 영향 | 대응 방향 |
|---|---|---|---|
| 버전 제약 | v1.16.0 에서는 weighted RRF 미사용 | branch 간 상대 강도를 수치로 조정 불가 | 모든 branch 검색 + branch-specific e5-instruct prefix + evidence selection 으로 대응 |
| 버전 제약 | 각 named vector raw similarity 의 직접 가중합은 범위 밖 | 정밀한 score engineering 이 제한됨 | deterministic evidence + gate 로 최종 판단 이동 |
| 모델 제약 | LLM 은 검색 밖의 사실을 알지 못함 | 환각 위험 | evidence card 와 validator 규칙 강화 |
| 구조 제약 | 전문가 1 point 구조는 item-level ranking 이 약함 | 특정 논문 1건이 아주 좋아도 전문가 전체 점수에 묻힐 수 있음 | 향후 `expert_id` grouping 구조 검토 |
| 데이터 제약 | 기관명/날짜/동일인 식별 품질에 민감 | 필터/추천 품질 하락 가능 | 정규화와 검증 파이프라인 강화 |
| 운영 제약 | 질의 유형이 다양함 | planner 출력 흔들림 가능 | planner 평가셋과 운영자 피드백 누적 |

### 9.2 추천 시스템 관점의 주의점

- 좋은 검색 점수 = 좋은 평가위원은 아니다.
- 추천에는 적합성 외에도 최근성, 실적 분포, 기관 중복, 데이터 공백이 함께 고려되어야 한다.
- 따라서 retrieval 점수 상위 순서를 그대로 사용자에게 노출하면 품질이 낮아질 수 있다.

### 9.3 데이터 품질 관점의 주의점

- 기관명 표기가 흔들리면 제외기관 필터가 실패할 수 있다.
- 날짜 필드 매핑이 잘못되면 최근 3년/5년 조건이 왜곡된다.
- 논문/특허/과제 텍스트가 지나치게 길면 임베딩 입력이 불안정해질 수 있다.
- IRIS 와 NTIS 의 동일 인물 식별 키 정합성이 약하면 enrichment 결과가 불안정해질 수 있다.

---

## 10. 방향성 지침

### 10.1 설계 원칙

- 단순한 retrieval 구조를 먼저 안정화하고, 사유 요약은 LLM 에 위임한다.
- 필터는 항상 deterministic 하게 구현하고, 벡터는 의미 검색에만 집중시킨다.
- 한 번에 많은 기능을 넣기보다, 운영 로그를 모아 오류 유형을 먼저 분류한다.
- 추천 품질 개선은 score 미세조정보다 evidence quality 와 prompt 품질 개선에서 시작한다.
- 모델 변경과 스키마 변경은 같은 시점에 동시에 하지 않는다.

### 10.2 단계별 로드맵

| 단계 | 목표 | 주요 작업 |
|---|---|---|
| MVP (완료) | 검색과 추천의 기본 루프 완성 | named vector 컬렉션, prefetch, Planner, Evidence Selector, Reasoner, Validator, 운영 로그 |
| Stabilization (진행 중) | 필터 정확도와 설명 품질 안정화 | 기관명 정규화, payload index, candidate card 개선, bilingual evidence_aspects, phrase-based coverage |
| Expansion | 추천 근거 다양화 | IRIS enrichment, 평가위원 활동 데이터 반영, explain 계열 API 재도입 여부 검토 |
| Future | 구조적 확장 검토 | `expert_id` grouping, item-level collection, 버전 업그레이드 검토 |

### 10.3 나중에 검토할 수 있으나 지금은 고정하지 않는 항목

- Weighted RRF 도입 여부
- `FormulaQuery` 기반 payload score boost 도입 여부
- Cross-encoder 또는 별도 re-ranker 도입 여부
- expert 와 item 을 분리한 2컬렉션 구조
- feedback-driven retrieval 개선
- `/explain` 류 추가 API 의 필요성 재평가

---

## 11. 품질 평가와 수용 기준

### 11.1 Retrieval 평가

- Recall@50 / Recall@100
- NDCG@10
- 필터 정확도(기관 제외, 최근성, 최소 실적)
- branch 별 기여도 관찰

### 11.2 Recommendation 평가

- Top-5 추천 적중률
- 운영자 수정률
- 추천 사유의 faithfulness
- Validator fallback 비율 (너무 높으면 prompt/guardrail 점검 신호)
- 데이터 공백 탐지율
- 동일 질의 반복 시 일관성

### 11.3 수용 기준 예시(초안)

| 항목 | 초기 목표 | 설명 |
|---|---|---|
| Hard filter 위반률 | 0% | 기관 제외, 학위 조건 등 위반 금지 |
| 운영자 전면 재선정 비율 | 20% 미만 | 추천 결과가 완전히 부적합한 경우 비율 |
| 응답 시간 | 실운영 기준에서 합의 | 토큰 비용과 latency 를 함께 고려 |
| 근거 누락 | 상위 추천 100% 근거 포함 | 추천마다 논문/특허/과제 등 최소 1개 이상 근거 |

골든 시나리오는 [`../operation/GOLDEN_TESTS.md`](../operation/GOLDEN_TESTS.md) 참조.

---

## 12. 최종 권고안 요약

> **최종 권고**
>
> - 전문가 1명 = 1 point 구조를 유지한다.
> - `basic / art / pat / pjt` named vector 와 sparse vector 를 함께 쓴다.
> - Qdrant v1.16.0 에서는 `prefetch + equal-weight RRF` 를 기본으로 한다.
> - 가중치와 직접 score 융합은 설계에서 제외한다.
> - Evidence 는 서버가 선택하고, LLM 은 선택된 evidence 만 요약한다.
> - Hard filter 는 시스템이 강제 보장한다.
> - Candidate card 단계로 LLM 입력을 압축한다.
> - Validator 로 사유의 사실성·일관성을 deterministic 하게 검증한다.
> - 운영 로그와 평가셋을 먼저 쌓고, 개선은 점진적으로 한다.

---

## 부록 A. 업로드 설계안 정합성 메모

아래 항목은 업로드된 스프레드시트([`../api/DATA_STRUCTURE_V0.3.xlsx`](../api/DATA_STRUCTURE_V0.3.xlsx)) 기준으로 구현 전에 한 번 더 확인해야 하는 부분이다.

| 항목 | 관찰 내용 | 판단 |
|---|---|---|
| 과제 sparse vector 명 | 구현 기준은 `pjt_vector_bm25` 로 확정 | 정정 반영 |
| 과제 날짜 매핑 | 구현 기준은 `project_start_date = TOT_RSCH_START_DT`, `project_end_date = TOT_RSCH_END_DT`, `reference_year = STAN_YR` | 정정 반영 |
| 기술분류 반영 | 기술분류 시트는 있으나 메인 스키마 반영이 약함 | 권장 보완 |
| IRIS 결합 방식 | IRIS payload 는 많지만 retrieval 스키마로 바로 넣기엔 복잡도 큼 | enrichment 용도 권장 |

---

## 부록 B. 참고 자료 (공식 문서)

- https://api.qdrant.tech/api-reference/search/query-points
- https://qdrant.tech/documentation/concepts/hybrid-queries/
- https://qdrant.tech/documentation/manage-data/collections/
- https://qdrant.tech/documentation/search/filtering/
- https://qdrant.tech/documentation/manage-data/indexing/
- https://qdrant.tech/documentation/search/search/
- https://qdrant.tech/documentation/search-precision/automate-filtering-with-llms/
- https://qdrant.tech/blog/qdrant-1.16.x/
- https://qdrant.tech/blog/qdrant-1.17.x/

---

## 변경 이력

| 버전 | 기준일 | 주요 변경 |
|---|---|---|
| v2.0 | 2026-04-17 | Map-Reduce Judge 제거, Evidence Selector / Shortlist Gate / Summary-only Reasoner / Validator 기반 파이프라인 반영. `/explain` API 삭제 반영. `evidence_aspects` v0.7.0, equal-weight RRF, branch-specific e5-instruct prefix 반영. 링크 4건 정리 (`CONTRACT.md → api/DATA_CONTRACT.md`, `SERVICE_FLOW.md §1-6~1-7` / `RUNBOOK.md §7` 섹션 참조 제거). Planner 필드 설명을 `retrieval_core / semantic_query / must_aspects / evidence_aspects / intent_flags` 기준으로 재구성. |
| v1.3 | 2026-04-14 | LLM 구조적 토큰 다이어트, 기관명 정규화 튜닝, 동적 최신성 가산점 반영 (※ 이후 v2.0 에서 Map-Reduce 구조 자체 폐기). |
