# 데이터 규약 (Data Contract) — flat chunk 파이프라인

**문서 버전:** v2.1 (flat payload 정합, 2026-06-02)
**데이터 전제:** [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md)

내부 단계 간 구조화된 계약을 정의한다. 외부 노출 응답 규격은 [`API_SPECIFICATION.md`](API_SPECIFICATION.md), 변경 이력은 [`EXTERNAL_API_CHANGELOG.md`](EXTERNAL_API_CHANGELOG.md).

> **실데이터 전제(flat payload).** 적재 단위는 `1 chunk = 1 Qdrant Point`이며 payload는 **평탄(flat)** 하다.
> 연구자 공통 메타(이름·소속·학위·누적 실적 count)는 **payload root에 비정규화 반복** 저장되고, doc_type별 상세만 `doc_attrs{}`에 들어간다.
> `researcher_meta` 같은 중첩 객체는 **없다.** 날짜는 단일 `doc_date`(문자열, 결측은 `"NONE"`).
> doc_type은 **정확히 5종**: `paper` / `patent` / `project` / `assessor_activity` / `specialty`. (구 11종 분기 폐기)
> 적재는 외부 제공자 소관이며(이전 `apps/ingest`는 `legacy_v1x/`로 격리), 본 서비스는 검색·집계·근거 선별·사유 생성만 담당한다.
> 런타임 evidence 식별자는 **payload root의 `chunk_id`** 가 authoritative 하다. Qdrant Point ID는 운영 컬렉션에 따라 `chunk_id`와 같을 수도 있고 UUID일 수도 있으나, evidence resolve와 trace는 `payload.chunk_id`를 기준으로 한다.

---

## 1. 플래너 출력 (Planner Output)

`PlannerOutput`은 플래너와 검색 엔진 사이의 계약이다.

```json
{
  "intent_summary": "드론 화재 진압 전문가 찾기",
  "retrieval_core": ["화재 진압", "드론"],
  "core_keywords": ["화재 진압", "드론"],
  "semantic_query": "드론 기반 화재 진압 기술 전문가",
  "task_terms": ["전문가 추천"],
  "role_terms": ["전문가"],
  "action_terms": ["추천"],
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "intent_flags": {},
  "top_k": 5
}
```

**규칙:**
- 하나의 JSON 객체만 반환한다(첫 글자 `{`, 끝 글자 `}`). 마크다운/`<thinking>` 금지.
- `core_keywords`/`retrieval_core`는 검색에 안전한 도메인 명사·명사구만.
- `role_terms`/`action_terms`("평가위원", "추천" 등)는 검색 텍스트(임베딩 입력)에 넣지 않는다(벡터 오염 방지).
- `hard_filters`는 허용 키만 사용한다(아래 §1.1). 미허용 키는 검색 컴파일러가 무시한다.
- 명시 요청 파라미터(`top_k`, `filters_override`, `exclude_orgs`)는 자연어 추출보다 우선.
- 출력이 무효이거나 `core_keywords`가 비면 1회 재시도, 그래도 비면 검색 생략.
- planner는 doc_type on/off를 결정하지 않는다. 검색 대상 doc_type 축소는 운영 화이트리스트(`NTIS_RETRIEVAL_DOC_TYPES`)로만 한다.

### 1.1 `hard_filters` 허용 키 (flat chunk 모델)

모든 백엔드 필드는 **flat root** 또는 `doc_attrs.*`(실제 payload 키)를 가리킨다. `researcher_meta.*` 같은 중첩 경로는 더 이상 존재하지 않는다.

| 키 | 의미 | flat 백엔드 경로 |
|---|---|---|
| `highest_degree` | 학위 | root `highest_degree` (MatchAny) |
| `recent_years` | 최근 N년 | `doc_date >= (올해-N)-01-01` (datetime range) |
| `recent_doc_types` | 최근성 적용 doc_type/family | recency 결합 대상 지정(아래 OR 가드 참조) |
| `publication_count_min` | 최소 논문 수 | root `publication_count` (Range gte) |
| `scie_publication_count_min` | 최소 SCIE 논문 수 | root `scie_publication_count` (Range gte) |
| `intellectual_property_count_min` | 최소 지식재산 수 | root `intellectual_property_count` (Range gte) |
| `research_project_count_min` | 최소 연구과제 수 | root `research_project_count` (Range gte) |
| `researcher_assessor_activity_count_min` | 최소 평가위원 활동 수 | root `researcher_assessor_activity_count` (Range gte) |
| `journal_class` | 등재구분 | `doc_attrs.indexing_database` (paper 등재구분, MatchAny) |

> **단일 assessor count.** 실데이터는 평가위원 활동을 단일 `researcher_assessor_activity_count`로 보유한다.
> 구 분리 키 `researcher_assessor_count_min`/`expert_assessor_count_min`과 `assessor_activity_count_min`은 모두 이 단일 필드로 흡수된다(하위호환 별칭).
>
> **flat 정합(v2.1).** v1.x의 `art_recent_years`/`pat_recent_years`/`pjt_recent_years`(doc_type별 분리·`event_year` 기준)는 통합 `recent_years` + `recent_doc_types`로 대체되며, recency 기준은 **단일 `doc_date`(datetime)** 다(`event_date`/`event_year` 폐기). `*_cnt_min` → `*_count_min`으로 명칭 통일.

> **HARD 제약 — 다중 doc_type recency는 OR.** `recent_doc_types`가 2개 이상이면 `(doc_type=X AND doc_date>=cutoff)` 조건들을 `min_should(min_count=1)`로 묶는다. AND로 묶으면 0건 회귀가 난다(DATA_MODEL §3.2 교훈). `recent_doc_types`에 family명(`achievement`/`assessment`/`expertise`)을 주면 그 family의 doc_type 전체로 확장된다. doc_type 미지정이면 "어떤 chunk든 최근 `doc_date`면 통과".

> `doc_date`가 `"NONE"`/결측이면 datetime range에 매칭되지 않는다(= recency 필터에서 자연 제외).

---

## 2. 쿼리 빌더 규약 (Query Builder Contract)

`QueryTextBuilder`는 플래너 출력과 사용자 원문으로 `SearchQueryPlan`을 만든다.
- `raw_query`: 사용자 원문. 로그, trace, dense 의미 검색의 기준.
- `dense_query`: 기본적으로 `raw_query`. Dense encoder에는 원문 의미를 보존해 넣는다.
- `sparse_joint_query`: SPLADE 1차 회수용 짧은 자연문/명사구. 예: `인공지능 반도체 연구개발 산업 경험`.
- `sparse_concept_queries`: 필수 개념별 보조 SPLADE query. 예: `ai`, `semiconductor`, `semiconductor_experience`.
- `required_concepts`: 연구자 집계 후 반드시 coverage를 확인할 개념. 현재 좁은 gate는 `ai`, `semiconductor`를 지원한다.
- `optional_concepts`: 점수/근거 확보 힌트이며 hard gate가 아니다. 예: `research_development`, `industry_experience`.
- SPLADE query에는 `전문`, `분야`, `연구자`, `가진`, `또는`, `상세` 같은 역할/일반어를 넣지 않는다. `연구/개발/산업/경험`은 단독어가 아니라 `반도체 연구개발`, `반도체 산업 경험` 같은 도메인 결합 문맥에서만 남긴다.
- v1.x의 확장 사전(`basic`/`art`/`pat`/`pjt`, `bundle_ids`)은 active 검색 경로에서 제거됐다. trace 호환용 stable/expanded 쌍은 유지하지만, 현재 검색기의 stable/expanded는 dense query와 동일하다.

---

## 3. 검색 규약 (Retrieval Contract)

`QdrantHybridRetriever`는 모드 `grouped_hybrid_rrf` 고정. 컬렉션 1개(`researcher_recommend_v1` 또는 운영 override), 단일 dense named vector `vector_e5i`(1024, Cosine) + 단일 sparse named vector `vector_splade`.

- **검색:** `query_points_groups(group_by="researcher_id")` 1회. prefetch는 `dense_full` 1개와 `sparse_joint`, `sparse_<concept>` 여러 개로 구성한다. 모든 prefetch는 `FusionQuery(RRF)` (**equal RRF — Qdrant 가중 RRF 금지**)로 융합한다.
- **chunk gate:** `required_concepts`가 있으면 concept hit가 없는 sibling chunk는 점수와 evidence에서 제거한다.
- **연구자 coverage gate:** 남은 chunk들이 `required_concepts`를 모두 덮지 못하면 후보를 `reason="relevance_concepts_missing"`로 제거한다.
- **집계:** 남은 chunk hit을 `researcher_id`로 묶어 RRF 누적(연구자 점수). 한 연구자의 동일 doc_type은 상위 N chunk(`doc_type_chunk_cap`, 기본 3)만 점수에 기여하고, 추가 chunk는 harmonic decay로 체감 반영한다. doc_type prior는 기본 equal이며 앱단 랭크 가중일 뿐 Qdrant score 가중합이 아니다.
- `trace.query_payload`는 `search_query_plan`, `group_count`, `aggregated_candidate_count`, `relevance_gate_active_concepts`, `relevance_kept_chunk_count`, `relevance_dropped_chunk_count`, `relevance_filtered_candidate_count`, `org_filtered_count`를 포함한다.
- `trace.query_payload`는 검색 키워드/텍스트만 노출하며 dense/sparse 벡터 값과 전체 payload는 노출하지 않는다.

---

## 4. 후보 카드 규약 (Candidate Card Contract)

`CandidateCard`는 `/search/candidates`와 `/recommend` 공통 내부 계약.
- 카드 순서는 검색 정렬 순서를 엄격히 따른다.
- `rank_score`(또는 `shortlist_score`)는 RRF 집계 점수를 0~100으로 정규화한 값.
- 각 후보는 **flat root 기반 누적 카운트**(`counts`: `publication_count`/`scie_publication_count`/`intellectual_property_count`/`research_project_count`/`researcher_assessor_activity_count`), 이번 검색에서 hit한 doc_type 목록(`doc_types_present`), doc_type별 evidence 묶음(`evidence_by_type`, 각 항목에 `chunk_id` 포함)을 보유한다.
- `/recommend` 직전, 후보 내부 chunk을 `core_keywords`/query 관련도로 재랭크한다(후보 순위에는 영향 없음).

---

## 5. evidence 선별 규약 (Evidence Selection Contract)

`EvidenceSelector` Protocol. 구현: `CrossEncoderEvidenceSelector`(`evidence_reranker_backend="cross_encoder"`, 모델 부재 시 lexical 강등) / lexical 기본(`KeywordEvidenceSelector`).

- 입력: 한 후보의 매칭 chunk 집합.
- 출력: family별 관련도 상위 chunk 묶음. 각 항목(`ChunkEvidence`)은 `chunk_id`, `doc_type`, `title`, `date`, `snippet`, `doc_attrs`, `score`를 보유한다. **evidence 참조 id == `chunk_id`.**
- family는 doc_type에서 파생: `achievement`={paper,patent,project} / `assessment`={assessor_activity} / `expertise`={specialty} / `identity`=합성 profile(전용 doc_type 없음 — root 필드로 구성).
- 캡: family별 상위 N(`evidence_family_cap` 기본 achievement 10 / assessment 6 / expertise 6 / identity 1, 운영 튜닝 가능). 후보 순위에는 영향을 주지 않는다(grounding 선별 한정).

---

## 6. 추천 사유 생성 규약 (Reason Generation Contract)

`OpenAICompatReasonGenerator`는 정렬된 상위 K명만 수신한다.

**출력 스키마:**
```json
{
  "items": [
    {
      "expert_id": "M1006328",
      "fit": "높음",
      "recommendation_reason": "화재 대응 관련 논문·과제와 평가위원 활동 이력이 있음.",
      "selected_evidence_ids": ["paper_100000045256_c000", "project_100000031245_c000"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

**규칙:**
- LLM은 후보 순위를 바꾸지 않고, 후보를 누락시키지 않으며, 새 ID를 만들지 않는다(re-rank/drop/invent 금지).
- `selected_evidence_ids`는 제공된 풀의 **`chunk_id`를 정확히 복사**한다(형식 `<doc_type>_<숫자doc_id>_c<NNN>`, 예 `paper_100000045256_c000`). 풀에 없거나 형식이 깨진 id는 무효 처리되고 결정론적 fallback으로 evidence를 조립한다.
- 후보별 evidence는 LLM 전달 전 관련도 재랭크 + family별 캡이 적용된다.
- 심사는 배치(`llm_judge_batch_size`, 기본 10) 단위 순차/병렬로 진행한다.

> v1.x의 evidence id 형식 `paper:N`/`project:N`/`patent:N`은 폐기되고 `chunk_id`로 통일된다. 상세 정책은 [`REASONER_RUNTIME_POLICY.md`](REASONER_RUNTIME_POLICY.md).
