
# 평가위원 추천 RAG 시스템
## 설계 · 구현 · 제한 · 방향성 지침서 (flat chunk 모델)

**기준선:** Qdrant 하이브리드 검색 / chunk 단위 Point / 단일 dense+sparse 벡터 / flat payload / LLM 추천
**문서 버전:** v2.1 (flat payload 정렬, 2026-06-02)
**선행 문서:** [`DATA_MODEL.md`](DATA_MODEL.md) — 본 문서의 데이터 전제는 모두 여기서 확정된다.

---

## 0. v2.x가 v1.x에서 바꾼 것 / 유지한 것

| 구분 | v1.x | v2.x |
|---|---|---|
| **저장 단위** | 연구자 1명 = 1 Point + nested 배열 | **chunk 1개 = 1 Point** (flat payload) |
| **검색 표현** | 4브랜치 named vector(basic/art/pat/pjt) | **단일 dense+sparse**, doc_type은 payload 필터 |
| **도메인 수** | 4 | **5 doc_type (4 family)** — 평가이력·전문성 신호 신규 |
| **payload 구조** | nested(researcher_meta + 배열) | **flat** — 공통 메타는 root 비정규화, 상세만 `doc_attrs{}` |
| **후보 단위** | Point가 곧 연구자 | chunk 검색 후 `researcher_id`로 **집계**하여 후보 생성 |
| **evidence 참조** | `paper:0` (배열 인덱스) | **`chunk_id`** (불변 id, 코덱) |
| 검색 철학 | recall 우선, 넓게 확보 | **유지** |
| 최종 판단 | LLM 추천·설명 | **유지** |
| hard filter | 시스템이 deterministic 보장 | **유지** |
| LLM 역할 | 재정렬/탈락/ID생성 금지 | **유지** |
| 임베딩 정제 | role/action 불용어 배제 | **유지** |

> **핵심:** "데이터·저장 구조"는 갈아엎되, "역할 분리(검색=후보확보 / LLM=판단·설명) · 결정론적 필터 · 단순한 단계 경계"라는 **운영 철학은 유지**한다. 이 철학은 2026-04 회고([`ADR/retrospective-recommendation-pipeline-iteration-2026-04.md`](ADR/retrospective-recommendation-pipeline-iteration-2026-04.md))에서 "똑똑한 중간 레이어를 얹는 것보다 단계 분리·역할 고정이 유지보수에 유리하다"는 결론으로 검증됐다.

> **적재 책임 경계.** 실제 적재 데이터는 **외부 제공자**가 만든 flat chunk payload다. 본 시스템은 그 payload를 **검색·집계·추천**하는 역할만 지며, ingestion(변환·정규화·임베딩 적재) 코드는 quarantine(`legacy_v1x/`)되어 active path가 아니다.

---

## 1. 문서 목적과 범위

평가위원 추천 RAG 시스템의 기준 설계를 flat chunk 데이터 모델 위에서 고정하기 위한 실행 문서다.

- **포함:** 컬렉션/벡터 설계, 검색·집계 전략, hard filter, evidence 선별, LLM 추천 규칙, 제한, 방향성
- **비포함:** 프런트 UI 상세, IaC, 인증 세부 구현, 적재(ingestion) 구현 (→ 외부 제공자 소관 / 폐기된 내부 구현은 `legacy_v1x/`)
- **우선순위:** 정확한 기준선 > 운영 가능성 > 확장성 > 시각적 완성도

---

## 2. 확정된 설계 기준

구현 편의로 아래 기준을 흔드는 것은 기준선 위반으로 본다.

| 항목 | 최종 결정 | 설계 의미 |
|---|---|---|
| 저장 단위 | chunk 1개 = 1 Point, payload `chunk_id` authoritative | 데이터 도착 단위와 저장 단위 일치, evidence id 안정성 |
| payload 형태 | **flat** — 공통 메타 root 비정규화 + `doc_attrs{}` | 실데이터 형태와 일치, nested 폐기 |
| 후보 단위 | 검색 후 `researcher_id` 집계 = 연구자 1명 | 저장은 chunk, 추천은 연구자 |
| 검색 표현 | 단일 `vector_e5i`(dense) + `vector_splade`(sparse) | doc_type은 named vector가 아니라 payload 필터 |
| 융합 방식 | Qdrant `prefetch + RRF`(equal) | weighted RRF는 Qdrant 융합에 쓰지 않음 (§6.3) |
| 집계 가중 | 앱단 doc_type prior (기본 equal) | intent 기반 prior는 옵트인, 기본 비활성 (§6) |
| 최종 판단 | LLM 추천·설명 | 검색 점수는 신뢰 근거가 아님 |
| 필터 처리 | 시스템 deterministic 보장 | LLM 환각·조건 누락 방지, recency는 OR 결합 |
| evidence 참조 | `chunk_id` 불변 id (코덱) | 인덱스 기반 id 폐기 |
| 임베딩 정제 | role/action 불용어 배제 | 벡터 오염 차단 |
| 출력 원칙 | 추천 + 근거 + 제외 사유 + 데이터 공백 | 운영자 검토 가능성 |

### 2.1 doc_type / family / chunk_id 코덱 (단일 출처)

- **doc_type 5종:** `paper` / `patent` / `project` / `assessor_activity` / `specialty`. (실측 분포: paper 64.6% / patent 18.7% / assessor_activity 9.7% / project 6.5% / specialty 0.5%.)
- **family 4종:** `achievement`={paper, patent, project} / `assessment`={assessor_activity} / `expertise`={specialty} / `identity`=합성(전용 doc_type 없음; 연구자 신원·누적 실적은 모든 chunk의 flat root에 반복 저장되며, 합성 profile evidence가 identity family를 채운다).
- **chunk_id 코덱:** `<doc_type>_<숫자doc_id>_c<NNN>` (예: `paper_100000045256_c000`). chunk index는 3자리 zero-pad. doc_id = `<doc_type>_<숫자doc_id>` (예: `paper_100000045256`).
- 단일 출처는 [`apps/search/doc_types.py`](../../search/doc_types.py)이며, 검색·필터·evidence·집계는 모두 여기를 참조한다(손수 재정의 금지).

### 2.2 flat payload 키 (실데이터 계약)

- **root 공통 메타(모든 chunk 동일):** `researcher_id`, `researcher_name`, `affiliated_organization`, `highest_degree`, `publication_count`, `scie_publication_count`, `intellectual_property_count`, `research_project_count`, `researcher_assessor_activity_count`. (assessor count는 단일 필드로 병합되어 있다 — split 키는 없다.)
- **단일 날짜:** `doc_date` (문자열, 결측이면 `"NONE"`). 별도 `event_date`/`event_year`는 없다.
- **doc_type별 상세는 `doc_attrs{}`:**
  - `paper`: `indexing_database`, `journal_name`, `keywords`, `main_language_title`, `sub_language_title`, `is_scie`, `publication_year_month`
  - `project`: `project_title_korean`, `project_title_english`, `project_period`, `performing_organization`, `managing_agency`
  - `patent`: `intellectual_property_title`, `intellectual_property_type`, `application_registration_type`, `application_country`, `application_number`, `application_date`, `intellectual_property_foreign_type`
  - `assessor_activity` / `specialty`: **doc_attrs 키 미상** → 타입 미지정 passthrough로 취급(인덱스/필터 대상 아님). 본 문서는 이 둘의 doc_attrs 키를 임의로 만들지 않는다.
- 알 수 없는 추가 키는 무시(extra=ignore)하여 스키마 진화에 견딘다.

---

## 3. 서비스 목표

자연어 요청을 입력받아 적합한 평가위원 후보군을 넓게 확보하고, 검색 근거(chunk)를 바탕으로 LLM이 최종 추천·설명을 생성한다. 핵심은 **정답 문장 회수**가 아니라 **후보 비교와 추천 근거 제시**다.

성공 기준:
- Hard filter 위반 0건으로 후보 반환
- 상위권에 실제 검토 가능한 전문가 포함
- 추천 사유가 `chunk_text` 근거와 일치(faithfulness)
- 운영자 전면 재선정 비율 점진 감소

---

## 4. 전체 아키텍처

```text
[사용자 질의]
   ↓
[LLM Query Planner]
   - intent_summary / core_keywords / semantic_query
   - hard_filters / include·exclude_orgs / top_k
   - (선택) intent flags: 최근성·평가이력 강조 등 prior 힌트
   ↓
[Search Query Builder]
   - raw_query 보존
   - dense_query = planner semantic_query 우선, 없으면 사용자 원문 fallback
   - sparse_joint_query = SPLADE용 짧은 핵심 자연문/명사구
   - sparse_concept_queries = 필수 개념별 보조 검색문
   - required_concepts = 연구자 집계/coverage gate 조건
   ↓
[Retrieval Orchestrator]  (chunk 단위, flat payload)
   1) view별 query_points
      - dense_full: dense_query
      - sparse_raw: raw_query
      - sparse_focus: sparse_joint_query
      - concept:<id>: sparse_concept_queries
      - 앱단 등수 기반 RRF 융합
   2) chunk gate: chunk_text/doc_id evidence term으로 필수 개념 확정(doc_attrs 값은 확정 근거 제외)
   3) researcher coverage gate: required_concepts를 모두 만족하는 연구자만 후보화
   4) 집계: researcher_id로 chunk hit 묶어 RRF 누적 (doc_type별 chunk cap) → 연구자 후보
   5) hard filter: doc_date 최근성 / flat root *_count / exclude org (deterministic, recency=OR)
   6) 결정론적 정렬: score desc → name asc → researcher_id asc
   ↓
[Evidence Selector]  (chunk이 곧 근거)
   - 후보별 매칭 chunk을 doc_type별로 모아 query 관련도 재랭크(cross-encoder→lexical 강등)
   - family별 top-N chunk만 LLM 입력으로 선별 (chunk_id 보존)
   ↓
[LLM Reasoner]
   - 후보 + 선별 chunk → fit / recommendation_reason / selected_evidence_ids(chunk_id) / risks
   - 재정렬·탈락·새 ID 생성 금지. 빈 사유는 서버 결정론적 fallback
   ↓
[응답 생성기] → 추천 + 근거 + 제외 사유 + 데이터 공백
   ↓
[가시성 계층] Trace ID 전파, 단계별 한글 로그
```

| 구성요소 | 역할 | 절대 하지 않는 것 |
|---|---|---|
| Query Planner | 의도/키워드/필터/top_k 추출 (JSON-only) | 최종 추천 판단, doc_type on/off 결정 |
| Search Query Builder | 채널별 검색 쿼리와 required concept 생성 | SPLADE에 사용자 원문 그대로 투입 |
| Retrieval Orchestrator | chunk 하이브리드 검색 + concept coverage gate + 연구자 집계 + 필터 + 정렬 | 최종 추천 판단 |
| Evidence Selector | 후보 내부 chunk 관련도 선별 | 후보(연구자) 순위 변경 |
| LLM Reasoner | 적합도·사유·근거 chunk 선택 | 재정렬·탈락·ID 생성 |

---

## 5. 검색 설계 (flat chunk 기반)

### 5.1 철학
검색의 목적은 정답 1명을 고르는 것이 아니라 **적절한 후보군을 넓게 확보**하는 것이다. retrieval은 recall 우선, recommendation은 reasoning 우선.

### 5.2 채널별 검색 쿼리 (`multiview_flat_relevance`)
검색 쿼리는 사용자 원문 하나를 그대로 모든 채널에 넣지 않는다. `SearchQueryPlan`으로 목적별 쿼리를 분리한다.

```python
SearchQueryPlan = {
    "raw_query": "...",
    "dense_query": "...",
    "sparse_joint_query": "...",
    "sparse_concept_queries": {"ai": "...", "semiconductor": "..."},
    "required_concepts": ["ai", "semiconductor"],
    "optional_concepts": [...],
}
```

1. **dense_full:** `dense_query`는 planner `semantic_query`를 우선 사용하고, 없을 때 사용자 원문으로 fallback한다.
2. **sparse_focus:** view source 이름은 `sparse_focus`이며 텍스트는 `sparse_joint_query`(SPLADE용 짧은 핵심 자연문/명사구)다. `전문`, `분야`, `연구자`, `또는`, `상세` 같은 일반어를 제거하고, 예: `인공지능 반도체 연구개발 산업 경험`.
3. **concept:<id>:** 필수 개념별 SPLADE 보조 검색으로 근거 chunk를 넓게 확보한다(view source 이름은 `concept:<id>`, 융합 weight 키는 `sparse_concept`). 예: `concept:ai`, `concept:semiconductor`, `concept:semiconductor_experience`.
4. **multiview RRF:** dense 1개 + sparse 여러 개를 view별 `query_points`로 회수한 뒤, 앱단에서 view별 등수 기반 RRF로 chunk hit을 융합한다.

> SPLADE에 사용자 원문을 그대로 넣으면 일반어가 과확장되고, 순수 키워드 나열은 OR 검색처럼 넓어진다. 따라서 SPLADE는 짧은 자연문/명사구, dense는 planner semantic_query 중심으로 분리한다.

> v1.x의 "브랜치"는 v2.x에서 "doc_type(또는 family) 경로"로 대체된다. 기본 검색 모드는 `multiview`(retrieval_mode=`multiview_flat_relevance`)다. 구 `keyword_pool_then_hybrid`의 1차 후보 풀 아이디어는 사용자 선택형 `keyword_similarity` 모드(retrieval_mode=`keyword_then_dense_similarity`)로 재설계되어 opt-in으로만 동작한다(§6.5).

### 5.3 연구자 집계 (v2.x 핵심)
chunk hit을 `researcher_id`로 묶어 연구자 후보 1건으로 만든다.
- 각 연구자 점수 = 그 연구자의 chunk hit들에 대한 **RRF 누적**.
- **점수 경로 분기(required 유무):** `required_concepts`가 **있으면** capped evidence 점수(joint/balance/concept/support, `score_researcher`)를 쓴다. **없으면**(query_exact 합성/도메인 미감지) 이 점수는 0으로 붕괴하므로 — concept 충족이 전제라서 — 대신 **검색 융합 관련도(`_score_generic`)** 로 순위를 매긴다. 그러지 않으면 전 후보가 동점(0)→이름순으로 정렬되는 사고가 난다. optional concept 확정분은 표시/증거선별용 `matched_concepts`로 보존한다.
- **concept coverage gate:** `required_concepts`가 있는 질의는 연구자 단위로 필수 개념을 모두 만족해야 한다. 개념 근거가 없는 chunk는 제거하고, 남은 chunk의 concept union이 부족하면 해당 연구자는 탈락한다.
- **doc_type별 캡:** 한 doc_type에서 상위 `N`개 chunk까지만 점수에 기여(`doc_type_chunk_cap`, 기본 3). 다작 연구자가 한 영역 chunk 수로 순위를 독식하지 못하게 한다.
- **doc_type prior(기본 equal):** intent에 따라 family별 기여 가중을 줄 수 있으나 기본값은 equal. (§6)
- 연구자당 최종 1 Point로 dedupe.
- 서버사이드 대안: Qdrant `query_points_groups(group_by="researcher_id")`로 그룹 단위 회수도 가능. 다만 cross-doc_type RRF 누적·캡을 앱단에서 제어하는 편이 결정성이 높아 기본은 앱단 집계.

### 5.4 hard filter (deterministic)
- `doc_date >= 올해-N` (+ `doc_type` 한정) — 최근성. `doc_date`는 단일 datetime 필드이며 `"NONE"`/결측은 datetime range에 매칭되지 않는다(= recency 제외). **여러 doc_type recency는 OR(min_should, min_count=1)** 로 결합한다(AND로 묶으면 0건 회귀, [`DATA_MODEL.md §3`](DATA_MODEL.md)).
- flat root `*_count >= 임계값` — 최소 실적 (`publication_count` / `scie_publication_count` / `intellectual_property_count` / `research_project_count` / `researcher_assessor_activity_count`).
- `highest_degree` — 학위 (root).
- **소속 기관 include/exclude:** 정규화된 root 소속 필드가 없으므로 Qdrant exact pre-filter에 의존하지 않고 Python post-filter에서 root `affiliated_organization`만 비교한다. `doc_attrs.performing_organization` / `doc_attrs.managing_agency`는 과제 속성이며 소속기관 필터 대상이 아니다.
- 필터는 LLM이 아니라 시스템이 보장한다.

### 5.5 결정론적 최종 정렬
1. score 내림차순 → 2. `researcher_name` 오름차순 → 3. `researcher_id` 오름차순.

---

## 6. 융합·가중·리랭커 결정 (재검토 결과 명문화)

> 2026-05-28 사용자가 "일부 고정 제약 재검토 개방"을 택했다. chunk 단위 score가 생기면서 weighted fusion·후보 리랭커가 기술적으로 더 자연스러워졌기 때문이다. 아래는 그 재검토의 **확정 결정**이다.

### 6.1 Qdrant 융합: equal RRF 유지
컬렉션 내 dense+sparse 융합은 앱단 등수 기반 RRF로 고정한다. Qdrant 단의 weighted RRF/score 가중합은 **여전히 쓰지 않는다**. 이유: RRF는 스케일이 다른 dense·sparse score를 순위로만 결합해 안정적이고, 버전·재현성 리스크가 낮다.

### 6.2 집계 가중(doc_type prior): 옵트인, 기본 equal
연구자 집계 단계의 family/doc_type 가중은 **앱단에서** 조정 가능하다(이미 v1.x도 `BRANCH_WEIGHTS`로 앱단 가중을 적용했다). v2.x 기본값은 **equal**. intent가 명확할 때만(예: "평가 경험이 풍부한" → 평가이력 family prior↑) 옵트인으로 켠다. 이는 Qdrant weighted RRF가 아니라 **랭크 누적 가중**이며, 켜더라도 hard filter와 결정론적 정렬을 침해하지 않는다. 설정: `NTIS_DOC_TYPE_PRIORS`(기본 미설정=equal).

### 6.3 후보(연구자) cross-encoder 리랭커: 기본 OFF, 옵트인 실험만
- **결정:** 후보 **순위의 척추는 RRF로 유지**한다. cross-encoder로 후보를 재정렬하는 것은 기본 비활성(`NTIS_CANDIDATE_RERANKER=off`)으로 두고, 켜더라도 **score 동률 밴드 내 재배열만** 허용하며 **후보를 탈락·생성하지 않는다**.
- **이유:** (1) RRF 순위는 결정론적·설명가능·재현가능하다. (2) 회고 교훈상 "판단을 한 레이어에 몰면" 경계가 무너진다. (3) 최종 비교·추천은 LLM이 맡는다는 역할 분리를 깨지 않기 위함.
- 즉, "재검토 개방"은 **레버를 설계에 명시하고 끌 수 있게 두되, 기본 추천 품질의 토대는 단순·결정론적으로 유지**한다는 결론이다.

### 6.4 evidence cross-encoder 리랭커: 1급 채택
chunk이 1급 단위가 되면서 evidence 선별에 cross-encoder가 자연스럽게 맞는다. 후보 **내부** chunk을 query 관련도로 재랭크해 family별 top-N만 LLM에 넘긴다(모델 부재 시 lexical 자동 강등). 목적은 **토큰 절감 + grounding 품질 + 정규화**이며 **후보 순위에는 영향을 주지 않는다**. ([`../api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md))

### 6.5 사용자 선택형 검색 모드 (`search_mode`, 요청별 opt-in)
요청 본문 `search_mode`로 회수/스코어링 전략을 고른다(기본 `multiview`). 후처리(chunk_id 병합·concept gate·capped evidence 재점수·org post-filter·결정론적 정렬·LLM no-rerank)는 모든 모드 공통이며, **회수와 chunk 점수 산출 방식만** 달라진다.

| 모드 | 회수 | chunk 점수 | 고정 제약과의 관계 |
|---|---|---|---|
| `multiview`(기본) | dense_full + sparse_raw/focus + concept 멀티뷰 | view별 등수 RRF × view weight | §6.1~6.3 제약 그대로 유지 |
| `hybrid` | dense_full + sparse_raw 2뷰 | 등수 RRF × 균등 가중(`hybrid_view_weights` 1.0/1.0) | equal RRF 유지(가중 RRF 아님) |
| `keyword_similarity` | SPLADE 1차 풀 → 그 id 집합 한정 dense | **dense 유사도(raw)** | cascade 재정렬 = 기본 "리랭커 금지"의 **명시적 예외**(사용자가 선택할 때만) |

- **결정:** 기본값 `multiview`는 §2/§6의 고정 제약(가중 RRF·리랭커 금지)을 **변함없이** 유지한다. `hybrid`/`keyword_similarity`는 사용자가 요청에서 명시 선택해야만 동작하는 opt-in 대체 전략이다. 특히 `keyword_similarity`의 dense 재정렬은 cascade 리랭크 성격이라 기본 제약을 비켜가므로 **기본 경로로 승격하지 않는다.**
- **이유:** 기본 추천 품질의 토대는 단순·결정론적으로 유지하되(§6.3), 운영/실험이 다른 회수 전략을 비교할 수 있게 레버를 설계에 명시한다. 모드는 L3 캐시 키에 포함돼 모드별 결과가 섞이지 않는다.

#### 6.5.1 multiview source-aware 가중 (정밀도 보강, 고정 제약 미위반)
multiview에서 **required concept이 없는**(query_exact 합성/도메인 미감지) 질의는 dense 의미신호가 순위를 주도해야 한다. 그런데 `fuse_chunk_score`가 한 chunk의 **모든 view weight를 합산**하므로, `제안평가`/`시스템` 같은 흔한 토큰이 `sparse_raw`+`sparse_focus`+`concept:<id>` 여러 sparse 뷰에서 동시에 잡히면 그 누적(예: 0.25+0.7+0.5+0.5)이 단일 dense 뷰(1.0)를 이겨, substring 빈도가 의미 관련도를 누르는 저하가 있었다.

- **동작:** required concept이 없으면 dense-우세 가중(`multiview_generic_view_weights`, 기본 dense_full=2.0)을 쓰고, query_exact 합성 concept의 `concept:<id>` sparse 뷰는 만들지 않는다(`multiview_drop_query_exact_concept_views=true`). required concept 질의(gate 활성)는 기존 `search_view_weights`·concept 뷰를 **그대로** 쓴다.
- **제약 관계:** 앱단 **등수 기반 RRF는 그대로**이고 view weight 값만 source별로 바꾼 것이다(§6.1의 가중 RRF/raw score 합산 금지를 위반하지 않음). 리랭커도 아니다 — 순위 산출 입력(view 구성·가중)만 정밀화했다.

---

## 7. LLM 추천 전략

### 7.1 원칙
- 벡터·RRF 점수 자체를 신뢰 근거로 삼지 않는다.
- retrieval이 넘긴 chunk 안의 근거만 사용한다.
- 추천 사유는 반드시 `chunk_id`로 evidence와 연결한다.
- 근거 부족 시 `추천 보류`/`근거 부족`을 출력한다(추측 금지).

### 7.2 입력 구성
- 후보 카드(머리): `profile`/flat root 메타(organization·degree·counts) 요약 + 평가이력 요약.
- 근거 풀: evidence selector가 family별로 고른 chunk(각 `chunk_id` 포함).
- Top-k만 LLM에 전달하며, 최대 5명 배치로 처리.

### 7.3 가드레일
- 질문에 없는 조건 추가 금지 / hard filter 미충족 후보 추천 금지.
- 동일 근거 반복 금지 / 1·2순위 비교 문장 포함.
- `rank_score`는 상대 RRF 점수이며 절대 적합도 백분율이 아님을 명시.
- `selected_evidence_ids`는 제공된 `chunk_id`만 그대로 인용(코덱 `<doc_type>_<숫자doc_id>_c<NNN>`). 새 id 생성 금지. 단, 이는 사유 문장의 인용 힌트·trace 용도이며, 최종 `recommendation.evidence`는 selector가 선별한 relevant 풀 전체로 결정론적으로 조립된다(선택 id가 evidence를 좌우하지 않음).

---

## 8. 제한 사항

| 구분 | 제한 | 영향 | 대응 |
|---|---|---|---|
| 구조 | 다작 연구자가 chunk 수로 집계 점수를 끌어올릴 수 있음 | 순위 왜곡 | doc_type별 chunk cap + 집계 정규화 |
| 데이터 | flat root 공통 메타가 모든 chunk에 비정규화 반복 → 적재 불일치 시 필터 비결정 | 필터 오작동 | 외부 적재 계약 + 불변식 검증([`DATA_MODEL.md`](DATA_MODEL.md)) |
| 데이터 | `assessor_activity`/`specialty`의 doc_attrs 키 미상 → 타입 필터 불가 | 세밀 필터 제한 | passthrough 취급, chunk_text/dense 검색에 의존 |
| 데이터 | 기관명/날짜(`doc_date`)/동일인 식별 품질 민감 | 필터·추천 품질 | 외부 정규화·검증에 의존 |
| 융합 | equal RRF는 영역 상대강도 미세조정 불가 | score engineering 제한 | 집계 prior(옵트인) + LLM 판단 |
| 모델 | LLM은 검색 밖 사실 모름 | 환각 | chunk grounding + 금지 규칙 |

---

## 9. 방향성 지침

- 단순한 retrieval을 먼저 안정화하고 추천 판단은 LLM에 위임.
- 필터는 항상 deterministic, 벡터는 의미 검색에만.
- 모델 변경과 스키마 변경을 같은 시점에 하지 않는다.
- planner/retrieval/reasoner/evidence_selector 중 둘 이상이 동시에 바뀌면 [`SERVICE_FLOW.md`](SERVICE_FLOW.md)와 [`../api/DATA_CONTRACT.md`](../api/DATA_CONTRACT.md)를 같은 커밋에서 갱신.
- 새 중간 레이어 도입 시 "언제 제거할지" 조건까지 ADR에 적는다.

### 9.1 지금 고정하지 않는 항목(향후 재검토)
- 후보 cross-encoder 리랭커 상시화(현재 옵트인·동률밴드 한정)
- doc_type prior 자동 학습(현재 수동 설정)
- `FormulaQuery` 기반 payload score boost
- feedback-driven retrieval 개선
- `assessor_activity`/`specialty` doc_attrs 키 확정 시 타입별 필터 추가

---

## 10. 품질 평가

- **Retrieval:** Recall@50/100, 필터 정확도(기관 제외/최근성/최소 실적), family별 기여도, 집계 캡 효과.
- **Recommendation:** Top-5 적중률, 운영자 수정률, 사유 faithfulness(`chunk_id` 근거 일치), 데이터 공백 탐지율, 반복 일관성.

| 항목 | 초기 목표 |
|---|---|
| Hard filter 위반률 | 0% |
| 운영자 전면 재선정 | 20% 미만 |
| 상위 추천 근거 포함 | 100% (chunk 근거 1개 이상) |

---

## 부록. 참고 자료
- https://qdrant.tech/documentation/concepts/hybrid-queries/
- https://qdrant.tech/documentation/concepts/search/#search-groups (group_by 집계)
- https://qdrant.tech/documentation/concepts/indexing/
- https://qdrant.tech/documentation/concepts/filtering/
