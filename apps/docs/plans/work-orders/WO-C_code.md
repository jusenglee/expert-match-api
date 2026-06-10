# WO-C · 코드 변경 (런타임 파이프라인 chunk화)

> MIGRATION_PLAN.md Phase C 대응. 가장 크고 상세한 WO. 본 문서의 공통 마이그레이션 컨텍스트(정체성/v1.x↔v2.0/HARD 제약/WO 그래프)는 워크플로 공통 블록을 참조하며 여기서 반복 서술하지 않는다.

---

## 1. 개요

- **목적:** 런타임 파이프라인(planner → retrieval → evidence_selector → reasoner) 전체를 "연구자 1명 = N Point / 4 named-vector 브랜치" 모델에서 "chunk 1개 = 1 Point / 단일 `dense_e5i`+`sparse_splade` / 11 doc_type(4 family)" 모델로 재작성한다.
- **한 줄 요약:** 브랜치(`basic/art/pat/pjt`) 기반 코드 전체를 doc_type 기반으로 교체하고, chunk hit → `researcher_id` 앱단 집계(equal RRF 누적 + doc_type별 chunk_cap + 옵트인 prior + dedupe)를 신설하며, evidence id를 `chunk_id`로 통일하고 `CrossEncoderEvidenceSelector`(신규)를 1급으로 도입한다.
- **완료 시 달성 상태:**
  - `apps/` 전체에서 `BRANCHES`/`BRANCH_WEIGHTS`/`PATH_WEIGHTS`/`_infer_point_branch`/`*_vector_e5i`/`*_vector_splade`/`paper:N|project:N|patent:N` 심볼이 제거된다.
  - `NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks`(WO-B 컬렉션)을 가리키는 단일 컬렉션에서 chunk 검색 → 연구자 집계가 동작한다.
  - `tests/test_cross_encoder_evidence_selector.py`(현재 CI red, import 실패)가 green이 된다.
  - 외부 응답이 `searched_doc_types`/`doc_type_coverage`/`matched_doc_types`/`evidence[*].chunk_id`로 전환되고, EXTERNAL_API_CHANGELOG v2.0 항목과 정합한다.
  - HARD 제약(equal RRF / 후보 리랭커 OFF / OR recency / LLM no-rerank / chunk_id evidence / meta 비정규화)을 코드가 강제한다.

---

## 2. 선행조건 & 의존성

- **선행 WO:** **WO-0 (안정화 & 계약 고정)** 필수. WO-0이 확정하는 산출물에 본 WO 전체가 의존한다:
  - `chunk_id` 코덱(파싱/생성: 형식 `PREFIX_researcher_id_seq_c#`, 예 `PUB_M1006328_0001_c0`) — C2/C6/C7/C8에서 사용.
  - doc_type 11종 enum + doc_type→family 매핑 상수(identity/achievement/assessment/expertise) — C2가 정의 위치를 제공받아 import.
  - CI green 기준선(현재 `tests/test_cross_encoder_evidence_selector.py:16-19`의 `CrossEncoderEvidenceSelector`/`rerank_source` import가 구현체 부재로 red). WO-0이 "red 상태를 known-failing으로 격리"했다면 C6에서 해제.
- **환경/VPN:** 코드 작성과 단위테스트는 **로컬 가능(VPN 불필요)**. 통합/live 검증(`/health/ready`, `ntis-validate-live`, 실제 `/recommend`)은 **WO-B가 생성한 `ntis_researcher_chunks` 컬렉션 + 임베딩/LLM 서버(VPN)** 가 있어야 한다.
- **차단 요소:**
  - WO-B 컬렉션 인덱스 필드 세트가 C2 `PAYLOAD_INDEX_FIELDS`(신규)와 **반드시 정합**해야 한다. 불일치 시 C8 readiness가 false. WO-0/WO-B와 인덱스 키 목록을 같은 소스에서 공유할 것(DATA_MODEL.md §5).
  - WO-A(소스→chunk 변환)가 적재한 payload 3층 구조/`researcher_meta` 8 count가 C4 필터·C7 카드의 전제다. 단위테스트는 픽스처로 대체 가능하나 통합 검증은 WO-A 데이터 필요.
- **병행성:** 본 WO의 "코드 골격"(C1~C7 작성)은 WO-A/WO-B와 병렬 가능. "통합 검증"(C8 + golden)은 WO-A·WO-B 완료 후.

---

## 3. 범위

### In Scope
- `apps/core/config.py`: 신규 env, `branch_*` → `doc_type_*` 개명, 브랜치 한정 설정 제거. (C1)
- `apps/search/schema_registry.py`: 브랜치 벡터 맵 제거, 단일 벡터 상수 + doc_type→family 매핑 + v2.0 인덱스 필드. (C2)
- `apps/search/retriever.py`: `search()`/`search_weighted()` 중복 통합, 브랜치 가중 제거, doc_type 경로 검색 + chunk→researcher 앱단 집계 신설. (C3)
- `apps/search/filters.py`: 도메인별 recency → 단일 `event_year` recency, OR(min_should) 가드 보존, org 필터를 chunk-level + cross-chunk post-filter로. (C4)
- `apps/recommendation/planner.py`: `hard_filters` 허용 키 재정의, intent_flags=prior 힌트만, 하위호환 shim 정리. (C5)
- `apps/recommendation/evidence_selector.py` + `reasoner.py`: `CrossEncoderEvidenceSelector` 신규 구현, item_id→chunk_id, `VALID_EVIDENCE_ID_PATTERN`/PAYLOAD_PROFILE 전환. (C6)
- `apps/recommendation/cards.py` + `apps/api/schemas.py` + `apps/domain/models.py`: `doc_type_coverage`/`matched_doc_types`/`researcher_meta` 채움, BREAKING 리네임, 신규 모델 필드. (C7)
- `apps/search/live_validator.py` + `apps/tools/validate_live.py`: 단일 벡터/v2.0 인덱스/`chunk_id` 샘플 점검. (C8)
- `apps/recommendation/service.py` + `apps/api/main.py`: BRANCHES import 제거, 단일 검색 경로로 통합, 응답 조립 필드 전환.

### Out of Scope
- chunk payload 변환기/적재 스크립트 (→ **WO-A**).
- Qdrant 컬렉션 생성·인덱스 부트스트랩(`qdrant_bootstrap.py`의 컬렉션 생성 로직) (→ **WO-B**). 단, C8 readiness가 읽는 컬렉션 스키마 가정은 WO-B와 정합 필요.
- cross-encoder 실제 모델 다운로드/서빙 (→ 운영). 본 WO는 인터페이스 + lexical 강등까지.
- 권위 문서(DATA_MODEL/SERVICE_FLOW/DATA_CONTRACT/API_SPEC/CHANGELOG) **신규 작성**(이미 v2.0으로 존재). 단, 코드 변경에 수반되는 CHANGELOG 갱신 의무는 C7에 포함.
- 컷오버/golden 전수 통과 판정 (→ **WO-D**). 본 WO는 단위테스트 + 핵심 golden 스모크까지.

---

## 4. 상세 작업 항목 (파일별 하위 섹션)

> 각 항목은 [대상 파일:심볼/라인] + 변경 + 근거 형식. 라인 번호는 작성 시점 스냅샷이며 **STEP 1에서 각자 재확인**한다.

### C1. `apps/core/config.py` — 설정 재정의

1. **[config.py:63 `qdrant_collection_name`]** default를 `"researcher_recommend_proto"` → `"ntis_researcher_chunks"`로 변경. env override 키는 기존 `NTIS_QDRANT_COLLECTION_NAME`(env_prefix `NTIS_` + `qdrant_collection_name`) 유지. 근거: DATA_MODEL.md §1, MIGRATION_PLAN Phase C `config.py` 행.
2. **[config.py:135 `branch_prefetch_limit` / :139 `branch_output_limit`]** → `doc_type_prefetch_limit` / `doc_type_output_limit`로 **개명**(값 100/40 유지). 주석의 "브랜치(논문/특허/기본)" 표현을 "doc_type 경로"로 교정. retriever.py(C3)에서 참조처 동시 변경. 근거: "브랜치 한정 설정 제거"(MIGRATION_PLAN), ADR 0003.
3. **[config.py:78-79 `support_rule_stable_min`/`expanded_min`]** v2.0에는 브랜치 교차 support 개념이 없다(집계는 RRF 누적 + chunk_cap). 두 설정은 **제거 또는 deprecated(기본 0 유지로 무력화)**. 제거 시 retriever.py:716-717/1179-1180, schemas/`SearchCandidateItem.stable_hits`(C7) 동시 정리. 권장: 제거하고 집계 trace로 대체. 근거: SERVICE_FLOW §2(집계 단계에 support rule 없음).
4. **[config.py 신규 — env 추가]** (전부 **신규 생성**)
   - `qdrant_collection_name`은 위 1번으로 충족.
   - `doc_type_priors: dict[str,float] | None = None` (env `NTIS_DOC_TYPE_PRIORS`, 미설정=equal). HARD: equal이 기본. ADR 0005 §2.
   - `candidate_reranker: Literal["off","band"] = "off"` (env `NTIS_CANDIDATE_RERANKER`). HARD: 기본 OFF, band는 동률 밴드 재배열만. ADR 0005 §3.
   - `retrieval_doc_types: list[str] | None = None` (env `NTIS_RETRIEVAL_DOC_TYPES`, 미설정=전체 11종 화이트리스트). ADR 0003(축소는 화이트리스트로만).
   - `doc_type_chunk_cap: int = 3` (env `NTIS_DOC_TYPE_CHUNK_CAP`). DESIGN_GUIDELINES §5.3.
   - `evidence_family_cap: dict[str,int]` 기본 `{"achievement":10,"assessment":6,"expertise":6,"identity":1}` (env `NTIS_EVIDENCE_FAMILY_CAP`). REASONER_RUNTIME_POLICY "Default family caps".
   - cross-encoder 모델 설정(신규): `evidence_reranker_backend: Literal["cross_encoder","lexical"] = "lexical"`(모델 부재 기본), `cross_encoder_model_name: str | None`, `cross_encoder_base_url`/`cross_encoder_api_key`(원격 서빙 옵션), `ce_relevance_floor: float = 0.30`, `ce_pregate_per_type: int = 20`, `ce_max_pairs_per_request: int = 256`, `ce_top_n_per_type: int = 5`. 값/이름은 `tests/test_cross_encoder_evidence_selector.py:79-87`의 생성자 인자(`top_n_per_type`/`relevance_floor`/`pregate_per_type`/`max_pairs_per_request`)와 정합. 근거: ADR 0005 §6.4, REASONER_RUNTIME_POLICY.
5. **[config.py:146-149 `final_recommendation_min/max`]** 의미 변화 없음(연구자 단위 추천 수). 주석만 "후보=연구자(집계 단위)"로 명확화. 유지.

### C2. `apps/search/schema_registry.py` — 스키마 레지스트리 재작성

1. **[schema_registry.py:19-24 `BRANCHES`]** 4-튜플 **제거**. 대신 `DOC_TYPES: tuple[str,...]`(11종, WO-0 enum import) + `FAMILIES`(identity/achievement/assessment/expertise) + `DOC_TYPE_TO_FAMILY: dict[str,str]`(WO-0 매핑 상수 import 또는 재노출) 추가(**신규**). 근거: DATA_MODEL §4.
2. **[schema_registry.py:31-36 `DENSE_VECTOR_BY_BRANCH` / :43-48 `SPARSE_VECTOR_BY_BRANCH`]** **제거**. 단일 상수 `DENSE_VECTOR_NAME = "dense_e5i"`, `SPARSE_VECTOR_NAME = "sparse_splade"` 추가(**신규**). 근거: DATA_MODEL §2.
3. **[schema_registry.py:101-113 `SearchSchemaRegistry`]** dataclass 필드 `dense_vector_by_branch`/`sparse_vector_by_branch`(dict) → `dense_vector_name: str` / `sparse_vector_name: str`로 교체. `default()`도 단일 이름 반환. retriever/live_validator의 `registry.dense_vector_by_branch[branch]` 참조(retriever.py:299/305/327/901/904, live_validator.py:239/247/254)를 전부 `registry.dense_vector_name` 단일 참조로 수정.
4. **[schema_registry.py:58-73 `FILTERABLE_ROOT_FIELDS`/`FILTERABLE_NESTED_FIELDS`]** nested(`publications`/`intellectual_properties`/`research_projects`) 키 기반 → v2.0 3층 payload 키로 교체. root 필터 후보: `researcher_id`, `doc_type`, `tags`, `researcher_meta.*`(8 count + `affiliated_organization`/`highest_degree`), `domain_attrs.*`(journal_class/ip_type/application_country/performing_organization/managing_agency/appointing_organization/evaluation_agency_name/tech_classification_system/specific_specialty_name), `event_date`/`event_year`. 근거: DATA_MODEL §5.
5. **[schema_registry.py:79-98 `PAYLOAD_INDEX_FIELDS`]** v1.x nested `publications[].*` 등 **전부 교체**. v2.0 인덱스 세트(keyword/integer/datetime)로 재작성하되 **WO-B 부트스트랩이 실제 생성하는 인덱스와 1:1 정합**. C8 live_validator가 이 상수로 readiness를 판단하므로 불일치 시 즉시 503. DATA_MODEL §5 표를 그대로 옮긴다.

### C3. `apps/search/retriever.py` — ★ 검색 통합 + chunk→researcher 집계 (가장 큰 단일 작업)

> **선행 리팩터(필수):** `search()`(:442)와 `search_weighted()`(:922)는 캐시 조회/payload 빌드/집계/support/정렬/캐시 저장이 대량 중복이다. v2.0에서 weighted 경로는 폐기되므로 **단일 `search()`로 통합**한다. `search_weighted` 전용 헬퍼(`_execute_single_vector_query` :878, `_minmax_normalize` :83, `WEIGHTED_HYBRID_DENSE/SPARSE` :79-80, `RETRIEVAL_MODE_WEIGHTED` :76) 제거.

1. **[retriever.py:62-67 `BRANCH_WEIGHTS` / :70-73 `PATH_WEIGHTS`]** **제거**. 집계 가중은 C1 `doc_type_priors`(기본 equal)로만. HARD: equal RRF. ADR 0005 §1·§2.
2. **[retriever.py:242-246 `_infer_point_branch` / :429 호출부]** **제거**. point_id는 이제 `chunk_id`이고 doc_type은 payload `doc_type` 필드로 직접 읽는다. trace의 `point_branch_hint`(:429) → `doc_type`로 교체. `_build_retrieval_score_trace`(:411-440)의 `branch_matches` → `doc_type_matches`로 키 개명(SERVICE_FLOW §6 `*_doc_type_counts`).
3. **[retriever.py:75-76 `RETRIEVAL_MODE`]** `"keyword_pool_then_hybrid"` 유지(DATA_CONTRACT §3). `RETRIEVAL_MODE_WEIGHTED` 제거.
4. **[retriever.py:276-284 `_build_path_tasks`]** `for branch in BRANCHES` + `art/pjt`만 expanded인 하드코딩 제거. v2.0 경로 = **doc_type(또는 family) 경로**. `retrieval_doc_types` 화이트리스트(C1) 기준으로 doc_type별 task 생성. DATA_CONTRACT §2(경로별 별도 hint 생성 안 함 — 동일 기본 텍스트 공유).
5. **[retriever.py:286-315 `_build_branch_query_payload`]** → `_build_doc_type_query_payload`로 개명. `using=self.registry.dense_vector_by_branch[branch]`/`sparse_vector_by_branch[branch]` → 단일 `registry.dense_vector_name`/`sparse_vector_name`. **doc_type 분기는 `using`이 아니라 `Prefetch.filter`/`query_filter`에 `FieldCondition(key="doc_type", match=MatchValue(value=<doc_type>))` 추가**로 표현(DATA_MODEL §2 "doc_type은 payload 필터"). `models.FusionQuery(fusion=models.Fusion.RRF)` 유지(:310). HARD: equal RRF.
6. **[retriever.py:317-332 `_build_keyword_query_payload`]** 1단계 sparse 키워드 쿼리. `using=sparse_vector_by_branch[branch]` → `sparse_vector_name`. limit = `doc_type_prefetch_limit`(C1 개명). 1단계는 doc_type 무관 전체 sparse 검색으로 `researcher_id` 풀 수집(DATA_CONTRACT §3 1단계). doc_type 한정은 화이트리스트 필터로만.
7. **[retriever.py:374-409 `_collect_keyword_candidate_pool`]** payload 검증을 `ExpertPayload.model_validate`(:395) → v2.0 chunk payload 모델(C7 신규, 예 `ChunkPayload`)로 교체. `researcher_id`는 `payload.basic_info.researcher_id`(:402) → `payload.researcher_id`(L1 필드). 풀 dedupe(중복 제거) 유지.
8. **[retriever.py:612-615 `_with_candidate_pool_filter`]** `key="basic_info.researcher_id"`(:358) → `key="researcher_id"`. min_should/must 보존 로직(:364-372) 유지(C4 OR 가드와 합성되므로 보존 필수).
9. **[retriever.py:650-705 집계 + RRF]** **신규 재작성 — 본 WO 핵심.**
   - 기존: `weight = BRANCH_WEIGHTS[branch] * PATH_WEIGHTS[path]`(:667), `rrf_contribution = weight * (1/(rank+60))`(:687).
   - v2.0: chunk hit을 `researcher_id`로 묶는다. 각 chunk의 RRF 기여 = `1/(rank+60)`(equal). **doc_type별 chunk_cap**(`doc_type_chunk_cap`, 기본 3): 한 연구자의 동일 doc_type에서 상위 cap개 chunk만 점수 누적(다작 독식 방지, DESIGN_GUIDELINES §5.3·§8). **prior**: `score += contribution * doc_type_priors.get(doc_type, 1.0)`(미설정=1.0=equal). **연구자당 1건 dedupe**.
   - aggregator 키/값을 expert/chunk 모델로 재정의: `branches`(:698) → `matched_doc_types: set`, `branch_matches`(:702) → `doc_type_matches`(각 항목 `{doc_type, chunk_id, rank, score}`). evidence 후보로 쓸 매칭 chunk 목록도 연구자별로 보존(C6 입력).
   - HARD: 후보 순위는 equal RRF 누적이 척추. prior는 옵트인 랭크누적 가중일 뿐 가중 RRF/score 가중합 아님(ADR 0005 §2).
10. **[retriever.py:707-757 support rule]** support_rule(stable/expanded) 분기(:716-726) **제거**(C1-3). 대신 C4 hard filter(event_year OR recency / meta count / 학위 / exclude org)는 가능한 한 Qdrant 필터로 1·2단계에 위임하고, **cross-chunk org post-filter만 앱단**에서 수행(C4). `exclude_orgs`(:720-724) 처리는 `data["payload"].basic_info.affiliated_organization` → `researcher_meta.affiliated_organization` + 매칭 chunk의 org 필드(performing/managing/appointing/evaluation_agency)로 확장(DESIGN_GUIDELINES §5.4).
11. **[retriever.py:728-741 / :793-796 SearchHit 조립]** `data_presence_flags`(:732-737, basic/art/pat/pjt) → `doc_type_coverage`(family 단위 dict) + `matched_doc_types`. `expanded_shadow`(:793-796, stable/expanded 기반) 제거. C7 `SearchHit` 모델과 정합.
12. **[retriever.py:227-235 `_sort_hits`]** 정렬 키 `(-score, researcher_name, expert_id)` 유지(결정론적 score DESC → name ASC → researcher_id ASC). `hit.payload.basic_info.researcher_name`(:232) → `hit.payload.researcher_name` 또는 집계 시 보존한 name. HARD: SERVICE_FLOW §2 5단계.
13. **[retriever.py:35 `SearchHit` import / :39 `BRANCHES,SearchSchemaRegistry` import]** BRANCHES import 제거. `ExpertPayload`(:36) → chunk payload 모델로 교체.
14. **[옵트인 후보 리랭커]** `candidate_reranker=="band"`일 때만 동작하는 hook(신규, 기본 OFF). **score 동률 밴드 내 재배열만**, 탈락/생성 금지. 기본 off에서는 호출 경로 없음. HARD: ADR 0005 §3. (본 WO는 hook + off 경로까지; 실제 cross-encoder 연결은 옵트인 실험.)

### C4. `apps/search/filters.py` — 단일 event_year recency + chunk-level org

1. **[filters.py:64-78 counts_mapping]** 키를 `*_cnt_min`(article/scie/patent/project)에서 DATA_CONTRACT §1.1 신규 키로 교체: `publication_count_min`/`scie_publication_count_min`/`intellectual_property_count_min`/`research_project_count_min` + 신규 `researcher_assessor_count_min`/`expert_assessor_count_min`. backend_key `researcher_profile.*` → `researcher_meta.*`. 근거: DATA_CONTRACT §1.1, DATA_MODEL §3.1.
2. **[filters.py:54-61 degree]** `hard_filters.get("degree_slct_nm")` → `highest_degree`(DATA_CONTRACT §1.1). `researcher_profile.highest_degree` → `researcher_meta.highest_degree`.
3. **[filters.py:83-166 도메인별 recency]** `art_recent_years`(:85)/`pat_recent_years`(:117)/`pjt_recent_years`(:149)와 nested `NestedCondition`(:102/135/152) **제거**. 신규: 단일 `recent_years`(DATA_CONTRACT §1.1) + `recent_doc_types`(적용 family/doc_type 목록). 각 대상 doc_type에 대해 `FieldCondition(key="event_year", range=Range(gte=올해-N))` AND `FieldCondition(key="doc_type", MatchValue(...))`를 한 조건으로 만들어 `recent_activity_conditions`에 적재. nested 배열 조건 불필요(chunk-level).
4. **[filters.py:168-182 OR 가드 — ★보존 필수]** `len(recent_activity_conditions) >= 2`일 때 `models.Filter(min_should=models.MinShould(conditions=..., min_count=1))`로 묶는 로직(:172-180)을 **반드시 보존**. 여러 doc_type recency를 AND로 묶으면 0건 회귀(v1.x 장애). HARD: OR recency. DATA_MODEL §3.2, DESIGN_GUIDELINES §5.4. 단일 조건일 때 직접 must(:181-182)도 유지.
5. **[filters.py:184-191 major]** `major_nm` → `researcher_meta` 또는 `domain_attrs`(profile/major)로 매핑 재검토. v2.0 hard_filters 허용 키에 major가 없으면(DATA_CONTRACT §1.1) **제거**하거나 `journal_class` 등 허용 키로 대체. WO-0 허용 키 확정에 맞춤.
6. **[filters.py:193-202 exclude_orgs / :40-51 include_orgs]** `basic_info.affiliated_organization_exact` → `researcher_meta.affiliated_organization`. **Qdrant 필터로는 `researcher_meta.affiliated_organization`만 배제 가능**하고, 매칭 chunk의 `performing_organization`/`managing_agency`/`appointing_organization`/`evaluation_agency_name` 교차 배제는 **retriever 앱단 post-filter(C3-10)** 로 보완. 근거: DESIGN_GUIDELINES §5.4.
7. **[전체]** `compile()`이 합성하는 must/must_not 구조(:205-207)는 유지. C3-8 `_with_candidate_pool_filter`가 이 필터에 `researcher_id MatchAny`를 합성할 때 min_should(OR 가드)가 깨지지 않도록 검증(filters 단위테스트에 OR+pool 합성 케이스 추가).

### C5. `apps/recommendation/planner.py` — hard_filters 키 재정의 + intent_flags prior 힌트

1. **[planner.py:139-235 `_build_system_prompt`]** 출력 스키마(:171-183)·규칙(:166 "명시적으로 지원되는 구조화 필터만 hard_filters")의 `hard_filters` 예시/허용 키를 DATA_CONTRACT §1.1로 교체: `highest_degree`/`recent_years`/`recent_doc_types`/`*_count_min`(6종)/`journal_class`. 구 `art_recent_years` 등은 프롬프트에서 제거.
2. **[planner.py:152 `intent_flags`]** 프롬프트 설명을 "집계 prior 힌트(예: 평가이력 강조)만, doc_type on/off는 결정하지 않음"으로 명문화. HARD: ADR 0003(planner는 doc_type on/off 금지). retriever는 `intent_flags`를 prior 힌트로만 소비(C3-9, 옵트인).
3. **[planner.py:249-266 retrieval_core/role_terms/action_terms 분리]** 유지(역할/행위어 누수 차감 :253-266은 1차 keyword stage 0건 사고 방지 — d57823d 커밋 교훈). 보존.
4. **[planner.py:271-273 하위호환 shim]** `output.core_keywords = list(output.retrieval_core)`(:272), `task_terms = role_terms+action_terms`(:273). v2.0에서도 evidence_selector/reasoner가 `core_keywords`를 읽으므로(DATA_CONTRACT §1 예시에 `core_keywords` 유지) **shim 유지하되 주석으로 "전환기 호환"임을 명시**. 완전 제거는 C6/C7 소비처를 `retrieval_core`로 통일한 뒤 별도 정리. (지시: shim을 무분별 제거하지 말 것 — empty keyword 회귀 위험.)
5. **[planner.py:512-522 fallback_output]** 필드 동일, 영향 없음. `hard_filters=dict(filters)` 유지.

### C6. `apps/recommendation/evidence_selector.py` + `reasoner.py` — ★ CrossEncoderEvidenceSelector 신규 + chunk_id 통일

> **현재 상태:** `evidence_selector.py`에는 `KeywordEvidenceSelector`(:126)만 존재. `tests/test_cross_encoder_evidence_selector.py:16-19`가 `CrossEncoderEvidenceSelector`를 import해 **CI red**. 이 테스트가 구현 계약을 정확히 규정하므로 그대로 충족시킨다.

**evidence_selector.py:**
1. **[신규 `CrossEncoderEvidenceSelector`]** 테스트(`:79-87`, `:90-207`)가 규정하는 계약을 그대로 구현:
   - 생성자: `scorer`(`.score(pairs)->list[float]`, `.model_name`), `fallback: KeywordEvidenceSelector`, `top_n_per_type`, `relevance_floor`, `pregate_per_type`, `max_pairs_per_request`. (C1 ce_* 설정과 매핑.)
   - 동작: query는 `semantic_query` 우선, 없으면 `core_keywords`(`test_prefers_semantic_query_over_core_keywords` :194-206 / `last_trace["query"]`). dedupe(title+year, `test_deduplicates...` :154-168, `dedup_dropped`) → pre-gate(`pregate_per_type`, `total_pairs` 제한 :143-151) → cross-encoder score → `relevance_floor` 미만 drop(`dropped_below_floor` :110-124) → score DESC 정렬 → `top_n_per_type` cap(:127-140).
   - `last_trace`: `mode`("cross_encoder"|"lexical_fallback"), `fallback_reason`("no_scorer"|"scorer_error"), `query`, `candidate_evidence_counts[*]`에 `dropped_below_floor`/`dedup_dropped`.
   - scorer가 None이거나 raise → `fallback.select(...)`로 lexical 강등(`test_falls_back...` :171-191).
2. **[family cap]** 테스트의 `top_n_per_type`는 paper/project/patent 타입 단위 cap. 운영에서는 추가로 **family cap**(C1 `evidence_family_cap` achievement10/assessment6/expertise6/identity1) 적용. REASONER_RUNTIME_POLICY "Default family caps".
3. **[item_id → chunk_id]** `KeywordEvidenceSelector`의 `item_id=f"paper:{index}"`(:210)/`project:{index}`(:252)/`patent:{index}`(:292) **제거**. `RelevantEvidenceItem.item_id`(:24)를 chunk의 `chunk_id`로 채운다(WO-0 코덱). `RelevantEvidenceItem`에 `doc_type`/`event_date` 필드 추가, `RelevantEvidenceBundle`(:34-44)을 family 단위 묶음으로 확장(DATA_CONTRACT §5: `chunk_id`/`doc_type`/`title`/`event_date`/`snippet`/`match_score`).
4. **[models.py rerank_source 등 — 신규]** `RelevantEvidenceItem`(또는 evidence chunk 모델)에 `rerank_source: str`(값 `"cross_encoder"`|`"lexical"`, `test_orders...` :106 `papers[0].rerank_source == "cross_encoder"`), `doc_type`, `chunk_id` 필드 추가. `apps/domain/models.py`에 chunk payload 모델(`ChunkPayload`: L1 `researcher_id`/`researcher_name`/`doc_type`/`chunk_id`/`chunk_text`/`researcher_meta`, L2 `event_date`/`event_year`/`tags`, L3 `domain_attrs`) **신규** + `researcher_meta` 모델(8 count + org + degree).
5. **HARD: evidence 리랭커는 grounding 선별만, 후보 순위 영향 0**(ADR 0005 §6.4, DESIGN_GUIDELINES §6.4). 후보 score/정렬에 절대 쓰지 않는다.

**reasoner.py:**
6. **[reasoner.py:44 `VALID_EVIDENCE_ID_PATTERN`]** `^(paper|project|patent):\d+$` → **chunk_id 형식 정규식**(WO-0 코덱: `^[A-Z]+_[^_]+_\d+_c\d+$` 등 확정 패턴). `_normalize_output`(:599-612)의 검증/dedupe 로직은 패턴만 교체.
7. **[reasoner.py:46-78 PRIMARY/RETRY_PAYLOAD_PROFILE]** chunk pool 기반으로 재정의. `relevant_limit`/`all_papers_limit` 등 nested 한정자(:50-61) → family별 chunk cap(achievement/assessment/expertise/identity)로 의미 전환. `_serialize_candidates`(:472-559)의 `relevant_papers`/`relevant_patents`/`relevant_projects`(:527-541)·`all_papers`/`all_patents`/`all_projects`(:542-556)를 chunk family 묶음 직렬화로 교체. `branch_presence_flags`(:509) → `doc_type_coverage`.
8. **[reasoner.py:253-254 프롬프트 evidence id 규칙]** `paper:<number>` 등 형식 안내(:254)를 "제공된 `chunk_id`를 정확히 복사"로 교체(REASONER_RUNTIME_POLICY "Evidence ID policy v2.0"). `_compact_retrieval_grounding`(:453-470)의 `branch`(:460/467) → `doc_type`.
9. **[reasoner.py:43 `REASON_GENERATION_MAX_TOKENS = 8192`]** 유지(REASONER_RUNTIME_POLICY "8192 completion-token hint"). batch size 5(service.py:44)도 유지.
10. **[reasoner.py:176-177 dead code — 정리]** `from tenacity import retry` 후 `payload_profile = RETRY_PAYLOAD_PROFILE if retry > 0 else PRIMARY_PAYLOAD_PROFILE`는 **버그성 dead code**(`retry`는 데코레이터, 정수 비교 무의미; `payload_profile` 미사용). `PassThroughReasonGenerator.generate`(:157-211)에서 제거. 서버 fallback reason 경로(service.py:678-718)는 유지.

**service.py 연동:**
11. **[service.py:30/35 import]** `VALID_EVIDENCE_ID_PATTERN`(:30) 재사용 유지(패턴만 C6-6에서 교체됨). `from apps.search.schema_registry import BRANCHES`(:35) **제거**.
12. **[service.py:643-718 `_build_*_evidence`/`_build_server_fallback_reason`]** `EvidenceItem.type` 4종(:689-694 type_labels paper/project/patent/profile)을 doc_type 11종 라벨로 확장. `EvidenceItem`(models.py:241-246, type Literal 4종)에 `chunk_id` 추가 + type을 doc_type 문자열로 확장(C7). 서버 fallback은 "최상위 chunk로 결정론적 조립"(REASONER_RUNTIME_POLICY) 유지.

### C7. `apps/recommendation/cards.py` + `apps/api/schemas.py` + `models.py` — 응답 계약 전환 (BREAKING)

1. **[cards.py:35-108 `_build_card`]** `payload.publications`/`intellectual_properties`/`research_projects`(:37-57) nested 접근 제거. 입력이 집계된 연구자(SearchHit + 매칭 chunk 목록)이므로 `researcher_meta` 기반 카드 구성. `branch_presence_flags=hit.data_presence_flags`(:92) → `doc_type_coverage`(family dict) + `matched_doc_types`. `counts`(:93-98) 키 `article_cnt`/`scie_cnt`/`patent_cnt`/`project_cnt` → `publication_count`/`scie_publication_count`/`intellectual_property_count`/`research_project_count`(+`researcher_assessor_count`/`expert_assessor_count`), 값 = `researcher_meta.*`. API_SPEC §2-2.
2. **[cards.py:60-72 matched_filter_summary]** `degree_slct_nm`/`art_sci_slct_nm`/`project_cnt_min`(:61/65/69) → C4 신규 hard_filters 키로 교체.
3. **[models.py:7 `BranchName` / :185 `SeedEvidencePoint.branch` / :225-226 `SearchHit.branch`/`data_presence_flags` / :255 `CandidateCard.branch_presence_flags`]** `BranchName` Literal **제거**. `SearchHit`에 `doc_type_coverage: dict[str,bool]`(family) + `matched_doc_types: list[str]` + 매칭 chunk 목록 필드 추가, `branch`/`data_presence_flags`/`stable_support_count`/`expanded_support_count`/`support_branches`(:229-231) 제거. `CandidateCard`(:248-267)도 `doc_type_coverage`/`matched_doc_types`/`researcher_meta` 카운트로. `EvidenceItem`(:241-246) type Literal 4종 → doc_type 문자열(str) + `chunk_id: str | None` 추가. `RecommendationDecision`(:270-285) `evidence` 항목이 chunk_id 보유.
4. **[schemas.py:63 `RecommendationResponse.searched_branches` / :106 `SearchCandidatesResponse.searched_branches`]** → `searched_doc_types`. **BREAKING.** API_SPEC §2, CHANGELOG A.
5. **[schemas.py:83 `SearchCandidateItem.branch_presence_flags`]** → `doc_type_coverage: dict[str,bool]`(family) + 신규 `matched_doc_types: list[str]`. `counts`(:86) 키 명칭 §C7-1과 동일. `stable_hits`/`expanded_hits`/`support_branches`(:92-94) 제거(C1-3). **BREAKING.**
6. **[main.py:63/346/570 `BRANCHES`]** import(:63) 제거. `/health`(:346) `searched_branches:list(BRANCHES)` → `searched_doc_types`(retrieval_doc_types 또는 전체 11종). `/search/candidates`(:570) `searched_branches=list(BRANCHES)` → `searched_doc_types`. `SearchCandidateItem` 조립(:492-505) `branch_presence_flags=card.branch_presence_flags` → `doc_type_coverage`+`matched_doc_types`, support 필드 제거.
7. **[main.py:46/207 `KeywordEvidenceSelector`]** import + 주입(:207)을 **`CrossEncoderEvidenceSelector`(C1 설정 기반, scorer 부재 시 lexical fallback=KeywordEvidenceSelector)** 로 교체. `build_app_runtime`(:127-220)에서 scorer 생성(설정 `evidence_reranker_backend`/`cross_encoder_*`)을 조립.
8. **[service.py:884 `searched_branches:list(BRANCHES)`]** → `searched_doc_types`. `_build_recommendation_response`(:858-922) 응답 필드 전환.
9. **[main.py:433-457 `/recommend/stream`]** `service.recommend_stream` 호출 — 현재 service.py에 `recommend_stream` **미존재**(메서드 없음). STEP 1에서 존재 여부 재확인하고, 없으면 본 WO에서 stub/제거 결정(WO-0 CI 기준선과 합의). 응답 evidence chunk_id 전환 시 동일 적용.
10. **[EXTERNAL_API_CHANGELOG.md 갱신 — 의무]** 코드 변경 커밋과 **같은 커밋**에서 CHANGELOG v2.0 항목(이미 초안 존재)의 커밋 해시/실 적용 표기를 갱신. DESIGN_GUIDELINES §9(둘 이상 단계 동시 변경 시 SERVICE_FLOW/DATA_CONTRACT 동기 갱신). API_SPEC/DATA_CONTRACT는 이미 v2.0이므로 코드가 그에 맞는지 정합만 확인.

### C8. `apps/search/live_validator.py` + `apps/tools/validate_live.py` — readiness 전환

1. **[live_validator.py:21 import / :239/247/254 `BRANCHES` 루프]** `BRANCHES`/`registry.dense_vector_by_branch[branch]` 제거. dense 체크(:238-242) → `registry.dense_vector_name("dense_e5i") in dense_names` 단일 확인. sparse 체크(:245-263) → `sparse_vector_name("sparse_splade")` 단일 + IDF modifier 정합(`_modifier_matches_expected` :130-133 유지: SPLADE=none / bm25 fallback=IDF). DATA_MODEL §2.
2. **[live_validator.py:135-162 `_build_sample_checks`]** `publications`/`intellectual_properties`/`research_projects`(:138-140) nested 키 → v2.0 chunk payload 키. 신규 체크: `sample_chunk_id_present`(point id == payload `chunk_id`), `sample_doc_type_valid`(11종 enum), `sample_researcher_meta_present`(8 count), `sample_event_year_consistent`(event_date 연도 일치, DATA_MODEL §6 불변식 4). `SAMPLE_COMPLETENESS_CHECKS`(:37-42)/`OPTIONAL_CHECKS`(:44) 갱신.
3. **[live_validator.py:266-271 payload index 체크]** `PAYLOAD_INDEX_FIELDS`(C2 신규)와 실제 컬렉션 인덱스 비교. WO-B 정합 전제.
4. **[validate_live.py]** 구조 변경 없음(`SearchSchemaRegistry.default()` :50 가 단일 벡터 반환하도록 C2에서 바뀌므로 자동 반영). `ntis-validate-live` CLI 진입점(`apps.tools.validate_live:main`) 유지. 출력 JSON에 `sample_point_id`가 `chunk_id` 형태인지 확인.
5. **[schemas.py:125-132 `ReadinessResponse`]** `sample_point_id`(:130) 의미가 chunk_id로. `collection_name` default가 C1 `ntis_researcher_chunks` 반영. API_SPEC §2-4.

---

## 5. 변경/생성 대상 파일

| 파일 | 변경유형 | 요지 |
|---|---|---|
| `apps/core/config.py` | 수정 | 컬렉션 default→`ntis_researcher_chunks`, `branch_*`→`doc_type_*` 개명, support_rule 제거, 신규 env(priors/reranker/whitelist/chunk_cap/family_cap/ce_*) (C1) |
| `apps/search/schema_registry.py` | 수정 | BRANCHES·브랜치 벡터맵 제거, 단일 `dense_e5i`/`sparse_splade` + doc_type→family + v2.0 인덱스 (C2) |
| `apps/search/retriever.py` | 수정 | search/search_weighted 통합, 브랜치 가중·`_infer_point_branch` 제거, chunk→researcher 집계(equal RRF+chunk_cap+prior+dedupe) 신설 (C3) |
| `apps/search/filters.py` | 수정 | 도메인별 recency→단일 `event_year`, OR(min_should) 가드 보존, org chunk-level+post-filter, `*_count_min` 키 (C4) |
| `apps/recommendation/planner.py` | 수정 | hard_filters 허용 키 재정의, intent_flags=prior 힌트, shim 주석화 (C5) |
| `apps/recommendation/evidence_selector.py` | 수정 | `CrossEncoderEvidenceSelector` 신규, item_id→chunk_id, family cap, lexical 강등 (C6) |
| `apps/recommendation/reasoner.py` | 수정 | `VALID_EVIDENCE_ID_PATTERN`→chunk_id, PAYLOAD_PROFILE→chunk pool, dead code 제거 (C6) |
| `apps/recommendation/cards.py` | 수정 | researcher_meta 기반 카드, `doc_type_coverage`/`matched_doc_types`/counts 명칭 (C7) |
| `apps/recommendation/service.py` | 수정 | BRANCHES import 제거, 단일 검색 경로, `searched_doc_types`, evidence chunk_id (C6/C7) |
| `apps/api/schemas.py` | 수정 | `searched_doc_types`/`doc_type_coverage`/`matched_doc_types`, support 필드 제거 (C7, BREAKING) |
| `apps/api/main.py` | 수정 | BRANCHES 제거, `/health`·`/search/candidates` 필드 전환, CrossEncoderEvidenceSelector 주입 (C7) |
| `apps/domain/models.py` | 수정/신규 | `BranchName`/SearchHit.branch 제거, `ChunkPayload`/`researcher_meta`/`rerank_source`·`chunk_id`·`doc_type` 신규 (C6/C7) |
| `apps/search/live_validator.py` | 수정 | 단일 벡터 체크, chunk payload 샘플 체크, v2.0 인덱스 (C8) |
| `apps/tools/validate_live.py` | 수정(경미) | 단일 벡터 레지스트리 자동 반영, chunk_id sample 확인 (C8) |
| `apps/docs/api/EXTERNAL_API_CHANGELOG.md` | 수정 | v2.0 항목 커밋/적용 표기 갱신(코드와 동일 커밋) (C7-10) |
| `tests/test_cross_encoder_evidence_selector.py` | (기존 활용) | 신규 selector 계약 검증 — 본 WO 구현으로 green 전환 |
| 신규 단위테스트 | 신규 | retriever 집계(chunk_cap/equal RRF/dedupe), filters OR recency, chunk_id evidence id 검증 |

---

## 6. 수용 기준 (Acceptance Criteria)

- [ ] `apps/` 전체에서 `BRANCHES`, `BRANCH_WEIGHTS`, `PATH_WEIGHTS`, `_infer_point_branch`, `*_vector_e5i`, `*_vector_splade`, `branch_presence_flags`, `searched_branches`, `paper:`/`project:`/`patent:` evidence id 출현이 **0회**(grep로 검증).
- [ ] `Settings`에 `doc_type_priors`(미설정=equal), `candidate_reranker`(default `off`), `retrieval_doc_types`(미설정=11종), `doc_type_chunk_cap`(=3), `evidence_family_cap`(ach10/ass6/exp6/id1), ce_* 설정이 존재하고 default가 사양과 일치.
- [ ] `qdrant_collection_name` default == `"ntis_researcher_chunks"`.
- [ ] `tests/test_cross_encoder_evidence_selector.py` 8개 테스트 전부 **green**(정렬/floor drop/top-N cap/pre-gate/dedup/lexical fallback/semantic_query 우선).
- [ ] retriever 집계 단위테스트: 동일 doc_type 4개 chunk hit인 연구자가 `doc_type_chunk_cap=3`에서 상위 3개만 점수에 기여; RRF 기여가 doc_type별 균등(prior 미설정 시); 연구자당 1건 dedupe.
- [ ] filters 단위테스트: `recent_years`+`recent_doc_types`가 2개 이상일 때 `min_should(min_count=1)` OR로 컴파일(0건 회귀 방지); `researcher_id MatchAny` pool 합성 후에도 OR 가드 보존.
- [ ] reasoner: `selected_evidence_ids`가 chunk_id 코덱 형식이 아니면 trace(`invalid_selected_evidence_ids_by_candidate`)에 기록(`paper:N` 형식 무효). evidence 조립은 `selected_evidence_ids`와 무관하게 선별 relevant 풀로 결정론적 수행(REASONER_RUNTIME_POLICY).
- [ ] evidence_selector가 후보 score/정렬에 영향 0(집계 결과는 evidence 선별 전후 동일 — 회귀 테스트).
- [ ] `pytest` 전체(WO-0 기준선 포함) green. ruff/타입 체크(프로젝트 표준) 통과.
- [ ] (통합, WO-B 필요) `GET /health/ready`가 `ntis_researcher_chunks`에 대해 단일 벡터/인덱스/chunk_id 샘플 점검 통과(200, ready=true).
- [ ] (통합) `/recommend` 응답이 `searched_doc_types`/`evidence[*].chunk_id`/`evidence[*].type`(doc_type 문자열)을 반환, `/search/candidates`가 `doc_type_coverage`/`matched_doc_types` 반환.

---

## 7. 검증 방법

**단위테스트 (로컬, VPN 불필요):**
```
pytest tests/test_cross_encoder_evidence_selector.py -v
pytest tests/ -q
```
- 신규 작성: `tests/test_retriever_aggregation.py`(chunk_cap/equal RRF/dedupe/정렬), `tests/test_filters_recency_or.py`(OR min_should), `tests/test_reasoner_chunk_id.py`(chunk_id 검증/fallback).

**정적 검증 (로컬):**
```
rg -n "BRANCHES|BRANCH_WEIGHTS|PATH_WEIGHTS|_infer_point_branch|_vector_e5i|_vector_splade|branch_presence_flags|searched_branches" apps/
rg -n "paper:|project:|patent:" apps/recommendation/
```
(추출 결과 0건이어야 함.)

**서버 기동 + health (통합, WO-B 컬렉션 + VPN 필요):**
```
$env:NTIS_QDRANT_COLLECTION_NAME = "ntis_researcher_chunks"
uvicorn apps.api.main:app --port 8011 --reload
```
- `GET http://localhost:8011/health` → `searched_doc_types` 포함 확인.
- `GET http://localhost:8011/health/ready` → ready=true, `sample_point_id`가 chunk_id 형태.

**Live 계약 검증 (통합, VPN 필요):**
```
ntis-validate-live
```
(종료 코드 0 = readiness 통과. 단일 `dense_e5i`/`sparse_splade` 존재·v2.0 인덱스·chunk 샘플 구조 점검.)

**수동 스모크 (통합):**
- `POST /search/candidates`로 `doc_type_coverage`/`matched_doc_types`/`counts`(신규 명칭) 확인.
- `POST /recommend`로 `evidence[*].chunk_id` + `evidence[*].type`(doc_type 문자열) + `searched_doc_types` 확인.

**Golden 스모크:** GOLDEN_TESTS.md 시나리오 중 chunk/집계/OR recency/chunk_id evidence 관련(7~16)을 로컬 가능 범위에서 점검(전수 통과 판정은 **WO-D**).

---

## 8. 리스크 & 가드레일

**위반 시 기준선 위반인 HARD 제약 (본 WO가 코드로 강제):**
1. **equal RRF only** — Qdrant 융합은 `FusionQuery(fusion=Fusion.RRF)` 고정(retriever.py:310 유지). 집계 누적도 equal(`1/(rank+60)`). doc_type 중요도는 `doc_type_priors`(옵트인, 기본 equal=1.0)로만. 가중 RRF/score 가중합 금지. (ADR 0005 §1·§2)
2. **후보 cross-encoder 리랭커 기본 OFF** — `candidate_reranker="off"` 기본. band 옵트인 시에도 score 동률 밴드 내 재배열만, 탈락/생성 금지. 후보 순위 척추 = RRF 고정. (ADR 0005 §3)
3. **LLM no-rerank** — reasoner는 이유 + `selected_evidence_ids`(chunk_id)만. 후보 재정렬/탈락/새 ID 생성 금지. service.py가 검색 순서를 유지. (SERVICE_FLOW §4)
4. **OR recency** — 다중 doc_type recency는 AND 아니라 `min_should(min_count=1)`. filters.py:172-180 OR 가드 **보존**(과거 0건 장애 교훈). (DATA_MODEL §3.2)
5. **evidence 리랭커 = grounding 선별만** — 후보 순위 영향 0. evidence score를 후보 score에 절대 합산 금지. (ADR 0005 §6.4)
6. **chunk_id evidence** — 모든 evidence id는 `chunk_id`(WO-0 코덱). 위치 기반 `paper:N` 폐기. (REASONER_RUNTIME_POLICY)
7. **researcher_meta 비정규화 일관성** — 본 WO는 소비측. 적재 lockstep/불변식은 WO-A; 코드는 한 연구자의 chunk에서 meta 불일치를 가정하지 않되 readiness(C8)가 표본 점검.

**회귀 위험 & 회피책:**
- **1차 keyword stage 0건 회귀**: planner shim(:272 core_keywords) + role/action 차감(:253-266)을 무분별 제거하면 d57823d 사고 재발. C5에서 보존, empty-keyword 단위테스트 유지.
- **recency AND 0건 회귀**: filters OR 가드 제거/우회 금지. C4-4 + 단위테스트.
- **search/search_weighted 통합 누락**: weighted 경로 제거 시 `/search/candidates` 핸들러(main.py:476 `search_weighted_candidates`)와 service.py:107-128가 끊긴다 — 단일 `search`로 재배선하고 main.py가 끊긴 호출을 남기지 않게 grep 검증.
- **WO-B 인덱스 불일치**: C2 `PAYLOAD_INDEX_FIELDS`와 WO-B 부트스트랩이 어긋나면 readiness 무한 503. WO-0/WO-B와 인덱스 키를 단일 소스로 공유.
- **dead code(reasoner.py:176-177)**: 제거 시 `PassThroughReasonGenerator` fallback 경로(service.py 우회)가 깨지지 않는지 fallback 단위테스트로 보장.

---

## 9. 작업 분할 & 예상 규모

> WO-C는 최대 규모 — 하위 단계로 분할하고 병렬/순서를 명시. 권장 순서: C1 → (C2 ∥ C4 ∥ C6) → C3 → (C5 ∥ C7) → C8 → 통합.

| 단계 | 의존 | 병렬성 | rough 난이도 |
|---|---|---|---|
| **C1 config** | WO-0(설정 키 명명 합의) | 단독 선행 | 하 — 값/이름 정의, 모든 후속이 import |
| **C2 schema_registry** | C1, WO-0(doc_type/family 상수) | C4·C6과 병렬 | 중 — 벡터맵/인덱스 교체, 참조처 다수 |
| **C4 filters** | C1, WO-0(허용 키) | C2·C6과 병렬 | 중 — OR 가드 보존이 핵심 |
| **C6 evidence+reasoner** | C1, WO-0(chunk_id 코덱) | C2·C4와 병렬 | 상 — CrossEncoderEvidenceSelector 신규, 테스트 계약 충족 |
| **C3 retriever** ★ | C1·C2·C4 | 직렬(가장 큼) | 상(최상) — search 통합 + 집계 신설 |
| **C5 planner** | C1 | C7과 병렬 | 하~중 — 프롬프트/키 + shim 보존 |
| **C7 cards/schemas/models/main/service** | C2·C3·C6 | C5와 병렬 | 상 — BREAKING 리네임 광범위 |
| **C8 live_validator/validate_live** | C2·C7 | 후행 | 중 — readiness 체크 전환 |
| **통합 검증** | C1~C8 + WO-A·WO-B | 최후행 | 중 — VPN/컬렉션 필요 |

- **rough 규모:** C3 + C7이 코드량의 절반 이상. C3는 단독 PR 권장(집계 로직 리뷰 집중). C6는 테스트가 계약을 못박아 비교적 안전.
- **권장 커밋 분할:** ①C1+C2(설정/스키마 골격) ②C6(evidence/reasoner, 테스트 green) ③C3(검색 통합+집계) ④C4(필터) ⑤C5+C7(planner+응답 계약, CHANGELOG 동봉) ⑥C8(readiness). 각 커밋 단위로 `pytest` green 유지.

---

## 10. 산출물 (Deliverables)

- 위 표의 14개 `apps/` 파일 수정 + `models.py` chunk 모델 신규.
- `CrossEncoderEvidenceSelector` 구현(lexical 강등 포함) — `tests/test_cross_encoder_evidence_selector.py` green.
- 신규 단위테스트 3종(retriever 집계 / filters OR recency / reasoner chunk_id).
- 정적 검증 통과 로그(브랜치/위치 evidence id 0건 grep 결과).
- EXTERNAL_API_CHANGELOG.md v2.0 항목 적용 표기(코드 커밋과 동일 커밋).
- 통합 검증 증빙(WO-B 컬렉션 가용 시): `/health/ready` 200 + `ntis-validate-live` exit 0 캡처.
- WO-D 인계 메모: 단위 green 범위 / 통합 미검 항목(WO-A·WO-B 의존) / 옵트인 레버(`NTIS_DOC_TYPE_PRIORS`/`NTIS_CANDIDATE_RERANKER`/`NTIS_EVIDENCE_FAMILY_CAP`) 기본값 정리.
