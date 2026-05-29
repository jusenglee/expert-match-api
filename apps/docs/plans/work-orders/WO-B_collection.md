# WO-B · Qdrant 컬렉션 생성 & 인덱스

> 본 WO는 공통 마이그레이션 컨텍스트(정체성·HARD 제약·WO 의존성 그래프)를 전제로 한다. 컨텍스트 본문을 여기 반복하지 않고 참조만 한다. MIGRATION_PLAN의 **Phase B (B-1 ~ B-4)** 에 해당한다.

## 1. 개요

- **목적:** v2.0 chunk 모델용 신규 Qdrant 컬렉션 `ntis_researcher_chunks`를, 단일 `dense_e5i`(1024d, Cosine) + 단일 `sparse_splade` 벡터와 v2.0 payload 인덱스 세트로 생성하고, sparse modifier 정합·표본 upsert 스모크까지 검증한다.
- **한 줄 요약:** `apps/search/qdrant_bootstrap.py`의 4×dense + 4×sparse named-vector 생성 로직을 단일 dense + 단일 sparse로 재작성하되, **기존 `researcher_recommend_proto`는 절대 건드리지 않는 blue/green 분기**를 도입한다.
- **완료 시 달성 상태:**
  - `ntis_researcher_chunks` 컬렉션이 `dense_e5i`(1024, Cosine) + `sparse_splade`(modifier는 활성 sparse backend에 정합) 구성으로 존재.
  - `DATA_MODEL.md §5`의 payload 인덱스(keyword/integer/datetime)가 모두 생성됨.
  - 표본 chunk Point(ID=`chunk_id`) upsert 후 `query_points`로 검색·doc_type 필터·집계 동작이 확인됨(B-4).
  - 구 컬렉션 `researcher_recommend_proto`는 변형/삭제 없이 그대로 보존됨.
  - `live_validator.py` / `qdrant_bootstrap.py`가 단일 벡터 스키마를 기준으로 일관 동작(상세 readiness 점검 로직 정합은 WO-C 본구현, 본 WO에서는 컬렉션 측 정합만 책임).

## 2. 선행조건 & 의존성

- **선행 WO:**
  - **WO-0 (안정화 & 계약 고정)** — `doc_type` 11종 enum / 4 family 매핑 / payload 인덱스 필드 상수, `chunk_id` 코덱(형식 `PREFIX_researcher_id_seq_c#`)이 확정되어 있어야 한다. 본 WO의 인덱스 필드명·`doc_type` keyword 필터 키는 **WO-0이 고정한 상수를 그대로 사용**한다(자체 정의 금지).
  - **WO-A (데이터 적재, chunk payload 형식)** 와 정합 필수 — 본 WO가 만드는 payload 인덱스 키 경로(예: `researcher_meta.publication_count`, `domain_attrs.journal_class`)는 WO-A가 적재하는 chunk payload의 실제 키 경로와 1:1 일치해야 한다. 불일치 시 인덱스가 빈 필드를 가리켜 필터가 무효화된다.
- **VPN:** **필요.** Qdrant 서버(`NTIS_QDRANT_URL` 기본 `http://203.250.234.159:8005`)와 임베딩 서버(B-4 표본 임베딩 생성 시) 접근이 VPN 경유로만 가능. B-1~B-4 전 단계가 VPN 구간이다.
- **환경:** Python 3.12+, `python -m pip install -e .[dev]`(RUNBOOK §1). sparse backend resolver(`resolve_sparse_runtime`)가 로컬 `models/PIXIE-Splade-v1.0`를 로드 가능해야 modifier=none 경로가 검증된다(미존재 시 bm25 fallback=IDF 경로로 검증됨 — 둘 다 정합 케이스).
- **차단 요소:** WO-0 미완 시 인덱스 필드 상수가 유동적이라 착수 불가. WO-A의 payload 키 경로 초안이 없으면 §4-(2) 인덱스 키를 확정할 수 없다(초안 확정 전까지는 `DATA_MODEL.md §5` 기준으로 진행하되 WO-A 합의 후 재확인).

## 3. 범위

### In Scope
- `apps/search/qdrant_bootstrap.py::QdrantBootstrapper.ensure_collection`(현 `:54-103`) 재작성: 단일 `dense_e5i` + 단일 `sparse_splade` 생성.
- blue/green 안전 분기 도입: 컬렉션명 기반으로 v1.x(4브랜치) / v2.0(단일) 스키마를 분기하는 "capability flag".
- v2.0 payload 인덱스 세트 정의 및 생성(`ensure_payload_indexes`, 현 `:189-201`).
- sparse modifier 정합: 단일 `sparse_splade`에 SPLADE=none / bm25 fallback=IDF 적용 및 복구(`ensure_sparse_vector_modifiers`, 현 `:125-187`).
- B-4 표본 upsert + `query_points` 스모크 절차 정의.
- modifier 불일치 시 PATCH vs drop&recreate 정책 open question 정리 + 권고.

### Out of Scope
- `schema_registry.py`의 상수 자체(`DENSE/SPARSE_VECTOR_BY_BRANCH` → 단일 벡터, family 매핑) 변경 → **WO-C**(이 WO는 그 결과 상수를 소비). 단, 본 WO 진행을 위해 임시 단일-벡터 상수가 필요하면 WO-C 골격과 합의해 최소 정의.
- retriever의 `query_points` 호출/RRF 융합/연구자 집계 로직 → **WO-C**.
- `live_validator.py` / `validate_live.py`의 BRANCHES 루프(현 `:239,246,253`)를 단일 벡터 기준으로 바꾸는 코드 수정 → **WO-C**(본 WO는 컬렉션이 검증 통과 가능한 상태인지만 확인).
- 실데이터 대량 적재 → **WO-A / WO-D**.
- 환경변수 컷오버(`NTIS_QDRANT_COLLECTION_NAME` 프로덕션 전환) → **WO-D**.

## 4. 상세 작업 항목

### B-1. `ensure_collection` 단일 벡터 재작성 + blue/green 분기

1. **[`apps/search/qdrant_bootstrap.py:82-99` `ensure_collection` create_collection 블록]** — 현재 `vectors_config`는 `BRANCHES`(`schema_registry.py:19-24`)를 순회하며 `basic/art/pat/pjt_vector_e5i` 4개 dense를 만들고, `sparse_vectors_config`도 `basic/art/pat/pjt_vector_splade` 4개 sparse를 만든다. 이를 **단일 벡터**로 교체:
   - `vectors_config = {"dense_e5i": models.VectorParams(size=self.settings.embedding_vector_size, distance=models.Distance.COSINE)}` — size는 `config.py:114 embedding_vector_size`(기본 1024), distance는 현 `:87`과 동일 Cosine 유지.
   - `sparse_vectors_config = {"sparse_splade": models.SparseVectorParams(modifier=sparse_modifier)}` — `sparse_modifier`는 현 `:71-72` 로직(`_requires_idf_modifier()` → IDF or None) 그대로 재사용.
   - 벡터명 문자열 `"dense_e5i"`/`"sparse_splade"`는 **하드코딩 금지**, WO-C가 `schema_registry.py`에 정의할 단일 벡터 상수(신규: 예 `DENSE_VECTOR_NAME = "dense_e5i"`, `SPARSE_VECTOR_NAME = "sparse_splade"`)를 import해 참조. 근거: `DATA_MODEL.md §2` 표 / `ENVIRONMENT.md:30`.
2. **[`apps/search/qdrant_bootstrap.py:54` `ensure_collection` 진입부 — 신규 capability flag]** — HARD: 구 컬렉션 무변형(MIGRATION_PLAN §0 blue/green). **컬렉션명으로 스키마 모드를 분기**하는 헬퍼 신규 추가:
   - `_is_v2_collection(name) -> bool` 등 신규 메서드. 판정 기준은 컬렉션명이 v2.0 컬렉션(`ntis_researcher_chunks` 또는 그 override)일 때만 단일-벡터 스키마를 적용. 구 `researcher_recommend_proto`로 호출되면 v1.x 4브랜치 스키마 경로(기존 코드)로 가도록 보존.
   - 가장 단순·안전한 구현: `ensure_collection`이 v2.0 모드에서만 단일-벡터 `create_collection`을 수행하고, 구 컬렉션은 **이 부트스트래퍼로 절대 재생성하지 않는다**(운영 가드: `recreate=True`라도 컬렉션명이 `researcher_recommend_proto`면 거부/로그 후 no-op). 근거: HARD "기존 researcher_recommend_proto를 절대 변형/삭제하지 않도록" + MIGRATION_PLAN §5 롤백(환경변수만 되돌리면 v1.x 복귀).
3. **[`apps/search/qdrant_bootstrap.py:60-66` recreate 가드]** — `recreate=True` 시 `delete_collection`(현 `:62`)이 구 컬렉션을 지우지 않도록, 위 capability flag와 결합해 **v2.0 컬렉션에 대해서만 삭제 허용**. 시드 경로(`NTIS_SEED_ALLOW_RECREATE_COLLECTION`, ENVIRONMENT.md:106)와의 상호작용도 v2.0 컬렉션 한정으로 명시.

### B-2. payload 인덱스 세트 (v2.0)

4. **[`apps/search/qdrant_bootstrap.py:189-201` `ensure_payload_indexes` + WO-C가 옮길 인덱스 상수]** — 현 인덱스는 `schema_registry.py:79-98 PAYLOAD_INDEX_FIELDS`(nested `publications[].*`, `intellectual_properties[].*`, `research_projects[].*` 경로 + root `basic_info.*`/`researcher_profile.*`)에 묶여 있다. v2.0은 nested 배열이 없으므로(`DATA_MODEL.md §0,§3`) **전면 교체**. `FIELD_SCHEMA_MAP`(현 `:24-28`, keyword/integer/datetime)는 그대로 재사용 가능. v2.0 인덱스 세트(`DATA_MODEL.md §5` 확정):
   - **keyword:** `researcher_id`, `doc_type`, `tags`, `researcher_meta.affiliated_organization`, `researcher_meta.highest_degree`, `domain_attrs.journal_class`, `domain_attrs.ip_type`, `domain_attrs.application_country`, `domain_attrs.application_registration_type`, `domain_attrs.performing_organization`, `domain_attrs.managing_agency`, `domain_attrs.appointing_organization`, `domain_attrs.appointing_organization_type`, `domain_attrs.evaluation_agency_name`, `domain_attrs.assessment_type_class`, `domain_attrs.assessment_class`, `domain_attrs.tech_classification_system`, `domain_attrs.specific_specialty_name`.
   - **integer:** `event_year`, `researcher_meta.publication_count`, `researcher_meta.scie_publication_count`, `researcher_meta.intellectual_property_count`, `researcher_meta.research_project_count`, `researcher_meta.researcher_assessor_count`, `researcher_meta.expert_assessor_count`(이상 6 + 아래 2 = `researcher_meta.*_count` 8종 = 추가로 `researcher_meta.researcher_tech_count`/유사 2종은 WO-0이 확정한 "8 count" 정의에 맞춤 — DATA_MODEL §3.1은 6종만 표기하나 컨텍스트는 "8 count"이므로 **WO-0의 8종 확정 목록을 단일 소스로 사용**), `domain_attrs.tech_rank`, `domain_attrs.specialty_count`.
   - **datetime:** `event_date` (필요 시 도메인 세부 일자 `application_date`/`registration_date`/`project_start_date` 등은 선택, `DATA_MODEL.md §5` 주석대로 운영 필요 시만).
   - `chunk_text`는 text 인덱스 **선택**(운영 full-text 매칭 필요 시만; 기본 생성 안 함).
   - **근거:** `DATA_MODEL.md §5`, `RUNBOOK.md:21,47`. **HARD-4 보존 함의:** `event_year`(integer range)와 `doc_type`(keyword)는 OR(min_should) recency 필터가 의존하는 인덱스이므로 누락 시 recency 0건 회귀 위험 → 필수.
5. **현재 인덱스 대비 추가/변경/삭제 표 (감사용 산출물):**

   | 현 v1.x 인덱스 (`schema_registry.py:79-98`) | v2.0 처리 |
   |---|---|
   | `basic_info.researcher_id` (keyword) | → `researcher_id` (keyword) **경로 변경** |
   | `basic_info.affiliated_organization_exact` (keyword) | → `researcher_meta.affiliated_organization` (keyword) **경로 변경** |
   | `researcher_profile.highest_degree` (keyword) | → `researcher_meta.highest_degree` (keyword) **경로 변경** |
   | `researcher_profile.publication_count` (integer) | → `researcher_meta.publication_count` (integer) **경로 변경** |
   | `researcher_profile.scie_publication_count` (integer) | → `researcher_meta.scie_publication_count` |
   | `researcher_profile.intellectual_property_count` (integer) | → `researcher_meta.intellectual_property_count` |
   | `researcher_profile.research_project_count` (integer) | → `researcher_meta.research_project_count` |
   | `publications[].journal_index_type` (keyword) | → `domain_attrs.journal_class`/`indexing_class` (keyword) **nested 제거** |
   | `publications[].publication_year_month` (datetime) | → `event_date`(datetime)/`event_year`(integer)로 통합 **삭제** |
   | `intellectual_properties[].application_registration_type` (keyword) | → `domain_attrs.application_registration_type` **nested 제거** |
   | `intellectual_properties[].application_country` (keyword) | → `domain_attrs.application_country` |
   | `intellectual_properties[].application_date` (datetime) | → `event_date` 통합(선택 잔존) |
   | `intellectual_properties[].registration_date` (datetime) | → `event_date` 통합(선택 잔존) |
   | `research_projects[].project_start_date` (datetime) | → `event_date` 통합(선택 잔존) |
   | `research_projects[].project_end_date` (datetime) | → 통합/삭제 |
   | `research_projects[].reference_year` (integer) | → `event_year` 통합 **삭제** |
   | `research_projects[].performing_organization` (keyword) | → `domain_attrs.performing_organization` |
   | `research_projects[].managing_agency` (keyword) | → `domain_attrs.managing_agency` |
   | (없음) | **신규:** `doc_type`, `tags`, `event_year`, `researcher_meta.researcher_assessor_count`, `researcher_meta.expert_assessor_count`, `domain_attrs.ip_type`, `domain_attrs.appointing_organization(_type)`, `domain_attrs.evaluation_agency_name`, `domain_attrs.assessment_type_class`, `domain_attrs.assessment_class`, `domain_attrs.tech_classification_system`, `domain_attrs.tech_rank`, `domain_attrs.specific_specialty_name`, `domain_attrs.specialty_count` |

### B-3. sparse modifier 정합 (단일 `sparse_splade`)

6. **[`apps/search/qdrant_bootstrap.py:125-187` `ensure_sparse_vector_modifiers`]** — 현재 `BRANCHES` 순회(현 `:155`)로 `basic/art/pat/pjt_vector_splade` 4개 modifier를 점검/복구한다. v2.0에서는 **단일 `sparse_splade`만 점검**하도록 루프 제거. `_requires_idf_modifier()`(현 `:49-52`)와 `_modifier_is_idf()`(현 `:105-123`)는 그대로 재사용.
   - 정합 규칙(`sparse_runtime.py:199-200 model_requires_idf_modifier`, `resolve_sparse_runtime` 반환 `requires_idf_modifier`): SPLADE 계열(`custom_splade`, `sparse_runtime.py:99` `requires_idf_modifier=False`) → modifier **없음(None)**; `Qdrant/bm25` fallback(`sparse_runtime.py:174` `requires_idf_modifier=True`) → modifier **IDF**. 근거: `ENVIRONMENT.md:65`, `RUNBOOK.md:41,58`, `DATA_MODEL.md §2` 표.
   - `QdrantBootstrapper`가 `sparse_runtime`(생성자 `:42-47`)를 받았으면 그 값을, 아니면 `model_requires_idf_modifier(settings.sparse_model_name)`(`config.py:117` 기본 `models/PIXIE-Splade-v1.0` → 이름에 "splade" 포함 → False)을 사용 — 기존 우선순위 로직 보존.
   - `update_collection`(현 `:172-175`)의 `sparse_vectors_config`는 단일 `{"sparse_splade": SparseVectorParams(modifier=...)}`로 축소.

### B-4. 표본 upsert + `query_points` 스모크 (VPN 경유)

7. **표본 chunk upsert (신규 절차/스크립트):**
   - VPN 연결 후, WO-A의 chunk payload 형식을 따르는 **소수 표본 chunk**(doc_type별 1~2개씩, 최소 `profile` + `publication` + `research_project` + `researcher_assessor` 포함)를 준비. Point ID = WO-0 코덱이 생성한 `chunk_id` 문자열 그대로(`DATA_MODEL.md §1.1,§6-1`).
   - dense 임베딩은 임베딩 서버(`NTIS_EMBEDDING_*`, VPN 필요), sparse는 `resolve_sparse_runtime`로 선택된 backend로 `chunk_text` 단일 입력에서 생성(`DATA_MODEL.md §2,§6-6`).
   - `client.upsert`로 `dense_e5i`/`sparse_splade` 두 벡터를 함께 적재. **동일 chunk_id 재upsert 시 동일 Point 덮어쓰기(멱등) 확인**(`DATA_MODEL.md §1.1,§6-1`).
8. **`query_points` 스모크 (retriever 본구현 전 최소 확인):**
   - `client.query_points(collection_name=..., using="dense_e5i", query=<표본 dense>, limit=...)` 로 dense 검색 hit 확인.
   - `using="sparse_splade"` sparse 검색 hit 확인.
   - `doc_type` keyword 필터(`models.FieldCondition(key="doc_type", match=...)`)로 family별 분기가 동작하는지 확인 → `DATA_MODEL.md §2` "doc_type은 payload 필터" 정합.
   - **HARD-4 회귀 가드 사전 확인:** `event_year >= (올해-N)` range 조건을 둘 이상 doc_type에 대해 OR(`min_should`, `min_count=1`)로 조립해 0건이 아닌지 확인(`filters.py:169-175` 가드의 컬렉션-측 전제). 본 WO는 OR 필터가 인덱스 상에서 동작 가능함만 확인하고, 조립 코드는 WO-C.
   - **RRF 융합 동작은 WO-C 책임**이나, 본 WO 스모크에서 `models.FusionQuery(fusion=models.Fusion.RRF)`(retriever.py:310에서 사용)가 단일 컬렉션 + prefetch(dense/sparse)로 정상 응답하는지 가벼운 확인 권장.

### B-5. modifier 불일치 PATCH vs drop&recreate (Open Question + 권고)

9. **Open Question:** 컬렉션이 이미 잘못된 modifier(예: SPLADE backend인데 IDF로 생성됨)로 존재할 때, Qdrant `update_collection`(현 `:172`)의 `sparse_vectors_config` PATCH로 modifier를 실시간 교정 가능한가, 아니면 drop&recreate가 필요한가?
   - **현 코드 동작:** `ensure_sparse_vector_modifiers`는 PATCH(`update_collection`)를 시도하고, 실패 시 경고만 남기고 통과(현 `:182-187`). 즉 코드는 PATCH 낙관 경로.
   - **권고:** (a) **신규 v2.0 컬렉션은 처음부터 올바른 modifier로 생성**(B-1에서 `sparse_modifier` 결정)하여 PATCH 의존을 회피한다. (b) 운영 중 backend가 바뀌어(예: 로컬 PIXIE 깨져 bm25 fallback) modifier 기대값이 바뀌면, **표본 단계에서는 PATCH 시도 → readiness가 `sparse_vectors_idf` 불일치로 잡힘**(`live_validator.py:251-263`)이므로 PATCH 실패가 확인되면 **drop&recreate**(데이터 재적재 동반)로 처리. blue/green이므로 신규 컬렉션 재생성은 구 컬렉션에 무영향(MIGRATION_PLAN §5). (c) PATCH 가능 여부는 환경 Qdrant 버전에 의존하므로 **B-4 스모크에서 실측 1회 기록**하여 결론을 RUNBOOK에 남길 것을 권고.

## 5. 변경/생성 대상 파일

| 파일 | 변경유형 | 요지 |
|---|---|---|
| `apps/search/qdrant_bootstrap.py` | 수정 | `ensure_collection`(`:82-99`) 단일 `dense_e5i`+`sparse_splade` 생성으로 재작성; capability flag(`_is_v2_collection`, 신규) 분기; `ensure_sparse_vector_modifiers`(`:125-187`)·`ensure_payload_indexes`(`:189-201`)를 단일 벡터/v2.0 인덱스 세트로 축소; recreate를 v2.0 컬렉션 한정으로 가드 |
| `apps/search/schema_registry.py` | 수정(WO-C 주관, 본 WO 합의) | 단일 벡터 상수(`DENSE_VECTOR_NAME`/`SPARSE_VECTOR_NAME`, 신규) + v2.0 `PAYLOAD_INDEX_FIELDS` 재정의. 본 WO는 이 상수를 소비 |
| `apps/tools/`(신규 스크립트 또는 RUNBOOK 절차) | 신규 | B-4 표본 upsert + `query_points` 스모크 스크립트/명령 절차(VPN 경유) |
| `apps/docs/operation/RUNBOOK.md` | 수정 | B-5 PATCH vs drop&recreate 실측 결론 1줄 추가(§2 영역) |
| `apps/search/live_validator.py` | 비변경(WO-C에서 수정) | 본 WO에서는 코드 미수정. 컬렉션이 이 검증기를 통과 가능한 구조인지만 확인 |
| `researcher_recommend_proto` (구 컬렉션) | **무변경(보존)** | 절대 삭제/변형 금지 |

## 6. 수용 기준 (Acceptance Criteria)

- [ ] `ntis_researcher_chunks` 컬렉션이 정확히 `dense_e5i`(size=`embedding_vector_size`=1024, distance=Cosine) 1개 + `sparse_splade` 1개만 갖는다(4브랜치 벡터 부재).
- [ ] 활성 sparse backend가 SPLADE면 `sparse_splade` modifier가 None, `Qdrant/bm25` fallback이면 IDF다(`_modifier_is_idf` 기준).
- [ ] `DATA_MODEL.md §5` keyword/integer/datetime 인덱스가 모두 생성됨 — 특히 `researcher_id`, `doc_type`, `tags`, `event_year`, `researcher_meta.*_count` 8종이 존재(`get_collection().payload_schema` 키로 확인).
- [ ] Point ID=`chunk_id` 문자열로 표본 upsert 성공, 동일 chunk_id 재upsert가 Point 수를 늘리지 않음(멱등).
- [ ] `query_points`로 dense / sparse 각각 검색 hit ≥1, `doc_type` 필터로 family 분기 동작, 둘 이상 doc_type `event_year` range OR(min_should) 질의가 0건이 아님.
- [ ] `recreate=True`를 구 컬렉션명으로 호출해도 `researcher_recommend_proto`가 삭제되지 않음(가드 동작) — 구 컬렉션 Point 수 불변 확인.
- [ ] B-5 PATCH/drop&recreate 실측 결과가 RUNBOOK에 1줄로 기록됨.

## 7. 검증 방법

- **단위/부트스트랩(로컬, in-memory Qdrant 가능 구간):**
  - `python -m pytest tests/ -k "bootstrap or schema_registry"` — 단일 벡터 생성·인덱스 세트·modifier 결정 로직(가능하면 `QdrantClient(":memory:")` 또는 모킹).
  - 전체 회귀: `python -m pytest -q` (WO-0이 CI green 보장 — 본 WO가 red를 추가하지 않을 것).
- **스모크(VPN 필요, 실 Qdrant):**
  1. VPN 연결 확인.
  2. `ensure_collection(recreate=False)` 호출(또는 부트스트랩 진입점) → 컬렉션 생성.
  3. `python -c "from qdrant_client import QdrantClient; c=QdrantClient(url=...); print(c.get_collection('ntis_researcher_chunks'))"` 로 vectors/sparse_vectors/payload_schema 확인.
  4. B-4 표본 upsert 스크립트 실행 → `query_points` dense/sparse/`doc_type` 필터/OR recency 확인.
- **Readiness(VPN, 앱 기동):**
  - `NTIS_QDRANT_COLLECTION_NAME=ntis_researcher_chunks ntis-validate-live` (CLI, RUNBOOK §3) — sparse modifier 기대값·인덱스 존재·샘플 구조 점검.
  - `uvicorn apps.api.main:app --host 0.0.0.0 --port 8011 --reload` 후 `GET /health` → `GET /health/ready`.
  - 주의: `live_validator.py`의 `dense_vectors_present`/`sparse_vectors_present`/`sparse_vectors_idf`/`payload_indexes_present` 체크(현 `:238-271`)는 **현재 `BRANCHES`/`PAYLOAD_INDEX_FIELDS` 기준**이라 v2.0 단일 벡터에서 false가 난다 → **WO-C 미완 시 ready:false가 정상**. 본 WO 단독 검증은 `get_collection` 직접 확인 + 스모크로 수행하고, ready 그린은 WO-C/WO-D에서 확정.
- **구 컬렉션 보존:** `c.count('researcher_recommend_proto')` 를 전/후 비교해 불변 확인.

## 8. 리스크 & 가드레일

- **HARD-1 (equal RRF):** 본 WO는 컬렉션 생성만 하므로 융합 가중을 도입하지 않는다. sparse modifier(IDF/none)는 **인코딩 정합**일 뿐 점수 가중이 아니다. doc_type 중요도는 절대 컬렉션/벡터 레벨에 넣지 않고 앱단 `NTIS_DOC_TYPE_PRIORS`(WO-C)로만 표현.
- **HARD-4 (OR recency 보존):** `event_year`(integer)·`doc_type`(keyword) 인덱스 누락 금지 — `filters.py:169-175`의 `min_should`(`min_count=1`) OR 가드가 의존. AND 묶음 0건 장애 교훈. B-4에서 OR 질의 비-0건을 명시 확인.
- **HARD-6 (researcher_meta 비정규화 lockstep):** 본 WO는 `researcher_meta.*` 인덱스를 만든다 — 한 연구자의 모든 chunk에 동일 값이 들어 있어야 필터가 결정적(`DATA_MODEL.md §3.1 주의`, §6-2). 값 일관성 보장은 WO-A 적재 책임이나, B-4 표본도 동일 값으로 넣어 인덱스가 의도대로 동작함을 보인다.
- **`chunk_id` Point ID (계약):** Point ID는 WO-0 코덱의 `chunk_id` 문자열 그대로 — UUID 변환·정수 ID 금지(멱등성·evidence 참조 안정성, `DATA_MODEL.md §1.1`).
- **blue/green 무변형:** 구 `researcher_recommend_proto` 삭제/스키마 변경 절대 금지(MIGRATION_PLAN §0,§5). `recreate`/시드 재생성은 v2.0 컬렉션 한정 가드 필수.
- **WO-A 정합 회귀:** 인덱스 키 경로가 적재 payload 키와 어긋나면 인덱스가 무효(빈 필드 가리킴) → 필터가 조용히 0건/전수 통과. 착수 전 WO-A payload 키 경로와 대조하고, B-4 표본으로 실측.
- **참조 무영향(HARD-2/3/5):** 본 WO는 후보 순위(RRF 척추)·후보 리랭커·LLM·evidence 리랭커를 건드리지 않는다 — 컬렉션 스키마 작업이 이들 제약에 영향 0임을 명시.

## 9. 작업 분할 & 예상 규모

- **순서(권장):** B-1(단일 벡터 + capability flag) → B-2(인덱스 세트) → B-3(modifier 정합) → [VPN] B-4(스모크) → B-5(open question 실측·기록).
- **병렬성:** B-1·B-2·B-3은 같은 파일(`qdrant_bootstrap.py`)을 만지므로 직렬 권장. B-4 스모크 스크립트 작성은 B-1~B-3와 병렬 가능(VPN 실행만 뒤로). WO-A(payload 형식)·WO-C(상수)와는 인터페이스만 합의되면 독립 진행 가능.
- **난이도(rough):**
  - B-1: 중 (분기·가드 설계가 핵심, 코드량은 적음).
  - B-2: 하 (상수 교체 위주, 단 WO-A 키 정합 확인 필요).
  - B-3: 하 (루프 제거 + 단일 키).
  - B-4: 중 (VPN·임베딩·스모크 스크립트 신규).
  - B-5: 하 (문서·실측 1회).
- 전체: 소~중 규모. 가장 큰 리스크는 코드량이 아니라 **WO-A 키 경로 정합**과 **VPN 구간 실측**.

## 10. 산출물 (Deliverables)

1. 단일 벡터(`dense_e5i`+`sparse_splade`) + blue/green capability flag로 재작성된 `apps/search/qdrant_bootstrap.py`.
2. v2.0 payload 인덱스 세트(`DATA_MODEL.md §5` 정합) 및 §4-5의 **현행 대비 추가/변경/삭제 매핑 표**.
3. B-4 표본 upsert + `query_points` 스모크 스크립트/명령 절차(VPN 경유, doc_type 필터·OR recency·멱등 확인 포함).
4. sparse modifier 정합 확인 결과(SPLADE=none / bm25=IDF 실측 로그).
5. B-5 PATCH vs drop&recreate 권고 결론(RUNBOOK 1줄 반영).
6. 구 `researcher_recommend_proto` Point 수 전/후 불변 증빙(보존 확인).
7. 후속 핸드오프 메모: `live_validator.py`/`schema_registry.py`의 단일 벡터 정합 코드 수정은 **WO-C**, ready 그린·골든 통과는 **WO-D**에서 마감됨을 명시.
