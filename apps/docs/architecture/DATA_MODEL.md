# 데이터 모델 (Data Model) — chunk 기반 컬렉션

**기준선:** Qdrant 컬렉션 1개 / 1 chunk = 1 Point / 단일 dense + 단일 sparse named vector / payload 3층 구조
**문서 버전:** v2.0 (chunk 재설계)
**기준일:** 2026-05-28
**원천 규격:** [`../전체 샘플 payload (도메인별).txt`](../전체%20샘플%20payload%20(도메인별).txt) — 본 문서는 이 payload 규격을 시스템 관점에서 정규화·확정한다.

---

## 0. 한 줄 요약

이전 모델은 **"연구자 1명 = 1 Point"** + nested 배열(`publications[]` 등) + 4브랜치 named vector였다. v2.0부터는 **"chunk 1개 = 1 Point"** 로 전환한다. 한 연구자는 여러 doc_type에 걸쳐 다수의 chunk(=다수 Point)로 존재하고, **연구자 후보는 검색 시점에 `researcher_id`로 집계**해서 만든다.

> **왜 바꾸나 (2026-05-28 사용자 결정):** 적재 데이터가 `chunk_id`/`doc_id`/`doc_type`/`chunk_text` 단위로 도착한다. 데이터가 chunk 단위인데 저장을 연구자 단위로 압축하면 (1) 도메인별 표현이 한 벡터로 뭉개지고 (2) evidence가 chunk와 분리되어 grounding이 약해진다. chunk를 1급 단위로 두면 의미 검색·evidence 선별·근거 추적이 모두 자연스러워진다. 자세한 결정 근거는 [`ADR/0002-chunk-level-point-model.md`](ADR/0002-chunk-level-point-model.md).

---

## 1. 컬렉션 (Collection)

| 항목 | 값 | 비고 |
|---|---|---|
| 컬렉션 이름 | `ntis_researcher_chunks` (기본값) | `NTIS_QDRANT_COLLECTION_NAME`으로 override. 구 `researcher_recommend_proto`는 폐기 |
| Point 단위 | **chunk 1개 = Point 1개** | nested 배열 없음. 한 연구자 = 다수 Point |
| Point ID | `chunk_id` 문자열 그대로 (예: `PUB_M1006328_0001_c0`) | 결정론적 ID → upsert 멱등성 + evidence 참조 안정성 |
| Named vector | `dense_e5i` (dense) + `sparse_splade` (sparse) **각 1개** | doc_type별 named vector를 두지 않는다. doc_type은 payload 필터 |
| 거리 함수 | dense: Cosine | e5-instruct 권장 |

### 1.1 Point ID를 `chunk_id`로 고정하는 이유

- **멱등 적재:** 같은 chunk를 다시 적재해도 같은 Point를 덮어쓴다(중복 Point 방지). 재적재·증분 적재가 안전하다.
- **evidence 참조 안정성:** LLM이 고른 근거를 `chunk_id`로 그대로 참조한다. 과거 `paper:0` 같은 **배열 인덱스 기반 id가 사라진다** — 인덱스는 직렬화 순서가 바뀌면 깨지지만 `chunk_id`는 불변이다. ([`api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md) 참조)
- **운영 추적성:** trace/로그에 찍힌 id를 Qdrant에서 바로 조회할 수 있다.

---

## 2. 벡터 레이아웃 (Vector Layout)

chunk 1개당 벡터는 **dense 1 + sparse 1** 뿐이다. 입력 텍스트는 둘 다 `chunk_text`(도메인별로 이미 목적에 맞게 직렬화된 본문)이다.

| 벡터명 | 종류 | 모델 | 입력 | 비고 |
|---|---|---|---|---|
| `dense_e5i` | Dense (1024) | `multilingual-e5-large-instruct` | `chunk_text` | 쿼리 시 e5 instruct 프리픽스 사용. `NTIS_EMBEDDING_*` |
| `sparse_splade` | Sparse | `PIXIE-Splade-v1.0` (로컬 우선) | `chunk_text` | SPLADE면 modifier 없음 / `Qdrant/bm25` fallback이면 modifier `IDF`. `NTIS_SPARSE_*` |

> **doc_type별 named vector를 두지 않는 이유:** 한 Point는 정확히 하나의 doc_type에 속한다. doc_type을 named vector로 만들면 "한 Point가 한 벡터만 채우고 나머지는 비는" 낭비 구조가 된다. doc_type은 **payload 필터**로 분기하는 것이 chunk 모델에 정합적이다. "컬렉션 분리 효과"는 named vector가 아니라 `doc_type` 인덱스 + 쿼리별 필터로 얻는다.

> **임베딩 정제 원칙 유지:** `chunk_text`는 적재 시점에 순수 도메인 텍스트로 직렬화되어야 한다. "평가위원/추천" 같은 요청 어투(role/action 용어)는 쿼리 측에서도 임베딩 텍스트에 넣지 않는다(벡터 오염 방지). 이 원칙은 v1.x에서 그대로 계승된다.

---

## 3. payload 3층 구조

모든 doc_type이 공통으로 1·2층을 갖고, 3층(`domain_attrs`)만 doc_type별로 다르다.

### 3.1 1층 · 공통 식별 (모든 chunk 동일 형식)

| 필드 | 타입 | 인덱스 | 설명 |
|---|---|---|---|
| `researcher_id` | keyword | ✅ keyword | 연구자 식별자. **집계·dedupe·후보 단위의 기준 키** |
| `researcher_name` | keyword | — | 성명 (동점 정렬 보조) |
| `doc_type` | keyword | ✅ keyword | 11종 중 하나. 검색 분기·faceting의 핵심 |
| `doc_id` | keyword | — | 실적/문서 식별자 (예: `PUB_..._0001`) |
| `chunk_id` | keyword | (= Point ID) | chunk 식별자. evidence 참조 단위 |
| `chunk_text` | text | (선택) full-text | 임베딩 입력 원문. 운영 full-text 매칭이 필요하면 text index |
| `chunk_text_len` | integer | — | 본문 길이 (디버깅/품질 모니터링) |
| `researcher_meta` | object | 하위 필드 인덱스 | 연구자 집계 메타 (아래) |

**`researcher_meta` (모든 chunk에 비정규화 반복 저장):**

| 하위 필드 | 타입 | 인덱스 | 용도 |
|---|---|---|---|
| `affiliated_organization` | keyword | ✅ | 소속 — 제외 기관 필터 후보 |
| `highest_degree` | keyword | ✅ | 학위 hard filter |
| `publication_count` | integer | ✅ | 최소 논문 수 필터 |
| `scie_publication_count` | integer | ✅ | 최소 SCIE 수 필터 |
| `intellectual_property_count` | integer | ✅ | 최소 특허 수 필터 |
| `research_project_count` | integer | ✅ | 최소 과제 수 필터 |
| `researcher_assessor_count` | integer | ✅ | 평가위원 활동량 필터/신호 |
| `expert_assessor_count` | integer | ✅ | 전문 평가 활동량 필터/신호 |

> **`researcher_meta` 비정규화 주의:** 같은 연구자의 모든 chunk에 동일 값이 반복 저장된다. 이로써 **별도 profile 조인 없이 chunk 단위에서 연구자 hard filter**가 가능하다(예: `publication_count >= 5` AND `doc_type = research_project`). 대가로 **적재 시 값 일관성을 적재 파이프라인이 보장**해야 한다(한 연구자의 chunk마다 meta가 어긋나면 필터가 비결정적이 됨). [`operation/RUNBOOK.md`](../operation/RUNBOOK.md) 적재 검증 참조.

### 3.2 2층 · 도메인 통합 정규화 (모든 chunk 공통)

| 필드 | 타입 | 인덱스 | 설명 |
|---|---|---|---|
| `event_date` | datetime \| null | ✅ datetime | 도메인 통합 대표 일자(논문 게재/특허 등록/과제 시작/임명일 등). 시점이 없는 doc_type은 `null` |
| `event_year` | integer \| null | ✅ integer | `event_date`의 연도. **최근성(recency) 필터의 단일 기준** |
| `tags` | keyword[] | ✅ keyword | 소문자 정규화 키워드/전문분야. sparse 보강·faceting |

> **통합 recency의 의미:** v1.x는 `art_recent_years`/`pat_recent_years`/`pjt_recent_years`로 도메인마다 따로 날짜를 다뤘다. v2.0은 **모든 doc_type이 `event_year` 하나**를 공유하므로 "최근 N년" 조건이 `doc_type` 필터 + `event_year >= (올해-N)` 하나로 표현된다. 여러 doc_type에 대한 recency는 반드시 **OR(min_should)** 로 결합한다 — AND로 묶으면 0건이 된다(과거 v1.x 장애에서 확인된 교훈).

### 3.3 3층 · 도메인 고유 (`domain_attrs`, doc_type별)

아래는 원천 payload 규격에서 확정한 doc_type별 `domain_attrs`다. ✅는 필터/인덱스 권장 필드.

**publication**

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `title_primary` / `title_secondary` | str | — | 논문명(국/영) |
| `journal_name` | str | — | 학술지명 |
| `journal_class` / `indexing_class` | keyword | ✅ | SCIE 등 등재구분 |
| `publication_year_month` | str(YYYY-MM) | — | (`event_*`로 정규화됨) |
| `abstract` | str | — | 초록 |
| `keywords` | str[] | — | 원문 키워드(소문자판은 `tags`) |

**intellectual_property**

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `ip_type` | keyword | ✅ | 특허권/실용신안 등 |
| `ip_title` | str | — | 지식재산권명 |
| `application_registration_type` | keyword | ✅ | 출원/등록 |
| `application_country` | keyword | ✅ | 출원국 |
| `application_number` / `registration_number` | str | — | 번호 |
| `application_date` / `registration_date` | date | (선택) | 세부 일자(통합값은 `event_date`) |
| `ip_summary` | str | — | 요약 |
| `ip_foreign_class` | keyword | (선택) | 해외출원 구분 |

**research_project**

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `project_title_korean` / `project_title_english` | str | — | 과제명 |
| `performing_organization` | keyword | ✅ | 수행기관 — 제외 기관 필터 후보 |
| `managing_agency` | keyword | ✅ | 전문/관리기관 — 제외 기관 필터 후보 |
| `research_summary_korean` / `_english` | str \| null | — | 요약 |
| `research_period_start` / `_end` | date | (선택) | 수행기간(통합 시작값은 `event_date`) |

**researcher_assessor / expert_assessor** (평가위원·전문 평가 활동 — 평가위원 적합성의 직접 신호)

| 필드(researcher_assessor) | 타입 | 인덱스 |
|---|---|---|
| `appointing_organization` / `appointing_organization_type` | keyword | ✅ |
| `evaluation_committee_name` | str | (선택) |
| `appointment_period_type` | keyword | (선택) |
| `appointment_period_start` / `_end` / `appointment_date` | date | (선택) |

| 필드(expert_assessor) | 타입 | 인덱스 |
|---|---|---|
| `evaluation_agency_name` | keyword | ✅ |
| `assessment_type_class` / `assessment_class` | keyword | ✅ |
| `assessment_content` | str | — |
| `valid_period_start` / `_end` | date | (선택) |

**researcher_tech / expert_tech** (기술분류)

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `tech_classification_system` | keyword | ✅ | 분류체계명 |
| `tech_classification` | str | (선택) | `A > B > C` 경로 |
| `keywords` (tech) | str[] | — | researcher_tech만 |
| `tech_rank` | integer | (선택) | expert_tech만 |

**researcher_core / researcher_major / expert_specific** (전문분야 선언)

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `specialty_names` | str[] | — | core/major |
| `specialty_count` | integer | (선택) | core/major |
| `specific_specialty_name` | keyword | ✅ | expert_specific |
| `career_description` | str | — | expert_specific |

**profile** (신원 앵커)

| 필드 | 타입 | 인덱스 | 비고 |
|---|---|---|---|
| `researcher_number` | keyword | — | |
| `position_title` | keyword | (선택) | 직위 |
| `highest_degree` / `major_field` | keyword | ✅ | 학위/전공 |
| `publication_count` … `expert_assessor_count` | integer | ✅ | `researcher_meta`와 동일 집계 |

---

## 4. doc_type 분류 체계 (Taxonomy)

11개 doc_type을 검색·집계·evidence에서 일관되게 다루기 위해 **4개 기능군(family)** 으로 묶는다.

| family | doc_type | 추천에서의 역할 |
|---|---|---|
| **신원 (identity)** | `profile` | 소속·직위·학위·전공·누적 실적 앵커. 후보 카드의 머리 |
| **성과 (achievement)** | `publication`, `intellectual_property`, `research_project` | 주제 적합성의 핵심 근거(논문/특허/과제) |
| **평가이력 (assessment)** | `researcher_assessor`, `expert_assessor` | **평가위원 적합성의 직접 신호** — 과거 어떤 위원회/기관에서 무엇을 심사했는가 |
| **전문성 (expertise)** | `researcher_tech`, `expert_tech`, `researcher_core`, `researcher_major`, `expert_specific` | 기술분류·전문분야 선언으로 도메인 정합성 보강 |

> **v2.0의 새 신호 — 평가이력:** 구 4브랜치(basic/art/pat/pjt)에는 없던 `*_assessor` doc_type이 들어왔다. 이는 "평가위원 추천"이라는 과업에 가장 직접적인 신호다. 검색·evidence·reasoner는 평가이력 family를 1급 근거로 취급한다. (집계 가중 기본값은 equal이며, intent에 따른 prior 조정은 [`DESIGN_GUIDELINES.md §6`](DESIGN_GUIDELINES.md) 참조)

---

## 5. payload 인덱스 권장안 (요약)

| 인덱스 종류 | 대상 필드 |
|---|---|
| keyword | `researcher_id`, `doc_type`, `tags`, `researcher_meta.affiliated_organization`, `researcher_meta.highest_degree`, `domain_attrs.journal_class`, `domain_attrs.ip_type`, `domain_attrs.application_country`, `domain_attrs.performing_organization`, `domain_attrs.managing_agency`, `domain_attrs.appointing_organization`, `domain_attrs.evaluation_agency_name`, `domain_attrs.tech_classification_system`, `domain_attrs.specific_specialty_name` |
| integer | `event_year`, `researcher_meta.*_count` 8종, `domain_attrs.tech_rank`, `domain_attrs.specialty_count` |
| datetime | `event_date` (필요 시 도메인별 세부 일자) |
| text (선택) | `chunk_text` — 운영 full-text 매칭이 필요할 때만 |

원칙(v1.x 계승):
- 필터에 자주 쓰는 필드는 반드시 인덱스 대상.
- exact match가 필요한 기관명/학위/구분값은 `keyword`.
- 날짜·집계값은 임베딩에 녹이지 말고 payload 필터로 처리.

---

## 6. 적재 정합성 체크리스트 (Ingestion Invariants)

적재 파이프라인이 보장해야 하는 불변식. 위반 시 검색·필터가 비결정적이 된다.

1. **Point ID == `chunk_id`** 이고 컬렉션 전역에서 유일하다.
2. 한 `researcher_id`의 모든 chunk에서 `researcher_meta`/`researcher_name` 값이 **동일**하다.
3. `doc_type`은 정의된 11종 enum 중 하나다.
4. `event_year`는 `event_date`의 연도와 일치한다(둘 다 null 가능).
5. `tags`는 소문자·trim 정규화되어 있다.
6. dense/sparse 벡터는 동일 `chunk_text`에서 생성된다(불일치 금지).
7. `chunk_text`에는 요청 어투(role/action 불용어)가 포함되지 않는다.

> 적재 코드는 현재 repo 밖에 일부 존재한다. 운영 컬렉션에서 위 불변식을 표본 점검하는 절차는 [`operation/RUNBOOK.md`](../operation/RUNBOOK.md)와 [`operation/GOLDEN_TESTS.md`](../operation/GOLDEN_TESTS.md)에 둔다.

---

## 7. v1.x → v2.0 매핑 (요약)

| v1.x (연구자=1 Point) | v2.0 (chunk=1 Point) |
|---|---|
| `basic` 브랜치 / `basic_vector_e5i` | `profile`(+`researcher_*`/`expert_*` 전문성 doc_type), 단일 `dense_e5i` |
| `art` 브랜치 / `art_vector_*` | `publication` doc_type |
| `pat` 브랜치 / `pat_vector_*` | `intellectual_property` doc_type |
| `pjt` 브랜치 / `pjt_vector_*` | `research_project` doc_type |
| (없음) | `researcher_assessor`, `expert_assessor`, `*_tech`, `researcher_core/major`, `expert_specific` |
| nested `publications[]` 등 | 개별 chunk Point들 |
| `basic_info.researcher_id` (root) | `researcher_id` (L1, 모든 chunk) |
| evidence id `paper:0` | evidence id = `chunk_id` |

전체 단계별 전환 계획은 [`../plans/MIGRATION_PLAN.md`](../plans/MIGRATION_PLAN.md).
