# 데이터 모델 (Data Model) — flat chunk 컬렉션

**기준선:** Qdrant 컬렉션 1개 / 1 chunk = 1 Point / 단일 dense + 단일 sparse named vector / **FLAT payload**
**문서 버전:** v2.1 (flat payload 정준 계약)
**기준일:** 2026-06-02
**원천 규격:** [`../전체 샘플 payload (도메인별).txt`](../전체%20샘플%20payload%20(도메인별).txt) — 실제 라이브 컬렉션에 적재된 chunk payload 샘플. 본 문서는 그 FLAT 구조를 시스템 관점에서 확정한다.
**코드 단일 출처:** `apps/search/doc_types.py`(doc_type/family/codec), `apps/domain/models.py`(`ChunkPayload`), `apps/search/schema_registry.py`(벡터/인덱스), `apps/core/config.py`(컬렉션·파이프라인)

---

## 0. 한 줄 요약

이전 모델은 **"연구자 1명 = 1 Point"** + nested 배열(`publications[]` 등) + 4브랜치 named vector였다. 현재는 **"chunk 1개 = 1 Point"** 다. 한 연구자는 여러 doc_type에 걸쳐 다수의 chunk(=다수 Point)로 존재하고, **연구자 후보는 검색 시점에 `researcher_id`로 집계**해서 만든다.

payload는 **FLAT**이다. 연구자 공통 메타는 payload **ROOT에 비정규화 반복** 저장되고, doc_type별 상세만 `doc_attrs{}`에 들어간다. 대표 일자는 단일 `doc_date`(문자열) 하나다.

> **적재 소유권:** 실제 적재는 **외부 제공자**가 소유한다. 구 적재 코드(`apps/ingest`)는 [`legacy_v1x/`](../../../legacy_v1x/)로 격리됐다. 본 API는 아래 FLAT 구조를 **읽기 계약**으로만 신뢰한다.

> **이 문서가 v2.0 초안을 대체하는 이유 (2026-06-02 결정):** 직전 v2.0 초안은 `researcher_meta` 중첩 / `domain_attrs` / `event_date`+`event_year` / `tags` / 11종 doc_type 등 **실데이터와 어긋난 설계안**을 담고 있었고, 이 불일치가 검색 0건의 근본 원인이었다. 본 v2.1은 실제 적재된 FLAT payload에 코드를 정렬한 결과를 정준으로 고정한다.

---

## 1. 컬렉션 (Collection)

| 항목 | 값 | 비고 |
|---|---|---|
| 컬렉션 이름 | `ntis_researcher_chunks` (기본값) | `NTIS_QDRANT_COLLECTION_NAME`으로 override. 구 `researcher_recommend_proto`는 레거시 v1.x(blue/green 가드로 보호, 절대 재생성 안 함) |
| Point 단위 | **chunk 1개 = Point 1개** | nested 배열 없음. 한 연구자 = 다수 Point |
| Point ID | `chunk_id` 문자열 그대로 (예: `paper_100000045256_c000`) | 결정론적 ID → upsert 멱등성 + evidence 참조 안정성 |
| Named vector | `vector_e5i` (dense) + `vector_splade` (sparse) **각 1개** | doc_type별 named vector를 두지 않는다. doc_type은 **payload 필터** |
| 거리 함수 | dense: **Cosine** (1024차원) | multilingual-e5-large-instruct 권장 |

### 1.1 Point ID를 `chunk_id`로 고정하는 이유

- **멱등 적재:** 같은 chunk를 다시 적재해도 같은 Point를 덮어쓴다(중복 Point 방지). 재적재·증분 적재가 안전하다.
- **evidence 참조 안정성:** LLM이 고른 근거를 `chunk_id`로 그대로 참조한다. 과거 `paper:0` 같은 **배열 인덱스 기반 id가 사라진다** — 인덱스는 직렬화 순서가 바뀌면 깨지지만 `chunk_id`는 불변이다. ([`api/REASONER_RUNTIME_POLICY.md`](../api/REASONER_RUNTIME_POLICY.md) 참조)
- **운영 추적성:** trace/로그에 찍힌 id를 Qdrant에서 바로 조회할 수 있다.

> **HARD 제약 — evidence 참조 id == `chunk_id`.** LLM은 후보를 재정렬/탈락/생성하지 않으며, 근거 인용 id는 반드시 컬렉션의 `chunk_id`여야 한다.

---

## 2. 벡터 레이아웃 (Vector Layout)

chunk 1개당 벡터는 **dense 1 + sparse 1** 뿐이다. 입력 텍스트는 둘 다 `chunk_text`(도메인별로 이미 목적에 맞게 직렬화된 본문)이다.

| 벡터명 | 종류 | 모델 | 입력 | 비고 |
|---|---|---|---|---|
| `vector_e5i` | Dense (1024, Cosine) | `multilingual-e5-large-instruct` | `chunk_text` | 쿼리 시 e5 instruct 프리픽스 사용. `NTIS_EMBEDDING_*` |
| `vector_splade` | Sparse | `PIXIE-Splade-v1.0` (로컬 우선) | `chunk_text` | SPLADE면 modifier 없음 / `Qdrant/bm25` fallback이면 modifier `IDF`. `NTIS_SPARSE_*` |

> **doc_type별 named vector를 두지 않는 이유:** 한 Point는 정확히 하나의 doc_type에 속한다. doc_type을 named vector로 만들면 "한 Point가 한 벡터만 채우고 나머지는 비는" 낭비 구조가 된다. doc_type은 **payload 필터**로 분기하는 것이 chunk 모델에 정합적이다. "컬렉션 분리 효과"는 named vector가 아니라 `doc_type` 인덱스 + 쿼리별 필터로 얻는다.

> **임베딩 정제 원칙 유지(HARD):** `chunk_text`는 적재 시점에 순수 도메인 텍스트로 직렬화되어야 한다. "평가위원/추천" 같은 요청 어투(role/action 불용어)는 쿼리 측에서도 임베딩 텍스트에 넣지 않는다(벡터 오염 방지). 이 원칙은 v1.x에서 그대로 계승된다.

---

## 3. FLAT payload 구조

payload는 **2개 영역**만 갖는다: (A) ROOT 평탄 필드(공통 식별 + 연구자 공통 메타), (B) `doc_attrs{}`(doc_type별 상세). **중첩 `researcher_meta` 없음**, **`domain_attrs` 명칭 폐기(`doc_attrs`)**, **`event_date`/`event_year`/`tags`/`chunk_text_len` 없음**.

코드 정준은 `apps/domain/models.py`의 `ChunkPayload`다(`extra="ignore"` — 알 수 없는 추가 키는 무시하여 스키마 진화에 견딘다).

### 3.1 ROOT · 공통 식별 (모든 chunk 동일)

| 필드 | 타입 | 인덱스 | 설명 |
|---|---|---|---|
| `researcher_id` | keyword | ✅ keyword | 연구자 식별자. **집계·dedupe·후보 단위의 기준 키** (예: `M1001176`) |
| `researcher_name` | keyword | — | 성명 (동점 정렬 보조) |
| `doc_type` | keyword | ✅ keyword | **5종 중 하나**. 검색 분기·faceting의 핵심 (§4) |
| `doc_id` | keyword | — | 문서 식별자 `<doc_type>_<숫자doc_id>` (예: `paper_100000045256`) |
| `chunk_id` | keyword | (= Point ID) | chunk 식별자. evidence 참조 단위 (예: `paper_100000045256_c000`) |
| `chunk_text` | text | (선택) full-text | 임베딩 입력 원문. 운영 full-text 매칭이 필요하면 text index |
| `doc_date` | string | ✅ datetime | **도메인 통합 단일 대표 일자.** 결측은 `"NONE"`(또는 빈값)일 수 있다. recency 필터의 유일 기준 (§3.4) |

### 3.2 ROOT · 연구자 공통 메타 (비정규화 반복 저장)

연구자 단위 누적 메타가 **그 연구자의 모든 chunk ROOT에 동일하게 반복** 저장된다. 별도 profile 조인 없이 chunk 단위에서 연구자 hard filter가 가능하다.

| 필드 | 타입 | 인덱스 | 용도 |
|---|---|---|---|
| `affiliated_organization` | keyword | ✅ | 소속 — include/제외 기관 필터 |
| `highest_degree` | keyword | ✅ | 학위 hard filter |
| `publication_count` | integer | ✅ | 최소 논문 수 필터 |
| `scie_publication_count` | integer | ✅ | 최소 SCIE 수 필터 |
| `intellectual_property_count` | integer | ✅ | 최소 특허 수 필터 |
| `research_project_count` | integer | ✅ | 최소 과제 수 필터 |
| `researcher_assessor_activity_count` | integer | ✅ | **평가위원 활동량** 필터/신호 (단일 병합 카운트) |

> **assessor count는 단일 필드다.** 구 초안의 `researcher_assessor_count` / `expert_assessor_count` 2분할은 실데이터에 없다. 실데이터는 **`researcher_assessor_activity_count` 하나**로 병합되어 있고, 필터 계층(`apps/search/filters.py`)은 구 split 키들을 이 단일 필드로 흡수(하위호환)한다.

> **비정규화 주의:** 같은 연구자의 모든 chunk에 동일 값이 반복 저장된다. 대가로 **적재 시 값 일관성을 외부 적재 제공자가 보장**해야 한다(한 연구자의 chunk마다 meta가 어긋나면 필터가 비결정적이 됨). 운영 표본 점검은 [`operation/RUNBOOK.md`](../operation/RUNBOOK.md) 참조.

### 3.3 `doc_attrs{}` · doc_type별 상세

`doc_attrs`는 doc_type별로 키가 다른 단일 객체다. **paper/project/patent는 키가 알려져 있고**, **assessor_activity/specialty는 키가 미상**이다.

**paper** (known)

| 키 | 비고 |
|---|---|
| `indexing_database` | 등재 DB(예: `SCIE`, `SCOPUS`, `NONE`). hard_filter `journal_class`가 이 키로 매핑됨 |
| `journal_name` | 학술지명 |
| `keywords` | 키워드(문자열, `;` 구분일 수 있음) |
| `main_language_title` / `sub_language_title` | 논문명(주/부 언어) |
| `is_scie` | `Y`/`N` |
| `publication_year_month` | `YYYY-MM` (대표 일자는 ROOT `doc_date`) |

**project** (known)

| 키 | 비고 |
|---|---|
| `project_title_korean` / `project_title_english` | 과제명(국/영) |
| `project_period` | `YYYY-MM-DD ~ YYYY-MM-DD` 문자열 |
| `performing_organization` | 수행기관 — 교차-chunk 제외 기관 필터 후보 |
| `managing_agency` | 관리(전문)기관 — 교차-chunk 제외 기관 필터 후보 |

**patent** (known)

| 키 | 비고 |
|---|---|
| `intellectual_property_title` | 지식재산권명 |
| `intellectual_property_type` | 권리 유형(예: `특허권`) |
| `application_registration_type` | 출원/등록 구분(예: `등록`) |
| `application_country` | 출원국(예: `대한민국`) |
| `application_number` | 출원번호 |
| `application_date` | 출원일(`YYYY-MM-DD`) |
| `intellectual_property_foreign_type` | 해외출원 구분(예: `해외미출원(국내)`) |

**assessor_activity / specialty** (UNKNOWN — passthrough)

> 이 두 doc_type의 `doc_attrs` 키 구성은 **현재 미상**이다. 본 API는 이를 **타입 없는 passthrough**로 취급한다:
> - 스키마로 키를 강제하지 않는다(추측 키를 만들지 않는다).
> - 필터/인덱스 대상으로 삼지 않는다(§5).
> - title/date 파생은 best-effort이고, 1차 표시는 `chunk_text`를 사용한다(`apps/domain/chunk_view.py`).
>
> 실제 키가 확정되면 본 문서와 `chunk_view._TITLE_KEYS`/`_DATE_KEYS`를 함께 갱신한다.

### 3.4 단일 `doc_date`와 recency

v1.x는 `art_recent_years`/`pat_recent_years`/`pjt_recent_years`로 도메인마다 따로 날짜를 다뤘다. 현재는 **모든 doc_type이 ROOT `doc_date` 하나**를 공유한다.

- `doc_date`는 **문자열**이며 Qdrant에 **datetime 인덱스**로 색인된다. `"NONE"`/결측은 datetime range에 매칭되지 않으므로 자연히 recency 대상에서 빠진다.
- "최근 N년"은 `doc_date >= (올해-N)-01-01` 로 표현된다.
- **여러 doc_type에 대한 recency는 반드시 OR(`min_should`, `min_count=1`)로 결합한다(HARD).** AND로 묶으면 0건이 된다(v1.x 장애에서 확인된 교훈). 구현은 `apps/search/filters.py`의 `QdrantFilterCompiler`.

---

## 4. doc_type 분류 체계 (Taxonomy)

doc_type은 **정확히 5종**이며, 검색·집계·evidence에서 일관되게 다루기 위해 **4개 기능군(family)** 으로 묶는다. 단일 출처는 `apps/search/doc_types.py`.

| doc_type | family | 분포(참고) | 추천에서의 역할 |
|---|---|---|---|
| `paper` | achievement | 64.6% | 논문 — 주제 적합성 핵심 근거 |
| `patent` | achievement | 18.7% | 특허 — 주제 적합성 핵심 근거 |
| `project` | achievement | 6.5% | 과제 — 주제 적합성 핵심 근거 |
| `assessor_activity` | assessment | 9.7% | **평가위원 활동** — 평가위원 적합성의 직접 신호 |
| `specialty` | expertise | 0.5% | 전문분야 선언 — 도메인 정합성 보강 |

| family | 구성 doc_type | 비고 |
|---|---|---|
| **achievement** | `paper`, `patent`, `project` | 성과 근거 |
| **assessment** | `assessor_activity` | 평가이력 — 1급 근거 |
| **expertise** | `specialty` | 전문성 선언 |
| **identity** | (전용 doc_type 없음) | **합성(synthetic).** 연구자 신원/누적 실적은 모든 chunk의 ROOT 평탄 필드(§3.1·3.2)에서 구성된다. 합성 profile evidence가 identity family를 차지한다 |

> **identity는 doc_type이 아니다.** 신원 앵커(소속·학위·누적 실적)는 별도 `profile` doc_type이 아니라 모든 chunk의 ROOT 메타에서 합성된다. evidence/카드의 `profile` 항목은 이 합성 식별 근거다(`EvidenceItem.type`에 `profile`이 포함되는 이유).

> **family별 evidence cap은 grounding 선별 한정이다 — 후보 순위에 영향 0.** 기본값: achievement 10 / assessment 6 / expertise 6 / identity 1 (`FAMILY_EVIDENCE_CAP`, `NTIS_EVIDENCE_FAMILY_CAP`).

---

## 5. payload 인덱스 (현행안)

코드 정준은 `apps/search/schema_registry.py`의 `PAYLOAD_INDEX_FIELDS`이고, 부트스트랩(`apps/search/qdrant_bootstrap.py`)이 이를 1:1로 생성한다.

| 인덱스 종류 | 대상 필드 |
|---|---|
| keyword | `researcher_id`, `doc_type`, `affiliated_organization`, `highest_degree`, `doc_attrs.is_scie`, `doc_attrs.indexing_database`, `doc_attrs.intellectual_property_type`, `doc_attrs.application_registration_type`, `doc_attrs.application_country`, `doc_attrs.performing_organization`, `doc_attrs.managing_agency` |
| integer | `publication_count`, `scie_publication_count`, `intellectual_property_count`, `research_project_count`, `researcher_assessor_activity_count` |
| datetime | `doc_date` (`"NONE"`/결측은 range 비매칭 = recency 자동 제외) |
| text (선택) | `chunk_text` — 운영 full-text 매칭이 필요할 때만 |

원칙(v1.x 계승):
- 필터에 자주 쓰는 필드는 반드시 인덱스 대상.
- exact match가 필요한 기관명/학위/구분값은 `keyword`.
- 날짜·집계값은 임베딩에 녹이지 말고 payload 필터로 처리.
- **assessor_activity/specialty의 `doc_attrs` 키는 미상이므로 인덱스 대상에서 제외**(passthrough).

---

## 6. 적재 정합성 체크리스트 (Read 계약 불변식)

외부 적재 제공자가 보장해야 하는 불변식. 위반 시 검색·필터가 비결정적이 된다.

1. **Point ID == `chunk_id`** 이고 컬렉션 전역에서 유일하다.
2. `chunk_id`는 코덱 `<doc_type>_<숫자doc_id>_c<NNN>`(3자리 zero-pad)을 따르고, `doc_id == <doc_type>_<숫자doc_id>`다.
3. 한 `researcher_id`의 모든 chunk에서 ROOT 공통 메타(`researcher_name`·소속·학위·count 5종)가 **동일**하다.
4. `doc_type`은 정의된 **5종** enum(`paper`/`patent`/`project`/`assessor_activity`/`specialty`) 중 하나다.
5. `doc_date`는 단일 대표 일자 문자열이며, 결측은 `"NONE"`(또는 빈값)으로 표현된다(별도 `event_year` 없음).
6. dense/sparse 벡터는 동일 `chunk_text`에서 생성된다(불일치 금지).
7. `chunk_text`에는 요청 어투(role/action 불용어)가 포함되지 않는다.

> 적재 코드는 본 repo 밖(외부 제공자) + 구 코드는 [`legacy_v1x/`](../../../legacy_v1x/)에 있다. 운영 컬렉션에서 위 불변식을 표본 점검하는 절차는 [`operation/RUNBOOK.md`](../operation/RUNBOOK.md)와 [`operation/GOLDEN_TESTS.md`](../operation/GOLDEN_TESTS.md)에 둔다.

---

## 7. v1.x → flat(v2.1) 매핑 (요약)

| v1.x (연구자=1 Point, nested, 4브랜치) | flat v2.1 (chunk=1 Point) |
|---|---|
| `basic` 브랜치 / `basic_vector_e5i` | doc_type 아님 — ROOT 메타에서 합성되는 **identity**(synthetic profile evidence), 단일 `vector_e5i` |
| `art` 브랜치 / `art_vector_*` | `paper` doc_type |
| `pat` 브랜치 / `pat_vector_*` | `patent` doc_type |
| `pjt` 브랜치 / `pjt_vector_*` | `project` doc_type |
| (없음) | `assessor_activity`(평가이력), `specialty`(전문분야) |
| 4브랜치 named vector | 단일 `vector_e5i`(dense) + `vector_splade`(sparse), doc_type은 payload 필터 |
| nested `publications[]` 등 | 개별 chunk Point들 |
| `researcher_meta{}` 중첩(구 초안) | ROOT 평탄 필드(§3.2) |
| `domain_attrs{}`(구 초안) | `doc_attrs{}` |
| `event_date` + `event_year`(구 초안) | 단일 `doc_date`(문자열, `"NONE"` 가능) |
| `tags` / `chunk_text_len`(구 초안) | 없음 |
| split `researcher_assessor_count` + `expert_assessor_count`(구 초안) | 단일 `researcher_assessor_activity_count` |
| `basic_info.researcher_id` (root) | `researcher_id` (ROOT, 모든 chunk) |
| evidence id `paper:0` | evidence id = `chunk_id` |

전체 단계별 전환 계획은 [`../plans/MIGRATION_PLAN.md`](../plans/MIGRATION_PLAN.md).

---

## 8. 보존된 HARD 제약 (요약)

추천 품질·재현성을 위해 아래 제약은 코드/문서 전반에서 변경 불가다.

- **equal RRF.** Qdrant 가중 RRF/score 가중합 금지. doc_type prior는 앱단 랭크 누적 가중일 뿐이며 기본값은 equal.
- **후보 cross-encoder 리랭커 기본 OFF.** 켜더라도 동점 밴드 내 재배열만 허용(탈락/생성 금지). 후보 순위 척추는 RRF 고정.
- **다중 doc_type recency는 OR(`min_should`).** AND 결합 금지(0건 회귀 방지).
- **LLM은 후보를 재정렬·탈락·생성하지 않는다.** 주어진 후보 집합 위에서 근거를 인용해 판단만 한다.
- **embedding 텍스트는 role/action 불용어를 배제한다.**
- **evidence 참조 id == `chunk_id`.**
