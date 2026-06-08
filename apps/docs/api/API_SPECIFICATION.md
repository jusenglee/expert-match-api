# NTIS Person API 명세서 (flat chunk 모델)

**문서 버전:** v2.1 (flat payload 정합, 2026-06-02)

본 문서는 `Ntis_person_API`의 외부 노출 API 입출력 규격을 정의한다. 데이터 모델은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md), 내부 계약은 [`DATA_CONTRACT.md`](DATA_CONTRACT.md), breaking change는 [`EXTERNAL_API_CHANGELOG.md`](EXTERNAL_API_CHANGELOG.md).

## 1. 개요
자연어 질의를 LLM으로 분석하고, flat chunk 단위 하이브리드 검색 후 `researcher_id`로 집계하여 평가위원 후보를 만들고, chunk 근거로 LLM이 최종 추천·사유를 생성하는 백엔드 API.

> **검색 대상 = doc_type 5종.** `paper` / `patent` / `project` / `assessor_activity` / `specialty`. (구 11종 분기 폐기)
> 적재 단위는 `1 chunk = 1 Point`이며 payload는 flat이다(연구자 공통 메타는 root 비정규화, doc_type별 상세는 `doc_attrs{}`). 적재는 외부 제공자 소관이다.

---

## 2. 주요 API 명세

### 1) 전문가 추천 — `POST /recommend`

자연어 질의로 의도를 분석하고 필터·쿼리를 생성해 후보군을 검색·집계한 뒤 최종 추천 명단을 반환한다.

> [!IMPORTANT]
> 검색 결과가 없어도 `404`/예외가 아니라 `200 OK` + `recommendations=[]`를 반환한다.

#### 요청
```json
{
  "query": "인공지능 및 반도체 분야 박사급 평가위원 추천",
  "top_k": 5,
  "filters_override": { "highest_degree": "박사" },
  "exclude_orgs": ["A대학교", "B연구소"]
}
```
- `query` (string, 필수): 자연어 질의. 여러 줄은 `, `로 합쳐 단일 질의로 정규화.
- `top_k` (integer, 선택): 반환 최대 인원 (1~15). 미지정 시 planner `top_k`를 따르되 런타임은 최대 15명으로 clamp한다.
- `filters_override` (dict, 선택): hard filter 강제 지정(허용 키는 [`DATA_CONTRACT.md §1.1`](DATA_CONTRACT.md)).
- `include_orgs` / `exclude_orgs` (list[string], 선택): 포함/배제 기관(root `affiliated_organization` 기준).

#### 응답 (`200 OK`)
```json
{
  "intent_summary": "AI·반도체 분야 박사 학위 평가위원",
  "applied_filters": { "highest_degree": "박사" },
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"],
  "retrieved_count": 27,
  "recommendations": [
    {
      "rank": 1,
      "expert_id": "M1006328",
      "name": "홍길동",
      "fit": "높음",
      "recommendation_reason": "AI 반도체 국책 과제 수행과 관련 평가위원 활동 이력이 있습니다.",
      "match_badges": ["AI 충족", "반도체 충족", "직접 수행 근거 있음"],
      "match_summary": "AI, 반도체 근거가 확인되었습니다.",
      "match_details": {
        "matched_concepts": ["ai", "semiconductor"],
        "missing_concepts": [],
        "coverage_type": "joint",
        "matched_doc_types": ["project"],
        "direct_evidence_count": 1
      },
      "score_explanation": {
        "final_score": 0.77,
        "rank_score": 95.8,
        "score_breakdown": { "joint": 0.5, "concept": 0.2 },
        "top_chunks": [
          {
            "doc_type": "project",
            "title": "차세대 지능형 반도체 설계",
            "concepts": ["ai", "semiconductor"],
            "sources": ["dense_full", "concept:semiconductor"],
            "score": 0.42
          }
        ]
      },
      "evidence_summary": {
        "total_profile_counts": { "article_cnt": 15, "scie_cnt": 5, "patent_cnt": 3, "project_cnt": 5, "assessor_cnt": 5 },
        "matched_evidence_count": 1,
        "shown_evidence_count": 1
      },
      "evidence": [
        {
          "type": "project",
          "chunk_id": "project_100000031245_c000",
          "title": "차세대 지능형 반도체 설계",
          "date": "2023-01-01",
          "detail": "수행기관: 주식회사 미소테크 · 기간: 2021-01-01 ~ 2023-12-31",
          "snippet": "국문과제명: 차세대 지능형 반도체 설계 수행기관: 주식회사 미소테크 ..."
        }
      ],
      "risks": ["최근 논문 실적 다소 부족"],
      "rank_score": 95.8
    }
  ],
  "data_gaps": [],
  "not_selected_reasons": [],
  "trace": { "...": "디버깅 데이터" }
}
```

**응답 필드:**
- `intent_summary` (string): 추출된 의도 요약.
- `applied_filters` (dict): 실제 검색에 적용된 hard filter.
- `searched_branches` (list[string]): 검색 대상 doc_type 목록. 기본은 5종 전체(`paper`/`patent`/`project`/`assessor_activity`/`specialty`), 운영 화이트리스트(`NTIS_RETRIEVAL_DOC_TYPES`)로 축소 가능. *(필드명은 하위호환상 `searched_branches`로 유지되며, 값은 doc_type 문자열이다.)*
- `retrieved_count` (integer): 필터·집계 후 랭킹된 연구자 후보 수.
- `recommendations` (list[object]):
  - `rank` (integer): 1부터.
  - `expert_id` (string): `researcher_id`.
  - `name` (string): 성명.
  - `fit` (string): "높음" / "중간" / "보통".
  - `recommendation_reason` (string): LLM 생성 단일 사유.
  - `match_badges` (list[string]): UI 카드 조건 충족 배지. 기존 사유 필드를 대체하지 않는 보조 정보다.
  - `match_summary` (string): 카드 기본 영역에 쓰는 짧은 매칭 요약.
  - `match_details` (object): `matched_concepts`, `missing_concepts`, `coverage_type`(`joint`/`separate`/`partial`/빈 문자열), `matched_doc_types`, `direct_evidence_count`.
  - `score_explanation` (object): 검색 점수 상세 펼침용 `final_score`, `rank_score`, `score_breakdown`, `top_chunks`.
  - `evidence_summary` (object): 누적 실적(`total_profile_counts`)과 이번 질의 매칭 근거 수(`matched_evidence_count`), 화면 표시 근거 수(`shown_evidence_count`)를 분리한다.
  - `evidence` (list[object]): 추천 근거 chunk.
    - `type` (string): doc_type — `paper` / `patent` / `project` / `assessor_activity` / `specialty` 중 하나, 또는 합성 신원 근거 `profile`.
    - `chunk_id` (string): 근거 chunk 식별자(불변, evidence 참조 id). 형식 `<doc_type>_<숫자doc_id>_c<NNN>`(예: `paper_100000045256_c000`).
    - `title` (string): 근거 제목/내용(`doc_attrs`에서 파생).
    - `date` (string, 선택): chunk의 `doc_date`(결측 시 생략).
    - `detail` (string, 선택): 기관·구분 등 세부.
    - `snippet` (string, 선택): 근거 chunk 본문 요약. 플레이그라운드와 디버깅에서 제목만으로 부족한 수행 실적 맥락을 보강한다.
  - `risks` (list[string]): 약점·편중 등 잠재 리스크.
  - `rank_score` (float): RRF 집계 점수(0~100 정규화). 절대 적합도 아님.
- `data_gaps` (list[string]): 상위 추천자 공통 데이터 공백.
- `not_selected_reasons` (list[string]): 숏리스트에 올랐으나 제외된 사유.
- `trace` (dict): 디버깅용 추적 데이터. `top_k_used`는 실제 적용된 반환 상한이다. `retrieval_score_traces[*].matches[*]`는 RRF 기여점수(`contribution`), `chunk_id`, 제목, 날짜, snippet을 포함한다. `trace.strict_filter`는 concept gate 활성 여부, required concept, `relevance_concepts_missing`으로 제외된 후보와 미충족 concept를 기록한다.

`/recommend`의 기본 추천 목록에는 `missing_concepts`가 없는 후보만 포함한다. required concept를 일부만 충족한 후보는 운영/디버그 확인용으로 `trace.strict_filter.excluded_reasons`에 남는다.

---

### 2) 후보 목록 조회 — `POST /search/candidates`

추천(사유 생성) 없이 검색·집계·정렬된 후보 숏리스트를 반환한다. 사용자 노출 후보 수는 `/recommend`와 동일하게 최대 15명이다.

#### 요청
- `/recommend`와 동일 스키마.

#### 응답 (`200 OK`)
```json
{
  "intent_summary": "...",
  "applied_filters": {},
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"],
  "keywords": ["반도체", "인공지능"],
  "retrieved_count": 27,
  "candidates": [
    {
      "expert_id": "M1006328",
      "name": "홍길동",
      "organization": "주식회사 미소테크",
      "doc_types_present": ["paper", "project", "assessor_activity"],
      "counts": {
        "publication_count": 15,
        "scie_publication_count": 5,
        "intellectual_property_count": 3,
        "research_project_count": 5,
        "researcher_assessor_activity_count": 5
      },
      "data_gaps": [],
      "risks": [],
      "shortlist_score": 95.8
    }
  ],
  "trace": { "...": "디버깅 데이터" }
}
```

**`candidates[*]` 필드:**
- `expert_id` (string): `researcher_id`.
- `name` (string): 성명.
- `organization` (string|null): root `affiliated_organization`.
- `doc_types_present` (list[string]): 이번 검색에서 실제 hit한 chunk의 doc_type 목록. *(family 단위 boolean 플래그가 아니라 doc_type 문자열 목록이다.)*
- `counts` (dict): **flat root 기반** 연구자 누적 실적 수 — `publication_count` / `scie_publication_count` / `intellectual_property_count` / `research_project_count` / `researcher_assessor_activity_count`. (평가위원 활동은 단일 카운트로 병합)
- `data_gaps`, `risks` (list[string]).
- `shortlist_score` (float): RRF 집계 정규화 점수(0~100).

추가로 `intent_summary`, `applied_filters`, `searched_branches`, `keywords`(검색에 사용된 핵심 키워드), `retrieved_count`, `trace`를 최상위에 반환한다. `/search/candidates.trace.top_k_used`는 실제 적용된 후보 반환 상한이다. `/search/candidates.trace.strict_filter`는 `/recommend`와 동일한 strict-filter 요약 구조를 제공한다.

---

### 3) 피드백 — `POST /feedback`
```json
{
  "query": "인공지능 및 반도체 분야 평가위원 추천",
  "selected_expert_ids": ["M1006328"],
  "rejected_expert_ids": ["M1009999"],
  "notes": "적합한 평가위원이 배정됨"
}
```
응답: `{ "feedback_id": 1, "stored": true }`

---

### 4) 준비 상태 — `GET /health/ready`
중요 의존성(LLM, 임베딩, Qdrant 컬렉션/벡터/인덱스) 심층 점검. 일부 실패 시 `503`, 본문은 동일 구조.
```json
{
  "ready": true,
  "checks": {
    "llm_backend": true,
    "embedding_backend": true,
    "qdrant_collection_exists": true,
    "vectors_present": true,
    "payload_indexes_present": true
  },
  "issues": [],
  "collection_name": "ntis_researcher_chunks",
  "sample_point_id": "11008395"
}
```

`sample_point_id`는 Qdrant의 실제 Point ID 표본이다. 운영 컬렉션에서는 UUID/숫자형 문자열일 수 있으며, evidence 식별자는 payload root의 `chunk_id`를 기준으로 한다.

### 5) 헬스체크 — `GET /health`
앱 생존만 반환(의존성 미검증).
```json
{
  "status": "ok",
  "collection_name": "ntis_researcher_chunks",
  "searched_branches": ["paper", "patent", "project", "assessor_activity", "specialty"]
}
```

## 3. 에러 처리
- 입력 범위 초과: `422 Unprocessable Entity`
- 백엔드 타임아웃/초기화 미완료: `503 Service Unavailable`
- 일반 오류: `500 Internal Server Error`
- 공통 포맷: `{"detail": "오류 사유"}`
