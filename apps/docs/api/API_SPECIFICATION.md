# NTIS Person API 명세서 (chunk 재설계)

**문서 버전:** v2.0 (2026-05-28)

본 문서는 `Ntis_person_API`의 외부 노출 API 입출력 규격을 정의한다. 데이터 모델은 [`../architecture/DATA_MODEL.md`](../architecture/DATA_MODEL.md), 내부 계약은 [`DATA_CONTRACT.md`](DATA_CONTRACT.md), breaking change는 [`EXTERNAL_API_CHANGELOG.md`](EXTERNAL_API_CHANGELOG.md).

## 1. 개요
자연어 질의를 LLM으로 분석하고, chunk 단위 하이브리드 검색 후 `researcher_id`로 집계하여 평가위원 후보를 만들고, chunk 근거로 LLM이 최종 추천·사유를 생성하는 백엔드 API.

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
- `top_k` (integer, 선택): 추천 최대 인원 (1~15).
- `filters_override` (dict, 선택): hard filter 강제 지정(허용 키는 [`DATA_CONTRACT.md §1.1`](DATA_CONTRACT.md)).
- `include_orgs` / `exclude_orgs` (list[string], 선택): 포함/배제 기관.

#### 응답 (`200 OK`)
```json
{
  "intent_summary": "AI·반도체 분야 박사 학위 평가위원",
  "applied_filters": { "highest_degree": "박사" },
  "searched_doc_types": ["profile", "publication", "intellectual_property", "research_project", "researcher_assessor", "expert_assessor", "researcher_tech", "expert_tech", "researcher_core", "researcher_major", "expert_specific"],
  "retrieved_count": 27,
  "recommendations": [
    {
      "rank": 1,
      "expert_id": "M1006328",
      "name": "홍길동",
      "fit": "높음",
      "recommendation_reason": "AI 반도체 국책 과제 수행과 관련 평가위원 활동 이력이 있습니다.",
      "evidence": [
        {
          "type": "research_project",
          "chunk_id": "PJT_M1006328_0001_c0",
          "title": "차세대 지능형 반도체 설계",
          "date": "2023-01-01",
          "detail": "수행기관: 주식회사 미소테크"
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
- `searched_doc_types` (list[string]): 검색 대상 doc_type 목록. *(v1.x `searched_branches` 대체)*
- `retrieved_count` (integer): 필터·집계 후 랭킹된 연구자 후보 수.
- `recommendations` (list[object]):
  - `rank` (integer): 1부터.
  - `expert_id` (string): `researcher_id`.
  - `name` (string): 성명.
  - `fit` (string): "높음" / "중간" / "보통".
  - `recommendation_reason` (string): LLM 생성 단일 사유.
  - `evidence` (list[object]): 추천 근거 chunk.
    - `type` (string): doc_type (예: `publication`, `research_project`, `researcher_assessor`, `profile` …).
    - `chunk_id` (string): 근거 chunk 식별자(불변). *(v2.0 신규)*
    - `title` (string): 근거 제목/내용.
    - `date` (string, 선택): `event_date`.
    - `detail` (string, 선택): 기관·구분 등 세부.
  - `risks` (list[string]): 약점·편중 등 잠재 리스크.
  - `rank_score` (float): RRF 집계 점수(0~100 정규화). 절대 적합도 아님.
- `data_gaps` (list[string]): 상위 추천자 공통 데이터 공백.
- `not_selected_reasons` (list[string]): 숏리스트에 올랐으나 제외된 사유.
- `trace` (dict): 디버깅용 추적 데이터.

---

### 2) 후보 목록 조회 — `POST /search/candidates`

추천(사유 생성) 없이 검색·집계·정렬된 숏리스트 전체를 반환한다.

#### 요청
- `/recommend`와 동일 스키마.

#### 응답 (`200 OK`)
```json
{
  "intent_summary": "...",
  "applied_filters": {},
  "searched_doc_types": ["profile", "publication", "..."],
  "retrieved_count": 27,
  "candidates": [
    {
      "expert_id": "M1006328",
      "name": "홍길동",
      "organization": "주식회사 미소테크",
      "doc_type_coverage": {
        "achievement": true,
        "assessment": true,
        "expertise": true,
        "identity": true
      },
      "counts": {
        "publication_count": 15,
        "scie_publication_count": 5,
        "intellectual_property_count": 3,
        "research_project_count": 5,
        "researcher_assessor_count": 5,
        "expert_assessor_count": 2
      },
      "matched_doc_types": ["publication", "research_project", "researcher_assessor"],
      "data_gaps": [],
      "risks": [],
      "shortlist_score": 95.8
    }
  ]
}
```

**`candidates[*]` 필드:**
- `expert_id`, `name`, `organization` (`researcher_meta.affiliated_organization`).
- `doc_type_coverage` (dict): family(identity/achievement/assessment/expertise)별 보유 여부. *(v1.x `branch_presence_flags` 대체)*
- `counts` (dict): `researcher_meta` 기반 누적 실적 수.
- `matched_doc_types` (list[string]): 이번 검색에서 실제 hit한 doc_type. *(v2.0 신규)*
- `data_gaps`, `risks` (list[string]).
- `shortlist_score` (float): RRF 집계 정규화 점수(0~100).

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
  "sample_point_id": "PUB_M1006328_0001_c0"
}
```

### 5) 헬스체크 — `GET /health`
앱 생존만 반환(의존성 미검증).
```json
{
  "status": "ok",
  "collection_name": "ntis_researcher_chunks",
  "searched_doc_types": ["profile", "publication", "..."]
}
```

## 3. 에러 처리
- 입력 범위 초과: `422 Unprocessable Entity`
- 백엔드 타임아웃/초기화 미완료: `503 Service Unavailable`
- 일반 오류: `500 Internal Server Error`
- 공통 포맷: `{"detail": "오류 사유"}`
