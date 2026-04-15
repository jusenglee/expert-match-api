# NTIS Person API 명세서

본 문서는 `Ntis_person_API` 프로젝트의 주요 API 기능 및 입출력 규격을 정의합니다.

## 1. 개요
NTIS 전문가 추천 시스템은 자연어 질의를 바탕으로 LLM 기반 의도 분석, 벡터 데이터베이스 기반 하이브리드 검색, 전문가 적합성을 검증하는 지능형 백엔드 API를 제공합니다.

---

## 2. 주요 API 명세

### 1) 전문가 추천 (Recommend)
**`POST /recommend`**  
자연어 질의를 입력받아 LLM(또는 규칙 기반)으로 의도를 분석하고, 필터와 최적 쿼리를 생성해 관련 후보군을 검색한 뒤 최종 추천 전문가 명단을 반환합니다.

> [!IMPORTANT]
> 검색된 결과가 없을 경우 `404`나 예외를 던지지 않고, `200 OK`와 함께 `recommendations=[]` 를 응답 본문에 반환합니다.

#### 요청 (Request payload)
```json
{
  "query": "인공지능 및 반도체 분야 대학 교수 추천",
  "top_k": 5, 
  "filters_override": {
    "degree_slct_nm": "박사"
  },
  "exclude_orgs": ["A대학교", "B연구소"]
}
```
- **`query`** (string, 필수): 자연어 검색 질의
- **`top_k`** (integer, 선택): 추천 받을 최대 인원 수 (1 ~ 5)
- **`filters_override`** (dict, 선택): 검색 필터 시스템 강제 지정
- **`exclude_orgs`** (list[string], 선택): 추천 목록에서 배제할 기관

#### 응답 (Response payload) 
- HTTP Status: `200 OK`
```json
{
  "intent_summary": "AI 및 반도체 분야 전공 박사 학위자",
  "applied_filters": {"degree_slct_nm": "박사"},
  "searched_branches": ["basic", "art", "pat", "pjt"],
  "retrieved_count": 27,
  "recommendations": [
    {
      "rank": 1,
      "expert_id": "12345678",
      "name": "홍길동",
      "fit": "높음",
      "recommendation_reason": "AI 반도체 분야 국책 과제 수행 이력이 있습니다.",
      "reasons": ["AI 반도체 분야 국책 과제 수행 이력이 있습니다."],
      "evidence": [
        {"type": "project", "title": "차세대 지능형 반도체 설계", "date": "2023-01-01", "detail": "주관연구책임자"}
      ],
      "risks": ["최근 논문 실적 다소 부족"],
      "rank_score": 95.8
    }
  ],
  "data_gaps": [],
  "not_selected_reasons": [],
  "trace": { /* 디버깅 데이터 */ }
}
```

**응답 필드 세부 명세:**
- **`intent_summary`** (string): 질의에서 추출된 의도 요약
- **`applied_filters`** (dict): 실제 데이터베이스 검색에 적용된 제한 필터
- **`searched_branches`** (list[string]): 검색 대상 코퍼스 영역
- **`retrieved_count`** (integer): 필터링을 거쳐 랭킹된 전체 후보의 수
- **`recommendations`** (list[object]): 최종 추천된 전문가들의 상세 정보 객체 배열
  - `rank` (integer): 추천 순위, 1부터 시작
  - `expert_id` (string): 전문가 고유 식별자(ID)
  - `name` (string): 전문가 성명
  - `fit` (string): 질의 대비 적합도 ("높음", "중간", "보통")
  - `recommendation_reason` (string): LLM이 생성한 단일 추천 사유
  - `reasons` (list[string]): 하위호환용 별칭. 현재는 `recommendation_reason` 1개를 배열로 노출
  - `evidence` (list[object]): 추천 사유를 뒷받침하기 위해 자동 추출된 대표 실적 증빙 자료 모음
    - `type` ("paper", "patent", "project", "profile"): 실적 분류
    - `title` (string): 실적 및 프로필 데이터의 내용 또는 제목
    - `date` (string, 선택): 실적의 발행/등록/수행 시작 날짜
    - `detail` (string, 선택): 세부 역할(예: 주관연구책임자, SCIE 논문 등) 정보
  - `risks` (list[string]): 추천은 되었으나 이 데이터에서 고려해야 할 약점이나 편중 사항 등 잠재 리스크
  - `rank_score` (float): RRF 랭킹 알고리즘에 따른 최종 검색 점수(0~100 정규화)
- **`data_gaps`** (list[string]): 최상위 추천자들이 공통으로 부족한 데이터 항목 요약
- **`not_selected_reasons`** (list[string]): 초기 후보군(숏리스트)에 올랐으나 최종 추천 명단에서 제외된 사유 요약
- **`trace`** (dict): API 디버깅 및 분석용 과정 추적 데이터 모음

---

### 2) 후보자 목록 조회 (Search Candidates)
**`POST /search/candidates`**  
추천(심사) 알고리즘 과정을 거치지 않고, 검색 엔진(Qdrant)에서 RRF 기반으로 랭킹이 매겨진 숏리스트(초기 후보군) 전체를 조회합니다.

#### 요청 (Request payload)
- 💡 `/recommend` 와 완벽히 동일 (RecommendationRequest 스키마 구조 사용)

#### 응답 (Response payload) 
- HTTP Status: `200 OK`
```json
{
  "intent_summary": "...",
  "applied_filters": {},
  "retrieved_count": 27,
  "candidates": [
    {
      "expert_id": "12345678",
      "name": "홍길동",
      "organization": "A대학교",
      "branch_coverage": {"art": true, "pat": false, "pjt": true},
      "counts": {"article_cnt": 10, "scie_cnt": 3, "patent_cnt": 0, "project_cnt": 4},
      "data_gaps": ["특허 실적 없음"],
      "risks": [],
      "shortlist_score": 95.8
    }
  ]
}
```

**응답 필드 세부 명세 (`intent_summary`, `applied_filters`, `retrieved_count` 등 공통 필드는 `/recommend`와 동일합니다):**
- **`candidates`** (list[object]): 심사 이전 단계에서 랭킹이 매겨진 전문가 초기 후보군(숏리스트) 객체 배열
  - `expert_id` (string): 전문가 고유 식별자(ID)
  - `name` (string): 전문가 성명
  - `organization` (string, 선택): 소속 기관 이름
  - `branch_coverage` (dict): 논문("art"), 특허("pat"), 과제("pjt") 등 어떤 영역의 실적 데이터를 보유하고 있는지에 대한 불리언 플래그맵
  - `counts` (dict): 해당 전문가의 원시 데이터상 누적 실적 개수 (`article_cnt`, `scie_cnt`, `patent_cnt`, `project_cnt` 등)
  - `data_gaps` (list[string]): 필수 기준 대비 누락된 데이터가 있는지에 대한 알림 (예: 특허 실적 없음)
  - `risks` (list[string]): 데이터 편향이나 평가에 지장을 줄 수 있는 초기 위험 요소 알림
  - `shortlist_score` (float): 필터와 랭킹을 거친 후 부여받은 1차 후보 초기 점수 (RRF 정규화 점수, 0~100)

---

### 3) 사용자 피드백 제출 (Feedback)
**`POST /feedback`**  
사용자가 프론트엔드에서 어떤 전문가를 선택하고 제외했는지를 받아 시스템 운영 데이터베이스 테이블에 저장합니다. 모델 개선의 핵심이 됩니다.

#### 요청 (Request payload)
```json
{
  "query": "인공지능 및 반도체 분야 대학 교수 추천",
  "selected_expert_ids": ["12345678", "87654321"],
  "rejected_expert_ids": ["11112222"],
  "notes": "요청에 부합하는 적당한 전문가가 배정됨"
}
```

#### 응답 (Response payload)
- HTTP Status: `200 OK`
```json
{
  "feedback_id": 1,
  "stored": true
}
```

---

### 4) 서비스 준비 상태 확인 (Readiness)
**`GET /health/ready`**  
LLM 인퍼런스 서버, Qdrant 벡터 검색 엔진 등 중요 의존성 백엔드가 모두 연결되었는지 전체 심층 건강 상태(Health Check)를 검사합니다.

> [!WARNING]
> K8s 등의 Readiness Probe로 활용 시 유용합니다. 일부라도 실패 시 `503`을 반환하지만, 구조화된 동일한 JSON 본문을 제공합니다.

#### 응답 (Response payload)
- HTTP Status: `200 OK` 또는 `503 Service Unavailable`
```json
{
  "ready": true,
  "checks": {
    "llm_backend": true,
    "embedding_backend": true,
    "qdrant_collection_exists": true
  },
  "issues": [],
  "collection_name": "researcher_recommend_proto",
  "sample_point_id": "a1b2c3d4-..."
}
```

---

### 5) 웹서버 헬스체크 (Health)
**`GET /health`**  
애플리케이션 자체의 L4/L7 생존 여부만을 반환합니다. 의존성 서버 검증은 포함되지 않습니다.

#### 응답 (Response payload)
- HTTP Status: `200 OK`
```json
{
  "status": "ok",
  "collection_name": "researcher_recommend_proto",
  "searched_branches": ["basic", "art", "pat", "pjt"]
}
```

## 3. 에러 처리 규약
- 입력 데이터가 허용 범위를 넘었을 경우: `422 Unprocessable Entity`
- 내부 백엔드 타임아웃 및 초기화 미완료: `503 Service Unavailable`
- 일반 로직 오류: `500 Internal Server Error` 
  - (공통적으로 `{"detail": "오류 사유"}` 포맷 사용)
