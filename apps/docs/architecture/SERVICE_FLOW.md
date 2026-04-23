# 서비스 동작 흐름 (Service Flow)

## 런타임 흐름 (Runtime Flow)

### 1. 플래너 (Planner)

`RecommendationService.search_candidates()`는 플래너 실행과 함께 시작됩니다.

**플래너의 역할:**
- 입력된 자연어 질의 정규화
- 순수 도메인 명사인 `core_keywords` 추출
- 요청의 목적(역할) 언어를 `task_terms`로 분리
- 명시적 필터, 포함/제외 기관, `top_k` 설정 보존

**플래너가 하지 않는 것:**
- 검색용 재작성 문장 생성
- 검색 뷰(View) 생성
- 브랜치 힌트 생성

### 2. 검색 및 추출 (Retrieval)

`QueryTextBuilder`는 1단계 sparse 키워드 검색에는 `retrieval_core`/`core_keywords` 기반 쿼리를, 2단계 hybrid 검색에는 `semantic_query` 기반 쿼리를 생성합니다.

`QdrantHybridRetriever`의 동작:
- 검색 모드는 `keyword_pool_then_hybrid`로 고정
- 1단계에서 각 브랜치/경로별 sparse 키워드 검색을 수행하고 `basic_info.researcher_id` 후보 풀 수집
- 2단계에서 후보 풀을 `basic_info.researcher_id MatchAny` 필터로 제한한 뒤 각 브랜치(기본, 논문, 특허, 과제)별 Dense + Sparse 검색 수행
- RRF(Reciprocal Rank Fusion) 알고리즘을 통해 브랜치 내 결과 통합
- 모든 브랜치의 결과를 다시 RRF로 최종 통합
- 각 결과 항목에 브랜치별 매칭 근거(`retrieval_score_traces`) 기록
- **결정론적 최종 정렬 적용:**
  1. 점수(Score) 내림차순
  2. 성명(Name) 오름차순 (점수 동점 시)
  3. 전문가 ID(Expert ID) 오름차순 (최종 순위 고정)

### 3. 후보자 반환 (Candidate Return)

`/search/candidates` 엔드포인트는 정렬된 후보자 목록을 즉시 반환합니다.

**동작 특징:**
- 요청 시 `top_k`가 명시된 경우 해당 수만큼 제한하여 반환
- 명시되지 않은 경우 전체 검색 결과 반환

### 4. 추천 결과 생성 (Recommendation Return)

`/recommend` 엔드포인트는 검색 과정을 거친 후 다음 단계를 추가로 수행합니다.

**추천 파이프라인:**
- 정렬된 상위 K명의 후보자 선택
- 플래너의 `core_keywords`를 기준으로 각 후보자의 내부 증빙 자료(논문, 과제, 특허) 재랭킹
- 후보자당 최대 논문 4건, 과제 4건, 특허 4건으로 구성된 **LLM 증빙 풀(Pool)** 구축
- 최대 5명의 후보자를 한 배치로 묶어 순차적으로 LLM에 전달
- LLM에 후보자의 기본 맥락(검색 근거, 평가 활동, 기술 분류 등)과 요약 정보 전달
- LLM으로부터 적합도(`fit`), 추천 사유(`recommendation_reason`), 선택된 증빙 ID를 수신
- **검색 시의 원본 순서 유지**
- LLM이 선택한 증빙 ID를 바탕으로 최종 `recommendation.evidence` 구성
- LLM이 응답을 누락하거나 사유가 없는 경우, 서버 측에서 증빙 기반의 보수적 사유(Fallback) 자동 생성

**LLM이 하지 않는 것:**
- 후보자 재정렬 (Reranking)
- 후보자 탈락 시키기 (Filtering)
- 새로운 전문가 ID 생성

### 5. 검색 결과 없음 또는 실패 처리

플래너가 재시도 후에도 빈 `core_keywords`를 반환하는 경우:
- 검색 단계를 건너跳
- `/search/candidates`는 빈 목록 반환
- `/recommend`는 구조화된 사유와 함께 빈 추천 목록 반환

## 추적 기록 (Trace Behavior)

현재 활성화된 추적 필드 목록:
- `planner`: 플래너 출력물
- `planner_trace`: 플래너 실행 상세
- `reason_generation_trace`: 추천 사유 생성 상세
- `raw_query`: 원본 사용자 질의
- `planner_keywords`: 플래너 추출 키워드
- `retrieval_keywords`: 실제 검색에 사용된 키워드
- `branch_queries`: 브랜치별 쿼리 내역
- `retrieval_score_traces`: 검색 점수 근거
- `query_payload.retrieval_mode`: 고정 2단계 검색 모드
- `query_payload.keyword_stage_candidate_count`: 1차 sparse 키워드 검색에서 수집한 후보 ID 수
- `query_payload.hybrid_stage_raw_branch_counts`: 2차 hybrid 검색의 branch/path별 raw hit 수
- `query_payload.aggregated_candidate_count`: 최종 support rule 적용 전 집계 후보 수
- `query_payload.support_pass_count` / `support_filtered_count`: support rule 통과/탈락 수
- `server_logs`: Trace ID로 캡처된 사용자 질의, 플래너, 1차 검색, 2차 검색 단계별 운영 로그
- `timers`: 구간별 실행 시간
