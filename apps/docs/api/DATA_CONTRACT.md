# 데이터 규약 (Data Contract)

## 플래너 출력 (Planner Output)

`PlannerOutput`은 플래너와 검색 엔진 사이의 구조화된 규약입니다.

```json
{
  "intent_summary": "드론 화재 진압 전문가 찾기",
  "hard_filters": {},
  "include_orgs": [],
  "exclude_orgs": [],
  "task_terms": ["전문가 추천"],
  "core_keywords": ["화재 진압", "드론"],
  "top_k": 5
}
```

**플래너 규칙:**
- 하나의 JSON 객체만 반환합니다.
- `core_keywords`는 검색에 안전한 도메인 명사 또는 명사구만 포함합니다.
- `task_terms`는 요청의 목적이나 액션 관련 용어만 포함합니다.
- 명시적인 요청 파라미터는 자연어 추출 결과보다 우선합니다.
- 출력값이 유효하지 않거나 키워드가 비어있으면 1회 재시도합니다.

## 쿼리 빌더 규약 (Query Builder Contract)

`QueryTextBuilder`는 플래너 출력을 바탕으로 단일 검색 쿼리를 생성합니다.

**현재 동작:**
- 원본 사용자 질의는 검색 텍스트로 사용하지 않습니다.
- `intent_summary` 및 `task_terms`는 검색 텍스트에서 제외됩니다.
- 검색 텍스트는 `core_keywords`를 줄바꿈(`\n`)으로 결합하여 생성합니다.
- 4개의 모든 브랜치(기본, 논문, 특허, 과제)는 동일한 검색 텍스트를 수신합니다.

## 검색 규약 (Retrieval Contract)

`QdrantHybridRetriever`는 항상 `basic`, `art`, `pat`, `pjt` 4개 브랜치를 검색합니다.

**브랜치별 동작:**
- Dense 검색: 브랜치별 밀집 벡터 공간에서 수행
- Sparse 검색: BM25 알고리즘을 사용하여 희소 벡터 공간에서 수행
- RRF 결합: 브랜치 내에서 Dense와 Sparse 결과를 통합
- 최상위 RRF: 모든 브랜치의 결과를 하나로 통합

## 후보자 카드 규약 (Candidate Card Contract)

`CandidateCard`는 내부 검색 결과 규약으로, `/search/candidates`와 `/recommend`에서 공통으로 사용됩니다.

**주요 특징:**
- 카드 순서는 검색 엔진의 정렬 순서를 엄격히 따릅니다.
- `rank_score`는 검색 점수를 정규화한 값입니다.
- `/recommend` 단계 이전에 각 후보자의 증빙 자료(논문, 특허, 과제)를 미리보기 형태로 포함합니다.

## 추천 사유 생성 규약 (Reason Generation Contract)

`OpenAICompatReasonGenerator`는 정렬된 상위 K명의 후보자 정보만 수신합니다.

**출력 스키마:**
```json
{
  "items": [
    {
      "expert_id": "11008395",
      "fit": "높음",
      "recommendation_reason": "화재 대응 관련 논문 및 과제 수행 이력이 다수 존재함.",
      "selected_evidence_ids": ["paper:0", "project:1"],
      "risks": []
    }
  ],
  "data_gaps": []
}
```

**생성 규칙:**
- LLM은 후보자의 순위를 변경하지 않습니다.
- LLM은 후보자를 누락시키지 않습니다.
- LLM은 새로운 전문가 ID를 임의로 생성하지 않습니다.
- 후보자별 증빙 자료는 LLM 전달 전 `core_keywords`를 기준으로 재정렬됩니다.
- LLM은 후보자당 최대 논문 4건, 과제 4건, 특허 4건의 증빙 후보를 수신합니다.
- 심사는 최대 5명 단위의 순차 배치(Batch)로 진행됩니다.
