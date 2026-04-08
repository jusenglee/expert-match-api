# 코드 기준 보강 다이어그램

## 1) 실제 구현 기준 전체 흐름

```mermaid
graph TD
    U[사용자 질의] --> API[FastAPI /recommend 또는 /search/candidates]
    API --> SVC[RecommendationService]

    SVC --> P[Planner]
    P --> P1[HeuristicPlanner 또는 OpenAICompatPlanner]
    P1 --> P2[PlannerOutput 생성<br/>intent_summary / hard_filters / exclude_orgs / branch_query_hints / top_k]

    P2 --> F[QdrantFilterCompiler]
    F --> F1[degree/count/recent/exclude_orgs 를 Qdrant Filter로 컴파일]

    P2 --> QB[QueryTextBuilder]
    QB --> QB1[basic/art/pat/pjt branch query 생성]

    F1 --> R[QdrantHybridRetriever]
    QB1 --> R

    subgraph Q[Qdrant Hybrid Retrieval]
        R --> D1[basic dense: basic_vector_e5i]
        R --> S1[basic sparse: basic_vector_bm25]
        D1 --> RRF1[branch 내부 RRF]
        S1 --> RRF1

        R --> D2[art dense: art_vector_e5i]
        R --> S2[art sparse: art_vector_bm25]
        D2 --> RRF2[branch 내부 RRF]
        S2 --> RRF2

        R --> D3[pat dense: pat_vector_e5i]
        R --> S3[pat sparse: pat_vector_bm25]
        D3 --> RRF3[branch 내부 RRF]
        S3 --> RRF3

        R --> D4[pjt dense: pjt_vector_e5i]
        R --> S4[pjt sparse: pjt_vector_bm25]
        D4 --> RRF4[branch 내부 RRF]
        S4 --> RRF4

        RRF1 --> RRF_TOP[최상위 branch 간 RRF]
        RRF2 --> RRF_TOP
        RRF3 --> RRF_TOP
        RRF4 --> RRF_TOP
    end

    RRF_TOP --> H[SearchHit 리스트]
    H --> H1[ExpertPayload 복원]
    H1 --> H2[branch_coverage는 실제 매치점수 분해가 아니라 payload 존재 여부로 계산]

    H2 --> C[CandidateCardBuilder]
    C --> C1[대표 논문/특허/과제 최근순 추출]
    C --> C2[shortlist_score 계산<br/>branch coverage + SCIE 수 + 과제 수 + 특허 수]
    C2 --> C3[카드 정렬 후 shortlist 상위 limit]

    C3 --> J{Judge 선택}
    J -->|문서 설명상| JL[OpenAICompatJudge]
    J -->|코드 기본값| JP[PythonSelector]
    J -->|테스트용| JH[HeuristicJudge]

    JP --> JP1[Qdrant raw_rank 기반 find_score]
    JP --> JP2[evidence bonus: 논문/특허/과제/SCIE/박사]
    JP --> JP3[동일 기관 중복 penalty -15]
    JP --> JP4[35점 미만 탈락]

    JL --> O1[JudgeOutput JSON]
    JP4 --> O1
    JH --> O1

    O1 --> RESP[최종 추천 응답]
```

## 2) 실제 점수/선발 흐름

```mermaid
graph LR
    A[Qdrant 최종 hit 순위] --> B[CandidateCard.raw_rank 저장]
    A --> C[CandidateCardBuilder가 shortlist_score 재계산]

    subgraph S1[카드 압축 단계]
        C --> C1[branch coverage 개수 x 10]
        C --> C2[SCIE count x 3]
        C --> C3[project count x 2]
        C --> C4[patent count x 2]
        C1 --> C5[shortlist_score]
        C2 --> C5
        C3 --> C5
        C4 --> C5
    end

    C5 --> D[shortlist 상위 N명]

    subgraph S2[PythonSelector 단계]
        D --> E[base_score = raw_rank percentile x 60]
        D --> F[evidence bonus<br/>논문 +8 / 과제 +8 / 특허 +5 / SCIE +4 / 박사 +3]
        D --> G[기관 중복 penalty -15]
        E --> H[final_score]
        F --> H
        G --> H
    end

    H --> I{35점 이상?}
    I -->|Yes| J[최종 추천 후보 채택]
    I -->|No| K[탈락]
    J --> L[대표 evidence 1건씩 구성]
    L --> M[RecommendationDecision 반환]
```

## 3) 코드 기준으로 반드시 고쳐야 할 설명 문구

- 현재 구현은 **Qdrant Multi-vector**가 아니라 **Qdrant Named Vector + Sparse Vector** 구조다.
- 현재 구현은 **Weighted Prefetch / Weighted RRF**를 사용하지 않는다.
- branch 중요도는 가중치로 주는 것이 아니라 **branch_query_hints 텍스트 보정**으로만 반영한다.
- branch별 매치 근거를 정교하게 추적하지 않고, `branch_coverage`는 현재 **payload 존재 여부**로만 계산한다.
- 최종 판정은 문서 표현과 달리 항상 Judge LLM이 아니다. 설정에 따라 **PythonSelector**가 기본 경로가 될 수 있다.
- 현재 shortlist 단계는 Qdrant score를 그대로 쓰지 않고, **counts 기반 재정렬**을 한 번 더 수행한다.

## 4) 코드 점검 중 발견된 즉시 수정 필요 항목

1. `RecommendationService.search_candidates()`는 `card_builder.build_small_cards(...)`를 호출하지만, 현재 `CandidateCardBuilder`에는 해당 메서드가 없다.
2. `Settings` 기본값은 `llm_backend=python_selector`인데, `strict_runtime_validation=True` 기본값은 `openai_compat`만 허용한다.
3. `HeuristicJudge`는 현재 도메인 모델과 맞지 않는 레거시 필드명(`paper_nm`, `jrnl_pub_dt`, `ipr_invention_nm` 등)을 참조한다.
4. `/recommend` 빈 결과를 200 + 빈 배열로 준다는 문서와 달리, 실제 서비스 코드는 추천 결과가 비면 예외를 발생시킨다.
5. `/health/ready` 실패 시 응답이 top-level body로 반환된다는 문서와 달리, 현재 코드는 `HTTPException(detail=...)` 경로를 사용한다.
