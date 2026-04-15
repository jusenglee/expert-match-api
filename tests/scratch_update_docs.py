import os

file_path = r"/apps/docs/SERVICE_FLOW.md"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Recency Bonus
content = content.replace(
    "이 단계가 필요한 이유는, LLM이 판단해야 할 핵심 근거만 선별해서 보여줘야 비교 품질이 좋아지기 때문이다.",
    "이 단계가 필요한 이유는, LLM이 판단해야 할 핵심 근거만 선별해서 보여줘야 비교 품질이 좋아지기 때문이다. 특히 최근 개편을 통해 **동적 최신성 가산점(Recency Bonus)** 시스템이 적용되어, 최근 실적(최근 5년 이내)을 보유한 전문가가 카드 숏리스트(Shortlist) 상단에 우선적으로 배치되도록 품질을 보정한다."
)

# 2. Map Batch Size
content = content.replace(
    "숏리스트 전체(기본 40명)를 `NTIS_LLM_JUDGE_BATCH_SIZE`(기본 10명) 단위로 분할한 뒤",
    "숏리스트 전체(기본 40명)를 `NTIS_LLM_JUDGE_BATCH_SIZE`(기본 20명) 단위로 분할한 뒤"
)

# 3. Reduce Round Details
content = content.replace(
    "- **전체 직렬화**: 후보의 논문 초록(150자 절단), 키워드, 과제 연구 요약 등 상세 정보를 포함한다.\n- **상세 프롬프트**: 추천 순위, 적합도, 추천 이유, 대표 근거, 리스크, 비선택 사유, 데이터 공백까지 모두 요구한다.\n- **max_tokens 제한 없음**: 충분한 분량의 상세 응답을 허용한다.",
    "- **전체 직렬화**: 후보의 논문 초록과 연구 요약(150자 → 80자 하드 리밋으로 초강력 다이어트 적용), 핵심 키워드(최대 3개 제한) 등 엄선된 상세 정보를 포함한다. 빈 값이거나 `None`인 데이터는 아예 JSON에서 삭제하여 컨텍스트 윈도우 점유율을 대폭 낮춘다.\n- **출력 토큰 제한(Clamp)**: 추천 사유(`reasons`)와 구체적 증거(`evidence`)의 출력 개수를 각각 \"최대 2개\"로 엄격히 통제하여 LLM의 응답 시간(Generation Latency) 지연을 원천 차단한다.\n- **최종 비교**: 줄어든 입력 토큰 덕분에 더 정확한 밀도로 추천 순위, 적합도, 추천 이유, 대표 근거, 리스크, 비선택 사유, 데이터 공백을 생성한다."
)

# 4. Planner Output Schema
content = content.replace(
    "(`intent_summary`, `core_keywords`, `hard_filters`, `exclude_orgs`, `soft_preferences`, `branch_weights`, `branch_query_hints`, `top_k`)를 예시와 함께 제시한다.",
    "(`intent_summary`, `core_keywords`, `hard_filters`, `exclude_orgs`, `soft_preferences`, `branch_query_hints`, `top_k`)를 예시와 함께 제시한다. (과거에 사용되던 `branch_weights` 필드는 가중치 배제 원칙에 따라 설계 구조에서 제거되었다.)"
)

# 5. Service Quality (exact match normalization)
content = content.replace(
    "- 기관명 exact 필드가 정규화되었는지",
    "- 기관명 exact 필드가 파이썬 후처리가 아닌 **시스템 사전 정규화 단계**(`normalize_org_name`을 통한 괄호 안 영문 약어 및 `주식회사` 등 제거)를 거쳐 Qdrant에 올바르게 매칭되는지"
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete for SERVICE_FLOW.md")
